# Third‑party libraries
import equinox as eqx
import jax
import jax.numpy as jnp
import optax

from adaptkan.jax.layers import AdaptKANLayerJax
from adaptkan.jax.utils import copy_state, copy_state_subset

# Implement a custom vjp in the future if needed
def _call_impl(model: 'AdaptKANJax', x: jnp.ndarray, state: eqx.nn.State, update=True, add_counts=False):

    # This the main function used during training. Returns information about layers needed for the loss function
    # This is what other functions use to call the model

    layer_act_norms, layer_alphas, layer_indices, layer_outputs, layer_postsplines = [], [], [], [], []

    # During inference, we don't adapt or update the weight counts
    for idx, layer in enumerate(model.layers):

        # Define a custom derivative for this function
        # x, state, coefs, indices = layer(x, state)
        x, state, act_norms, alphas, indices, postsplines = layer(x, state, update, add_counts)

        layer_alphas.append(alphas)
        layer_indices.append(indices)
        layer_act_norms.append(act_norms)
        layer_outputs.append(x)
        layer_postsplines.append(postsplines)

    return {
        "x": x,
        "state": state,
        "layer_outputs": layer_outputs,
        "layer_act_norms": layer_act_norms,
        "layer_alphas": layer_alphas,
        "layer_indices": layer_indices,
        "layer_postsplines": layer_postsplines
    }

class AdaptKANJax(eqx.Module):

    width: list[int]
    layers: list[AdaptKANLayerJax]
    update_data_counts: bool
    prune_patience: int
    min_delta: float
    k: int

    def __init__(self,
                 width=None,
                 activation_noise=0.1, # Need to pass this to other networks
                 update_data_counts=True,
                 ema_alpha=0.01, # Lower this to give more weight to past data
                 prune_patience=1, # How many batches to wait before pruning
                 initialization_range=[-1, 1],
                 num_grid_intervals=5,  # Need to pass this to other networks. This can be a list or an int
                 activation_strategy='linear', # Either 'linear', 'zero', or 'kan'. Linear is simple and works well
                 min_delta=1e-4, # The minimium distance between the upper and lower bounds of the domain
                 rounding_precision_eps=0.0,
                 k=3, # Bspline order (ignored if using chebyshev)
                 stretch_mode="max", # Either "mean", "max", "half_max", "edge", or "relative"
                 stretch_threshold=None, # Used with the relative stretch mode
                 prune_mode="default", # Either "default" or "relative". Default works well
                 exact_refit=True, # Turning this to false sacrifices accuracy but is faster
                 basis_type="bspline", # Either "bspline" or "chebyshev"
                 constraints_in=None,  # Per-layer constraints: (n_layers, in_dim, n_constraints, 2) or broadcast (n_layers, n_constraints, 2)
                 constraints_y=None,   # Per-layer targets: (n_layers, out_dim, in_dim, n_constraints) or broadcast (n_layers, out_dim, n_constraints)
                 seed = 0):

        super(AdaptKANJax, self).__init__()

        depth = len(width) - 1

        self.width = width
        self.prune_patience = prune_patience
        self.min_delta = min_delta
        self.k = k

        # Whether or not we should save the data_counts (histograms) as we go
        self.update_data_counts = update_data_counts

        # Set the number of activation weights per layer
        if isinstance(num_grid_intervals, list):
            assert len(num_grid_intervals) == depth, "Number of weights per activation list length should match depth of the network."
            grid_intervals_list = num_grid_intervals
        else:
            grid_intervals_list = [num_grid_intervals for _ in range(depth)]

        # For now this will default to [-1, 1] because the network will adapt
        assert len(initialization_range) == 2, "Initialization range should be a list of 2 integers."
        assert initialization_range[1] - initialization_range[0] >= min_delta, "Second number of init range needs to be at least min_delta larger than first number"

        # Create the keys for our different layers
        key = jax.random.key(seed)
        keys = jax.random.split(key, depth)
        keys = [None for i in range(depth)]

        # Process constraints for each layer
        layer_constraints = self._process_constraints(constraints_in, constraints_y, width, depth)

        self.layers = []

        for i in range(depth):
            in_dim = width[i]
            out_dim = width[i+1]

            # Get constraints for this layer (or None)
            layer_cons_in, layer_cons_y = layer_constraints[i]

            layer = AdaptKANLayerJax(in_dim=in_dim,
                                    out_dim=out_dim,
                                    activation_noise=activation_noise,
                                    activation_strategy=activation_strategy,
                                    num_grid_intervals=grid_intervals_list[i],
                                    initialization_range=initialization_range,
                                    prune_patience=prune_patience,
                                    ema_alpha=ema_alpha,
                                    min_delta=min_delta,
                                    stretch_threshold=stretch_threshold,
                                    stretch_mode=stretch_mode,
                                    prune_mode=prune_mode,
                                    exact_refit=exact_refit,
                                    rounding_precision_eps=rounding_precision_eps,
                                    basis_type=basis_type,
                                    k=k,
                                    constraints_in=layer_cons_in,
                                    constraints_y=layer_cons_y,
                                    key=keys[i])
            self.layers.append(layer)

    def _process_constraints(self, constraints_in, constraints_y, width, depth):
        """
        Process constraint inputs into per-layer format.

        Args:
            constraints_in: List of arrays (or None) for each layer.
                - Each element: (n_constraints, 2) for broadcast, (in_dim, n_constraints, 2) for full, or None
                - Example: [jnp.array([[0,0], [1,0]]), None, jnp.array([[0,0]])]
            constraints_y: List of arrays (or None) for each layer.
                - Each element: (out_dim, n_constraints) for broadcast, (out_dim, in_dim, n_constraints) for full, or None
                - Example: [jnp.array([[0, 0]]), None, jnp.array([[0]])]

        Returns:
            List of (layer_constraints_in, layer_constraints_y) tuples, with None for unconstrained layers.
        """
        # No constraints at all
        if constraints_in is None and constraints_y is None:
            return [(None, None) for _ in range(depth)]

        # Must provide both or neither
        assert constraints_in is not None and constraints_y is not None, \
            "Must provide both constraints_in and constraints_y, or neither"

        assert len(constraints_in) == depth, \
            f"constraints_in list length must match depth ({depth}), got {len(constraints_in)}"
        assert len(constraints_y) == depth, \
            f"constraints_y list length must match depth ({depth}), got {len(constraints_y)}"

        layer_constraints = []
        for i in range(depth):
            cons_in = constraints_in[i]
            cons_y = constraints_y[i]

            # Skip this layer if either is None
            if cons_in is None or cons_y is None:
                assert cons_in is None and cons_y is None, \
                    f"Layer {i}: constraints_in and constraints_y must both be None or both be arrays"
                layer_constraints.append((None, None))
                continue

            in_dim = width[i]
            out_dim = width[i + 1]

            layer_cons_in = jnp.asarray(cons_in)
            layer_cons_y = jnp.asarray(cons_y)

            layer_cons_in, layer_cons_y = self._broadcast_layer_constraints(
                layer_cons_in, layer_cons_y, in_dim, out_dim, i
            )
            layer_constraints.append((layer_cons_in, layer_cons_y))

        return layer_constraints

    def _broadcast_layer_constraints(self, layer_cons_in, layer_cons_y, in_dim, out_dim, layer_idx):
        """
        Broadcast constraints to full shape for a single layer.

        Input formats:
        - layer_cons_in: (n_constraints, 2) for broadcast, or (in_dim, n_constraints, 2) for full
        - layer_cons_y: (out_dim, n_constraints) for broadcast, or (out_dim, in_dim, n_constraints) for full

        Output formats:
        - layer_cons_in: (in_dim, n_constraints, 2)
        - layer_cons_y: (out_dim, in_dim, n_constraints)
        """
        # Handle broadcast format for constraints_in
        # Broadcast: (n_constraints, 2) -> (in_dim, n_constraints, 2)
        if layer_cons_in.ndim == 2:
            layer_cons_in = jnp.broadcast_to(
                layer_cons_in[None, :, :],
                (in_dim, layer_cons_in.shape[0], 2)
            )

        # Handle broadcast format for constraints_y
        # Broadcast: (out_dim, n_constraints) -> (out_dim, in_dim, n_constraints)
        if layer_cons_y.ndim == 2:
            n_constraints = layer_cons_y.shape[1]
            layer_cons_y = jnp.broadcast_to(
                layer_cons_y[:, None, :],
                (out_dim, in_dim, n_constraints)
            )

        # Validate shapes
        assert layer_cons_in.shape == (in_dim, layer_cons_in.shape[1], 2), \
            f"Layer {layer_idx}: constraints_in shape mismatch. Expected (in_dim={in_dim}, n_constraints, 2), got {layer_cons_in.shape}"
        assert layer_cons_y.shape[0] == out_dim and layer_cons_y.shape[1] == in_dim, \
            f"Layer {layer_idx}: constraints_y shape mismatch. Expected (out_dim={out_dim}, in_dim={in_dim}, n_constraints), got {layer_cons_y.shape}"

        return layer_cons_in, layer_cons_y
    
    @eqx.filter_jit
    def adapt(self, state):
        # Adapt the model and return the new model and state
        layers = []
        conds = []
        for idx, layer in enumerate(self.layers):
            layer, state, adapted = layer.adapt(state)
            layers.append(layer)
            conds.append(adapted)

        model = eqx.tree_at(lambda m: m.layers, self, replace=layers)
        return model, state, jnp.any(jnp.stack(conds))

    @eqx.filter_jit
    def manual_adapt(self, state):
        # Adapt the model and return the new model and state
        layers = []
        conds = []
        for idx, layer in enumerate(self.layers):
            layer, state, adapted = layer.manual_adapt(state)
            layers.append(layer)
            conds.append(adapted)

        model = eqx.tree_at(lambda m: m.layers, self, replace=layers)
        return model, state, jnp.any(jnp.stack(conds))

    def get_activations(self, layer, x, state):
        # Like call layer but we don't sum the outputs
        clipped_x = jnp.clip(x, state.get(layer.a), state.get(layer.b))
        substate = state.substate([layer.a, layer.b, layer.last_alphas, layer.last_indices])
        return eqx.filter_vmap(layer.get_activations, in_axes=(0, None))(clipped_x, substate)
    
    def refine(self, state, new_num_grid_intervals):
        new_model, new_state = eqx.nn.make_with_state(AdaptKANJax)(width=self.width,
                                                   num_grid_intervals=new_num_grid_intervals, 
                                                   prune_patience=self.prune_patience,
                                                   k=self.k,
                                                   min_delta=self.min_delta,
                                                   activation_strategy=self.layers[0].activation_strategy,
                                                   activation_noise=self.layers[0].activation_noise,
                                                   ema_alpha=self.layers[0].ema_alpha,
                                                   stretch_mode=self.layers[0].stretch_mode,
                                                   prune_mode=self.layers[0].prune_mode)
        refined_layers = []
        for idx, (new_layer, layer) in enumerate(zip(new_model.layers, self.layers)):

            new_state = copy_state(new_layer, layer, new_state, state, exclude=[(new_layer.data_counts, state.get(layer.data_counts))])

            new_layer, new_state = new_layer.refine(layer.weights, state.get(layer.data_counts), new_state, new_num_grid_intervals)

            # NEW: Added in new pieces for KAN initialization
            if isinstance(new_layer.scale_base, jax.Array):
                new_layer = eqx.tree_at(lambda l: l.scale_base, new_layer, replace=layer.scale_base)

            if isinstance(new_layer.scale_sp, jax.Array):
                new_layer = eqx.tree_at(lambda l: l.scale_sp, new_layer, replace=layer.scale_sp)

            refined_layers.append(new_layer)

        model = eqx.tree_at(lambda m: m.layers, self, replace=refined_layers)
        return model, new_state
    
    def reset_data_counts(self, state):

        # Reset the histograms (out-of-domain and in domain)

        new_model, new_state = eqx.nn.make_with_state(AdaptKANJax)(width=self.width,
                                                   num_grid_intervals=[self.layers[i].num_grid_intervals for i in range(len(self.layers))], 
                                                   prune_patience=self.prune_patience,
                                                   k=self.k,
                                                   min_delta=self.min_delta)
        
        for idx, (new_layer, layer) in enumerate(zip(new_model.layers, self.layers)):

            new_state = copy_state(new_layer, layer, new_state, state, exclude=[(new_layer.data_counts,
                                                                                 state.get(layer.data_counts)),
                                                                                 (new_layer.ood_data_counts,
                                                                                 state.get(layer.ood_data_counts))])

        return new_state
            
    def __call__(self, x, state, update=True):

        out = _call_impl(self, x, state, update)

        return out["x"], out["state"]
    
    def call_with_details(self, x, state, update=True, add_counts=False): # call_w_act_norms

        out = _call_impl(self, x, state, update, add_counts)

        return out
    
    def prune(self, x, state, thresh=1e-2):
        # This prunes the nodes and edges of the network
        # Note that this is different from shrinking the domain

        out = self.call_with_details(x, state, update=False)

        # Added this
        prune_masks = []
        new_widths = [self.width[0]]

        # Implements auto-pruning from the original KAN by first masking out the nodes
        for i in range(len(out["layer_act_norms"])-1):
            max_in = out["layer_act_norms"][i].max(1)
            max_out = out["layer_act_norms"][i+1].max(0)
            prune_mask = (max_in > thresh) * (max_out > thresh)
            prune_masks.append(prune_mask)
            new_widths.append(prune_mask.sum().item())

        new_widths.append(self.width[-1])

        num_grid_intervals = [self.layers[i].num_grid_intervals for i in range(len(self.layers))]

        new_model, new_state = eqx.nn.make_with_state(AdaptKANJax)(width=new_widths,
                                                num_grid_intervals=num_grid_intervals, 
                                                prune_patience=self.prune_patience,
                                                k=self.k,
                                                min_delta=self.min_delta,
                                                activation_strategy=self.layers[0].activation_strategy,
                                                activation_noise=self.layers[0].activation_noise,
                                                ema_alpha=self.layers[0].ema_alpha,
                                                stretch_mode=self.layers[0].stretch_mode,
                                                prune_mode=self.layers[0].prune_mode)
        
        if 0 in new_model.width:
            return self, state
        
        # Replace the weights in the first layer
        idx = jnp.nonzero(prune_masks[0], size=new_widths[1])[0]
        new_state = copy_state(new_model.layers[0], self.layers[0], new_state, state)

        # Prune the outputs of the first layer
        new_layer = eqx.tree_at(lambda l: l.weights, new_model.layers[0], replace=jnp.take(self.layers[0].weights, idx, axis=0))
        
        # NEW: Added in new pieces for kan initialization
        if isinstance(new_layer.scale_base, jax.Array):
            new_layer = eqx.tree_at(lambda l: l.scale_base, new_layer, replace=jnp.take(self.layers[0].scale_base, idx, axis=1))

        if isinstance(new_layer.scale_sp, jax.Array):
            new_layer = eqx.tree_at(lambda l: l.scale_sp, new_layer, replace=jnp.take(self.layers[0].scale_sp, idx, axis=1))

        new_layers = [new_layer]

        for i in range(len(out["layer_act_norms"])-2):
            idx = jnp.nonzero(prune_masks[i], size=new_widths[i+1])[0]

            new_state = copy_state_subset(new_model.layers[i+1], self.layers[i+1], new_state, state, idx)
            # Prune the inputs of the next layer
            new_layer = eqx.tree_at(lambda l: l.weights, new_model.layers[i+1], replace=jnp.take(self.layers[i+1].weights, idx, axis=1))

            # NEW: Added in new pieces for kan initialization
            if isinstance(new_layer.scale_base, jax.Array):
                new_layer = eqx.tree_at(lambda l: l.scale_base, new_layer, replace=jnp.take(self.layers[i+1].scale_base, idx, axis=2))

            if isinstance(new_layer.scale_sp, jax.Array):
                new_layer = eqx.tree_at(lambda l: l.scale_sp, new_layer, replace=jnp.take(self.layers[i+1].scale_sp, idx, axis=2))

            # Prune the outputs of the next layer
            next_idx = jnp.nonzero(prune_masks[i+1], size=new_widths[i+2])[0]
            new_layer = eqx.tree_at(lambda l: l.weights, new_layer, replace=jnp.take(new_layer.weights, next_idx, axis=0))

            # NEW: Added in new pieces for kan initialization
            if isinstance(new_layer.scale_base, jax.Array):
                new_layer = eqx.tree_at(lambda l: l.scale_base, new_layer, replace=jnp.take(new_layer.scale_base, next_idx, axis=1))

            if isinstance(new_layer.scale_sp, jax.Array):
                new_layer = eqx.tree_at(lambda l: l.scale_sp, new_layer, replace=jnp.take(new_layer.scale_sp, next_idx, axis=1))
                
            new_layers.append(new_layer)

        # Prune the inputs of the final layer
        idx = jnp.nonzero(prune_masks[-1], size=new_widths[-2])[0]
        new_state = copy_state_subset(new_model.layers[-1], self.layers[-1], new_state, state, idx)
        new_layer = eqx.tree_at(lambda l: l.weights, new_model.layers[-1], replace=jnp.take(self.layers[-1].weights, idx, axis=1))

        # NEW: Added in new pieces for kan initialization
        if isinstance(new_layer.scale_base, jax.Array):
            new_layer = eqx.tree_at(lambda l: l.scale_base, new_layer, replace=jnp.take(self.layers[-1].scale_base, idx, axis=2))

        if isinstance(new_layer.scale_sp, jax.Array):
            new_layer = eqx.tree_at(lambda l: l.scale_sp, new_layer, replace=jnp.take(self.layers[-1].scale_sp, idx, axis=2))

        new_layers.append(new_layer)

        new_model = eqx.tree_at(lambda m: m.layers, new_model, replace=new_layers)

        return new_model, new_state
