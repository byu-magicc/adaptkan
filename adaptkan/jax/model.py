# Third‑party libraries
import equinox as eqx
import jax
import jax.numpy as jnp
import optax

from adaptkan.jax.layers import AdaptKANLayerJax
from adaptkan.jax.utils import copy_state, copy_state_subset, build_combined_constraint_matrix, compute_constraint_projection_operator

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

    # Network-wide constraints (applied to last layer)
    has_network_constraints: bool = eqx.field(static=True)
    network_constraints_in: jax.Array | None   # (n_constraints, network_in_dim) - stored as module attribute
    network_constraints_out: jax.Array | None  # (n_constraints, network_out_dim) - stored as module attribute

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
                 network_constraints_in=None,   # (n_constraints, network_in_dim) - network-wide constraints
                 network_constraints_out=None,  # (n_constraints, network_out_dim) - target outputs
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

        # Handle network-wide constraints
        if network_constraints_in is not None and network_constraints_out is not None and basis_type == "chebyshev":
            self.has_network_constraints = True
            self.network_constraints_in = jnp.asarray(network_constraints_in)
            self.network_constraints_out = jnp.asarray(network_constraints_out)

            n_constraints = self.network_constraints_in.shape[0]
            assert self.network_constraints_in.shape == (n_constraints, width[0]), \
                f"network_constraints_in must have shape (n_constraints, network_in_dim={width[0]})"
            assert self.network_constraints_out.shape == (n_constraints, width[-1]), \
                f"network_constraints_out must have shape (n_constraints, network_out_dim={width[-1]})"
        else:
            self.has_network_constraints = False
            self.network_constraints_in = None
            self.network_constraints_out = None

        self.layers = []

        for i in range(depth):
            in_dim = width[i]
            out_dim = width[i+1]

            # For the last layer with network constraints, pass placeholder constraints
            # that will be populated by _setup_network_constraints
            is_last_layer = (i == depth - 1)
            if is_last_layer and self.has_network_constraints:
                # Create placeholder constraints for the last layer
                # These will be overwritten by _setup_network_constraints
                n_constraints = self.network_constraints_in.shape[0]
                placeholder_in = jnp.zeros((n_constraints, in_dim))
                placeholder_out = jnp.zeros((n_constraints, out_dim))
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
                                        constraints_in=placeholder_in,
                                        constraints_out=placeholder_out,
                                        key=keys[i])
            else:
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
                                        key=keys[i])
            self.layers.append(layer)

    def _setup_network_constraints(self, state):
        """
        Setup network-wide constraints on the last layer.

        Forwards network_constraints_in through all but the last layer to get z_c,
        then sets up constraints on the last layer with z_c as input points.

        Args:
            state: Current state

        Returns:
            Updated state with last layer constraints initialized
        """
        if not self.has_network_constraints:
            return state

        # Forward constraint inputs through all but last layer
        z_c = self.network_constraints_in  # (n_constraints, network_in_dim)
        for layer in self.layers[:-1]:
            z_c, state, *_ = layer(z_c, state, update=False)

        # z_c is now (n_constraints, last_layer_in_dim)
        last_layer = self.layers[-1]

        # Get current domain bounds of last layer
        a = state.get(last_layer.a)
        b = state.get(last_layer.b)

        # Compute constraints_a and constraints_b for domain protection
        cons_a = jnp.min(z_c, axis=0)  # (in_dim,)
        cons_b = jnp.max(z_c, axis=0)  # (in_dim,)

        # Expand domain to include constraint points
        new_a = jnp.minimum(a, cons_a)
        new_b = jnp.maximum(b, cons_b)
        state = state.set(last_layer.a, new_a)
        state = state.set(last_layer.b, new_b)

        # Build combined constraint matrix and projection operator
        C = build_combined_constraint_matrix(z_c, new_a, new_b, last_layer.k)
        P = compute_constraint_projection_operator(C)

        # Store in state
        state = state.set(last_layer.constraints_in, z_c)
        state = state.set(last_layer.constraints_out, self.network_constraints_out)
        state = state.set(last_layer.constraints_C, C)
        state = state.set(last_layer.constraints_P, P)
        state = state.set(last_layer.constraints_a, cons_a)
        state = state.set(last_layer.constraints_b, cons_b)

        return state

    def _recompute_network_constraints(self, state):
        """
        Recompute last layer constraints after any layer adapts.

        When intermediate layers adapt (domain changes), the mapping from
        network inputs to last layer inputs changes. We must:
        1. Forward network_constraints_in through layers[:-1] to get new z_c
        2. Recompute constraints_C and constraints_P for the last layer
        3. Update constraints_a and constraints_b for domain protection

        Args:
            state: Current state

        Returns:
            Updated state with recomputed last layer constraints
        """
        if not self.has_network_constraints:
            return state

        # Forward network constraint inputs through all but last layer
        z_c = self.network_constraints_in  # (n_constraints, network_in_dim)
        for layer in self.layers[:-1]:
            z_c, state, *_ = layer(z_c, state, update=False)

        # z_c is now (n_constraints, last_layer_in_dim)
        last_layer = self.layers[-1]

        # Update constraints_in (the z_c values)
        state = state.set(last_layer.constraints_in, z_c)

        # Recompute C and P matrices with new z_c and current domain
        a = state.get(last_layer.a)
        b = state.get(last_layer.b)
        C = build_combined_constraint_matrix(z_c, a, b, last_layer.k)
        P = compute_constraint_projection_operator(C)

        state = state.set(last_layer.constraints_C, C)
        state = state.set(last_layer.constraints_P, P)

        # Update constraints_a, constraints_b (domain protection bounds)
        cons_a = jnp.min(z_c, axis=0)  # (in_dim,)
        cons_b = jnp.max(z_c, axis=0)  # (in_dim,)
        state = state.set(last_layer.constraints_a, cons_a)
        state = state.set(last_layer.constraints_b, cons_b)

        return state

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
        adapted_any = jnp.any(jnp.stack(conds))

        # Recompute network constraints after ANY adaptation
        # This is critical: when intermediate layers change domain,
        # the z_c values for the last layer change!
        if self.has_network_constraints:
            state = jax.lax.cond(
                adapted_any,
                lambda s: model._recompute_network_constraints(s),
                lambda s: s,
                state
            )

        return model, state, adapted_any

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
        adapted_any = jnp.any(jnp.stack(conds))

        # Recompute network constraints after adaptation
        if self.has_network_constraints:
            state = model._recompute_network_constraints(state)

        return model, state, adapted_any

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
