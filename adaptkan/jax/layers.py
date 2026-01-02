"""Layer implementations for JAX backend."""
import equinox as eqx

import jax
import jax.numpy as jnp
from adaptkan.jax.utils import (compute_marginal_log_likelihood,
                                spline_interpolate_jax,
                                chebyshev_interpolate_jax,
                                shrink_weights_and_counts_jax,
                                stretch_weights_and_counts_jax,
                                refit_weights_and_counts_jax,
                                coefs_from_curve_jax)
from adaptkan.jax.constants import (M1, M2, M3, M4, M5)

class AdaptKANLayerJax(eqx.Module):
    in_dim: int = eqx.field(static=True)
    out_dim: int
    weights: jax.Array
    num_grid_intervals: int = eqx.field(static=True)
    activation_noise: float 
    activation_strategy: str = eqx.field(static=True)
    base_fun: callable = eqx.field(static=True)
    save_weight_counts: bool
    ema_alpha: float = eqx.field(static=True)
    prune_patience: int
    rounding_precision_eps: float = eqx.field(static=True)
    initialization_range: list = eqx.field(static=True)
    min_delta: float
    prune_scale_factor: float
    k: int = eqx.field(static=True)
    prune_mode: str = eqx.field(static=True)
    stretch_mode: str = eqx.field(static=True)
    stretch_threshold: float
    exact_refit: bool = eqx.field(static=True)
    basis_type: str = eqx.field(static=True)

    # Added in new variables for kan initialization
    base_fun: str = eqx.field(static=True)
    scale_base: float | jax.Array
    scale_sp: float | jax.Array

    # These define the different "buffers" we are keeping track of
    a: eqx.nn.StateIndex
    b: eqx.nn.StateIndex
    data_counts: eqx.nn.StateIndex
    ood_data_counts: eqx.nn.StateIndex
    ood_a: eqx.nn.StateIndex
    ood_b: eqx.nn.StateIndex
    counts_nonempty: eqx.nn.StateIndex
    
    # This allows us to mask out the weights that are not being used
    mask_indices: eqx.nn.StateIndex

    def __init__(self,
                 in_dim=3,
                 out_dim=2,
                 num_grid_intervals=10, # Number of grid intervals used for bsplines or degree of chebyshev polynomials
                 activation_noise=.1, # How much noise to add to the activation functions at initialization
                 activation_strategy='linear', # Either linear, zero, or kan. linear is simplest and works well.
                 base_fun=lambda x: x,
                 save_weight_counts=True,
                 ema_alpha=0.01, # Change this to be closer to zero to change the EMA to more heavily weight past data
                 prune_patience=1, # Set this to be equal to the number of batches per epoch. Affects the how often we shrink the domain
                 rounding_precision_eps=0.0,
                 initialization_range=[-1., 1.],
                 min_delta=1e-4,
                 prune_scale_factor=1.0,
                 k=3, 
                 prune_mode="default", # Options are "default", "relative", "default" usually works best
                 stretch_mode="max", # Options are "mean", "half_max", "max", "edge" or "relative"
                 stretch_threshold=None, # Used alongside the "relative" stretch mode
                 exact_refit=False, # turn this on for slightly more accurate results in some scenarios. However exact_refit=False is faster for larger models
                 basis_type="bspline", # Adding in chebyshev basis functions
                 key=None):
        
        if key is None:
            key = jax.random.key(0)
        
        # Save the different parameters
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_grid_intervals = num_grid_intervals
        self.activation_noise = activation_noise
        self.activation_strategy = activation_strategy
        self.save_weight_counts = save_weight_counts
        self.ema_alpha = ema_alpha
        self.prune_patience = prune_patience
        self.rounding_precision_eps = rounding_precision_eps
        self.initialization_range = initialization_range
        self.min_delta = min_delta
        self.prune_scale_factor = prune_scale_factor
        self.k = k
        self.prune_mode = prune_mode
        self.stretch_mode = stretch_mode
        self.stretch_threshold = stretch_threshold
        self.exact_refit = exact_refit
        self.basis_type = basis_type
        
        default_dtype = jnp.array(0.).dtype
        a = jnp.full((in_dim,), float(initialization_range[0]), dtype=default_dtype)
        b = jnp.full((in_dim,), float(initialization_range[1]), dtype=default_dtype)

        # Initialize the weights
        self.weights = self.initialize_weights(in_dim, out_dim, num_grid_intervals, k, activation_noise, activation_strategy, key, a, b, basis_type)

        if activation_strategy == 'kan':
            key, subkey = jax.random.split(key)
            # Use the kan default initialization
            scale_base_mu = 0.0
            scale_base_sigma = 1.0
            scale_sp = 1.0
            self.scale_base = scale_base_mu * 1. / jnp.sqrt(in_dim) + \
                scale_base_sigma * (jax.random.uniform(subkey, (1, out_dim, in_dim)) * 2 - 1) * 1. / jnp.sqrt(in_dim) 
            self.scale_sp = jnp.ones((1, out_dim, in_dim)) * scale_sp * 1. / jnp.sqrt(in_dim)
            self.base_fun = jax.nn.silu
        else:
            self.scale_base = 0.0
            self.scale_sp = 1.0
            self.base_fun = base_fun
        
        # Initialize the buffers
        self.a = eqx.nn.StateIndex(jnp.full((in_dim,), float(initialization_range[0]), dtype=default_dtype))
        self.b = eqx.nn.StateIndex(jnp.full((in_dim,), float(initialization_range[1]), dtype=default_dtype))
        self.ood_a = eqx.nn.StateIndex(jnp.full((in_dim,), float(initialization_range[0]), dtype=default_dtype))
        self.ood_b = eqx.nn.StateIndex(jnp.full((in_dim,), float(initialization_range[1]), dtype=default_dtype))
        self.data_counts = eqx.nn.StateIndex(jnp.zeros((in_dim, num_grid_intervals), dtype=default_dtype))
        self.ood_data_counts = eqx.nn.StateIndex(jnp.zeros((in_dim, 2), dtype=default_dtype))
        self.counts_nonempty = eqx.nn.StateIndex(False)
        
        # What is the final index that we are looking at?
        self.mask_indices = eqx.nn.StateIndex(jnp.full((in_dim,), num_grid_intervals, dtype=jnp.int32))

    @property
    def M(self):
        if self.k == 1:
            return M1
        elif self.k == 2:
            return M2
        elif self.k == 3:
            return M3
        elif self.k == 4:
            return M4
        elif self.k == 5:
            return M5
        else:
            raise ValueError(f"bspline_order {self.k} is not supported.")


    def initialize_weights(self, input_dim, output_dim, num_grid_intervals, k, activation_noise, activation_strategy, key, a, b, basis_type):
        # Initialize the weights to fall within the specified interval with the specified activation strategy
        if basis_type == "bspline":
            if activation_strategy == 'linear':
                linspace = jnp.linspace(-1, 1, num_grid_intervals + k)
                weights = jnp.tile(linspace, (output_dim, input_dim, 1)) + \
                    jax.random.normal(key, (output_dim, input_dim, num_grid_intervals + k)) * activation_noise
            elif activation_strategy == 'zero':
                weights = jax.random.normal(key, (output_dim, input_dim, num_grid_intervals + k)) * activation_noise
            elif activation_strategy == 'kan':
                noise = (jax.random.uniform(key, (output_dim, input_dim, num_grid_intervals + 1)) - 1/2) * activation_noise / num_grid_intervals
                weights = coefs_from_curve_jax(noise, a, b, num_grid_intervals, k=k, n_pts=num_grid_intervals+1)
            else:
                # TODO define more activation strategies
                raise ValueError(f"{activation_strategy} is an invalid activation strategy")
        elif basis_type == "chebyshev":
            # Recommended by Gemini
            if activation_strategy == "linear":
                weights = jax.random.normal(key, (output_dim, input_dim, k + 1)) * 0.01

                # Set T1 (index 1) to a 'LeCun' scale to preserve signal variance
                # This makes the KAN behave like a standard linear layer at start
                std = 1.0 / jnp.sqrt(input_dim)
                weights = weights.at[:, :, 1].set(jax.random.normal(key, (output_dim, input_dim)) * std)

        return weights
    
    def get_shrink_threshold(self, state):
        # Get the prune (shrink) threshold
        if self.prune_mode == "default": # <-- default mode
            thresh = (1 - self.ema_alpha)**self.prune_patience * self.ema_alpha
        elif self.prune_mode == "relative":
            thresh = state.get(self.data_counts).max() * self.ema_alpha
        return thresh
    
    def get_deltas(self, state):
        return (state.get(self.b) - state.get(self.a)) / self.num_grid_intervals
    
    def adapt(self, state):  

        # Prune the model if it is the right time to do so
        prune_out = shrink_weights_and_counts_jax(self.get_shrink_threshold(state),
                                                self.weights,
                                                state.get(self.a),
                                                state.get(self.b),
                                                self.num_grid_intervals,
                                                data_counts=state.get(self.data_counts),
                                                ood_counts=state.get(self.ood_data_counts),
                                                k=self.k,
                                                rounding_eps=self.rounding_precision_eps,
                                                min_delta=self.min_delta,
                                                exact_shrink=self.exact_refit,
                                                basis_type=self.basis_type)
    
        updated_weights, updated_dist, updated_ood_counts, updated_a, updated_b, pruned = prune_out
            
        stretch_out = stretch_weights_and_counts_jax(updated_weights,
                                                   updated_dist,
                                                   updated_ood_counts,
                                                   updated_a,
                                                   updated_b,
                                                   state.get(self.ood_a),
                                                   state.get(self.ood_b),
                                                   k=self.k,
                                                   num_grid_intervals=self.num_grid_intervals,
                                                   rounding_eps=self.rounding_precision_eps,
                                                   stretch_mode=self.stretch_mode,
                                                   stretch_threshold=self.stretch_threshold,
                                                   exact_stretch=self.exact_refit,
                                                   basis_type=self.basis_type)
        
        updated_weights, updated_dist, updated_ood_counts, updated_a, updated_b, stretched = stretch_out

        # Set the new state "buffers"
        state = state.set(self.a, updated_a)
        state = state.set(self.b, updated_b)
        state = state.set(self.data_counts, updated_dist)
        state = state.set(self.ood_data_counts, updated_ood_counts)

        model = eqx.tree_at(lambda m: m.weights, self, updated_weights)

        adapted = pruned | stretched
        
        return model, state, adapted
    
    def manual_adapt(self, state):
        # This will force the domain to fit the current histograms which have a max and min value of self.ood_b and self.ood_a

        new_a = state.get(self.ood_a)
        new_b = state.get(self.ood_b)

        updated_weights, updated_counts, updated_ood_counts = refit_weights_and_counts_jax(
            self.weights,
            state.get(self.data_counts),
            state.get(self.ood_data_counts),
            state.get(self.a),
            state.get(self.b),
            new_num_grid_intervals=self.num_grid_intervals,  # Use stored value directly
            new_a=new_a,
            new_b=new_b,
            k=self.k,
            rounding_eps=self.rounding_precision_eps,
            exact_refit=self.exact_refit,
            basis_type=self.basis_type
        )

        # Set the new state "buffers"
        state = state.set(self.a, new_a)
        state = state.set(self.b, new_b)
        state = state.set(self.data_counts, updated_counts)
        state = state.set(self.ood_data_counts, updated_ood_counts)
        
        # Update the model with the new weights
        # Stop the gradient from flowing through
        model = eqx.tree_at(lambda m: m.weights, self, updated_weights)
        
        return model, state, jnp.array(True)
    
    def update_counts(self, last_indices, lower_ood_mask, upper_ood_mask, last_ood_a, last_ood_b, state, add_only=False):

        # This updates the histograms (out-of-domain and in domain) with the latest data

        combined_mask = lower_ood_mask | upper_ood_mask

        dtype = state.get(self.data_counts).dtype
        
        # This refers to the count values in our domain
        # This includes everything inside the domain
        values = (~combined_mask).astype(dtype)
        row_indices = jnp.arange(self.in_dim)[:, None]
        row_indices = jnp.broadcast_to(row_indices, (self.in_dim, values.shape[0]))

        # Define per-row scatter-add
        def scatter_add_row(idx, val):
            row_out = jnp.zeros(self.num_grid_intervals, dtype=dtype)
            return row_out.at[idx].add(val)

        # Change this to last_indices[:,:,0].T for different results WHY??
        last_data_counts = jax.vmap(scatter_add_row)(last_indices.T, values.T) 

        if not add_only:
            data_counts = jax.lax.cond(
                state.get(self.counts_nonempty),
                lambda: (1 - self.ema_alpha) * state.get(self.data_counts) + self.ema_alpha * last_data_counts,
                lambda: last_data_counts
            )
        else:
            data_counts = jax.lax.cond(
                state.get(self.counts_nonempty),
                lambda: state.get(self.data_counts) + last_data_counts, # Just add directly, no ema smoothing
                lambda: last_data_counts
            )

        # Get the out of distribution counts
        last_ood_data_counts = (jnp.vstack([lower_ood_mask.sum(0), upper_ood_mask.sum(0)]).T).astype(dtype)

        if not add_only:
            ood_data_counts = jax.lax.cond(
                state.get(self.counts_nonempty),
                lambda: (1 - self.ema_alpha) * state.get(self.ood_data_counts) + self.ema_alpha * last_ood_data_counts,
                lambda: last_ood_data_counts
            )
        else:
            ood_data_counts = jax.lax.cond(
                state.get(self.counts_nonempty),
                lambda: state.get(self.ood_data_counts) + last_ood_data_counts,
                lambda: last_ood_data_counts
            )

        state = state.set(self.counts_nonempty, True)
        state = state.set(self.data_counts, data_counts)

        # Set the out of distribution information
        state = state.set(self.ood_data_counts, ood_data_counts)
        state = state.set(self.ood_a, jnp.minimum(state.get(self.ood_a), last_ood_a).astype(dtype))
        state = state.set(self.ood_b, jnp.maximum(state.get(self.ood_b), last_ood_b).astype(dtype))
        
        return state

    def basic_forward(self, x, state):

        if self.basis_type == "bspline":
            postsplines, alphas, indices = spline_interpolate_jax(x,
                                                        state.get(self.a),
                                                        state.get(self.b),
                                                        self.weights,
                                                        self.k,
                                                        self.rounding_precision_eps)
        elif self.basis_type == "chebyshev":
            postsplines, alphas, indices = chebyshev_interpolate_jax(x,
                                                        state.get(self.a),
                                                        state.get(self.b),
                                                        self.weights,
                                                        self.num_grid_intervals,
                                                        self.rounding_precision_eps)

        base = self.base_fun(x)
        y = self.scale_base * base[:,None,:] + self.scale_sp * postsplines

        act_norms = jnp.abs(y).sum(0) / y.shape[0]

        # Sum over the last dimension
        y = jnp.sum(y, axis=2)

        return y, act_norms, alphas, indices, postsplines

    def __call__(self, x, state, update=True, add_counts=False):

        lower_clip_mask, upper_clip_mask = (x < state.get(self.a)), (x > state.get(self.b))
        ood_a, ood_b = x.min(axis=0), x.max(axis=0)
        clipped_x = jnp.clip(x, state.get(self.a), state.get(self.b))
        y, act_norms, alphas, indices, postsplines = self.basic_forward(clipped_x, state)

        state = jax.lax.cond(
            update,
            lambda s: self.update_counts(indices, lower_clip_mask, upper_clip_mask, ood_a, ood_b, s, add_counts),
            lambda s: s,
            state
        )

        return y, state, act_norms, alphas, indices, postsplines
    
    @eqx.filter_jit
    def get_log_prob(self, indices, state, key=None):
        """
        Performs out-of-distribution (OOD) detection by computing the marginal likelihood of the data under the current layer.
        Args:
            indices (jnp.ndarray): Indices of the data points for which to compute the marginal likelihood.
            state (Any): State object containing the layer's data counts.
            key (jax.random.PRNGKey, optional): Random key for noise generation. Defaults to None.
            noise_scale (float, optional): Standard deviation of the noise added to the CDF values. Defaults to 0.05.
        Returns:
            jnp.ndarray: Marginal likelihood values for the given data points, possibly with added noise.
        """

        if key is None:
            key = jax.random.PRNGKey(0)

        # Get the cdf for the layer
        layer_histogram = state.get(self.data_counts)
        layer_histogram = layer_histogram / layer_histogram.sum(-1)[:,None] 

        log_vals = compute_marginal_log_likelihood(jnp.clip(indices, 0, self.num_grid_intervals - 1), layer_histogram)

        return log_vals
    
    def refine(self, weights, data_counts, new_state, new_num_grid_intervals):

        if self.basis_type == "bspline":
            # For B-splines: increase num_grid_intervals (more control points)
            refined_weights, refined_counts, refined_ood_counts = refit_weights_and_counts_jax(
                weights,
                data_counts,
                new_state.get(self.ood_data_counts),
                new_state.get(self.a),
                new_state.get(self.b),
                new_num_grid_intervals,
                new_a=new_state.get(self.a),
                new_b=new_state.get(self.b),
                k=self.k,
                exact_refit=True,
                basis_type=self.basis_type
            )

            new_state = new_state.set(self.data_counts, refined_counts)
            new_state = new_state.set(self.ood_data_counts, refined_ood_counts)
            layer = eqx.tree_at(lambda l: l.weights, self, refined_weights)
            layer = eqx.tree_at(lambda l: l.num_grid_intervals, layer, new_num_grid_intervals)

        elif self.basis_type == "chebyshev":
            # For Chebyshev: increase k (polynomial degree)
            # Here new_num_grid_intervals is interpreted as new_k
            new_k = new_num_grid_intervals
            refined_weights, refined_counts, refined_ood_counts = refit_weights_and_counts_jax(
                weights,
                data_counts,
                new_state.get(self.ood_data_counts),
                new_state.get(self.a),
                new_state.get(self.b),
                self.num_grid_intervals,  # num_grid_intervals stays the same
                new_a=new_state.get(self.a),
                new_b=new_state.get(self.b),
                k=new_k,  # Pass new_k for Chebyshev degree
                exact_refit=True,
                basis_type=self.basis_type
            )

            new_state = new_state.set(self.data_counts, refined_counts)
            new_state = new_state.set(self.ood_data_counts, refined_ood_counts)
            layer = eqx.tree_at(lambda l: l.weights, self, refined_weights)
            layer = eqx.tree_at(lambda l: l.k, layer, new_k)

        return layer, new_state
