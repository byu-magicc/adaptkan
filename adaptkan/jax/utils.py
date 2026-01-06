"""Utility stubs for the JAX backend."""
import jax.numpy as jnp
from adaptkan.jax.constants import (
    M1, M2, M3, M4, M5
)
import jax
from jax import lax
from jax import vmap, jit
from jax import tree_util
# from spline import bspline
from functools import partial
from jax.scipy.stats import norm

def parallel_linspace_jax(start: jnp.ndarray, end: jnp.ndarray, steps: int) -> jnp.ndarray:
    """
    Generates multiple linearly spaced sequences in parallel.

    Args:
        start (jnp.ndarray): 1D array of start points, shape (n,)
        end (jnp.ndarray): 1D array of end points, shape (n,)
        steps (int): Number of points in each sequence

    Returns:
        jnp.ndarray: 2D array of shape (n, steps) where each row is a linspace from start[i] to end[i]
    """
    # Ensure start and end are 1D arrays of the same length
    if start.ndim != 1 or end.ndim != 1:
        raise ValueError("Start and end arrays must be 1D.")
    if start.shape != end.shape:
        raise ValueError("Start and end arrays must have the same shape.")

    # Reshape start and end to (n, 1) for broadcasting
    start = start[:, None]  # Shape: (n, 1)
    end = end[:, None]      # Shape: (n, 1)

    # Generate a linspace vector from 0 to 1 with 'steps' points
    # Shape: (1, steps) to broadcast across all n sequences
    t = jnp.linspace(0, 1, steps)[None, :]

    # Compute the linear interpolation
    # Formula: start + t * (end - start)
    linspace = start + t * (end - start)  # Shape: (n, steps)

    return linspace

def get_coef_matrix_jax(k):
    """
    Returns the coefficient matrix for a given spline order k.
    """
    if k == 1:
        return M1
    elif k == 2:
        return M2
    elif k == 3:
        return M3
    elif k == 4:
        return M4
    elif k == 5:
        return M5
    else:
        raise ValueError("Spline order k must be between 1 and 5.")

def get_interp_coefs_jax(alphas, k):
    """
    Returns the interpolation coefficients for a given spline order k.
    """
    alphas = alphas.T
    M = get_coef_matrix_jax(k)
    exponents = jnp.arange(k, -1, -1)
    
    alphas_expanded = alphas[None, :, :]
    exponents_expanded = exponents[:, None, None]
    
    # Completely avoid any power operations - handle each exponent explicitly
    # This is needed to avoid nans in certain training setups
    def completely_safe_power(base, exp):
        # Clean the base first
        safe_base = jnp.where(jnp.isnan(base) | jnp.isinf(base), 0.0, base)
        
        return jnp.where(
            exp == 0,
            jnp.ones_like(safe_base),
            jnp.where(
                exp == 1,
                safe_base,
                jnp.where(
                    exp == 2,
                    safe_base * safe_base,
                    jnp.where(
                        exp == 3,
                        safe_base * safe_base * safe_base,
                        jnp.where(
                            exp == 4,
                            safe_base * safe_base * safe_base * safe_base,
                            # For any higher exponents, just return 0 to be safe
                            jnp.where(
                                exp == 5,
                                safe_base * safe_base * safe_base * safe_base * safe_base,
                                # For any higher exponents, just return 0 to be safe
                                jnp.zeros_like(safe_base)
                            )
                        )
                    )
                )
            )
        )
    
    result = completely_safe_power(alphas_expanded, exponents_expanded)
    
    out = jnp.einsum('ij,jkl->ikl', M, result)
    
    return jnp.transpose(out, (2, 1, 0))

def get_deriv_interp_coefs_jax(alphas, deltas, k):
    """
    Returns the interpolation coefficients for a given spline order k.
    """
    alphas = alphas.T
    M = get_coef_matrix_jax(k)
    exp = jnp.arange(k, -1, -1)[:, None, None]
    deltas = deltas[None, :, None]
    alphas = alphas[None, :, :]
    result = exp * alphas ** (exp - 1) / deltas
    result = result.at[-1].set(0.)
    out = jnp.einsum('ij,jkl->ikl', M, result)
    return jnp.transpose(out, (2, 1, 0))

def get_interp_indices_jax(indices, max_items, k):
    """
    Returns the indices for interpolation for a given spline order k.
    """
    # Create an array of offsets: [0, 1, ..., k-1]. We reshape it to broadcast along the
    # expanded dimension. Its shape will be [1, ..., k, ..., 1] with k in the `axis` position.
    indices = indices.T
    indices = jnp.expand_dims(indices, axis=2)
    
    offsets_shape = [1] * indices.ndim
    offsets_shape[-1] = k + 1
    offsets = jnp.arange(k + 1).reshape(offsets_shape)

    # Add offsets to the base indices and clip to valid range.
    new_indices = jnp.clip(indices + offsets, min=0, max=max_items - 1)
    return jnp.transpose(new_indices, (1, 0, 2))

def spline_interpolate_jax(x, a, b, control_pts, k=1, rounding_eps=1e-5, clip_end=False):
    
    # Grid is the number of grid intervals
    # weights size should be grid + k

    out_dim, _, n_ctrl_pts = control_pts.shape
    num_grid_intervals = n_ctrl_pts - k
    batch_size = x.shape[0]
    
    x_adjusted = x - a[None, :]
    deltas = ((b - a) / num_grid_intervals)[None, :]
    indices = jnp.floor(x_adjusted / deltas + rounding_eps).astype(jnp.int32)

    largest_index = num_grid_intervals - 1
    if not clip_end:
        largest_index = num_grid_intervals

    indices = jnp.clip(indices, 0, largest_index) # Added in the -1 so that no matter what we lie in the last interval 

    # Clip the indices out of bounds and get a mask of these indices
    # indices_clipped = jnp.clip(indices, 0, num_grid_intervals)
    
    interp_indices = get_interp_indices_jax(indices, n_ctrl_pts, k=k)
    # interp_indices = jnp.transpose(interp_indices, (1, 0, 2))

    # Calculate alphas for input data
    alphas = x_adjusted / deltas + rounding_eps - jnp.floor(x_adjusted / deltas + rounding_eps)
    interp_coefs = get_interp_coefs_jax(alphas, k=k) # Should be (in_dim, new_num_grid_intervals+1, k+1)

    out = spline_interpolate_with_coefs_and_indices(interp_coefs, interp_indices, control_pts, k)

    # Change this to interp_indices for different results WHY??
    return out, alphas, indices

# Generated by Gemini
def compute_chebyshev_basis(x, a, b, degree):
    """
    Maps x -> [-1, 1] and computes Chebyshev polynomials recursively.
    Crucially, it clips x to [a, b] to prevent recursion explosion outside domain.

    Args:
        x: Input data, shape (batch, in_dim) or (M+1, in_dim)
        a, b: Domain boundaries, shape (in_dim,)
        degree: Polynomial degree

    Returns:
        basis: Chebyshev basis, shape (batch, in_dim, degree+1)
    """
    # Map x to [-1, 1], broadcasting a and b for batch dimension
    x_scaled = 2.0 * (x - a[None, :]) / (b[None, :] - a[None, :]) - 1.0
    # --- CRITICAL STEP FOR EXTENSIONS ---
    # Clip to avoid NaNs in recursive loops outside [-1, 1]
    x_scaled = jnp.clip(x_scaled, -1.0, 1.0)
    
    basis = [jnp.ones_like(x_scaled), x_scaled]
    for _ in range(2, degree + 1):
        next_t = 2.0 * x_scaled * basis[-1] - basis[-2]
        basis.append(next_t)
    return jnp.stack(basis, axis=-1)

# Generated by Gemini
def chebyshev_interpolate_jax(x, a, b, weights, num_grid_intervals, rounding_eps=1e-5, clip_end=False):
    """
    Chebyshev interpolation in JAX.

    Args:
        x: Input data
        a, b: Domain boundaries
        weights: Chebyshev coefficients, shape (out_dim, in_dim, k+1)
        num_grid_intervals: Number of histogram bins for domain adaptation
        rounding_eps: Small epsilon for numerical stability
        clip_end: Whether to clip to last interval
    """
    out_dim, in_dim, n_weights = weights.shape
    k = n_weights - 1  # Derive polynomial degree from weights
    batch_size = x.shape[0]

    # Use num_grid_intervals for histogram binning
    x_adjusted = x - a[None, :]
    deltas = ((b - a) / num_grid_intervals)[None, :]
    indices = jnp.floor(x_adjusted / deltas + rounding_eps).astype(jnp.int32)

    largest_index = num_grid_intervals - 1
    if not clip_end:
        largest_index = num_grid_intervals

    indices = jnp.clip(indices, 0, largest_index)

    # Calculate alphas for input data
    alphas = x_adjusted / deltas + rounding_eps - jnp.floor(x_adjusted / deltas + rounding_eps)

    # Use k (derived from weights) for polynomial degree
    basis = compute_chebyshev_basis(x, a, b, degree=k)
    out = jnp.einsum('bik,oik->boi', basis, weights)

    return out, alphas, indices

# Generated by Gemini
def cheby_summation_oversampled_refit(old_weights, old_a, old_b, new_a, new_b, new_k, oversample_ratio=1.5):
    """
    Refit Chebyshev coefficients from old domain/degree to new domain/degree.

    Args:
        old_weights: Old Chebyshev coefficients, shape (out_dim, in_dim, old_k+1)
        old_a, old_b: Old domain boundaries
        new_a, new_b: New domain boundaries
        new_k: New polynomial degree
        oversample_ratio: Oversampling ratio for least squares fitting

    Returns:
        new_weights: New Chebyshev coefficients, shape (out_dim, in_dim, new_k+1)
    """
    out_dim, in_dim, _ = old_weights.shape
    old_k = old_weights.shape[-1] - 1  # Derive old degree from weights
    M = int(max(old_k, new_k) * oversample_ratio)  # Sample enough points for both

    # 1. Get M Gauss-Lobatto nodes in the NEW domain
    # nodes_std has shape (M+1,), new_a/new_b have shape (in_dim,)
    # x_phys should have shape (M+1, in_dim)
    i = jnp.arange(M + 1)
    nodes_std = jnp.cos(jnp.pi * i / M)  # (M+1,)
    # Broadcast: (M+1, 1) with (in_dim,) -> (M+1, in_dim)
    x_phys = new_a[None, :] + (nodes_std[:, None] + 1.0) * (new_b[None, :] - new_a[None, :]) / 2.0

    # 2. Evaluate OLD function at these M nodes
    # Using the original weights and bounds captures the 'Retention' data
    # basis has shape (M+1, in_dim, old_k+1)
    basis = compute_chebyshev_basis(x_phys, old_a, old_b, degree=old_k)
    # y_targets has shape (M+1, out_dim, in_dim) after einsum
    y_targets = jnp.einsum('mik,oik->moi', basis, old_weights)

    # 3. Discrete Least Squares via DCT logic
    # Scale endpoints for the trapezoidal rule weights
    weights_vec = jnp.ones(M + 1)
    weights_vec = weights_vec.at[0].set(0.5)
    weights_vec = weights_vec.at[-1].set(0.5)

    # Normalize targets: y_targets is (M+1, out_dim, in_dim), weights_vec is (M+1,)
    # Broadcast weights_vec to (M+1, 1, 1)
    y_weighted = y_targets * weights_vec[:, None, None]

    # 4. Generate the Projection Matrix (Size: new_k+1 x M+1)
    k_idx = jnp.arange(new_k + 1)
    j_idx = jnp.arange(M + 1)
    # Cosine basis: cos(k * pi * j / M)
    projection = jnp.cos(k_idx[:, None] * j_idx[None, :] * jnp.pi / M)  # (new_k+1, M+1)

    # 5. Calculate New Coefficients
    # Contract: projection (new_k+1, M+1) with y_weighted (M+1, out_dim, in_dim)
    # Result should be (out_dim, in_dim, new_k+1)
    new_weights = (2.0 / M) * jnp.einsum('km,moi->oik', projection, y_weighted)

    # Apply the standard Chebyshev c0 scaling to the first coefficient (k=0)
    new_weights = new_weights.at[:, :, 0].multiply(0.5)

    return new_weights


def spline_interpolate_with_coefs_and_indices(interp_coefs, interp_indices, control_pts, k):
    """
    interp_coefs should be shape (batch_size, in_dim, k+1)
    interp_indices should be shape (batch_size, in_dim, k+1)
    """
    out_dim, _, _ = control_pts.shape
    batch_size = interp_coefs.shape[0]

    # Expand the indices to the correct shape for the weights
    control_pts_expanded = jnp.expand_dims(control_pts, axis=0).repeat(batch_size, axis=0)
    indices_expanded = jnp.expand_dims(interp_indices, axis=1).repeat(out_dim, axis=1)

    # Gather the k consecutive elements.
    gathered_coefs = jnp.take_along_axis(control_pts_expanded, indices_expanded, axis=-1)

    out = jnp.einsum('bijk,bjk->bij', gathered_coefs, interp_coefs)

    return out

def scatter_coeffs_to_matrix(coeffs, target_indices, n_weights):
    """
    Scatter interpolation coefficients into a staggered matrix.

    Args:
      coeffs: jnp.ndarray of shape (n_pts, k+1) containing interpolation coefficients.
      target_indices: jnp.ndarray of shape (n_pts, k+1) containing the target column indices
                      for each coefficient.
      n_weights: int, total number of columns for the output matrix.

    Returns:
      A jnp.ndarray of shape (n_pts, n_weights) with the coefficients scattered to the 
      positions specified by target_indices.
    """
    n_pts, k_plus_1 = coeffs.shape

    # Create row indices for each element.
    # For each row r, we need a copy of r for each coefficient.
    row_indices = jnp.arange(n_pts)[:, None]               # Shape: (n_pts, 1)
    row_indices = jnp.broadcast_to(row_indices, (n_pts, k_plus_1))  # Shape: (n_pts, k+1)

    # Create a zeros matrix of the target shape.
    C = jnp.zeros((n_pts, n_weights), dtype=coeffs.dtype)

    # Use the "at" update to scatter the values.
    C = C.at[(row_indices, target_indices)].add(coeffs)
    return C

def full_knot_vector(defined_knots, k):
    """
    Given defined knot points for a clamped B-spline and degree k,
    construct the full knot vector by repeating the endpoints k+1 times.
    
    Args:
      defined_knots: 1D jnp.ndarray of shape (g+1,) with the unique knots.
      k: degree of the spline.
      
    Returns:
      A 1D jnp.ndarray representing the full knot vector.
    """
    # Repeat the first and last defined knot k+1 times
    start = jnp.full((k+1,), defined_knots[0])
    end   = jnp.full((k+1,), defined_knots[-1])
    # The interior defined knots (if any)
    interior = defined_knots[1:-1]
    return jnp.concatenate([start, interior, end])

def lstsq_with_coefs_and_indices(y_eval, interp_coefs, interp_indices, n_weights):
    # Solve the least squares problem, over the batch
    mat = scatter_coeffs_to_matrix(interp_coefs, interp_indices, n_weights)
    coef, _, _, _ = jnp.linalg.lstsq(mat, y_eval, rcond=None)
    return coef

vmap_lstsq = vmap(vmap(lstsq_with_coefs_and_indices, in_axes=(0, 0, 0, None)), in_axes=(0, 0, 0, None))

def coefs_from_curve_jax(y_eval, a, b, num_grid_intervals, k=1, rounding_eps=1e-5, n_pts=101):
    """
    Compute interpolation coefficients from curve evaluations using JAX.
    It takes as input a set of curves to fit and will return the weights (B-spline control points) for each curve.
    Parameters
    ----------
    y_eval : jnp.ndarray
        Array of evaluated curve values. Its first dimension represents the output dimension.
    a : jnp.ndarray
        Lower bounds for each evaluation instance; used for grid adjustment.
    b : jnp.ndarray
        Upper bounds for each evaluation instance corresponding to 'a'.
    num_grid_intervals : int
        Number of intervals into which the grid (from a to b) is subdivided. This will influence the number of weights.
    k : int, optional
        Degree or order parameter for interpolation (default is 1).
    rounding_eps : float, optional
        A small epsilon value added to mitigate floating-point rounding issues during index computation (default is 1e-5).
    n_pts : int, optional
        Number of points in the linspace domain between a and b (default is 101).
    Returns
    -------
    new_weights : jnp.ndarray
        Array containing the newly computed weights based on least-squares interpolation.
    """
    
    x = parallel_linspace_jax(a, b, n_pts).T

    out_dim = y_eval.shape[0]

    x_adjusted = x - a[None, :]
    deltas = ((b - a) / num_grid_intervals)[None, :]
    indices = jnp.floor(x_adjusted / deltas + rounding_eps).astype(jnp.int32)
    n_weights = num_grid_intervals + k
    
    # Shouldn't need to clip these new indices
    alphas = x_adjusted / deltas - jnp.floor(x_adjusted / deltas + rounding_eps)
    # indices_expanded = jnp.expand_dims(indices, axis=0)
    # indices_expanded = jnp.tile(indices_expanded, (out_dim, 1, 1))
    
    interp_coefs = get_interp_coefs_jax(alphas, k=k)
    interp_indices = get_interp_indices_jax(indices, n_weights, k=k)
    interp_indices = jnp.transpose(interp_indices, (1, 0, 2))
    interp_indices = jnp.expand_dims(interp_indices, axis=0)
    interp_indices = jnp.tile(interp_indices, (out_dim, 1, 1, 1))

    # Make sure everything is the right shape
    interp_coefs = jnp.transpose(interp_coefs, (1, 0, 2)) # Should be (in_dim, new_num_grid_intervals+1, k+1)
    interp_coefs = jnp.expand_dims(interp_coefs, axis=0)
    interp_coefs = jnp.repeat(interp_coefs, repeats=out_dim, axis=0)
    # interp_indices = jnp.transpose(interp_indices, (0, 1, 3, 2))

    # Interp_coefs and interp_indices should be (out_dim, in_dim, n_weights, k+1)
    new_weights = vmap_lstsq(y_eval, interp_coefs, interp_indices, n_weights)

    return new_weights

def _compute_greville_jax(a, b, num_grid_intervals, k):
    """
    Compute Greville abscissae using lax.dynamic_slice for dynamic indexing.
    
    Args:
      knots: 1D jnp.ndarray representing the full knot vector.
      k: degree of the spline.
      
    Returns:
      A 1D jnp.ndarray containing the Greville abscissae.
    """
    defined_knots = jnp.linspace(a, b, num_grid_intervals + 1)

    full_knots = full_knot_vector(defined_knots, k)
    n_control = full_knots.shape[0] - (k + 1)

    # Number of control points = len(knots) - (k+1)
    n_control = full_knots.shape[0] - (k + 1)
    
    # Function to compute the mean of k knots starting at a dynamic index.
    def greville_for_index(i):
        # i is a scalar index; we need to slice knots[i+1 : i+k+1]
        # Since lax.dynamic_slice requires a tuple for the start indices, we do:
        start = (i + 1,)  # start index along dimension 0
        slice_size = (k,)  # we want k elements
        slice_ = lax.dynamic_slice(full_knots, start, slice_size)
        return jnp.mean(slice_)
    
    # Vectorize over control point indices.
    indices = jnp.arange(n_control)
    greville = jax.vmap(greville_for_index)(indices)
    return greville

def _compute_greville_jax(a, b, num_grid_intervals, k):
    """
    Compute Greville abscissae using lax.dynamic_slice for dynamic indexing.
    
    Args:
      knots: 1D jnp.ndarray representing the full knot vector.
      k: degree of the spline.
      
    Returns:
      A 1D jnp.ndarray containing the Greville abscissae.
    """
    defined_knots = jnp.linspace(a, b, num_grid_intervals + 1)

    full_knots = full_knot_vector(defined_knots, k)
    n_control = full_knots.shape[0] - (k + 1)

    # Number of control points = len(knots) - (k+1)
    n_control = full_knots.shape[0] - (k + 1)
    
    # Function to compute the mean of k knots starting at a dynamic index.
    def greville_for_index(i):
        # i is a scalar index; we need to slice knots[i+1 : i+k+1]
        # Since lax.dynamic_slice requires a tuple for the start indices, we do:
        start = (i + 1,)  # start index along dimension 0
        slice_size = (k,)  # we want k elements
        slice_ = lax.dynamic_slice(full_knots, start, slice_size)
        return jnp.mean(slice_)
    
    # Vectorize over control point indices.
    indices = jnp.arange(n_control)
    greville = jax.vmap(greville_for_index)(indices)
    return greville

def compute_greville_jax(a, b, num_grid_intervals, k):
    return vmap(_compute_greville_jax, in_axes=(0, 0, None, None))(a, b, num_grid_intervals, k)

def refit_weights_and_counts_jax(weights,
                                 counts,
                                 ood_counts,
                                 a,
                                 b,
                                 new_num_grid_intervals,
                                 new_a,
                                 new_b,
                                 k=1,
                                 rounding_eps=0.0,
                                 rescale_counts=True,
                                 exact_refit=True,
                                 basis_type="bspline"): # Change this to false for faster results
    """
    Refits weights and counts based on the new domain using linear interpolation and/or linear least squares.
    """
    
    end_weights = weights[:,:,-1]
    start_weights = weights[:,:,0]

    # First, refit the counts
    domain_counts = parallel_linspace_jax(a, b, new_num_grid_intervals + 1)
    domain_centers = (domain_counts[:, 1:] + domain_counts[:, :-1]) / 2 # This is equivalent to the number of grid intervals
    center_a, center_b = domain_centers[:, 0], domain_centers[:, -1]
    bin_widths = domain_counts[:, 1] - domain_counts[:, 0]


    new_domain_counts = new_a[:,None] + (domain_counts - a[:,None]) * (new_b[:,None] - new_a[:,None]) / (b[:,None] - a[:,None])
    new_domain_centers = (new_domain_counts[:, 1:] + new_domain_counts[:, :-1]) / 2

    new_center_a, new_center_b = new_domain_centers[:, 0], new_domain_centers[:, -1]
    new_bin_widths = new_domain_counts[:, 1] - new_domain_counts[:, 0]

    # Fixed bug here
    lower_mask = (new_domain_centers - new_bin_widths[:,None] / 2 < center_a[:,None] - bin_widths[:,None] / 2)
    upper_mask = (new_domain_centers + new_bin_widths[:,None] / 2 > center_b[:,None] + bin_widths[:,None] / 2)
    # Used to be
    # lower_mask = (new_domain_centers < center_a[:,None])
    # upper_mask = (new_domain_centers > center_b[:,None])

    # Fill with the out of domain counts if provided
    lower_mask_sum = lower_mask.sum(-1)
    upper_mask_sum = upper_mask.sum(-1)
    lower_mask_sum = jnp.where(lower_mask_sum == 0, 1, lower_mask_sum)
    upper_mask_sum = jnp.where(upper_mask_sum == 0, 1, upper_mask_sum)

    ood_bin_values = ood_counts.at[:, 0].divide(lower_mask_sum)
    ood_bin_values = ood_bin_values.at[:, -1].divide(upper_mask_sum)

    new_counts, _, _ = spline_interpolate_jax(new_domain_centers.T, center_a, center_b, counts[None,:,:], k=1, rounding_eps=rounding_eps, clip_end=False)
        
    new_counts = jnp.transpose(new_counts, (1, 2, 0))
    new_counts = jnp.where(lower_mask[None,:,:], ood_bin_values[None,:,0,None], new_counts)
    new_counts = jnp.where(upper_mask[None,:,:], ood_bin_values[None,:,1,None], new_counts)
    new_counts = jnp.squeeze(new_counts, axis=0)

    # Now refit the weights depending on the basis function we are using
    if basis_type == "bspline":
        
        greville_pts = compute_greville_jax(a, b, new_num_grid_intervals, k)
        new_greville_pts= new_a[:,None] + (greville_pts - a[:,None]) * (new_b[:,None] - new_a[:,None]) / (b[:,None] - a[:,None] + 1e-8)

        lower_mask = (new_greville_pts < a[:,None])
        upper_mask = (new_greville_pts > b[:,None])

        if exact_refit:
            domain = parallel_linspace_jax(new_a, new_b, 101)
            out, _, _ = spline_interpolate_jax(domain.T, a, b, weights, k=k, rounding_eps=rounding_eps, clip_end=False)

            out = jnp.transpose(out, (1, 2, 0))
            new_weights = coefs_from_curve_jax(out, new_a, new_b, new_num_grid_intervals, k=k)
            new_weights = jnp.where(lower_mask[None,:,:], start_weights[:,:,None], new_weights)
            new_weights = jnp.where(upper_mask[None,:,:], end_weights[:,:,None], new_weights)
        else:
            # Do a rough refitting of the weights via linear interpolation
            new_weights, _, _ = spline_interpolate_jax(new_greville_pts.T, a, b, weights, k=1, clip_end=False, rounding_eps=rounding_eps)
            new_weights = jnp.transpose(new_weights, (1, 2, 0))
            new_weights = jnp.where(lower_mask[None,:,:], start_weights[:,:,None], new_weights)
            new_weights = jnp.where(upper_mask[None,:,:], end_weights[:,:,None], new_weights)
    elif basis_type == "chebyshev":
        new_weights = cheby_summation_oversampled_refit(weights, a, b, new_a, new_b, k, oversample_ratio=1.5)
    else:
        raise ValueError("Unknown basis type: {}".format(basis_type))

    # This removes parts of the ood_counts that we aren't using anymore from stretching
    ood_mask = jnp.vstack([lower_mask.any(axis=-1), upper_mask.any(axis=-1)]).T
    new_ood_counts = jnp.where(ood_mask, 0., ood_counts) # .at[jnp.where(ood_mask)].set(0.)

    # This puts the pruned counts into the ood counts
    new_center_a = new_domain_centers[:, 0]
    new_center_b = new_domain_centers[:, -1]

    # Fixed bug here
    lower_prune_mask = domain_centers - bin_widths[:,None] / 2 < (new_center_a[:,None] - new_bin_widths[:,None] / 2 - rounding_eps)
    upper_prune_mask = domain_centers + bin_widths[:,None] / 2 > (new_center_b[:,None] + new_bin_widths[:,None] / 2 + rounding_eps)
    # Used to be
    # lower_prune_mask = domain_centers < (new_center_a[:,None] - rounding_eps)
    # upper_prune_mask = domain_centers > (new_center_b[:,None] + rounding_eps)

    counts_temp, _, _ = spline_interpolate_jax(domain_centers.T, center_a, center_b, counts[None,:,:], k=1, rounding_eps=rounding_eps)
    
    counts_temp = jnp.transpose(counts_temp, (1, 2, 0))

    lower_counts_sum = (counts_temp * lower_prune_mask).sum(axis=-1)
    upper_counts_sum = (counts_temp * upper_prune_mask).sum(axis=-1)

    ood_counts_add = jnp.vstack([lower_counts_sum, upper_counts_sum]).T
    new_ood_counts = new_ood_counts + ood_counts_add

    # Rescale the counts to have a consistant total count
    if rescale_counts:
        # Add on the out of domain counts
        # This should be equal to the batch size
        max_counts_before = counts.sum(axis=1, keepdims=True) + ood_counts.sum(axis=1, keepdims=True)
        max_before_subset = max_counts_before - new_ood_counts.sum(axis=1, keepdims=True)

        # This could be a subset or superset of the original counts
        max_counts_after = new_counts.sum(axis=1, keepdims=True) # These now include the out of domain counts

        # If the sum of our counts is 0, we set it to 1 to avoid division by zero
        max_counts_after = jnp.where(max_counts_after == 0, 1, max_counts_after)

        # We want this to be equal to the subset where we subtract off the new ood_counts
        new_counts = max_before_subset * new_counts / max_counts_after

    return new_weights, new_counts, new_ood_counts

def _stretch_weights_and_counts_jax(combined_mask, activation_weights, data_counts, ood_counts, a, b, ood_a, ood_b, k, num_grid_intervals, rounding_eps, exact_stretch, basis_type):
    # Determine the new domain boundaries
    new_a = jnp.where(combined_mask, ood_a, a) # TODO Replace combined_mask with a_mask and b_mask
    new_b = jnp.where(combined_mask, ood_b, b)

    new_weights, new_counts, new_ood_counts = refit_weights_and_counts_jax(
        activation_weights,
        data_counts,
        ood_counts,
        a,
        b,
        new_num_grid_intervals=num_grid_intervals, # This stays the same during stretching
        new_a=new_a,
        new_b=new_b,
        k=k,
        rounding_eps=rounding_eps,
        exact_refit=exact_stretch,
        basis_type=basis_type
    )

    return new_weights, new_counts, new_ood_counts, new_a, new_b

def stretch_weights_and_counts_jax(activation_weights, data_counts, ood_counts, a, b, ood_a, ood_b, k, num_grid_intervals, rounding_eps, stretch_mode="max", stretch_threshold=None, exact_stretch=False, basis_type="bspline"):

    # Figure out if we need to stretch the weights or not
    if stretch_mode == "max":
        a_mask = data_counts.max(-1) < ood_counts[:, 0] # data_counts[:, 0] < ood_counts[:, 0]
        b_mask = data_counts.max(-1) < ood_counts[:, -1] # data_counts[:, -1] < ood_counts[:, -1]
    elif stretch_mode == "half_max":
        a_mask = data_counts.max(-1)*0.5 < ood_counts[:, 0]
        b_mask = data_counts.max(-1)*0.5 < ood_counts[:, -1]
    elif stretch_mode == "mean":
        a_mask = data_counts.mean(-1) < ood_counts[:, 0]
        b_mask = data_counts.mean(-1) < ood_counts[:, -1]
    elif stretch_mode == "edge":
        a_mask = data_counts[:, 0] < ood_counts[:, 0]
        b_mask = data_counts[:, -1] < ood_counts[:, -1]
    elif stretch_mode == "threshold":
        a_mask = data_counts.max(-1) < stretch_threshold
        b_mask = data_counts.max(-1) < stretch_threshold
    elif stretch_mode == "relative":
        a_mask = data_counts.min(-1) < ood_counts[:, 0]*0.1
        b_mask = data_counts.min(-1) < ood_counts[:, -1]*0.1
    elif stretch_mode == "interval":
        a_mask = jnp.ones_like(data_counts[:, 0], dtype=bool)
        b_mask = jnp.ones_like(data_counts[:, -1], dtype=bool)

    combined_mask = a_mask | b_mask
    stretch = jnp.any(combined_mask)

    # TODO Replace combined_mask with a_mask, b_mask
    updated_weights, updated_counts, updated_ood_counts, updated_a, updated_b = jax.lax.cond(
        stretch,
        lambda: _stretch_weights_and_counts_jax(combined_mask, activation_weights, data_counts, ood_counts, a, b, ood_a, ood_b, k, num_grid_intervals, rounding_eps, exact_stretch, basis_type),
        lambda: (activation_weights, data_counts, ood_counts, a, b)
    )

    # if stretch:
    #     print("Stretching:", stretch, a_mask, b_mask, updated_a, updated_b)
    
    return updated_weights, updated_counts, updated_ood_counts, updated_a, updated_b, stretch

def shrink_domain(data_counts, ood_counts, thresh):
    # Compute combined mask for pruning
    # This checks to see if the first bins or last bins for the different counts are below the set threshold
    lower_mask = data_counts[:, 0] < thresh
    upper_mask = data_counts[:, -1] < thresh
    lower_ood_mask = ood_counts[:, 0] < thresh
    upper_ood_mask = ood_counts[:, -1] < thresh
    combined_mask = (lower_mask & lower_ood_mask) | (upper_mask & upper_ood_mask)
    return combined_mask

def _shrink_weights_and_counts_jax(weights, data_counts, ood_counts, a, b, thresh, num_grid_intervals, min_delta, k, rounding_eps, exact_refit, basis_type):
    # Create mask for counts above threshold
    # masked_weight_counts = jnp.where(combined_mask[:, None], weight_counts, -jnp.inf)
    above_thresh_mask = (data_counts > thresh).astype(float)
    # above_thresh_mask = (weight_counts[combined_mask] > thresh).astype(float)

    # Find first and last indices above threshold
    first_ones = jnp.argmax(above_thresh_mask, axis=1)
    last_ones = num_grid_intervals - jnp.argmax(
        jnp.flip(above_thresh_mask, axis=1), axis=1
    )

    domain = parallel_linspace_jax(a, b, num_grid_intervals + 1)
    new_a = domain[jnp.arange(first_ones.shape[0]), first_ones]
    new_b = domain[jnp.arange(last_ones.shape[0]), last_ones]

    # Enforce minimum delta
    min_delta_violated = ((new_b - new_a) / num_grid_intervals) < min_delta
    new_b = jnp.where(min_delta_violated, new_b + min_delta, new_b)

    new_weights, new_counts, new_ood_counts = refit_weights_and_counts_jax(
        weights,
        data_counts,
        ood_counts,
        a,
        b,
        new_num_grid_intervals=num_grid_intervals,
        new_a=new_a,
        new_b=new_b,
        k=k,
        rounding_eps=rounding_eps,
        exact_refit=exact_refit,
        basis_type=basis_type
    )

    return new_weights, new_counts, new_ood_counts, new_a, new_b

def shrink_weights_and_counts_jax(thresh, activation_weights, a, b, num_grid_intervals, data_counts, ood_counts, k, rounding_eps, min_delta=1e-4, exact_shrink=False, basis_type="spline"):
    combined_mask = shrink_domain(data_counts, ood_counts, thresh)
    needs_shrinking = jnp.any(combined_mask)
    counts_nonempty = jnp.any(data_counts > 0)
    shrunk = jnp.logical_and(needs_shrinking, counts_nonempty)

    new_weights, new_counts, new_ood_counts, new_a, new_b = jax.lax.cond(
        shrunk,
        lambda: _shrink_weights_and_counts_jax(activation_weights, data_counts, ood_counts, a, b, thresh, num_grid_intervals, min_delta, k, rounding_eps, exact_shrink, basis_type),
        lambda: (activation_weights, data_counts, ood_counts, a, b)
    )

    return new_weights, new_counts, new_ood_counts, new_a, new_b, shrunk

def copy_state(new_layer, layer, new_state, state, exclude=None, include=None):

    if include is None:
        include = [(new_layer.a, state.get(layer.a)),
                (new_layer.b, state.get(layer.b)),
                (new_layer.data_counts, state.get(layer.data_counts)),
                (new_layer.ood_data_counts, state.get(layer.ood_data_counts)),
                (new_layer.counts_nonempty, state.get(layer.counts_nonempty)),
                (new_layer.ood_a, state.get(layer.ood_a)),
                (new_layer.ood_b, state.get(layer.ood_b)),
                (new_layer.mask_indices, state.get(layer.mask_indices))]
    
    if exclude is not None:
        for item in exclude:
            include.remove(item)
    
    for item in include:
        # Set everything except for the data counts
        new_state = new_state.set(item[0], item[1])

    return new_state

def copy_state_subset(new_layer, layer, new_state, state, idx):
    new_a = jnp.take(state.get(layer.a), idx, axis=0)
    new_b = jnp.take(state.get(layer.b), idx, axis=0)
    new_data_counts = jnp.take(state.get(layer.data_counts), idx, axis=0)
    new_ood_data_counts = jnp.take(state.get(layer.ood_data_counts), idx, axis=0)
    new_counts_nonempty = state.get(layer.counts_nonempty)
    new_ood_a = jnp.take(state.get(layer.ood_a), idx, axis=0)
    new_ood_b = jnp.take(state.get(layer.ood_b), idx, axis=0)
    new_mask_indices = jnp.take(state.get(layer.mask_indices), idx, axis=0) # FIXME Remove this in the future!

    include = [(new_layer.a, new_a),
                (new_layer.b, new_b),
                (new_layer.data_counts, new_data_counts),
                (new_layer.ood_data_counts, new_ood_data_counts),
                (new_layer.counts_nonempty, new_counts_nonempty),
                (new_layer.ood_a, new_ood_a),
                (new_layer.ood_b, new_ood_b),
                (new_layer.mask_indices, new_mask_indices)]
    
    for item in include:
        # Set everything except for the data counts
        new_state = new_state.set(item[0], item[1])
    return new_state

def get_broadcasted_indices(layer, broadcast_shape):

    X, Y = jnp.meshgrid(jnp.arange(layer.out_dim), jnp.arange(layer.in_dim), indexing='ij')
    X = jnp.expand_dims(X, axis=2).repeat(layer.k+1, axis=2)
    Y = jnp.expand_dims(Y, axis=2).repeat(layer.k+1, axis=2)
    X_b = jnp.broadcast_to(X, broadcast_shape)
    Y_b = jnp.broadcast_to(Y, broadcast_shape)

    return X_b, Y_b

@jax.jit
def compute_marginal_log_likelihood(indices: jnp.ndarray,
                                    layer_histogram: jnp.ndarray) -> jnp.ndarray:
    """
    Args
    ----
    indices          (n, d)  : integer bin indices for each sample & dimension
    layer_histogram  (d, k)  : probability mass per bin for every dimension
                               (k = number of bins)

    Returns
    -------
    log_likelihood   (n,)    : joint log-likelihood for each sample
    """
    # gather the probability for each (dim, sample) pair
    #   indices.T has shape (d, n)
    #   result     has shape (d, n)
    bin_probs = jnp.take_along_axis(layer_histogram, indices.T, axis=1)

    # clip to avoid log(0), take log, then sum over dimensions
    log_probs = jnp.log(jnp.clip(bin_probs, a_min=1e-10))
    return jnp.sum(log_probs, axis=0)          # shape (n,)

# Generated by gemini
def cheby_coefs_from_curve_jax(y_eval, degree, n_pts=101):
    """
    Refits Chebyshev coefficients to a set of target y-values over a new domain [a, b].
    
    y_eval: (out_dim, in_dim, n_pts) - the target curve values
    a, b: (in_dim,) - the domain boundaries
    degree: int - the polynomial degree (D)
    """
    out_dim, in_dim, _ = y_eval.shape
    
    # 1. Create a standardized linspace in [-1, 1] for the solver
    # This represents the points in the new domain [a, b]
    x_standard = jnp.linspace(-1, 1, n_pts)
    
    # 2. Build the Vandermonde Matrix (The Basis Matrix)
    # Shape: (n_pts, degree + 1)
    def get_vandermonde(x):
        basis = [jnp.ones_like(x), x]
        for _ in range(2, degree + 1):
            basis.append(2.0 * x * basis[-1] - basis[-2])
        return jnp.stack(basis, axis=-1)

    # Phi is the matrix of basis functions evaluated at n_pts
    Phi = get_vandermonde(x_standard) 
    
    # 3. Solve Least Squares: Phi @ weights = y_eval
    # We want to solve for weights: (out_dim, in_dim, degree + 1)
    # JAX's lstsq handles the batching via vmap efficiently
    
    def solve_single_channel(target_y):
        # target_y shape: (n_pts,)
        # Phi shape: (n_pts, degree + 1)
        coefs, _, _, _ = jnp.linalg.lstsq(Phi, target_y)
        return coefs

    # Vmap over out_dim and in_dim
    # Resulting weights: (out_dim, in_dim, degree + 1)
    new_weights = jax.vmap(jax.vmap(solve_single_channel))(y_eval)
    
    return new_weights