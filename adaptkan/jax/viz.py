"""Visualization utilities for JAX backend."""
import matplotlib.pyplot as plt
import numpy as np
import os
import imageio
import io
import numpy as np
from PIL import Image
import jax.numpy as jnp

from adaptkan.jax.utils import (
    chebyshev_interpolate_jax,
    spline_interpolate_jax,
    parallel_linspace_jax,
    compute_chebyshev_basis)

def plot_layer(
    model,
    state,
    layer_index=0,
    square_size=1.3,
    fontsize=6,
    domain_input_size=100,
    title=None,
    dpi=100,
    ood_bin_width_ratio=0.75,  # allow customizing OOD bin thickness
    show_constraints=True,  # whether to show constraint markers
    tangent_length=0.15  # length of derivative tangent lines (as fraction of domain)
):
    if title is None:
        title = f"Layer {layer_index+1}"
    layer = model.layers[layer_index]
    layer_weights = layer.weights
    layer_a = state.get(layer.a)
    layer_b = state.get(layer.b)
    ood_data_counts = state.get(layer.ood_data_counts)
    num_grid_intervals = layer.num_grid_intervals
    data_counts = state.get(layer.data_counts)

    # Get projected weights if constraints are present (for plotting constrained curves)
    projected_weights = None
    if show_constraints and layer.has_constraints:
        projected_weights = layer.get_projected_weights(state)

    img = plot_layer_from_weights(
        layer_weights,
        layer_a,
        layer_b,
        num_grid_intervals,
        data_counts,
        ood_data_counts,
        layer_index,
        layer.k,
        layer.rounding_precision_eps,
        basis_type=layer.basis_type,
        square_size=square_size,
        fontsize=fontsize,
        domain_input_size=domain_input_size,
        title=title,
        dpi=dpi,
        ood_bin_width_ratio=ood_bin_width_ratio,
        projected_weights=projected_weights,
        tangent_length=tangent_length
    )
    # -------- wrap the RGBA array in a live figure --------
    h_px, w_px = img.shape[:2]
    fig = plt.figure(figsize=(w_px / dpi, h_px / dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])  # fill entire canvas
    ax.imshow(img)
    ax.axis("off")
    return fig        

# TODO Need to update this for higher order splines
def plot_layer_from_weights(
    layer_weights,
    layer_a,
    layer_b,
    layer_num_grid_intervals,
    layer_data_counts,
    layer_ood_data_counts,
    layer_index,
    layer_k,
    layer_rounding_precision_eps,
    basis_type="bspline",
    square_size=1.3,
    fontsize=6,
    domain_input_size=100,
    title=None,
    ood_bin_width_ratio=0.75,
    dpi=100,
    projected_weights=None,  # (out_dim, in_dim, k+1) - weights after projection (for constrained layers)
    tangent_length=0.15  # unused, kept for API compatibility
):
    """
    Plot one layer of activations plus a histogram row at the bottom.
    The layout is fixed so that each cell has square_size inches in each dimension.

    If projected_weights is provided, the constrained curves will be shown.
    Note: Combined constraints apply to layer outputs (sum of activations), not individual
    activation functions, so we only show the projected curves without per-activation markers.
    """

    # Pull out weights & counts
    domains = parallel_linspace_jax(layer_a, layer_b, domain_input_size+1)

    # Use projected weights for plotting if available (to show constrained curve)
    weights_to_plot = projected_weights if projected_weights is not None else layer_weights

    if basis_type == "bspline":
        preds, _, _ = spline_interpolate_jax(domains.T, layer_a, layer_b, weights_to_plot, layer_k, layer_rounding_precision_eps)
    elif basis_type == "chebyshev":
        preds, _, _ = chebyshev_interpolate_jax(domains.T, layer_a, layer_b, weights_to_plot, layer_num_grid_intervals, layer_rounding_precision_eps)
    else:
        raise NotImplementedError("Only B-spline and Chebyshev basis are currently supported in plotting.")
    
    m, n, _ = layer_weights.shape

    # Get the bin edges and centers for the histogram of the counts
    bin_count_edges = parallel_linspace_jax(layer_a, layer_b, layer_num_grid_intervals + 1)
    bin_count_centers = (bin_count_edges[:,:-1] + bin_count_edges[:,1:]) / 2
    bin_count_widths = bin_count_edges[:,1:2] - bin_count_edges[:,:1]

    ncols = n
    nrows = m + 1

    fig, axs = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(ncols * square_size, nrows * square_size),
        squeeze=False 
    )
    plt.subplots_adjust(left=0.05, right=0.95, top=0.90, bottom=0.1, wspace=0.0, hspace=0.0)

    # 1) Bottom row: histogram with domain labels on x-axis
    for col_idx in range(ncols):
        ax_bottom = axs[m, col_idx]
        counts = layer_data_counts[col_idx]
        ood_counts = layer_ood_data_counts[col_idx]
        
        bin_centers = bin_count_centers[col_idx]
        bin_width = bin_count_widths[col_idx][0]
        bin_edges = bin_count_edges[col_idx]
        
        # NEW: Make OOD bins thinner
        ood_bin_width = bin_width * ood_bin_width_ratio
        
        # Position OOD bins at the edges with thin width
        left_ood_center = bin_edges[0] - ood_bin_width / 2
        right_ood_center = bin_edges[-1] + ood_bin_width / 2
        
        # Plot regular bins
        ax_bottom.bar(bin_centers, counts, width=bin_width, color='gray')
        
        # Plot thin OOD bins
        ax_bottom.bar([left_ood_center], [ood_counts[0]], width=ood_bin_width, color='black')
        ax_bottom.bar([right_ood_center], [ood_counts[1]], width=ood_bin_width, color='black')
        
        # NEW: Extend x-axis to include the OOD bins
        # plot_xmin = left_ood_center - ood_bin_width / 2
        # plot_xmax = right_ood_center + ood_bin_width / 2
        # ax_bottom.set_xlim(plot_xmin, plot_xmax)
        ax_bottom.set_xlim(left_ood_center, right_ood_center)
        
        # Set x-ticks to show the domain boundaries
        ax_bottom.set_xticks([bin_edges[0], bin_edges[-1]])
        ax_bottom.set_xticklabels([
            f"{layer_a[col_idx]:.2f}",
            f"{layer_b[col_idx]:.2f}"
        ], fontsize=fontsize, rotation=45)
        ax_bottom.set_yticks([])

        ymin, ymax = counts.min(), counts.max()
        ax_bottom.text(0.1, 0.9, f"{ymax:.1f}",
                       transform=ax_bottom.transAxes,
                       va='top', ha='left', fontsize=fontsize,
                       bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
        ax_bottom.text(0.1, 0.1, f"{ymin:.1f}",
                       transform=ax_bottom.transAxes,
                       va='bottom', ha='left', fontsize=fontsize,
                       bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
        if col_idx == 0:
            ax_bottom.set_ylabel('Histogram', fontsize=fontsize)

    # 2) Activation rows
    for row_idx in range(m):
        for col_idx in range(ncols):
            ax = axs[row_idx, col_idx]
            x = domains[row_idx]
            y = preds[:, row_idx, col_idx]

            if row_idx == 0:
                ax.set_title(f'$\\phi^{{{layer_index+1}}}_{{\\cdot,{col_idx}}}$', fontsize=fontsize*2)

            ax.plot(x, y, color='green')

            # Set x-axis limits based on the actual domain (gray bins) plus fixed padding
            bin_edges = bin_count_edges[col_idx]
            bin_width = bin_count_widths[col_idx][0]
            padding = bin_width * ood_bin_width_ratio  # Fixed padding amount
            
            # Always pad by the same amount on each side
            plot_xmin = bin_edges[0] - padding
            plot_xmax = bin_edges[-1] + padding
            ax.set_xlim(plot_xmin, plot_xmax)
            
            ymin, ymax = y.min(), y.max()
            ax.text(0.1, 0.9, f"{ymax:.2f}",
                    transform=ax.transAxes,
                    va='top', ha='left', fontsize=fontsize,
                    color='red',
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
            ax.text(0.1, 0.1, f"{ymin:.2f}",
                    transform=ax.transAxes,
                    va='bottom', ha='left', fontsize=fontsize,
                    color='blue',
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

            if row_idx < m:
                ax.set_xticks([])

            ax.set_yticks([])
            if col_idx == 0:
                ax.set_ylabel(
                    f'$\\phi^{{{layer_index+1}}}_{{{row_idx+1},\\cdot}}$',
                    rotation=0,
                    fontsize=fontsize*2,
                    labelpad=2,
                    ha='right',
                    va='center'
                )
    
    title_y = 1.1
    fig.suptitle(title, fontsize=fontsize*2, y=title_y)
    
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches='tight', pad_inches=0.1)  # Use dpi here
    plt.close(fig)
    buf.seek(0)
    img = np.array(Image.open(buf))
    return img

def plot_model(model, state, square_size=1.3, fontsize=6, dpi=100, ood_bin_width_ratio=0.75):
    for layer_idx in range(len(model.layers)):
        fig = plot_layer(model, state, layer_idx, square_size, fontsize, dpi=dpi, ood_bin_width_ratio=ood_bin_width_ratio)
        plt.show()

def render_animations(frames_per_layer, save_path: str, save_name: str, fps: int, save_frames: bool = False):
    """Save per-layer GIFs and optionally individual frames after training."""
    base_dir = os.path.join(save_path, save_name)
    os.makedirs(base_dir, exist_ok=True)
    
    for layer_idx, frames in enumerate(frames_per_layer):
        # Save GIF
        gif_filename = os.path.join(base_dir, f"layer_{layer_idx+1}_progress.gif")
        imageio.mimsave(gif_filename, frames, fps=fps)
        print(f"Saved animation for Layer {layer_idx+1} -> {gif_filename}")
        
        # Save individual frames if requested
        if save_frames:
            frames_dir = os.path.join(base_dir, f"layer_{layer_idx+1}_frames")
            os.makedirs(frames_dir, exist_ok=True)
            
            for frame_idx, frame in enumerate(frames):
                frame_filename = os.path.join(frames_dir, f"frame_{frame_idx:04d}.png")
                imageio.imwrite(frame_filename, frame)
            
            print(f"Saved {len(frames)} frames for Layer {layer_idx+1} -> {frames_dir}")