"""
Demo: Constrained training with domain adaptation and visualization.

This demo shows:
1. Constraints (point and derivative) that remain satisfied during training
2. Domain adaptation triggered by out-of-distribution data
3. Comparison between learned function and target function
4. Histogram visualization of data distribution
"""

import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import matplotlib.pyplot as plt
import numpy as np

from adaptkan.jax.layers import AdaptKANLayerJax
from adaptkan.jax.viz import plot_layer


class ConstrainedKAN(eqx.Module):
    """Simple wrapper around a single constrained layer."""
    layer: AdaptKANLayerJax

    def __call__(self, x, state, update=True):
        """Forward pass. Set update=True to track data distribution for histograms."""
        return self.layer(x, state, update=update)

    @property
    def layers(self):
        """Expose layers list for visualization."""
        return [self.layer]

    def adapt(self, state):
        """Delegate adaptation to layer."""
        new_layer, state, adapted = self.layer.adapt(state)
        new_model = eqx.tree_at(lambda m: m.layer, self, new_layer)
        return new_model, state, adapted


def get_domain(model, state):
    """Helper to get current domain bounds."""
    layer = model.layers[0]
    a = float(state.get(layer.a)[0])
    b = float(state.get(layer.b)[0])
    return a, b


def run_demo():
    """
    Train a constrained network with domain adaptation.

    Constraints:
    - f(0) = 0  (point constraint)
    - f(1) = 0  (point constraint)
    - f'(0) = 0 (derivative constraint - smooth at boundary)
    - f'(1) = 0 (derivative constraint - smooth at boundary)

    Target function: sin^2(pi*x) which naturally satisfies these constraints.

    Training data is sampled from [-1, 2] to trigger domain adaptation.
    """

    print("=" * 60)
    print("Constrained Training with Domain Adaptation Demo")
    print("=" * 60)

    # ============================================================
    # 1. Define constraints
    # ============================================================
    # constraints_in shape: (n_constraints, 2) where each row is [x, derivative_order]
    # derivative_order: 0 = point constraint, 1 = first derivative, etc.
    constraints_in = jnp.array([
        [0.0, 0],  # f(0) = target (position constraint)
        [1.0, 0],  # f(1) = target (position constraint)
        [0.0, 1],  # f'(0) = target (derivative constraint)
        [1.0, 1],  # f'(1) = target (derivative constraint)
    ])

    # constraints_y shape: (out_dim, n_constraints)
    # Target values for each constraint
    constraints_y = jnp.array([
        [0.0, 0.0, 0.0, 0.0],  # f(0)=0, f(1)=0, f'(0)=0, f'(1)=0
    ])

    print("\nConstraints:")
    print("  f(0) = 0   (boundary value)")
    print("  f(1) = 0   (boundary value)")
    print("  f'(0) = 0  (zero slope at boundary)")
    print("  f'(1) = 0  (zero slope at boundary)")

    # ============================================================
    # 2. Create constrained layer
    # ============================================================
    layer, state = eqx.nn.make_with_state(AdaptKANLayerJax)(
        in_dim=1,
        out_dim=1,
        num_grid_intervals=9,
        k=9,  # Higher degree for derivative constraints
        basis_type='chebyshev',
        initialization_range=[0.0, 1.0],
        constraints_in=constraints_in,
        constraints_y=constraints_y,
        prune_patience=3,       # Wait 3 steps before shrinking domain
        stretch_threshold=0.05,  # Stretch if >5% data is OOD
        key=jax.random.PRNGKey(42),
    )
    model = ConstrainedKAN(layer)

    a_init, b_init = get_domain(model, state)
    print(f"\nInitial domain: [{a_init:.2f}, {b_init:.2f}]")

    # ============================================================
    # 3. Define target function and generate training data
    # ============================================================
    def target_fn(x):
        """
        Target: sin^2(pi*x)

        This function naturally satisfies our constraints:
        - f(0) = sin^2(0) = 0
        - f(1) = sin^2(pi) = 0
        - f'(0) = 2*sin(0)*cos(0)*pi = 0
        - f'(1) = 2*sin(pi)*cos(pi)*pi = 0
        """
        return jnp.sin(jnp.pi * x) ** 2

    # Training data OUTSIDE initial domain to trigger adaptation
    key = jax.random.PRNGKey(0)
    x_train = jax.random.uniform(key, (200, 1), minval=-1.0, maxval=2.0)
    y_train = target_fn(x_train)

    print(f"\nTraining data range: [{float(x_train.min()):.2f}, {float(x_train.max()):.2f}]")
    print("  (This is outside initial domain, so adaptation should occur)")

    # ============================================================
    # 4. Verify constraints before training
    # ============================================================
    x_test = jnp.array([[0.0], [1.0]])
    y_pred, state, *_ = model(x_test, state, update=False)
    print(f"\nBefore training (constraints should be satisfied):")
    print(f"  f(0) = {float(y_pred[0, 0]):.6f} (target: 0)")
    print(f"  f(1) = {float(y_pred[1, 0]):.6f} (target: 0)")

    # ============================================================
    # 5. Training loop with adaptation tracking
    # ============================================================
    optimizer = optax.adam(0.01)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    @eqx.filter_jit
    def train_step(model, state, opt_state, x, y):
        def loss_fn(model, state, x, y):
            # update=True to track data distribution for histograms
            pred, state, *_ = model(x, state, update=True)
            return jnp.mean((pred - y) ** 2), state

        (loss, state), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(
            model, state, x, y
        )

        # Apply adaptation (this will expand domain if data is OOD)
        model, state, adapted = model.adapt(state)

        # Apply optimizer updates
        updates, opt_state = optimizer.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)

        # Return domain bounds for logging (outside JIT we can print)
        layer = model.layers[0]
        a = state.get(layer.a)[0]
        b = state.get(layer.b)[0]

        return model, state, opt_state, loss, adapted, a, b

    print("\n" + "-" * 60)
    print("Training...")
    print("-" * 60)

    adaptation_events = []
    prev_domain = (a_init, b_init)

    num_steps = 300
    for step in range(num_steps):
        model, state, opt_state, loss, adapted, a, b = train_step(
            model, state, opt_state, x_train, y_train
        )

        current_domain = (float(a), float(b))

        # Check if adaptation happened by comparing domains
        if current_domain != prev_domain:
            adaptation_events.append((step, current_domain))
            print(f"  Step {step}: ADAPTED to [{current_domain[0]:.3f}, {current_domain[1]:.3f}]")
            prev_domain = current_domain

        # Regular progress logging
        if step % 50 == 0:
            y_pred, _, *_ = model(x_test, state, update=False)
            print(f"  Step {step:3d}: loss={float(loss):.6f}, "
                  f"f(0)={float(y_pred[0, 0]):.6f}, f(1)={float(y_pred[1, 0]):.6f}, "
                  f"domain=[{current_domain[0]:.2f}, {current_domain[1]:.2f}]")

    # ============================================================
    # 6. Final verification
    # ============================================================
    y_pred, state, *_ = model(x_test, state, update=False)
    a_final, b_final = get_domain(model, state)

    print("\n" + "-" * 60)
    print("After training:")
    print("-" * 60)
    print(f"  f(0) = {float(y_pred[0, 0]):.6f} (target: 0)")
    print(f"  f(1) = {float(y_pred[1, 0]):.6f} (target: 0)")
    print(f"  Final domain: [{a_final:.3f}, {b_final:.3f}]")
    print(f"  Adaptation events: {len(adaptation_events)}")

    # ============================================================
    # 7. Create comparison plot: learned vs target
    # ============================================================
    print("\n" + "-" * 60)
    print("Creating visualizations...")
    print("-" * 60)

    # Dense evaluation grid over final domain (with some padding)
    x_plot = jnp.linspace(a_final - 0.1, b_final + 0.1, 200).reshape(-1, 1)
    y_learned, _, *_ = model(x_plot, state, update=False)
    y_target = target_fn(x_plot)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left plot: Learned vs Target comparison
    ax1 = axes[0]
    ax1.plot(x_plot, y_target, 'b-', linewidth=2, label='Target: sin²(πx)')
    ax1.plot(x_plot, y_learned, 'g--', linewidth=2, label='Learned')

    # Mark constraint points
    ax1.scatter([0, 1], [0, 0], color='red', s=100, zorder=5,
                marker='o', label='Point constraints')

    # Mark training data range
    ax1.axvline(x=-1.0, color='gray', linestyle=':', alpha=0.5, label='Training range')
    ax1.axvline(x=2.0, color='gray', linestyle=':', alpha=0.5)

    # Mark initial domain
    ax1.axvspan(0, 1, alpha=0.1, color='yellow', label='Initial domain [0,1]')

    ax1.set_xlabel('x', fontsize=12)
    ax1.set_ylabel('f(x)', fontsize=12)
    ax1.set_title('Learned Function vs Target', fontsize=14)
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # Right plot: Error
    ax2 = axes[1]
    error = np.abs(np.array(y_learned) - np.array(y_target))
    ax2.plot(x_plot, error, 'r-', linewidth=2)
    ax2.axvline(x=0, color='red', linestyle='--', alpha=0.5)
    ax2.axvline(x=1, color='red', linestyle='--', alpha=0.5)
    ax2.set_xlabel('x', fontsize=12)
    ax2.set_ylabel('|Learned - Target|', fontsize=12)
    ax2.set_title('Absolute Error', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')

    plt.tight_layout()
    plt.savefig('constraint_comparison.png', dpi=150, bbox_inches='tight')
    print("  Saved: constraint_comparison.png")
    plt.show()

    # ============================================================
    # 8. Plot layer visualization with histograms
    # ============================================================
    fig = plot_layer(
        model, state, layer_index=0,
        title=f"Constrained Layer (domain: [{a_final:.2f}, {b_final:.2f}])",
        tangent_length=0.2,
        square_size=2.0,
        fontsize=10,
        dpi=150
    )
    plt.savefig('constraint_layer.png', dpi=150, bbox_inches='tight')
    print("  Saved: constraint_layer.png")
    plt.show()

    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)

    return model, state


if __name__ == "__main__":
    model, state = run_demo()
