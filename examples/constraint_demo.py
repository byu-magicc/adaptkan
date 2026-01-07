"""
Demonstration of hard constraints for AdaptKAN with Chebyshev basis.

This demo shows how to use hard constraints to enforce specific function values
and derivatives at given points. Constraints are exactly satisfied throughout training.

Example use cases:
- Boundary conditions for ODEs/PDEs
- Known function values at specific points
- Smoothness constraints (zero derivative)
- Physics-informed neural networks (PINNs)
"""

import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import matplotlib.pyplot as plt

from adaptkan.jax.layers import AdaptKANLayerJax
from adaptkan.jax.viz import plot_layer


# Helper to create a simple 1-layer model wrapper
class ConstrainedKAN(eqx.Module):
    """Simple wrapper around a single constrained layer."""
    layer: AdaptKANLayerJax

    def __call__(self, x, state, update=False):
        return self.layer(x, state, update=update)

    @property
    def layers(self):
        """Expose layers list for visualization."""
        return [self.layer]

    def adapt(self, state):
        """Delegate adaptation to layer."""
        new_layer, state, adapted = self.layer.adapt(state)
        # Update the model with new layer
        new_model = eqx.tree_at(lambda m: m.layer, self, new_layer)
        return new_model, state, adapted


def demo_point_constraints():
    """
    Demo 1: Point constraints only.

    We constrain the network to pass through specific points:
    - f(0) = 0
    - f(1) = 1

    Then train to fit a sine-like function while maintaining these constraints.
    """
    print("=" * 60)
    print("Demo 1: Point Constraints")
    print("=" * 60)
    print("Constraints: f(0) = 0, f(1) = 1")
    print()

    # Define constraints
    # constraints_in: (n_constraints, 2) where each row is (x_position, derivative_order)
    # derivative_order = 0 means value constraint
    constraints_in = jnp.array([
        [0.0, 0],  # f(0) = target
        [1.0, 0],  # f(1) = target
    ])

    # constraints_y: (out_dim, n_constraints) - target values for each output and constraint
    constraints_y = jnp.array([
        [0.0, 1.0],  # Output 0: f(0)=0, f(1)=1
    ])

    # Create layer with constraints
    layer, state = eqx.nn.make_with_state(AdaptKANLayerJax)(
        in_dim=1,
        out_dim=1,
        num_grid_intervals=7,
        k=7,
        basis_type='chebyshev',
        initialization_range=[0.0, 1.0],
        constraints_in=constraints_in,
        constraints_y=constraints_y,
        key=jax.random.PRNGKey(42),
    )
    model = ConstrainedKAN(layer)

    # Target function: a sine curve that happens to match our constraints
    def target_fn(x):
        return jnp.sin(jnp.pi * x / 2)  # sin(pi*x/2) gives 0 at x=0, 1 at x=1

    # Training data
    key = jax.random.PRNGKey(0)
    x_train = jax.random.uniform(key, (100, 1), minval=0.0, maxval=1.0)
    y_train = target_fn(x_train)

    # Verify constraints before training
    x_test = jnp.array([[0.0], [1.0]])
    y_pred, state, *_ = model(x_test, state)
    print(f"Before training:")
    print(f"  f(0) = {y_pred[0, 0]:.6f} (target: 0)")
    print(f"  f(1) = {y_pred[1, 0]:.6f} (target: 1)")

    # Training loop
    optimizer = optax.adam(0.01)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    @eqx.filter_jit
    def train_step(model, state, opt_state, x, y):
        def loss_fn(model, state, x, y):
            pred, state, *_ = model(x, state)
            return jnp.mean((pred - y) ** 2), state

        (loss, state), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(
            model, state, x, y
        )
        updates, opt_state = optimizer.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        return model, state, opt_state, loss

    print("\nTraining...")
    for step in range(200):
        model, state, opt_state, loss = train_step(model, state, opt_state, x_train, y_train)
        if step % 50 == 0:
            y_pred, state, *_ = model(x_test, state)
            print(f"  Step {step}: loss={loss:.6f}, f(0)={y_pred[0, 0]:.6f}, f(1)={y_pred[1, 0]:.6f}")

    # Final verification
    y_pred, state, *_ = model(x_test, state)
    print(f"\nAfter training:")
    print(f"  f(0) = {y_pred[0, 0]:.6f} (target: 0)")
    print(f"  f(1) = {y_pred[1, 0]:.6f} (target: 1)")

    # Plot
    fig = plot_layer(model, state, layer_index=0, title="Point Constraints: f(0)=0, f(1)=1")
    plt.show()

    return model, state


def demo_derivative_constraints():
    """
    Demo 2: Point and derivative constraints.

    We constrain the network:
    - f(0) = 0   (starts at origin)
    - f(1) = 0   (ends at zero)
    - f'(0) = 0  (flat at start)
    - f'(1) = 0  (flat at end)

    This creates a "bump" function with smooth endpoints.
    """
    print("\n" + "=" * 60)
    print("Demo 2: Point + Derivative Constraints")
    print("=" * 60)
    print("Constraints: f(0)=0, f(1)=0, f'(0)=0, f'(1)=0")
    print()

    # Define constraints with both position and derivative
    constraints_in = jnp.array([
        [0.0, 0],  # f(0) = target (position)
        [1.0, 0],  # f(1) = target (position)
        [0.0, 1],  # f'(0) = target (derivative)
        [1.0, 1],  # f'(1) = target (derivative)
    ])

    constraints_y = jnp.array([
        [0.0, 0.0, 0.0, 0.0],  # f(0)=0, f(1)=0, f'(0)=0, f'(1)=0
    ])

    # Create layer - use higher degree for derivative constraints
    layer, state = eqx.nn.make_with_state(AdaptKANLayerJax)(
        in_dim=1,
        out_dim=1,
        num_grid_intervals=9,
        k=9,
        basis_type='chebyshev',
        initialization_range=[0.0, 1.0],
        constraints_in=constraints_in,
        constraints_y=constraints_y,
        key=jax.random.PRNGKey(42),
    )
    model = ConstrainedKAN(layer)

    # Target function: a smooth bump
    def target_fn(x):
        # Smooth bump: sin^2(pi*x) has the right boundary conditions
        return jnp.sin(jnp.pi * x) ** 2

    # Training data
    key = jax.random.PRNGKey(0)
    x_train = jax.random.uniform(key, (100, 1), minval=0.0, maxval=1.0)
    y_train = target_fn(x_train)

    # Verify constraints before training
    x_test = jnp.array([[0.0], [1.0]])
    y_pred, state, *_ = model(x_test, state)
    print(f"Before training:")
    print(f"  f(0) = {y_pred[0, 0]:.6f} (target: 0)")
    print(f"  f(1) = {y_pred[1, 0]:.6f} (target: 0)")

    # Training loop
    optimizer = optax.adam(0.01)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    @eqx.filter_jit
    def train_step(model, state, opt_state, x, y):
        def loss_fn(model, state, x, y):
            pred, state, *_ = model(x, state)
            return jnp.mean((pred - y) ** 2), state

        (loss, state), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(
            model, state, x, y
        )
        updates, opt_state = optimizer.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        return model, state, opt_state, loss

    print("\nTraining...")
    for step in range(200):
        model, state, opt_state, loss = train_step(model, state, opt_state, x_train, y_train)
        if step % 50 == 0:
            y_pred, state, *_ = model(x_test, state)
            print(f"  Step {step}: loss={loss:.6f}, f(0)={y_pred[0, 0]:.6f}, f(1)={y_pred[1, 0]:.6f}")

    # Final verification
    y_pred, state, *_ = model(x_test, state)
    print(f"\nAfter training:")
    print(f"  f(0) = {y_pred[0, 0]:.6f} (target: 0)")
    print(f"  f(1) = {y_pred[1, 0]:.6f} (target: 0)")

    # Plot
    fig = plot_layer(model, state, layer_index=0,
                     title="Derivative Constraints: f(0)=f(1)=0, f'(0)=f'(1)=0",
                     tangent_length=0.2)
    plt.show()

    return model, state


def demo_constraints_with_adaptation():
    """
    Demo 3: Constraints with domain adaptation.

    Shows that constraints are maintained even when the domain adapts
    to out-of-distribution data.
    """
    print("\n" + "=" * 60)
    print("Demo 3: Constraints with Domain Adaptation")
    print("=" * 60)
    print("Initial domain: [0, 1], then adapt to data in [-1, 2]")
    print("Constraints: f(0)=0, f(1)=1 (within initial domain)")
    print()

    constraints_in = jnp.array([
        [0.0, 0],
        [1.0, 0],
    ])

    constraints_y = jnp.array([
        [0.0, 1.0],
    ])

    layer, state = eqx.nn.make_with_state(AdaptKANLayerJax)(
        in_dim=1,
        out_dim=1,
        num_grid_intervals=7,
        k=7,
        basis_type='chebyshev',
        initialization_range=[0.0, 1.0],
        constraints_in=constraints_in,
        constraints_y=constraints_y,
        prune_patience=1,
        key=jax.random.PRNGKey(42),
    )
    model = ConstrainedKAN(layer)

    print(f"Initial domain: [{state.get(layer.a)[0]:.2f}, {state.get(layer.b)[0]:.2f}]")

    # Verify constraints before adaptation
    x_test = jnp.array([[0.0], [1.0]])
    y_pred, state, *_ = model(x_test, state)
    print(f"Before adaptation: f(0)={y_pred[0, 0]:.6f}, f(1)={y_pred[1, 0]:.6f}")

    # Feed OOD data to trigger adaptation
    key = jax.random.PRNGKey(0)
    x_ood = jax.random.uniform(key, (200, 1), minval=-1.0, maxval=2.0)

    print("\nFeeding out-of-domain data and adapting...")
    for _ in range(3):
        _, state, *_ = model(x_ood, state, update=True)
        model, state, adapted = model.adapt(state)
        if adapted:
            layer = model.layers[0]
            print(f"  Adapted to: [{state.get(layer.a)[0]:.2f}, {state.get(layer.b)[0]:.2f}]")

    # Verify constraints after adaptation
    y_pred, state, *_ = model(x_test, state)
    print(f"\nAfter adaptation: f(0)={y_pred[0, 0]:.6f}, f(1)={y_pred[1, 0]:.6f}")

    layer = model.layers[0]
    print(f"Final domain: [{state.get(layer.a)[0]:.2f}, {state.get(layer.b)[0]:.2f}]")

    # Plot
    fig = plot_layer(model, state, layer_index=0,
                     title="Constraints maintained after domain adaptation")
    plt.show()

    return model, state


def demo_compare_with_without_constraints():
    """
    Demo 4: Compare training with and without constraints.

    Shows how constraints affect the learned function when trying
    to fit data that conflicts with the constraints.
    """
    print("\n" + "=" * 60)
    print("Demo 4: With vs Without Constraints")
    print("=" * 60)
    print("Target: linear function f(x) = 2x")
    print("Constraint (for constrained model): f(0) = 1 (conflicts with target!)")
    print()

    # Target function
    def target_fn(x):
        return 2 * x

    # Training data - linear function
    key = jax.random.PRNGKey(0)
    x_train = jax.random.uniform(key, (100, 1), minval=0.0, maxval=1.0)
    y_train = target_fn(x_train)

    # Constraint that conflicts with the target: f(0) = 1 instead of 0
    constraints_in = jnp.array([[0.0, 0]])
    constraints_y = jnp.array([[1.0]])  # f(0) = 1

    # Create unconstrained layer
    layer_u, state_u = eqx.nn.make_with_state(AdaptKANLayerJax)(
        in_dim=1,
        out_dim=1,
        num_grid_intervals=5,
        k=5,
        basis_type='chebyshev',
        initialization_range=[0.0, 1.0],
        key=jax.random.PRNGKey(42),
    )
    model_unconstrained = ConstrainedKAN(layer_u)
    state_unconstrained = state_u

    # Create constrained layer
    layer_c, state_c = eqx.nn.make_with_state(AdaptKANLayerJax)(
        in_dim=1,
        out_dim=1,
        num_grid_intervals=5,
        k=5,
        basis_type='chebyshev',
        initialization_range=[0.0, 1.0],
        constraints_in=constraints_in,
        constraints_y=constraints_y,
        key=jax.random.PRNGKey(42),
    )
    model_constrained = ConstrainedKAN(layer_c)
    state_constrained = state_c

    # Train both models
    optimizer = optax.adam(0.01)

    @eqx.filter_jit
    def train_step(model, state, opt_state, x, y, optimizer):
        def loss_fn(model, state, x, y):
            pred, state, *_ = model(x, state)
            return jnp.mean((pred - y) ** 2), state

        (loss, state), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(
            model, state, x, y
        )
        updates, opt_state = optimizer.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        return model, state, opt_state, loss

    opt_state_u = optimizer.init(eqx.filter(model_unconstrained, eqx.is_array))
    opt_state_c = optimizer.init(eqx.filter(model_constrained, eqx.is_array))

    print("Training both models...")
    for step in range(200):
        model_unconstrained, state_unconstrained, opt_state_u, loss_u = train_step(
            model_unconstrained, state_unconstrained, opt_state_u, x_train, y_train, optimizer
        )
        model_constrained, state_constrained, opt_state_c, loss_c = train_step(
            model_constrained, state_constrained, opt_state_c, x_train, y_train, optimizer
        )

    # Compare at x=0
    x_test = jnp.array([[0.0], [0.5], [1.0]])
    y_u, state_unconstrained, *_ = model_unconstrained(x_test, state_unconstrained)
    y_c, state_constrained, *_ = model_constrained(x_test, state_constrained)

    print("\nResults:")
    print("  x | Target | Unconstrained | Constrained (f(0)=1)")
    print("  --|--------|---------------|---------------------")
    for i, x_val in enumerate([0.0, 0.5, 1.0]):
        print(f"  {x_val:.1f} |  {target_fn(jnp.array([x_val]))[0]:.3f} |     {y_u[i, 0]:.3f}     |        {y_c[i, 0]:.3f}")

    # Create combined plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    x_plot = jnp.linspace(0, 1, 100).reshape(-1, 1)
    y_u_plot, _, *_ = model_unconstrained(x_plot, state_unconstrained)
    y_c_plot, _, *_ = model_constrained(x_plot, state_constrained)
    y_target = target_fn(x_plot)

    axes[0].plot(x_plot, y_target, 'b--', label='Target: f(x)=2x', linewidth=2)
    axes[0].plot(x_plot, y_u_plot, 'g-', label='Learned', linewidth=2)
    axes[0].set_title('Unconstrained')
    axes[0].legend()
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('f(x)')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(x_plot, y_target, 'b--', label='Target: f(x)=2x', linewidth=2)
    axes[1].plot(x_plot, y_c_plot, 'g-', label='Learned', linewidth=2)
    axes[1].scatter([0], [1], color='red', s=100, zorder=5, label='Constraint: f(0)=1')
    axes[1].set_title('Constrained: f(0)=1')
    axes[1].legend()
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('f(x)')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    return (model_unconstrained, state_unconstrained), (model_constrained, state_constrained)


if __name__ == "__main__":
    print("AdaptKAN Hard Constraints Demo")
    print("==============================\n")

    demo_point_constraints()
    demo_derivative_constraints()
    demo_constraints_with_adaptation()
    demo_compare_with_without_constraints()

    print("\n" + "=" * 60)
    print("All demos completed!")
    print("=" * 60)
