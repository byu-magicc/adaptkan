"""Comprehensive unit tests for combined (sum-of-activations) constraint system."""
import jax
import jax.numpy as jnp
import equinox as eqx
import pytest

from adaptkan.jax.layers import AdaptKANLayerJax
from adaptkan.jax.model import AdaptKANJax
from adaptkan.jax.utils import (
    build_combined_constraint_matrix,
    compute_chebyshev_basis,
    compute_constraint_projection_operator,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def key():
    return jax.random.PRNGKey(42)


# =============================================================================
# Test: Layer-level combined constraint initialization
# =============================================================================

class TestLayerConstraintInitialization:
    """Tests for layer-level combined constraint initialization."""

    def test_no_constraints_initialization(self, key):
        """Layer without constraints should have has_constraints=False."""
        layer = AdaptKANLayerJax(
            in_dim=3,
            out_dim=2,
            k=5,
            basis_type="chebyshev",
            constraints_in=None,
            constraints_out=None,
            key=key,
        )

        assert layer.has_constraints is False
        assert layer.constraints_in is None
        assert layer.constraints_out is None
        assert layer.constraints_C is None
        assert layer.constraints_P is None
        assert layer.constraints_a is None
        assert layer.constraints_b is None

    def test_bspline_ignores_constraints(self, key):
        """B-spline basis should ignore constraints."""
        out_dim, in_dim = 2, 3
        n_constraints = 2

        constraints_in = jnp.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
        constraints_out = jnp.array([[0.0, 0.0], [1.0, 1.0]])

        layer = AdaptKANLayerJax(
            in_dim=in_dim,
            out_dim=out_dim,
            k=3,
            basis_type="bspline",
            constraints_in=constraints_in,
            constraints_out=constraints_out,
            key=key,
        )

        assert layer.has_constraints is False

    def test_combined_constraint_shapes(self, key):
        """Test shapes of combined constraint arrays."""
        out_dim, in_dim, k = 3, 4, 5
        n_constraints = 2

        # constraints_in: (n_constraints, in_dim)
        # constraints_out: (n_constraints, out_dim)
        constraints_in = jnp.array([
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0],
        ])
        constraints_out = jnp.array([
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
        ])

        layer, state = eqx.nn.make_with_state(AdaptKANLayerJax)(
            in_dim=in_dim,
            out_dim=out_dim,
            k=k,
            basis_type="chebyshev",
            constraints_in=constraints_in,
            constraints_out=constraints_out,
            key=key,
        )

        assert layer.has_constraints is True

        # Check shapes of stored constraint arrays
        constraints_in_stored = state.get(layer.constraints_in)
        constraints_out_stored = state.get(layer.constraints_out)
        constraints_C = state.get(layer.constraints_C)
        constraints_P = state.get(layer.constraints_P)

        assert constraints_in_stored.shape == (n_constraints, in_dim)
        assert constraints_out_stored.shape == (n_constraints, out_dim)
        assert constraints_C.shape == (n_constraints, in_dim * (k + 1))
        assert constraints_P.shape == (in_dim * (k + 1), n_constraints)

    def test_domain_expansion_for_constraints(self, key):
        """Constraints outside initialization_range should expand domain."""
        out_dim, in_dim = 2, 2

        # Constraint at x=(2.0, 2.0), outside default [-1, 1]
        constraints_in = jnp.array([
            [-1.0, -1.0],  # at edge of default domain
            [2.0, 2.0],    # outside default domain
        ])
        constraints_out = jnp.array([
            [0.0, 0.0],
            [1.0, 1.0],
        ])

        layer, state = eqx.nn.make_with_state(AdaptKANLayerJax)(
            in_dim=in_dim,
            out_dim=out_dim,
            k=5,
            basis_type="chebyshev",
            initialization_range=[-1.0, 1.0],
            constraints_in=constraints_in,
            constraints_out=constraints_out,
            key=key,
        )

        # Domain should have expanded to include constraint points
        a = state.get(layer.a)
        b = state.get(layer.b)

        assert jnp.all(a <= -1.0)
        assert jnp.all(b >= 2.0)


# =============================================================================
# Test: Weight projection with combined constraints
# =============================================================================

class TestCombinedWeightProjection:
    """Tests for get_projected_weights with combined constraints."""

    def test_projection_satisfies_layer_output_constraints(self, key):
        """Projected weights should satisfy layer output constraints."""
        out_dim, in_dim, k = 2, 3, 5

        # Constraints on layer output at specific input points
        constraints_in = jnp.array([
            [0.0, 0.0, 0.0],  # input point 1
            [1.0, 1.0, 1.0],  # input point 2
        ])
        constraints_out = jnp.array([
            [0.0, 1.0],  # output at input point 1: y[0]=0, y[1]=1
            [1.0, 0.0],  # output at input point 2: y[0]=1, y[1]=0
        ])

        layer, state = eqx.nn.make_with_state(AdaptKANLayerJax)(
            in_dim=in_dim,
            out_dim=out_dim,
            k=k,
            basis_type="chebyshev",
            initialization_range=[0.0, 1.0],
            constraints_in=constraints_in,
            constraints_out=constraints_out,
            key=key,
        )

        projected_weights = layer.get_projected_weights(state)

        # Verify constraints are satisfied
        C = state.get(layer.constraints_C)  # (n_constraints, in_dim * (k+1))
        y = state.get(layer.constraints_out)  # (n_constraints, out_dim)

        # Flatten weights: (out_dim, in_dim, k+1) -> (out_dim, in_dim * (k+1))
        w_flat = projected_weights.reshape(out_dim, -1)

        # C @ w_flat.T gives (n_constraints, out_dim)
        Cw = C @ w_flat.T

        assert jnp.allclose(Cw, y, atol=1e-6)

    def test_no_constraints_returns_original_weights(self, key):
        """Without constraints, get_projected_weights should return original weights."""
        layer, state = eqx.nn.make_with_state(AdaptKANLayerJax)(
            in_dim=2,
            out_dim=2,
            k=3,
            basis_type="chebyshev",
            key=key,
        )

        assert layer.has_constraints is False
        projected = layer.get_projected_weights(state)

        assert jnp.allclose(projected, layer.weights)

    def test_projection_is_idempotent(self, key):
        """Projecting already-constrained weights should give same result."""
        out_dim, in_dim, k = 2, 2, 4

        constraints_in = jnp.array([
            [0.0, 0.0],
            [1.0, 1.0],
        ])
        constraints_out = jnp.zeros((2, out_dim))

        layer, state = eqx.nn.make_with_state(AdaptKANLayerJax)(
            in_dim=in_dim,
            out_dim=out_dim,
            k=k,
            basis_type="chebyshev",
            initialization_range=[0.0, 1.0],
            constraints_in=constraints_in,
            constraints_out=constraints_out,
            key=key,
        )

        # First projection
        projected1 = layer.get_projected_weights(state)

        # Manually apply projection formula again
        C = state.get(layer.constraints_C)
        P = state.get(layer.constraints_P)
        y = state.get(layer.constraints_out)

        w_flat = projected1.reshape(out_dim, -1)
        Cw = C @ w_flat.T
        violation = Cw - y
        correction = P @ violation
        projected2_flat = w_flat - correction.T
        projected2 = projected2_flat.reshape(out_dim, in_dim, k + 1)

        # Should be the same (projection is idempotent)
        assert jnp.allclose(projected1, projected2, atol=1e-5)


# =============================================================================
# Test: Forward pass with combined constraints
# =============================================================================

class TestForwardPassWithCombinedConstraints:
    """Tests for forward pass with combined constrained weights."""

    def test_forward_pass_satisfies_layer_output_constraints(self, key):
        """Forward pass at constraint points should match constraint values."""
        out_dim, in_dim, k = 2, 2, 5

        # Constraint: at input [0,0], output should be [0, 1]
        #            at input [1,1], output should be [1, 0]
        constraints_in = jnp.array([
            [0.0, 0.0],
            [1.0, 1.0],
        ])
        constraints_out = jnp.array([
            [0.0, 1.0],  # at [0,0]: y[0]=0, y[1]=1
            [1.0, 0.0],  # at [1,1]: y[0]=1, y[1]=0
        ])

        layer, state = eqx.nn.make_with_state(AdaptKANLayerJax)(
            in_dim=in_dim,
            out_dim=out_dim,
            k=k,
            basis_type="chebyshev",
            initialization_range=[0.0, 1.0],
            constraints_in=constraints_in,
            constraints_out=constraints_out,
            activation_strategy='linear',
            key=key,
        )

        # Evaluate at constraint points
        x_0 = jnp.array([[0.0, 0.0]])
        x_1 = jnp.array([[1.0, 1.0]])

        y_0, state, *_ = layer(x_0, state, update=False)
        y_1, state, *_ = layer(x_1, state, update=False)

        # Check outputs match constraint targets
        assert jnp.allclose(y_0[0], constraints_out[0], atol=1e-5)
        assert jnp.allclose(y_1[0], constraints_out[1], atol=1e-5)

    def test_forward_pass_batched(self, key):
        """Forward pass with multiple samples in a batch."""
        out_dim, in_dim, k = 2, 2, 4

        constraints_in = jnp.array([
            [0.0, 0.0],
            [1.0, 1.0],
        ])
        constraints_out = jnp.zeros((2, out_dim))  # All zeros

        layer, state = eqx.nn.make_with_state(AdaptKANLayerJax)(
            in_dim=in_dim,
            out_dim=out_dim,
            k=k,
            basis_type="chebyshev",
            initialization_range=[0.0, 1.0],
            constraints_in=constraints_in,
            constraints_out=constraints_out,
            activation_strategy='linear',
            key=key,
        )

        # Batch of inputs
        x = jnp.array([
            [0.0, 0.0],
            [1.0, 1.0],
            [0.5, 0.5],
        ])

        y, state, *_ = layer(x, state, update=False)

        # At constraint points, output should be zero
        assert jnp.allclose(y[0, :], 0.0, atol=1e-5)  # x = [0, 0]
        assert jnp.allclose(y[1, :], 0.0, atol=1e-5)  # x = [1, 1]


# =============================================================================
# Test: Network-wide constraints
# =============================================================================

class TestNetworkConstraints:
    """Tests for network-wide constraints applied to the last layer."""

    def test_network_constraint_shapes(self, key):
        """Test shapes when using network-wide constraints."""
        width = [2, 4, 3]  # 2 inputs, hidden layer of 4, 3 outputs
        n_constraints = 2

        network_constraints_in = jnp.array([
            [0.0, 0.0],
            [1.0, 1.0],
        ])
        network_constraints_out = jnp.array([
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
        ])

        model, state = eqx.nn.make_with_state(AdaptKANJax)(
            width=width,
            k=5,
            basis_type="chebyshev",
            network_constraints_in=network_constraints_in,
            network_constraints_out=network_constraints_out,
        )

        assert model.has_network_constraints is True

        # Last layer should have constraints
        last_layer = model.layers[-1]
        assert last_layer.has_constraints is True

        # Other layers should not have constraints
        for layer in model.layers[:-1]:
            assert layer.has_constraints is False

    def test_network_constraints_satisfied_after_setup(self, key):
        """Network constraints should be satisfied after _setup_network_constraints."""
        width = [2, 4, 2]
        n_constraints = 2

        network_constraints_in = jnp.array([
            [0.0, 0.0],
            [1.0, 1.0],
        ])
        network_constraints_out = jnp.array([
            [0.0, 0.0],
            [1.0, 1.0],
        ])

        model, state = eqx.nn.make_with_state(AdaptKANJax)(
            width=width,
            k=5,
            basis_type="chebyshev",
            initialization_range=[0.0, 1.0],
            network_constraints_in=network_constraints_in,
            network_constraints_out=network_constraints_out,
        )

        # Setup network constraints (initializes last layer constraints with z_c)
        state = model._setup_network_constraints(state)

        # Evaluate network at constraint input points
        y_0, state = model(network_constraints_in[0:1], state)
        y_1, state = model(network_constraints_in[1:2], state)

        # Outputs should match constraint targets
        assert jnp.allclose(y_0[0], network_constraints_out[0], atol=1e-4)
        assert jnp.allclose(y_1[0], network_constraints_out[1], atol=1e-4)


# =============================================================================
# Test: Gradient flow with constraints
# =============================================================================

class TestGradientFlowWithCombinedConstraints:
    """Tests to ensure gradients flow properly through constrained layers."""

    def test_gradients_flow_through_projection(self, key):
        """Gradients should flow through the weight projection."""
        out_dim, in_dim, k = 2, 2, 4

        constraints_in = jnp.array([
            [0.0, 0.0],
            [1.0, 1.0],
        ])
        constraints_out = jnp.zeros((2, out_dim))

        layer, state = eqx.nn.make_with_state(AdaptKANLayerJax)(
            in_dim=in_dim,
            out_dim=out_dim,
            k=k,
            basis_type="chebyshev",
            initialization_range=[0.0, 1.0],
            constraints_in=constraints_in,
            constraints_out=constraints_out,
            key=key,
        )

        x = jnp.array([[0.5, 0.5]])

        @eqx.filter_value_and_grad
        def loss_fn(layer, state, x):
            y, state, *_ = layer(x, state, update=False)
            return jnp.sum(y ** 2)

        loss, grads = loss_fn(layer, state, x)

        # Should have gradients for weights
        assert grads.weights is not None
        assert not jnp.all(grads.weights == 0)

    def test_constraints_satisfied_after_gradient_step(self, key):
        """After a gradient step, projected weights should still satisfy constraints."""
        import optax

        out_dim, in_dim, k = 2, 2, 4

        constraints_in = jnp.array([
            [0.0, 0.0],
            [1.0, 1.0],
        ])
        constraints_out = jnp.zeros((2, out_dim))

        layer, state = eqx.nn.make_with_state(AdaptKANLayerJax)(
            in_dim=in_dim,
            out_dim=out_dim,
            k=k,
            basis_type="chebyshev",
            initialization_range=[0.0, 1.0],
            constraints_in=constraints_in,
            constraints_out=constraints_out,
            key=key,
        )

        optimizer = optax.adam(0.01)
        opt_state = optimizer.init(eqx.filter(layer, eqx.is_array))

        x = jnp.array([[0.5, 0.5]])

        @eqx.filter_value_and_grad
        def loss_fn(layer, state, x):
            y, state, *_ = layer(x, state, update=False)
            return jnp.sum(y ** 2)

        # Perform gradient step
        loss, grads = loss_fn(layer, state, x)
        updates, opt_state = optimizer.update(grads, opt_state, layer)
        layer = eqx.apply_updates(layer, updates)

        # Check constraints are still satisfied
        projected_weights = layer.get_projected_weights(state)
        C = state.get(layer.constraints_C)
        y = state.get(layer.constraints_out)

        w_flat = projected_weights.reshape(out_dim, -1)
        Cw = C @ w_flat.T

        assert jnp.allclose(Cw, y, atol=1e-5)


# =============================================================================
# Test: Constraint recomputation during adaptation
# =============================================================================

class TestConstraintRecomputationDuringAdaptation:
    """Tests for constraint matrix recomputation when domain changes."""

    def test_layer_constraints_satisfied_after_adapt(self, key):
        """Layer constraints should still be satisfied after adaptation."""
        out_dim, in_dim, k = 2, 2, 5

        constraints_in = jnp.array([
            [0.0, 0.0],
            [1.0, 1.0],
        ])
        constraints_out = jnp.zeros((2, out_dim))

        layer, state = eqx.nn.make_with_state(AdaptKANLayerJax)(
            in_dim=in_dim,
            out_dim=out_dim,
            k=k,
            basis_type="chebyshev",
            initialization_range=[0.0, 1.0],
            constraints_in=constraints_in,
            constraints_out=constraints_out,
            prune_patience=1,
            key=key,
        )

        # Simulate out-of-domain data to trigger stretch
        x_ood = jax.random.uniform(key, (100, in_dim), minval=-2.0, maxval=3.0)
        _, state, *_ = layer(x_ood, state, update=True)

        # Run adaptation
        layer, state, adapted = layer.adapt(state)

        # Verify constraints are still satisfied
        projected = layer.get_projected_weights(state)
        C = state.get(layer.constraints_C)
        y = state.get(layer.constraints_out)

        w_flat = projected.reshape(out_dim, -1)
        Cw = C @ w_flat.T

        assert jnp.allclose(Cw, y, atol=1e-5)

    def test_domain_never_shrinks_past_constraints(self, key):
        """Domain should never shrink past constraint points."""
        out_dim, in_dim, k = 2, 2, 5

        # Constraints at positions outside initial domain
        constraints_in = jnp.array([
            [-0.5, -0.5],
            [1.5, 1.5],
        ])
        constraints_out = jnp.zeros((2, out_dim))

        layer, state = eqx.nn.make_with_state(AdaptKANLayerJax)(
            in_dim=in_dim,
            out_dim=out_dim,
            k=k,
            basis_type="chebyshev",
            initialization_range=[0.0, 1.0],
            constraints_in=constraints_in,
            constraints_out=constraints_out,
            prune_patience=1,
            key=key,
        )

        # Domain should already be expanded to include constraint points
        a = state.get(layer.a)
        b = state.get(layer.b)
        assert jnp.all(a <= -0.5)
        assert jnp.all(b >= 1.5)

        # Train on data in narrow range
        x_narrow = jax.random.uniform(key, (1000, in_dim), minval=0.25, maxval=0.75)
        for _ in range(5):
            _, state, *_ = layer(x_narrow, state, update=True)
            layer, state, adapted = layer.adapt(state)

        # After adaptation, domain should still include constraint points
        a_final = state.get(layer.a)
        b_final = state.get(layer.b)
        assert jnp.all(a_final <= -0.5)
        assert jnp.all(b_final >= 1.5)


# =============================================================================
# Test: Utility functions
# =============================================================================

class TestUtilityFunctions:
    """Tests for constraint utility functions."""

    def test_build_combined_constraint_matrix_shape(self):
        """Test shape of combined constraint matrix."""
        n_constraints = 3
        in_dim = 4
        k = 5

        z_c = jnp.zeros((n_constraints, in_dim))
        a = jnp.full((in_dim,), -1.0)
        b = jnp.full((in_dim,), 1.0)

        C = build_combined_constraint_matrix(z_c, a, b, k)

        assert C.shape == (n_constraints, in_dim * (k + 1))

    def test_projection_operator_properties(self):
        """Test properties of the projection operator P."""
        n_constraints = 2
        in_dim = 2
        k = 3

        z_c = jnp.array([
            [0.0, 0.0],
            [1.0, 1.0],
        ])
        a = jnp.full((in_dim,), 0.0)
        b = jnp.full((in_dim,), 1.0)

        C = build_combined_constraint_matrix(z_c, a, b, k)
        P = compute_constraint_projection_operator(C)

        # P should have shape (in_dim * (k+1), n_constraints)
        assert P.shape == (in_dim * (k + 1), n_constraints)

        # C @ P should be identity on constraint space
        CP = C @ P
        assert jnp.allclose(CP, jnp.eye(n_constraints), atol=1e-6)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
