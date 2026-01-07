"""Comprehensive unit tests for constraint initialization and forward pass."""
import jax
import jax.numpy as jnp
import equinox as eqx
import pytest

from adaptkan.jax.layers import AdaptKANLayerJax
from adaptkan.jax.utils import (
    build_constraint_matrix,
    compute_chebyshev_basis,
    compute_chebyshev_derivative_basis,
    compute_constraint_projection_operator,
    project_weights,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def key():
    return jax.random.PRNGKey(42)


@pytest.fixture
def simple_position_constraints():
    """Position constraints only: f(0)=0, f(1)=1 for all activations."""
    # (n_constraints, 2) - will be broadcast to all input dims
    constraints_in = jnp.array([
        [0.0, 0],  # position at x=0
        [1.0, 0],  # position at x=1
    ])
    return constraints_in


@pytest.fixture
def position_and_derivative_constraints():
    """Position + derivative constraints: f(0)=0, f(1)=1, f'(0)=0, f'(1)=0."""
    constraints_in = jnp.array([
        [0.0, 0],  # position at x=0
        [1.0, 0],  # position at x=1
        [0.0, 1],  # derivative at x=0
        [1.0, 1],  # derivative at x=1
    ])
    return constraints_in


# =============================================================================
# Test: Basic constraint initialization
# =============================================================================

class TestConstraintInitialization:
    """Tests for constraint array initialization and broadcasting."""

    def test_no_constraints_initialization(self, key):
        """Layer without constraints should have has_constraints=False."""
        layer = AdaptKANLayerJax(
            in_dim=3,
            out_dim=2,
            k=5,
            basis_type="chebyshev",
            constraints_in=None,
            constraints_y=None,
            key=key,
        )

        assert layer.has_constraints is False
        assert layer.constraints_in is None
        assert layer.constraints_y is None
        assert layer.constraints_C is None
        assert layer.constraints_P is None
        assert layer.constraints_a is None
        assert layer.constraints_b is None

    def test_bspline_ignores_constraints(self, key, simple_position_constraints):
        """B-spline basis should ignore constraints."""
        out_dim, in_dim = 2, 3
        n_constraints = simple_position_constraints.shape[0]
        constraints_y = jnp.zeros((out_dim, n_constraints))

        layer = AdaptKANLayerJax(
            in_dim=in_dim,
            out_dim=out_dim,
            k=3,
            basis_type="bspline",
            constraints_in=simple_position_constraints,
            constraints_y=constraints_y,
            key=key,
        )

        assert layer.has_constraints is False

    def test_constraints_with_broadcasting(self, key, simple_position_constraints):
        """Test broadcasting of constraints_in and constraints_y."""
        out_dim, in_dim, k = 3, 4, 5
        n_constraints = simple_position_constraints.shape[0]

        # constraints_y: (out_dim, n_constraints) should broadcast to (out_dim, in_dim, n_constraints)
        constraints_y = jnp.ones((out_dim, n_constraints))

        layer, state = eqx.nn.make_with_state(AdaptKANLayerJax)(
            in_dim=in_dim,
            out_dim=out_dim,
            k=k,
            basis_type="chebyshev",
            constraints_in=simple_position_constraints,
            constraints_y=constraints_y,
            key=key,
        )

        assert layer.has_constraints is True

        # Check shapes of stored constraint arrays
        constraints_in_stored = state.get(layer.constraints_in)
        constraints_y_stored = state.get(layer.constraints_y)
        constraints_C = state.get(layer.constraints_C)
        constraints_P = state.get(layer.constraints_P)

        assert constraints_in_stored.shape == (in_dim, n_constraints, 2)
        assert constraints_y_stored.shape == (out_dim, in_dim, n_constraints)
        assert constraints_C.shape == (in_dim, n_constraints, k + 1)
        assert constraints_P.shape == (in_dim, k + 1, n_constraints)

    def test_full_shape_constraints_no_broadcasting(self, key):
        """Test providing fully-shaped constraint arrays (no broadcasting needed)."""
        out_dim, in_dim, k = 2, 3, 4
        n_constraints = 2

        # Fully specified shapes
        constraints_in = jnp.zeros((in_dim, n_constraints, 2))
        constraints_in = constraints_in.at[:, 0, :].set(jnp.array([0.0, 0]))  # position at x=0
        constraints_in = constraints_in.at[:, 1, :].set(jnp.array([1.0, 0]))  # position at x=1

        constraints_y = jnp.ones((out_dim, in_dim, n_constraints))

        layer, state = eqx.nn.make_with_state(AdaptKANLayerJax)(
            in_dim=in_dim,
            out_dim=out_dim,
            k=k,
            basis_type="chebyshev",
            constraints_in=constraints_in,
            constraints_y=constraints_y,
            key=key,
        )

        assert layer.has_constraints is True

        constraints_in_stored = state.get(layer.constraints_in)
        assert constraints_in_stored.shape == (in_dim, n_constraints, 2)

    def test_domain_expansion_for_constraints(self, key):
        """Constraints outside initialization_range should expand domain."""
        out_dim, in_dim = 2, 1

        # Constraint at x=2.0, outside default [-1, 1]
        constraints_in = jnp.array([
            [-1.0, 0],  # at edge of default domain
            [2.0, 0],   # outside default domain
        ])
        constraints_y = jnp.zeros((out_dim, 2))

        layer, state = eqx.nn.make_with_state(AdaptKANLayerJax)(
            in_dim=in_dim,
            out_dim=out_dim,
            k=5,
            basis_type="chebyshev",
            initialization_range=[-1.0, 1.0],
            constraints_in=constraints_in,
            constraints_y=constraints_y,
            key=key,
        )

        # Domain should have expanded to include x=2.0
        a = state.get(layer.a)
        b = state.get(layer.b)

        assert a[0] <= -1.0
        assert b[0] >= 2.0

    def test_derivative_constraints_dont_affect_domain(self, key):
        """Derivative constraints (d>0) should not affect domain bounds."""
        out_dim, in_dim = 2, 1

        # Only derivative constraint at x=5.0
        constraints_in = jnp.array([
            [0.0, 0],   # position constraint at x=0
            [5.0, 1],   # derivative constraint at x=5 (should not expand domain)
        ])
        constraints_y = jnp.zeros((out_dim, 2))

        layer, state = eqx.nn.make_with_state(AdaptKANLayerJax)(
            in_dim=in_dim,
            out_dim=out_dim,
            k=5,
            basis_type="chebyshev",
            initialization_range=[-1.0, 1.0],
            constraints_in=constraints_in,
            constraints_y=constraints_y,
            key=key,
        )

        a = state.get(layer.a)
        b = state.get(layer.b)

        # Domain should not have expanded to x=5.0 (derivative constraint)
        assert a[0] == -1.0
        assert b[0] == 1.0


# =============================================================================
# Test: Weight projection
# =============================================================================

class TestWeightProjection:
    """Tests for get_projected_weights method."""

    def test_projection_satisfies_position_constraints(self, key):
        """Projected weights should satisfy position constraints exactly."""
        out_dim, in_dim, k = 2, 3, 5

        constraints_in = jnp.array([
            [0.0, 0],  # f(0) = target
            [1.0, 0],  # f(1) = target
        ])

        # Different target values for each output
        constraints_y = jnp.array([
            [0.0, 1.0],  # output 0: f(0)=0, f(1)=1
            [1.0, 0.0],  # output 1: f(0)=1, f(1)=0
        ])

        layer, state = eqx.nn.make_with_state(AdaptKANLayerJax)(
            in_dim=in_dim,
            out_dim=out_dim,
            k=k,
            basis_type="chebyshev",
            initialization_range=[0.0, 1.0],
            constraints_in=constraints_in,
            constraints_y=constraints_y,
            key=key,
        )

        projected_weights = layer.get_projected_weights(state)

        # Verify constraints are satisfied: C @ w = y
        C = state.get(layer.constraints_C)
        y = state.get(layer.constraints_y)

        # C[j]: (n_constraints, k+1), projected_weights[i, j]: (k+1,)
        Cw = jnp.einsum('jnk,ijk->ijn', C, projected_weights)

        assert jnp.allclose(Cw, y, atol=1e-6)

    def test_projection_satisfies_derivative_constraints(self, key):
        """Projected weights should satisfy derivative constraints."""
        out_dim, in_dim, k = 1, 1, 7  # Use higher degree for better conditioning

        # Position and derivative constraints
        constraints_in = jnp.array([
            [0.0, 0],  # f(0) = 0
            [1.0, 0],  # f(1) = 1
            [0.0, 1],  # f'(0) = 0
            [1.0, 1],  # f'(1) = 0
        ])

        constraints_y = jnp.array([
            [0.0, 1.0, 0.0, 0.0],  # targets for f(0), f(1), f'(0), f'(1)
        ])

        layer, state = eqx.nn.make_with_state(AdaptKANLayerJax)(
            in_dim=in_dim,
            out_dim=out_dim,
            k=k,
            basis_type="chebyshev",
            initialization_range=[0.0, 1.0],
            constraints_in=constraints_in,
            constraints_y=constraints_y,
            key=key,
        )

        projected_weights = layer.get_projected_weights(state)

        C = state.get(layer.constraints_C)
        y = state.get(layer.constraints_y)

        Cw = jnp.einsum('jnk,ijk->ijn', C, projected_weights)

        # Use looser tolerance due to derivative constraint scaling
        assert jnp.allclose(Cw, y, atol=1e-4)

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
            [0.0, 0],
            [1.0, 0],
        ])
        constraints_y = jnp.zeros((out_dim, 2))

        layer, state = eqx.nn.make_with_state(AdaptKANLayerJax)(
            in_dim=in_dim,
            out_dim=out_dim,
            k=k,
            basis_type="chebyshev",
            initialization_range=[0.0, 1.0],
            constraints_in=constraints_in,
            constraints_y=constraints_y,
            key=key,
        )

        # First projection
        projected1 = layer.get_projected_weights(state)

        # Manually apply projection formula again
        C = state.get(layer.constraints_C)
        P = state.get(layer.constraints_P)
        y = state.get(layer.constraints_y)

        Cw = jnp.einsum('jnk,ijk->ijn', C, projected1)
        violation = Cw - y
        correction = jnp.einsum('jkn,ijn->ijk', P, violation)
        projected2 = projected1 - correction

        # Should be the same (projection is idempotent)
        # Use reasonable tolerance for single precision floats
        assert jnp.allclose(projected1, projected2, atol=1e-5)


# =============================================================================
# Test: Forward pass with constraints
# =============================================================================

class TestForwardPassWithConstraints:
    """Tests for basic_forward with constrained weights."""

    def test_forward_pass_satisfies_position_constraints(self, key):
        """Forward pass at constraint points should match constraint values."""
        out_dim, in_dim, k = 2, 1, 5

        constraints_in = jnp.array([
            [0.0, 0],  # f(0)
            [1.0, 0],  # f(1)
        ])

        # Output 0: f(0)=0, f(1)=1
        # Output 1: f(0)=1, f(1)=0
        constraints_y = jnp.array([
            [0.0, 1.0],
            [1.0, 0.0],
        ])

        layer, state = eqx.nn.make_with_state(AdaptKANLayerJax)(
            in_dim=in_dim,
            out_dim=out_dim,
            k=k,
            basis_type="chebyshev",
            initialization_range=[0.0, 1.0],
            constraints_in=constraints_in,
            constraints_y=constraints_y,
            activation_strategy='linear',  # Standard initialization
            key=key,
        )

        # Evaluate at constraint points
        x_0 = jnp.array([[0.0]])  # batch_size=1, in_dim=1
        x_1 = jnp.array([[1.0]])

        y_0, state, *_ = layer(x_0, state)
        y_1, state, *_ = layer(x_1, state)

        # Output shape is (batch_size, out_dim) - summed over in_dim
        # The constraint is per activation, so output y[i] = sum_j(phi_{ij}(x_j))
        # With in_dim=1, output is just phi_{i0}(x_0)
        # Check outputs match constraint targets
        assert jnp.allclose(y_0[0, 0], 0.0, atol=1e-5), f"Expected f(0)[0]=0, got {y_0[0, 0]}"
        assert jnp.allclose(y_0[0, 1], 1.0, atol=1e-5), f"Expected f(0)[1]=1, got {y_0[0, 1]}"
        assert jnp.allclose(y_1[0, 0], 1.0, atol=1e-5), f"Expected f(1)[0]=1, got {y_1[0, 0]}"
        assert jnp.allclose(y_1[0, 1], 0.0, atol=1e-5), f"Expected f(1)[1]=0, got {y_1[0, 1]}"

    def test_forward_pass_batched(self, key):
        """Forward pass with multiple samples in a batch."""
        out_dim, in_dim, k = 2, 2, 4

        constraints_in = jnp.array([
            [0.0, 0],
            [1.0, 0],
        ])
        constraints_y = jnp.zeros((out_dim, 2))  # All zeros

        layer, state = eqx.nn.make_with_state(AdaptKANLayerJax)(
            in_dim=in_dim,
            out_dim=out_dim,
            k=k,
            basis_type="chebyshev",
            initialization_range=[0.0, 1.0],
            constraints_in=constraints_in,
            constraints_y=constraints_y,
            activation_strategy='linear',  # Standard initialization
            key=key,
        )

        # Batch of inputs
        x = jnp.array([
            [0.0, 0.0],
            [1.0, 1.0],
            [0.5, 0.5],
        ])

        y, state, *_ = layer(x, state)

        # Output shape is (batch_size, out_dim) - activations summed over in_dim
        # At constraint points, each activation output should be zero
        # y[batch, out] = sum_j(phi_{out,j}(x_j))
        # With constraints_y = 0 for all activations at x=0 and x=1:
        assert jnp.allclose(y[0, :], 0.0, atol=1e-5)  # x = [0, 0]
        assert jnp.allclose(y[1, :], 0.0, atol=1e-5)  # x = [1, 1]

    def test_forward_pass_without_constraints(self, key):
        """Forward pass without constraints should work normally."""
        layer, state = eqx.nn.make_with_state(AdaptKANLayerJax)(
            in_dim=3,
            out_dim=2,
            k=5,
            basis_type="chebyshev",
            key=key,
        )

        x = jax.random.normal(key, (10, 3))
        y, state, *_ = layer(x, state)

        # Output shape is (batch_size, out_dim) - activations summed over in_dim
        assert y.shape == (10, 2)
        assert not jnp.any(jnp.isnan(y))


# =============================================================================
# Test: Gradient flow with constraints
# =============================================================================

class TestGradientFlowWithConstraints:
    """Tests to ensure gradients flow properly through constrained layers."""

    def test_gradients_flow_through_projection(self, key):
        """Gradients should flow through the weight projection."""
        out_dim, in_dim, k = 2, 2, 4

        constraints_in = jnp.array([
            [0.0, 0],
            [1.0, 0],
        ])
        constraints_y = jnp.zeros((out_dim, 2))

        layer, state = eqx.nn.make_with_state(AdaptKANLayerJax)(
            in_dim=in_dim,
            out_dim=out_dim,
            k=k,
            basis_type="chebyshev",
            initialization_range=[0.0, 1.0],
            constraints_in=constraints_in,
            constraints_y=constraints_y,
            key=key,
        )

        x = jnp.array([[0.5, 0.5]])

        @eqx.filter_value_and_grad
        def loss_fn(layer, state, x):
            y, state, *_ = layer(x, state)
            return jnp.sum(y ** 2)

        loss, grads = loss_fn(layer, state, x)

        # Should have gradients for weights
        assert grads.weights is not None
        assert not jnp.all(grads.weights == 0)

    def test_constraints_still_satisfied_after_gradient_step(self, key):
        """After a gradient step, projected weights should still satisfy constraints."""
        import optax

        out_dim, in_dim, k = 2, 2, 4

        constraints_in = jnp.array([
            [0.0, 0],
            [1.0, 0],
        ])
        constraints_y = jnp.zeros((out_dim, 2))

        layer, state = eqx.nn.make_with_state(AdaptKANLayerJax)(
            in_dim=in_dim,
            out_dim=out_dim,
            k=k,
            basis_type="chebyshev",
            initialization_range=[0.0, 1.0],
            constraints_in=constraints_in,
            constraints_y=constraints_y,
            key=key,
        )

        optimizer = optax.adam(0.01)
        opt_state = optimizer.init(eqx.filter(layer, eqx.is_array))

        x = jnp.array([[0.5, 0.5]])

        @eqx.filter_value_and_grad
        def loss_fn(layer, state, x):
            y, state, *_ = layer(x, state)
            return jnp.sum(y ** 2)

        # Perform gradient step
        loss, grads = loss_fn(layer, state, x)
        updates, opt_state = optimizer.update(grads, opt_state, layer)
        layer = eqx.apply_updates(layer, updates)

        # Check constraints are still satisfied
        projected_weights = layer.get_projected_weights(state)
        C = state.get(layer.constraints_C)
        y = state.get(layer.constraints_y)

        Cw = jnp.einsum('jnk,ijk->ijn', C, projected_weights)

        assert jnp.allclose(Cw, y, atol=1e-5)


# =============================================================================
# Test: Edge cases
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and unusual configurations."""

    def test_single_constraint(self, key):
        """Layer with single constraint should work."""
        out_dim, in_dim, k = 1, 1, 3

        constraints_in = jnp.array([[0.5, 0]])  # Single constraint
        constraints_y = jnp.array([[1.0]])

        layer, state = eqx.nn.make_with_state(AdaptKANLayerJax)(
            in_dim=in_dim,
            out_dim=out_dim,
            k=k,
            basis_type="chebyshev",
            initialization_range=[0.0, 1.0],
            constraints_in=constraints_in,
            constraints_y=constraints_y,
            key=key,
        )

        assert layer.has_constraints is True

        # Verify constraint is satisfied
        C = state.get(layer.constraints_C)
        y = state.get(layer.constraints_y)
        projected = layer.get_projected_weights(state)

        Cw = jnp.einsum('jnk,ijk->ijn', C, projected)
        assert jnp.allclose(Cw, y, atol=1e-6)

    def test_max_constraints_equal_to_degree(self, key):
        """Number of constraints equal to degree+1 should work (fully determined)."""
        out_dim, in_dim, k = 1, 1, 2  # k+1 = 3 coefficients

        # 3 constraints = 3 coefficients (fully determined system)
        constraints_in = jnp.array([
            [0.0, 0],
            [0.5, 0],
            [1.0, 0],
        ])
        constraints_y = jnp.array([[0.0, 0.5, 1.0]])  # Linear function

        layer, state = eqx.nn.make_with_state(AdaptKANLayerJax)(
            in_dim=in_dim,
            out_dim=out_dim,
            k=k,
            basis_type="chebyshev",
            initialization_range=[0.0, 1.0],
            constraints_in=constraints_in,
            constraints_y=constraints_y,
            key=key,
        )

        C = state.get(layer.constraints_C)
        y = state.get(layer.constraints_y)
        projected = layer.get_projected_weights(state)

        Cw = jnp.einsum('jnk,ijk->ijn', C, projected)
        assert jnp.allclose(Cw, y, atol=1e-5)

    def test_high_derivative_order(self, key):
        """Second derivative constraints should work."""
        out_dim, in_dim, k = 1, 1, 5

        constraints_in = jnp.array([
            [0.0, 0],  # f(0) = 0
            [1.0, 0],  # f(1) = 0
            [0.5, 2],  # f''(0.5) = -2 (concave)
        ])
        constraints_y = jnp.array([[0.0, 0.0, -2.0]])

        layer, state = eqx.nn.make_with_state(AdaptKANLayerJax)(
            in_dim=in_dim,
            out_dim=out_dim,
            k=k,
            basis_type="chebyshev",
            initialization_range=[0.0, 1.0],
            constraints_in=constraints_in,
            constraints_y=constraints_y,
            key=key,
        )

        C = state.get(layer.constraints_C)
        y = state.get(layer.constraints_y)
        projected = layer.get_projected_weights(state)

        Cw = jnp.einsum('jnk,ijk->ijn', C, projected)
        assert jnp.allclose(Cw, y, atol=1e-5)

    def test_per_input_dim_constraints(self, key):
        """Different constraints per input dimension."""
        out_dim, in_dim, k = 1, 2, 4

        # Different constraints for each input dimension
        constraints_in = jnp.array([
            [[0.0, 0], [1.0, 0]],  # Input dim 0: f(0), f(1)
            [[0.0, 0], [0.5, 0]],  # Input dim 1: f(0), f(0.5) (different!)
        ])  # Shape: (in_dim, n_constraints, 2)

        constraints_y = jnp.array([
            [[0.0, 1.0],   # Output 0, Input dim 0
             [0.0, 0.5]],  # Output 0, Input dim 1
        ])  # Shape: (out_dim, in_dim, n_constraints)

        layer, state = eqx.nn.make_with_state(AdaptKANLayerJax)(
            in_dim=in_dim,
            out_dim=out_dim,
            k=k,
            basis_type="chebyshev",
            initialization_range=[0.0, 1.0],
            constraints_in=constraints_in,
            constraints_y=constraints_y,
            key=key,
        )

        C = state.get(layer.constraints_C)
        y = state.get(layer.constraints_y)
        projected = layer.get_projected_weights(state)

        Cw = jnp.einsum('jnk,ijk->ijn', C, projected)
        assert jnp.allclose(Cw, y, atol=1e-5)


# =============================================================================
# Test: Utility functions
# =============================================================================

class TestUtilityFunctions:
    """Tests for constraint utility functions in utils.py."""

    def test_build_constraint_matrix_position_only(self):
        """Test constraint matrix with position constraints only."""
        constraints = [
            (0.0, 1.0, 0),  # f(0) = 1
            (1.0, 2.0, 0),  # f(1) = 2
        ]
        a, b, degree = 0.0, 1.0, 3

        C, y_vec = build_constraint_matrix(constraints, a, b, degree)

        assert C.shape == (2, degree + 1)
        assert y_vec.shape == (2,)
        assert jnp.allclose(y_vec, jnp.array([1.0, 2.0]))

    def test_build_constraint_matrix_with_derivatives(self):
        """Test constraint matrix with derivative constraints."""
        constraints = [
            (0.0, 1.0, 0),  # f(0) = 1
            (1.0, 2.0, 0),  # f(1) = 2
            (0.5, 0.0, 1),  # f'(0.5) = 0
        ]
        a, b, degree = 0.0, 1.0, 4

        C, y_vec = build_constraint_matrix(constraints, a, b, degree)

        assert C.shape == (3, degree + 1)
        assert y_vec.shape == (3,)

    def test_projection_operator_properties(self):
        """Test properties of the projection operator P."""
        constraints = [
            (0.0, 0.0, 0),
            (1.0, 0.0, 0),
        ]
        a, b, degree = 0.0, 1.0, 3

        C, _ = build_constraint_matrix(constraints, a, b, degree)
        P = compute_constraint_projection_operator(C)

        # P should have shape (k+1, n_constraints)
        assert P.shape == (degree + 1, 2)

        # P @ C should project onto constraint subspace
        # (C @ P @ C^T) should equal identity on constraint space
        # Equivalently: C @ P = I (n_constraints x n_constraints)
        CP = C @ P
        assert jnp.allclose(CP, jnp.eye(2), atol=1e-6)

    def test_project_weights_1d(self):
        """Test project_weights with 1D weight vector."""
        constraints = [
            (0.0, 1.0, 0),
            (1.0, 2.0, 0),
        ]
        a, b, degree = 0.0, 1.0, 5

        C, y_vec = build_constraint_matrix(constraints, a, b, degree)
        P = compute_constraint_projection_operator(C)

        # Random weights
        key = jax.random.PRNGKey(0)
        weights = jax.random.normal(key, (degree + 1,))

        projected = project_weights(weights, C, P, y_vec)

        # Check constraints are satisfied
        assert jnp.allclose(C @ projected, y_vec, atol=1e-6)

    def test_project_weights_batched(self):
        """Test project_weights with batched weights."""
        constraints = [
            (0.0, 0.0, 0),
            (1.0, 1.0, 0),
        ]
        a, b, degree = 0.0, 1.0, 4

        C, y_vec = build_constraint_matrix(constraints, a, b, degree)
        P = compute_constraint_projection_operator(C)

        # Batched weights: (out_dim, in_dim, k+1)
        key = jax.random.PRNGKey(0)
        weights = jax.random.normal(key, (3, 2, degree + 1))

        projected = project_weights(weights, C, P, y_vec)

        # Check constraints satisfied for all activations
        # Cw shape: (out_dim, in_dim, n_constraints)
        Cw = jnp.einsum('nk,oik->oin', C, projected)

        assert jnp.allclose(Cw, y_vec, atol=1e-6)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
