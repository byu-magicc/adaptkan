"""Tests for Chebyshev basis and combined constraint matrix utilities."""
import jax
import jax.numpy as jnp
import pytest

from adaptkan.jax.utils import (
    build_combined_constraint_matrix,
    compute_chebyshev_basis,
    compute_constraint_projection_operator,
)


def run_combined_constraint_demo():
    """Runnable script version of the combined constraint projection check."""
    print("=" * 60)
    print("TEST: Combined Constraint Matrix and Weight Projection")
    print("=" * 60)

    # Setup: Layer with in_dim=2, k=5 on domain [0, 1]
    # Constraints on layer output y_i = sum_j w_{i,j}^T @ T_j(x_j)
    a = jnp.array([0.0, 0.0])
    b = jnp.array([1.0, 1.0])
    in_dim = 2
    k = 5  # degree k, so we have k+1 coefficients per activation
    out_dim = 2

    # Constraint points: at x=[0,0] output [0,0], at x=[1,1] output [1,1]
    z_c = jnp.array([
        [0.0, 0.0],
        [1.0, 1.0],
    ])
    y_targets = jnp.array([
        [0.0, 0.0],  # output at [0,0]
        [1.0, 1.0],  # output at [1,1]
    ])

    print(f"Domain: [{a}, {b}]")
    print(f"in_dim: {in_dim}, out_dim: {out_dim}, k: {k}")
    print(f"Number of constraints: {z_c.shape[0]}")
    print()

    # Build combined constraint matrix
    C = build_combined_constraint_matrix(z_c, a, b, k)
    print(f"Combined constraint matrix C shape: {C.shape}")
    print(f"Expected: ({z_c.shape[0]}, {in_dim * (k + 1)})")
    print()

    # Compute projection operator
    P = compute_constraint_projection_operator(C)
    print(f"Projection operator P shape: {P.shape}")
    print()

    # Start with random weights: (out_dim, in_dim, k+1)
    key = jax.random.PRNGKey(42)
    weights = jax.random.normal(key, (out_dim, in_dim, k + 1))
    print(f"Original weights shape: {weights.shape}")

    # Flatten weights: (out_dim, in_dim * (k+1))
    w_flat = weights.reshape(out_dim, -1)

    # Check constraints before projection
    Cw_before = C @ w_flat.T  # (n_constraints, out_dim)
    print(f"C @ w (before) shape: {Cw_before.shape}")
    print(f"Constraint violation (before):\n{Cw_before - y_targets}")
    print()

    # Project weights
    # ŵ_i = w_i - P @ (C @ w_i - y_i)
    violation = Cw_before - y_targets
    correction = P @ violation  # (in_dim * (k+1), out_dim)
    w_projected_flat = w_flat - correction.T

    # Reshape back
    projected = w_projected_flat.reshape(out_dim, in_dim, k + 1)
    print(f"Projected weights shape: {projected.shape}")

    # Check constraints after projection
    Cw_after = C @ w_projected_flat.T
    print(f"C @ w (after) shape: {Cw_after.shape}")
    print(f"Constraint violation (after):\n{Cw_after - y_targets}")
    print(f"Max violation: {jnp.abs(Cw_after - y_targets).max():.2e}")
    print()

    print("=" * 60)
    print("All tests passed!")


@pytest.fixture
def combined_constraint_setup():
    """Common fixtures for combined constraint tests."""
    a = jnp.array([0.0, 0.0])
    b = jnp.array([1.0, 1.0])
    in_dim = 2
    k = 5
    out_dim = 2

    z_c = jnp.array([
        [0.0, 0.0],
        [1.0, 1.0],
    ])
    y_targets = jnp.array([
        [0.0, 0.0],
        [1.0, 1.0],
    ])

    C = build_combined_constraint_matrix(z_c, a, b, k)
    P = compute_constraint_projection_operator(C)

    return {
        "a": a,
        "b": b,
        "in_dim": in_dim,
        "out_dim": out_dim,
        "k": k,
        "z_c": z_c,
        "y_targets": y_targets,
        "C": C,
        "P": P,
    }


def test_combined_constraint_matrix_shape(combined_constraint_setup):
    """Test that combined constraint matrix has correct shape."""
    C = combined_constraint_setup["C"]
    in_dim = combined_constraint_setup["in_dim"]
    k = combined_constraint_setup["k"]
    z_c = combined_constraint_setup["z_c"]

    n_constraints = z_c.shape[0]
    expected_shape = (n_constraints, in_dim * (k + 1))
    assert C.shape == expected_shape


def test_projection_operator_shape(combined_constraint_setup):
    """Test that projection operator has correct shape."""
    P = combined_constraint_setup["P"]
    in_dim = combined_constraint_setup["in_dim"]
    k = combined_constraint_setup["k"]
    z_c = combined_constraint_setup["z_c"]

    n_constraints = z_c.shape[0]
    expected_shape = (in_dim * (k + 1), n_constraints)
    assert P.shape == expected_shape


def test_projection_satisfies_combined_constraints(combined_constraint_setup):
    """Test that projection satisfies combined constraints."""
    C = combined_constraint_setup["C"]
    P = combined_constraint_setup["P"]
    y_targets = combined_constraint_setup["y_targets"]
    in_dim = combined_constraint_setup["in_dim"]
    out_dim = combined_constraint_setup["out_dim"]
    k = combined_constraint_setup["k"]

    key = jax.random.PRNGKey(42)
    weights = jax.random.normal(key, (out_dim, in_dim, k + 1))

    # Flatten weights
    w_flat = weights.reshape(out_dim, -1)

    # Project
    Cw = C @ w_flat.T
    violation = Cw - y_targets
    correction = P @ violation
    w_projected_flat = w_flat - correction.T

    # Check constraints are satisfied
    Cw_after = C @ w_projected_flat.T
    assert jnp.allclose(Cw_after, y_targets, atol=1e-6)


def test_projection_is_idempotent(combined_constraint_setup):
    """Test that projecting already-projected weights gives same result."""
    C = combined_constraint_setup["C"]
    P = combined_constraint_setup["P"]
    y_targets = combined_constraint_setup["y_targets"]
    in_dim = combined_constraint_setup["in_dim"]
    out_dim = combined_constraint_setup["out_dim"]
    k = combined_constraint_setup["k"]

    key = jax.random.PRNGKey(42)
    weights = jax.random.normal(key, (out_dim, in_dim, k + 1))

    # First projection
    w_flat = weights.reshape(out_dim, -1)
    Cw = C @ w_flat.T
    violation = Cw - y_targets
    correction = P @ violation
    w_projected_flat = w_flat - correction.T

    # Second projection
    Cw2 = C @ w_projected_flat.T
    violation2 = Cw2 - y_targets
    correction2 = P @ violation2
    w_projected_flat2 = w_projected_flat - correction2.T

    # Should be the same
    assert jnp.allclose(w_projected_flat, w_projected_flat2, atol=1e-6)


def test_CP_equals_identity(combined_constraint_setup):
    """Test that C @ P = I on constraint space."""
    C = combined_constraint_setup["C"]
    P = combined_constraint_setup["P"]
    z_c = combined_constraint_setup["z_c"]

    n_constraints = z_c.shape[0]
    CP = C @ P

    assert jnp.allclose(CP, jnp.eye(n_constraints), atol=1e-6)


if __name__ == "__main__":
    run_combined_constraint_demo()
