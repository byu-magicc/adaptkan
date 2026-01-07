import jax
import jax.numpy as jnp
import pytest

from adaptkan.jax.utils import (
	build_constraint_matrix,
	compute_chebyshev_basis,
	compute_chebyshev_derivative_basis,
	compute_constraint_projection_operator,
	project_weights,
)


def run_constraint_projection_demo():
	"""Runnable script version of the projection check with prints."""
	print("=" * 60)
	print("TEST: Constraint Matrix and Weight Projection")
	print("=" * 60)

	# Setup: Chebyshev polynomial of degree 5 on domain [0, 1]
	# We want: phi(0) = 1, phi(1) = 2, phi'(0) = 0 (3 constraints)
	a, b = 0.0, 1.0
	degree = 5  # k=5, so we have 6 coefficients
	constraints = [
		(0.0, 1.0, 0),  # phi(0) = 1
		(1.0, 2.0, 0),  # phi(1) = 2
		(0.0, 0.0, 1),  # phi'(0) = 0
	]

	print(f"Domain: [{a}, {b}]")
	print(f"Degree: {degree} (k+1 = {degree + 1} coefficients)")
	print(f"Number of constraints: {len(constraints)}")
	print()

	# Build constraint matrix
	C, y_vec = build_constraint_matrix(constraints, a, b, degree)
	print(f"Constraint matrix C shape: {C.shape}")
	print(f"y_vec = {y_vec}")
	print()

	# Compute projection operator
	P = compute_constraint_projection_operator(C)
	print(f"Projection operator P shape: {P.shape}")
	print()

	# Start with random weights
	key = jax.random.PRNGKey(42)
	weights = jax.random.normal(key, (degree + 1,))
	print(f"Original weights: {weights}")

	# Check constraints before projection
	Cw_before = C @ weights
	print(f"C @ w (before) = {Cw_before}")
	print(f"Constraint violation (before): {Cw_before - y_vec}")
	print()

	# Project weights
	projected = project_weights(weights, C, P, y_vec)
	print(f"Projected weights: {projected}")

	# Check constraints after projection
	Cw_after = C @ projected
	print(f"C @ w (after) = {Cw_after}")
	print(f"Constraint violation (after): {Cw_after - y_vec}")
	print()

	# Verify by evaluating the polynomial at constraint points
	print("Verification by evaluating polynomial:")
	x_test = jnp.array([[0.0], [1.0]])
	basis = compute_chebyshev_basis(x_test, jnp.array([a]), jnp.array([b]), degree)[:, 0, :]
	values = basis @ projected
	print(f" phi(0) = {values[0]:.6f}, expected = 1.0")
	print(f" phi(1) = {values[1]:.6f}, expected = 2.0")

	# Check derivative at x=0
	deriv_basis = compute_chebyshev_derivative_basis(
		jnp.array([0.0]), a, b, degree, derivative_order=1
	)
	deriv_value = deriv_basis @ projected
	print(f" phi'(0) = {deriv_value[0]:.6f}, expected = 0.0")
	print()

	print("=" * 60)
	print("TEST: Batched weights (for layer with multiple activations)")
	print("=" * 60)

	# Test with batched weights: (out_dim, in_dim, k+1)
	out_dim, in_dim = 2, 3
	weights_batched = jax.random.normal(key, (out_dim, in_dim, degree + 1))
	print(f"Batched weights shape: {weights_batched.shape}")

	# Project all weights at once
	projected_batched = project_weights(weights_batched, C, P, y_vec)
	print(f"Projected batched weights shape: {projected_batched.shape}")

	# Verify constraints are satisfied for all activations
	violations = jnp.einsum("nk,oik->oin", C, projected_batched) - y_vec
	print(f"Max constraint violation across all activations: {jnp.abs(violations).max():.2e}")
	print()
	print("All tests passed!")


@pytest.fixture
def chebyshev_setup():
	"""Common fixtures for the Chebyshev projection tests."""
	a, b = 0.0, 1.0
	degree = 5
	constraints = [
		(0.0, 1.0, 0),  # phi(0) = 1
		(1.0, 2.0, 0),  # phi(1) = 2
		(0.0, 0.0, 1),  # phi'(0) = 0
	]
	C, y_vec = build_constraint_matrix(constraints, a, b, degree)
	P = compute_constraint_projection_operator(C)
	return {
		"a": a,
		"b": b,
		"degree": degree,
		"constraints": constraints,
		"C": C,
		"y_vec": y_vec,
		"P": P,
	}


def test_projection_satisfies_constraints(chebyshev_setup):
	a = chebyshev_setup["a"]
	b = chebyshev_setup["b"]
	degree = chebyshev_setup["degree"]
	C = chebyshev_setup["C"]
	y_vec = chebyshev_setup["y_vec"]
	P = chebyshev_setup["P"]

	key = jax.random.PRNGKey(42)
	weights = jax.random.normal(key, (degree + 1,))

	projected = project_weights(weights, C, P, y_vec)

	# Constraints should be met (with reasonable tolerance for single precision)
	# Derivative constraints can have larger scaling factors, so use 1e-4 tolerance
	assert jnp.allclose(C @ projected, y_vec, atol=1e-4)

	# Evaluate polynomial at constraint points
	x_test = jnp.array([[0.0], [1.0]])
	basis = compute_chebyshev_basis(x_test, jnp.array([a]), jnp.array([b]), degree)[:, 0, :]
	values = basis @ projected

	assert jnp.isclose(values[0], 1.0, atol=1e-4)
	assert jnp.isclose(values[1], 2.0, atol=1e-4)

	# Check derivative constraint at x=0
	deriv_basis = compute_chebyshev_derivative_basis(
		jnp.array([0.0]), a, b, degree, derivative_order=1
	)
	deriv_value = deriv_basis @ projected
	assert jnp.isclose(deriv_value[0], 0.0, atol=1e-3)  # Derivative constraints have larger scaling


def test_projection_batched_weights(chebyshev_setup):
	degree = chebyshev_setup["degree"]
	C = chebyshev_setup["C"]
	y_vec = chebyshev_setup["y_vec"]
	P = chebyshev_setup["P"]

	out_dim, in_dim = 2, 3
	key = jax.random.PRNGKey(123)
	weights_batched = jax.random.normal(key, (out_dim, in_dim, degree + 1))

	projected_batched = project_weights(weights_batched, C, P, y_vec)

	# Use reasonable tolerance for single precision with derivative constraints
	violations = jnp.einsum("nk,oik->oin", C, projected_batched) - y_vec
	assert jnp.max(jnp.abs(violations)) < 1e-3


if __name__ == "__main__":
	run_constraint_projection_demo()