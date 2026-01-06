import jax
import jax.numpy as jnp
import pytest

from adaptkan.jax.layers import AdaptKANLayerJax


def test_constraint_initialization_dicts():
    constraints = {0: jnp.array([[0.0, 0], [1.5, 1]]),
                   2: jnp.array([[0.5, 0], [1.0, 0]])}
    out_dim, in_dim = 3, 4
    # Shape must be (out_dim, in_dim, N_constraints)
    constraint_values = {
        0: jnp.array(
                [[1.0, 2.0],
                 [5.0, 6.0],
                 [9.0, 10.0]]
        ),
        2: jnp.array(
                [[3.0, 4.0],
                 [7.0, 8.0],
                 [11.0, 12.0]]
        )
    }

    layer = AdaptKANLayerJax(
        in_dim=in_dim,
        out_dim=out_dim,
        k=5,
        basis_type="chebyshev",
        constraints=constraints,
        constraint_values=constraint_values,
        initialization_range=[-1.0, 1.0],
        key=jax.random.PRNGKey(0),
    )

    assert layer.has_constraints is True
    assert layer.constraint_C is not None
    assert layer.constraint_P is not None
    assert layer.constraint_C.shape == (len(constraints), layer.k + 1)
    assert layer.constraint_P.shape == (layer.k + 1, len(constraints))
    assert layer.constraint_y.shape == constraint_values.shape
    assert jnp.allclose(layer.constraint_y, constraint_values)
    assert layer.constraint_x_min == pytest.approx(0.0)
    assert layer.constraint_x_max == pytest.approx(1.5)


def test_constraint_initialization_disabled_without_constraints():
    layer = AdaptKANLayerJax(
        in_dim=2,
        out_dim=2,
        k=3,
        basis_type="chebyshev",
        constraints=None,
        constraint_values=None,
        key=jax.random.PRNGKey(1),
    )

    assert layer.has_constraints is False
    assert layer.constraint_C is None
    assert layer.constraint_P is None
    assert layer.constraint_y is None
    assert layer.constraint_x_min is None
    assert layer.constraint_x_max is None
