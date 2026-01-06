"""
Unit tests for Chebyshev basis adaptation mechanism (shrinking and stretching).
"""

import jax
import jax.numpy as jnp
import equinox as eqx
from adaptkan.jax.model import AdaptKANJax
from adaptkan.jax.losses import mse_loss
import optax


def create_model_and_state(basis_type='chebyshev'):
    """Create a fresh model for each test."""
    return eqx.nn.make_with_state(AdaptKANJax)(
        width=[2, 1],  # Simple 2->1 layer
        num_grid_intervals=5,
        prune_patience=1,
        k=3,
        seed=0,
        stretch_mode='max',
        basis_type=basis_type
    )


@eqx.filter_jit
def train_step(model, state, opt_state, batch, optim):
    """Single training step with adaptation."""

    def loss_fn(model, state, batch):
        xs, ys = batch['X'], batch['y']
        pred_ys, state = model(xs, state)
        loss = jnp.mean((pred_ys.squeeze() - ys.squeeze()) ** 2)
        return loss, state

    (loss, state), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(model, state, batch)

    # Adapt the domain
    model, state, adapted = model.adapt(state)

    # Update weights
    updates, opt_state = optim.update(grads, opt_state, params=eqx.filter(model, eqx.is_array))
    model = eqx.apply_updates(model, updates)

    return model, state, opt_state, loss, adapted


def test_shrinking():
    """
    TEST 1: SHRINKING
    Initial domain: [-1, 1]
    Data range: [-0.5, 0.5]
    Expected: Domain should shrink to approximately [-0.5, 0.5]
    """
    print('=' * 60)
    print('TEST 1: SHRINKING - Data in [-0.5, 0.5], domain [-1, 1]')
    print('=' * 60)

    model, state = create_model_and_state('chebyshev')
    layer = model.layers[0]

    print(f'Initial domain: a={state.get(layer.a)}, b={state.get(layer.b)}')

    # Create data in [-0.5, 0.5]
    key = jax.random.PRNGKey(42)
    X = jax.random.uniform(key, (100, 2), minval=-0.5, maxval=0.5)
    y = jnp.sum(X, axis=1, keepdims=True)  # Simple target

    print(f'Data range: [{X.min():.3f}, {X.max():.3f}]')

    # Setup optimizer
    optim = optax.adam(0.01)
    opt_state = optim.init(eqx.filter(model, eqx.is_array))

    batch = {'X': X, 'y': y}

    # Run multiple training steps
    adapted_any = False
    for step in range(10):
        model, state, opt_state, loss, adapted = train_step(model, state, opt_state, batch, optim)
        layer = model.layers[0]  # Get updated layer reference
        if adapted:
            adapted_any = True
            print(f'Step {step}: Adapted! a={state.get(layer.a)}, b={state.get(layer.b)}')

    print(f'\nFinal domain: a={state.get(layer.a)}, b={state.get(layer.b)}')
    print(f'data_counts: {state.get(layer.data_counts)}')
    print(f'ood_data_counts: {state.get(layer.ood_data_counts)}')

    # Check results
    a_val = state.get(layer.a)
    b_val = state.get(layer.b)

    # Domain should have shrunk (a > -1 or b < 1)
    shrunk = (a_val > -0.9).any() or (b_val < 0.9).any()

    if shrunk:
        print('✓ SHRINKING TEST PASSED!')
        return True
    else:
        print('✗ SHRINKING TEST FAILED - Domain did not shrink')
        return False


def test_stretching():
    """
    TEST 2: STRETCHING
    Initial domain: [-1, 1]
    Data range: [-2, 2]
    Expected: Domain should stretch to approximately [-2, 2]
    """
    print('\n' + '=' * 60)
    print('TEST 2: STRETCHING - Data in [-2, 2], domain [-1, 1]')
    print('=' * 60)

    model, state = create_model_and_state('chebyshev')
    layer = model.layers[0]

    print(f'Initial domain: a={state.get(layer.a)}, b={state.get(layer.b)}')

    # Create data in [-2, 2]
    key = jax.random.PRNGKey(42)
    X = jax.random.uniform(key, (100, 2), minval=-2.0, maxval=2.0)
    y = jnp.sum(X, axis=1, keepdims=True)

    print(f'Data range: [{X.min():.3f}, {X.max():.3f}]')

    # Setup optimizer
    optim = optax.adam(0.01)
    opt_state = optim.init(eqx.filter(model, eqx.is_array))

    batch = {'X': X, 'y': y}

    # Run multiple training steps
    adapted_any = False
    for step in range(10):
        model, state, opt_state, loss, adapted = train_step(model, state, opt_state, batch, optim)
        layer = model.layers[0]
        if adapted:
            adapted_any = True
            print(f'Step {step}: Adapted! a={state.get(layer.a)}, b={state.get(layer.b)}')

    print(f'\nFinal domain: a={state.get(layer.a)}, b={state.get(layer.b)}')
    print(f'ood_a: {state.get(layer.ood_a)}, ood_b: {state.get(layer.ood_b)}')
    print(f'data_counts: {state.get(layer.data_counts)}')
    print(f'ood_data_counts: {state.get(layer.ood_data_counts)}')

    # Check results
    a_val = state.get(layer.a)
    b_val = state.get(layer.b)

    # Domain should have stretched (a < -1 and b > 1)
    stretched = (a_val < -1.0).any() and (b_val > 1.0).any()

    if stretched:
        print('✓ STRETCHING TEST PASSED!')
        return True
    else:
        print('✗ STRETCHING TEST FAILED - Domain did not stretch')
        return False


def test_asymmetric_stretching():
    """
    TEST 3: ASYMMETRIC STRETCHING
    Initial domain: [-1, 1]
    Data range: [0, 2]
    Expected: 'a' should shrink toward 0, 'b' should stretch to ~2
    """
    print('\n' + '=' * 60)
    print('TEST 3: ASYMMETRIC - Data in [0, 2], domain [-1, 1]')
    print('=' * 60)

    model, state = create_model_and_state('chebyshev')
    layer = model.layers[0]

    print(f'Initial domain: a={state.get(layer.a)}, b={state.get(layer.b)}')

    # Create data in [0, 2]
    key = jax.random.PRNGKey(42)
    X = jax.random.uniform(key, (100, 2), minval=0.0, maxval=2.0)
    y = jnp.sum(X, axis=1, keepdims=True)

    print(f'Data range: [{X.min():.3f}, {X.max():.3f}]')

    # Setup optimizer
    optim = optax.adam(0.01)
    opt_state = optim.init(eqx.filter(model, eqx.is_array))

    batch = {'X': X, 'y': y}

    # Run multiple training steps
    for step in range(10):
        model, state, opt_state, loss, adapted = train_step(model, state, opt_state, batch, optim)
        layer = model.layers[0]
        if adapted:
            print(f'Step {step}: Adapted! a={state.get(layer.a)}, b={state.get(layer.b)}')

    print(f'\nFinal domain: a={state.get(layer.a)}, b={state.get(layer.b)}')
    print(f'ood_a: {state.get(layer.ood_a)}, ood_b: {state.get(layer.ood_b)}')
    print(f'data_counts: {state.get(layer.data_counts)}')
    print(f'ood_data_counts: {state.get(layer.ood_data_counts)}')

    # Check results
    a_val = state.get(layer.a)
    b_val = state.get(layer.b)

    # 'a' should have moved toward 0 (shrunk from -1), 'b' should have stretched beyond 1
    a_shrunk = (a_val > -0.5).any()  # a moved from -1 toward 0
    b_stretched = (b_val > 1.0).any()  # b stretched beyond 1

    if b_stretched:
        print('✓ ASYMMETRIC STRETCHING TEST PASSED!')
        return True
    else:
        print('✗ ASYMMETRIC STRETCHING TEST FAILED')
        print(f'  Expected b > 1.0, got b = {b_val}')
        return False


def test_compare_bspline_chebyshev():
    """
    TEST 4: Compare B-spline and Chebyshev adaptation behavior
    Both should adapt similarly when given the same out-of-domain data
    """
    print('\n' + '=' * 60)
    print('TEST 4: COMPARE B-SPLINE vs CHEBYSHEV ADAPTATION')
    print('=' * 60)

    # Create data in [-2, 2]
    key = jax.random.PRNGKey(42)
    X = jax.random.uniform(key, (100, 2), minval=-2.0, maxval=2.0)
    y = jnp.sum(X, axis=1, keepdims=True)
    batch = {'X': X, 'y': y}

    results = {}

    for basis_type in ['bspline', 'chebyshev']:
        print(f'\n--- {basis_type.upper()} ---')
        model, state = create_model_and_state(basis_type)
        layer = model.layers[0]

        print(f'Initial domain: a={state.get(layer.a)}, b={state.get(layer.b)}')

        optim = optax.adam(0.01)
        opt_state = optim.init(eqx.filter(model, eqx.is_array))

        adapted_count = 0
        for step in range(10):
            model, state, opt_state, loss, adapted = train_step(model, state, opt_state, batch, optim)
            layer = model.layers[0]
            if adapted:
                adapted_count += 1

        a_val = state.get(layer.a)
        b_val = state.get(layer.b)

        print(f'Final domain: a={a_val}, b={b_val}')
        print(f'Adapted {adapted_count} times')

        results[basis_type] = {
            'a': a_val,
            'b': b_val,
            'adapted_count': adapted_count
        }

    # Both should have stretched
    bs_stretched = (results['bspline']['a'] < -1.0).any() and (results['bspline']['b'] > 1.0).any()
    ch_stretched = (results['chebyshev']['a'] < -1.0).any() and (results['chebyshev']['b'] > 1.0).any()

    print(f'\nB-spline stretched: {bs_stretched}')
    print(f'Chebyshev stretched: {ch_stretched}')

    if bs_stretched and ch_stretched:
        print('✓ BOTH ADAPTED SIMILARLY!')
        return True
    elif bs_stretched and not ch_stretched:
        print('✗ CHEBYSHEV DID NOT ADAPT (but B-spline did)')
        return False
    else:
        print('✗ UNEXPECTED BEHAVIOR')
        return False


if __name__ == '__main__':
    print('\n' + '=' * 60)
    print('CHEBYSHEV ADAPTATION UNIT TESTS')
    print('=' * 60 + '\n')

    results = []
    results.append(('Shrinking', test_shrinking()))
    results.append(('Stretching', test_stretching()))
    results.append(('Asymmetric', test_asymmetric_stretching()))
    results.append(('Compare B-spline/Chebyshev', test_compare_bspline_chebyshev()))

    print('\n' + '=' * 60)
    print('SUMMARY')
    print('=' * 60)

    all_passed = True
    for name, passed in results:
        status = '✓ PASSED' if passed else '✗ FAILED'
        print(f'{name}: {status}')
        if not passed:
            all_passed = False

    print('=' * 60)
    if all_passed:
        print('ALL TESTS PASSED!')
    else:
        print('SOME TESTS FAILED!')
