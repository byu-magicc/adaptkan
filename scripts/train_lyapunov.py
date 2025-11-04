import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from jax import vmap
from adaptkan.jax.fit import fit
from adaptkan.jax.model import AdaptKANJax
from adaptkan.jax.utils import get_interp_indices_jax, spline_interpolate_with_coefs_and_indices
from adaptkan.jax.utils import get_deriv_interp_coefs_jax
from adaptkan.common.lyapunov_util import simulate_batch_trajectories, contour_plot, NeuralLyapunov, conformal_analysis, error_tolerance_to_confidence, lyapunov_loss
from adaptkan.common.dynamics import CrossCoupledCubic

import os
os.environ["JAX_PLATFORM_NAME"] = "cpu"

# Uncomment this line to enable 64 bit precision
# jax.config.update("jax_enable_x64", True)

class CrossCoupledCubic(eqx.Module):
    
    # Indices for the system
    X1 = 0
    X2 = 1
    n_dims: int = 2
    n_controls: int = 1
    
    def f(self, t, x, args):
        x1 = x[:, self.X1]
        x2 = x[:, self.X2]
        return jnp.array(
            [
                x2**3,
                -x1**3
            ]
        )
        
    def g(self, t, x, args):
        return jnp.array(
            [
                [1],
                [0]
            ]
        )

if __name__ == "__main__":

    # Define initial parameters
    num_samples = 10000
    test_split = 0.2
    init_min = [-3.0, -3.0]
    init_max = [3.0, 3.0]
    lr = 0.01
    steps = 1000
    optimizer = "Adam"
    model_type = "adaptkan" # "mlp" or "adaptkan"
    manual_adapt_every = None
    dynamics = CrossCoupledCubic()
    adapt_first = False
    calibration_points = 200

    args = {
        "lamb1": 10.0,
        "lamb2": 0.1, # LfV loss
        "lamb3": 1.0, # LgV loss
        "lamb4": 1.0, # Sandwich loss
        "lamb5": 0.0, # Positivity loss (don't care about this if we are enforcing positive definiteness)
        "k1": 0.001, # Lower bound of the bowl loss
        "k2": 10.0, # Upper bound of the bowl loss
        "k3": 0.001, # 0.02
        "dynamics": dynamics,
        "manual_adapt_every": manual_adapt_every,
        "adapt_first": adapt_first,
        "tau": 1.0, # 1.0 for cross coupled cubic system
    }

    # seeds = [1250738658,  520122828, 1722835854]
    seeds = [2364524912, 2691673004, 2927334345, 132970749, 4182733318]
    pooled_errors = []

    for seed in seeds:
        key = jax.random.PRNGKey(seed)

        if model_type == "mlp":
            V = eqx.nn.MLP(
                in_size=2,
                out_size=10,
                width_size=64,
                depth=4,
                activation=jax.nn.tanh,
                key=key,
                # final_activation=lambda x: scaled_parabola(x, scale=.5)
            )
            state = None # No state for the MLP
        else:
            V, state =  eqx.nn.make_with_state(AdaptKANJax)(width=[dynamics.n_dims, 10, 1],
                                                            initialization_range=[init_min[0], init_max[0]],
                                                            num_grid_intervals=3, 
                                                            k=3,
                                                            prune_patience=1,
                                                            seed=seed,
                                                            activation_strategy='linear')
        
        model = NeuralLyapunov(V, already_pos_definite=False)

        train_key, test_key = jax.random.split(key)

        X_train = jax.random.uniform(train_key,
                                    shape=(int(num_samples*(1-test_split)),dynamics.n_dims),
                                    minval=jnp.array(init_min),
                                    maxval=jnp.array(init_max))
        
        X_test = jax.random.uniform(test_key,
                                    shape=(int(num_samples*test_split),dynamics.n_dims),
                                    minval=jnp.array(init_min),
                                    maxval=jnp.array(init_max))

        model, state, results = fit(model,
                                    state,
                                    train_data={"X": X_train},
                                    test_data={"X": X_test},
                                    learning_rate=lr,
                                    steps=steps,
                                    opt=optimizer,
                                    loss_fn=lyapunov_loss,
                                    test_loss_fn=lyapunov_loss,
                                    loss_args=args,
                                    display_metrics=["train_loss", "test_loss"])
        
        # Evaluate on a few trajectories
        x_start = jnp.linspace(-2.5, 2.5, 5)
        y_start = jnp.array([-2.5, -1.5, 1.5, 2.5])
        x, y = jnp.meshgrid(x_start, y_start)
        initial_conditions = jnp.stack([x.flatten(), y.flatten()], axis=-1)
        T = 10.0 # Simulation time

        num_params = sum(x.size for x in jax.tree_util.tree_leaves(V) if hasattr(x, 'size'))
        print("Model has ", num_params, " parameters")

        # Uncomment out this section to use a known lyapunov function for this system
        # preset_V = lambda x: 0.25 * x[0]**4 + 0.25 * x[1]**4
        # model = NeuralLyapunov(preset_V, already_pos_definite=True)

        train_key, calibration_key = jax.random.split(train_key)

        all_trajs = simulate_batch_trajectories(model, state, dynamics, 
                                                initial_conditions, T, dt = 0.001) # model.dt)
        sols = [all_trajs[i] for i in range(initial_conditions.shape[0])]

        contour_plot(model,
                    state,
                    init_min, init_max,
                    title=model_type.capitalize(),
                    xlabel=r"$x_1$",
                    ylabel=r"$x_2$",
                    points=500,
                    sols=sols,
                    plot_vdot=False,
                    dynamics=dynamics)
        
        # Store the points for conformal prediction analysis
        X_calibration = jax.random.uniform(calibration_key,
                                    shape=(calibration_points, dynamics.n_dims),
                                    minval=jnp.array(init_min),
                                    maxval=jnp.array(init_max))
        
        calibration_trajs = simulate_batch_trajectories(model, state, dynamics, X_calibration, T, dt = 0.001)
        end_points = calibration_trajs[:, -1]
        end_points_cleaned = jnp.where(jnp.isnan(end_points), jnp.inf, end_points)
        # Generalize this to any equilibrium point later
        errors = jnp.linalg.norm(end_points_cleaned, axis=1)

        pooled_errors.append(np.array(errors))
    
# Print out the conformal prediction analysis
print('Conformal Analysis at 1-delta=0.8 gives:', conformal_analysis(np.concatenate(pooled_errors), confidence=0.8))
print('Conformal Analysis at 1-delta=0.9 gives:', conformal_analysis(np.concatenate(pooled_errors), confidence=0.9))
print('Conformal Analysis at 1-delta=0.95 gives:', conformal_analysis(np.concatenate(pooled_errors), confidence=0.95))

print('Percentage converging within 0.5', error_tolerance_to_confidence(np.concatenate(pooled_errors), 0.5)["confidence"])
print('Percentage converging within 0.25', error_tolerance_to_confidence(np.concatenate(pooled_errors), 0.2)["confidence"])
print('Percentage converging within 0.1', error_tolerance_to_confidence(np.concatenate(pooled_errors), 0.1)["confidence"])
    





    
    

