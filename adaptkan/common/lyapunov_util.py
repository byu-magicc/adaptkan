import jax.numpy as jnp
import equinox as eqx
import jax
import numpy as np

# VISUALIZATION FUNCTIONS
def contour_plot(model, state, init_min, init_max, title="Contour Plot", xlabel="x", ylabel="y", points=100, sols=None, extra_points=None, extra_points_mask=None, plot_vdot=False, dynamics=None, mode="LfV"):
    import matplotlib.pyplot as plt
    import numpy as np
    
    x = jnp.linspace(init_min[0], init_max[0], points)
    y = jnp.linspace(init_min[1], init_max[1], points)

    X, Y = np.meshgrid(x, y)
    V_input = jnp.stack([X, Y], axis=-1).reshape(-1, 2)

    if plot_vdot and dynamics is not None:
        f = dynamics.f(0, V_input, None)
        g = dynamics.g(0, V_input, None)
        dv_dx = model.dV_dx(V_input, state)
        u = model.u(V_input, state, dynamics)
        # Z = (dv_dx * f.T).sum(-1) + (dv_dx @ g * u).sum(-1)
        # Z = Z.reshape(X.shape)
        if mode == "LfV":
            Z = (dv_dx * f.T).sum(-1).reshape(X.shape)
        elif mode == "LgV":
            Z = (dv_dx @ g).sum(-1).reshape(X.shape)
        elif mode == "LgV_u":
            Z = (dv_dx @ g * u).sum(-1).reshape(X.shape)
        elif mode == "u":
            Z = u.squeeze().reshape(X.shape)
        else:
            Z = (dv_dx * f.T).sum(-1) + (dv_dx @ g * u).sum(-1)
    else:
        Z = model.V_forward(V_input, state)[0].reshape(X.shape)

    # print(f"Corner points check:")
    # print(f"Bottom-left: input={V_input[0]}, value={Z.flatten()[0]}")
    # print(f"Top-right: input={V_input[-1]}, value={Z.flatten()[-1]}")
    
    plt.figure()
    levels = np.array([-.5, -.2, -.1, -.05, -.01, .01, .05, .1, .2, .5])
    contour = plt.contour(X, Y, Z, levels=levels, colors='black', linewidths=0.3, alpha=0.6)
    plt.clabel(contour, inline=True, fontsize=5, fmt='%.2f')
    plt.contourf(X, Y, Z, levels=50, cmap='plasma')
    plt.colorbar(label='Value')
    plt.title(" ".join([title, "(V_dot)"]) if plot_vdot else title, fontsize=24)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.axis('square')
    
    if sols is not None:
        for i, sol in enumerate(sols):
            # Plot the trajectory
            plt.plot(sol[:,0], sol[:,1], color='gray', linewidth=1)
            
            # Add arrow at the end of trajectory
            # Use last two points to determine direction
            if len(sol) >= 2:
                # Get last two points
                x_end, y_end = sol[-1, 0], sol[-1, 1]
                x_prev, y_prev = sol[-2, 0], sol[-2, 1]
                
                # Calculate direction
                dx = x_end - x_prev
                dy = y_end - y_prev
                
                # Plot arrow
                plt.arrow(x_prev, y_prev, dx, dy, 
                         head_width=0.01, head_length=0.005, 
                         fc='gray', ec='gray', linewidth=2)
            
            # Optional: Mark starting point with a circle
            plt.plot(sol[0, 0], sol[0, 1], 'o', color='black', markersize=5)

    if extra_points is not None and extra_points_mask is not None:
        pos = extra_points[extra_points_mask]
        neg = extra_points[~extra_points_mask]
        plt.scatter(pos[:, 0], pos[:, 1], color='green', label='Stable', s=3)
        plt.scatter(neg[:, 0], neg[:, 1], color='darkred' if plot_vdot else "salmon", s=3)


    
    # plt.legend()
            
    plt.show()

@eqx.filter_jit
def simulate_batch_trajectories(model, state, dynamics, x0_batch, T, dt=0.001):
    """
    Simulate multiple trajectories in parallel (FASTEST)
    
    Args:
        model: NeuralLyapunov model
        state: State for adaptive models
        dynamics: System dynamics
        x0_batch: (n_traj, 2) array of initial conditions
        T: Total simulation time
        dt: Time step
    
    Returns:
        trajectories: (n_traj, N, 2) array
    """
    def closed_loop_dynamics(t, x_batch):
        # x_batch is (n_traj, 2)
        u = model.u(x_batch, state, dynamics)
        
        # Compute closed-loop dynamics for all trajectories at once
        f = dynamics.f(t, x_batch, None)
        g = dynamics.g(t, x_batch, None)
        
        # f is (2, n_traj), g is (2, 1), u is (n_traj, 1)

        dx_dt = (f + g @ u.T).T  # (n_traj, 2)
        # dx_dt = (f + jnp.sum((g.squeeze(1) * u.T), axis=0, keepdims=True)).T  # (n_traj, 2)
        return dx_dt
    
    def rk4_step_batch(f, t, y_batch, dt):
        """RK4 step for batch of states"""
        k1 = f(t, y_batch)
        k2 = f(t + dt/2, y_batch + dt*k1/2)
        k3 = f(t + dt/2, y_batch + dt*k2/2)
        k4 = f(t + dt, y_batch + dt*k3)
        return y_batch + dt*(k1 + 2*k2 + 2*k3 + k4)/6
    
    steps = int(T / dt)
    
    def scan_fn(x_batch, i):
        t = i * dt
        x_next = rk4_step_batch(closed_loop_dynamics, t, x_batch, dt)
        return x_next, x_next
    
    _, trajectories = jax.lax.scan(scan_fn, x0_batch, jnp.arange(steps))
    
    # Add initial conditions
    trajectories = jnp.concatenate([x0_batch.reshape(1, *x0_batch.shape), 
                                   trajectories], axis=0)
    
    # Transpose to get (n_traj, steps, 2)
    return trajectories.transpose(1, 0, 2)

class NeuralLyapunov(eqx.Module):
    V: eqx.Module
    c: float = 1.0
    dt: float = 0.01
    enforce_pos_definite: bool = False
    eps: float = 1e-6
    u_min: float = -10.0
    u_max: float = 10.0
    lgv_threshold: float = 0.1  # Threshold for sufficient control authority
    already_pos_definite: bool = True  

    def __init__(self, model, enforce_pos_definite=False, eps=1e-6, u_min=-30, u_max=30, lgv_threshold=0.5, already_pos_definite=True):
        self.V = model
        self.enforce_pos_definite = enforce_pos_definite
        self.eps = eps
        self.u_min = u_min
        self.u_max = u_max
        self.lgv_threshold = lgv_threshold
        self.already_pos_definite = already_pos_definite

    def __call__(self, x, state, dynamics):
        u_out = self.u(x, state, dynamics)
        return u_out
    
    def u(self, x, state, dynamics):

        # return -x[:,:1]**3

        dv_dx = self.dV_dx(x, state)
        f = dynamics.f(0, x, None) # (2, batch_size)
        g = dynamics.g(0, x, None) # (2, 1)

        LgV = (dv_dx @ g)
        # LgV = jnp.sum(dv_dx * g.squeeze(1).T, axis=-1)[:, None] # (dv_dx @ g)  # Lie derivative along g
        LfV = (dv_dx * f.T).sum(-1, keepdims=True)  # Lie derivative along f
        # v = self.V_forward(x, state)[0]

        # Use Sontag's Universal Formula instead

        u = -(LfV + jnp.sqrt(LfV**2 + LgV**4)) / (LgV + 1e-8)
        # print()

        # numer = - (jax.nn.relu(LfV + self.c * v) * LgV).sum(-1, keepdims=True)
        # #numer = - (jax.nn.relu(LfV + 0.001 * (x**2).sum(-1, keepdims=True)) * LgV).sum(-1, keepdims=True)
        # denom = (LgV * LgV + 1e-6).sum(-1, keepdims=True)
        # u = numer / denom

        # u = jnp.clip(u, self.u_min, self.u_max)

        return u
    
    def dV_dx(self, x, state):

        # Takes in a batch of inputs, returns a the gradients w.r.t. each input
        if hasattr(self.V, 'adapt'):
            def single_point_output(x_point):
                # Model single point -> single output
                v_out, _ = self.V(x_point.reshape(1, -1), state, update=False)  # Add batch dim
                # Squared output
                if self.enforce_pos_definite:
                    v_out = (0.5 * (v_out * v_out).sum() + self.eps) * jax.nn.tanh(jnp.sqrt((x_point * x_point).sum()))
                elif self.already_pos_definite:
                    v_out = jnp.squeeze(v_out)
                else:
                    v_out = 0.5 * (v_out * v_out).sum()
                return v_out
            
            gradients = jax.vmap(jax.grad(single_point_output))(x)

        else:
            def scalar_V(x_point):
                v_out = self.V(x_point)
                # Squared output
                if self.enforce_pos_definite:
                    v_out = (0.5 * (v_out * v_out).sum(-1) + self.eps) * jax.nn.tanh(jnp.sqrt((x_point * x_point).sum(-1)))
                elif self.already_pos_definite:
                    v_out = jnp.squeeze(v_out)
                else:
                    v_out = 0.5 * (v_out * v_out).sum(-1)

                return v_out  # Extract scalar from (1,) shaped output
            
            gradients = jax.vmap(jax.grad(scalar_V))(x)

        return gradients
        
    def V_forward(self, x, state, update=False):
        if hasattr(self.V, 'adapt'):
            v_out, state = self.V(x, state, update=update)
        else:
            v_out = jax.vmap(self.V)(x)

        if self.enforce_pos_definite:
            v_out = (0.5 * (v_out * v_out).sum(-1, keepdims=True) + self.eps) * jax.nn.tanh(jnp.sqrt((x * x).sum(-1, keepdims=True)))
        elif self.already_pos_definite:
            v_out = v_out
        else:
            v_out = 0.5 * (v_out * v_out).sum(-1, keepdims=True)

        return v_out, state
        
    # Define these three functions as a wrapper for AdaptKANJax
    def refine(self, state, new_num_grid_intervals):
        if hasattr(self.V, "refine"):
            new_V, new_state = self.V.refine(state, new_num_grid_intervals=new_num_grid_intervals)
            new_model = NeuralLyapunov(new_V, enforce_pos_definite=self.enforce_pos_definite, eps=self.eps)
            return new_model, new_state
        else:
            return self, state
        
    def adapt(self, state):
        if hasattr(self.V, "adapt"):
            new_V, new_state, adapted = self.V.adapt(state)
            new_model = NeuralLyapunov(new_V, enforce_pos_definite=self.enforce_pos_definite, eps=self.eps, already_pos_definite=self.already_pos_definite)
            return new_model, new_state, adapted
        else:
            return self, state, False

    def manual_adapt(self, state):
        if hasattr(self.V, "manual_adapt"):
            new_V, new_state, adapted = self.V.manual_adapt(state)
            new_model = NeuralLyapunov(new_V, enforce_pos_definite=self.enforce_pos_definite, eps=self.eps, already_pos_definite=self.already_pos_definite)
            return new_model, new_state, adapted
        else:
            return self, state, False
        
def error_tolerance_to_confidence(errors, tolerance):
    """
    Given calibration errors and a tolerance, return confidence level
    """
    K = len(errors)
    within_tolerance = sum(1 for e in errors if e <= tolerance)
    
    # Conservative conformal prediction estimate
    # We add + 1 here because we consider the test point being in the empirical distribution
    confidence = (within_tolerance) / (K + 1)
    
    return {
        'confidence': confidence,
        'count_within': within_tolerance,
        'total_samples': K,
        'statement': f"P(error ≤ {tolerance}) ≥ {confidence:.1%}"
    }

def conformal_analysis(errors, confidence=0.9):
    """Apply conformal prediction to error data"""
    K = len(errors)
    delta = 1 - confidence
    p = int(np.ceil((K + 1) * confidence))
    
    if p <= K:
        sorted_errors = sorted(errors)
        threshold = sorted_errors[p-1]  # (p-1) for 0-indexing
    else:
        threshold = float('inf')
    
    return threshold

@eqx.filter_jit
def lyapunov_loss(model, state, batch, args): 

    if args.get("index_batch", False):
        # We are indexing into a larger batch
        xs = batch["X"][args["current_epoch"]]
    else:
        xs = batch["X"]

    x0 = jnp.zeros((1, xs.shape[1]))  # Origin

    v0, state = model.V_forward(x0, state)
    v, state = model.V_forward(xs, state, update=args["update"])
    dv_dx = model.dV_dx(xs, state)
    f = args["dynamics"].f(0, xs, None)
    g = args["dynamics"].g(0, xs, None)

    # Origin Loss
    loss1 = (v0 ** 2).squeeze()

    LfV = jnp.sum(dv_dx * f.T, axis=-1)
    LgV = jnp.sum(dv_dx @ g, axis=-1)

    tau = args.get("tau", 1.0)

    large_LgV = jnp.abs(LgV) > tau # LgV is small
    small_LgV = jnp.abs(LgV) <= tau # LgV is large

    loss2 = jnp.mean(large_LgV * jax.nn.relu(-LfV)) + jnp.mean(small_LgV * jax.nn.relu(LfV))
    loss3 = jnp.mean(small_LgV * jax.nn.relu(tau - jnp.abs(LgV + 1e-8)))

    # Bowl losses
    loss4 = jnp.mean(jax.nn.relu(args["k1"]*(xs**2).sum(-1) - v.squeeze(-1)))
    loss5 = jnp.mean(jax.nn.relu(v.squeeze(-1) - args["k2"]*(xs**2).sum(-1)))

    loss6 = jnp.mean(jax.nn.relu(-v))

    loss = args["lamb1"] * loss1 + args["lamb2"] * loss2 + args["lamb3"] * loss3 + args["lamb4"] * (loss4 + loss5) + args["lamb5"] * loss6

    return loss, ({"loss": loss, "loss1": loss1, "loss2": loss2, "loss3": loss3, "loss4": loss4, "loss5": loss5, "loss6": loss6, "batch_size": xs.shape[0]}, state)