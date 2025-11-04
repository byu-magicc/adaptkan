# These dynamics were generated using Claude
import jax.numpy as jnp
import equinox as eqx

class UnicycleRobot(eqx.Module):
    
    # Indices for the system
    X = 0
    Y = 1
    THETA = 2
    n_dims: int = 3
    n_controls: int = 2
    
    def f(self, t, x, args):
        return jnp.array(
            [
                0,
                0,
                0
            ]
        )
        
    def g(self, t, x, args):
        theta = x[:, self.THETA]
        return jnp.array(
            [
                [jnp.cos(theta), 0],
                [jnp.sin(theta), 0],
                [0, 1]
            ]
        )

class BrockettsTwistedSystem(eqx.Module):
    
    # Indices for the system
    X1 = 0
    X2 = 1
    X3 = 2
    n_dims: int = 3
    n_controls: int = 2
    
    def f(self, t, x, args):
        x3 = x[:, self.X3]
        return jnp.array(
            [
                x3,
                x3**2,
                0
            ]
        )
        
    def g(self, t, x, args):
        x1 = x[:, self.X1]
        x2 = x[:, self.X2]
        return jnp.array(
            [
                [1, 0],
                [0, 1],
                [-x2, x1]
            ]
        )


class BrockettsSystem(eqx.Module):
    
    # Indices for the system
    X1 = 0
    X2 = 1
    n_dims: int = 3
    n_controls: int = 2
    
    def f(self, t, x, args):
        return jnp.array(
            [
                0,
                0,
                0
            ]
        )
        
    def g(self, t, x, args):
        x1 = x[:, self.X1]
        x2 = x[:, self.X2]
        return jnp.array(
            [
                [1, 0],
                [0, 1],
                [-x2, x1]
            ]
        )

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
    
class Backstepping(eqx.Module):
    """ """
    n_dims: int = 2
    n_controls: int = 1
    
    # Indices for the system
    X1 = 0
    X2 = 1
    
    def f(self, t, x, args):
        x1 = x[:, self.X1]
        x2 = x[:, self.X2]
        return jnp.array(
            [
                -x1**3 + x2,
                jnp.zeros_like(x1)
            ]
        )
        
    def g(self, t, x, args):
        return jnp.array(
            [
                [0],
                [1]
            ]
        )

class InvertedPendulum(eqx.Module):
    """ """
    n_dims: int = 2
    n_controls: int = 1
    b = 0.01
    m = 1
    L = 1
    G = 9.81
    
    # Indices for the system
    THETA = 0
    THETA_DOT = 1
    
    def f(self, t, x, args):
        theta = x[:, self.THETA]
        theta_dot = x[:, self.THETA_DOT]
        return jnp.array(
            [
                theta_dot,
                self.G * jnp.sin(theta) / self.L - self.b * theta_dot / (self.m * self.L ** 2)
            ]
        )
        
    def g(self, t, x, args):
        return jnp.array(
            [
                [0],
                [1/(self.m * self.L**2)]
            ]
        )
    
class DoubleIntegrator(eqx.Module):
    """Double integrator system: ẍ = u
    
    State vector x = [position, velocity]
    Control input u = acceleration
    """
    n_dims: int = 2
    n_controls: int = 1
    
    # Indices for the system
    POSITION = 0
    VELOCITY = 1
    
    def f(self, t, x, args):
        """Drift dynamics: ẋ = f(x)
        
        For double integrator:
        ẋ₁ = x₂ (position derivative is velocity)
        ẋ₂ = 0  (no drift in acceleration)
        """
        position = x[:, self.POSITION]
        velocity = x[:, self.VELOCITY]
        
        return jnp.array([
            velocity,
            jnp.zeros_like(velocity)
        ])
    
    def g(self, t, x, args):
        """Control matrix: how control affects state
        
        Control directly affects acceleration (velocity derivative)
        """
        return jnp.array([
            [0],  # Control doesn't directly affect position
            [1]   # Control directly affects velocity (acceleration)
        ])


class VanDerPolOscillator(eqx.Module):
    """Van der Pol oscillator - exhibits limit cycles and nonlinear damping.
    
    Challenge: The state-dependent damping term creates regions of negative
    damping (energy injection) and positive damping (energy dissipation),
    leading to limit cycle behavior that's challenging to stabilize.
    """
    n_dims: int = 2
    n_controls: int = 1
    mu: float = 1.0  # Nonlinearity parameter (larger = stronger nonlinearity)
    
    # State indices
    X = 0  # Position
    X_DOT = 1  # Velocity
    
    def f(self, t, x, args):
        pos = x[:, self.X]
        vel = x[:, self.X_DOT]
        return jnp.array([
            vel,
            self.mu * (1 - pos**2) * vel - pos
        ])
    
    def g(self, t, x, args):
        return jnp.array([
            [0],
            [1]
        ])


class DuffingOscillator(eqx.Module):
    """Duffing oscillator - bistable system with cubic nonlinearity.
    
    Challenge: Multiple equilibria and potential wells create complex
    basins of attraction. The cubic stiffness can cause hardening or
    softening spring behavior depending on parameters.
    """
    n_dims: int = 2
    n_controls: int = 1
    alpha: float = -1.0  # Linear stiffness (negative for double-well)
    beta: float = 1.0   # Cubic stiffness
    delta: float = 0.3  # Damping coefficient
    
    # State indices
    X = 0  # Position
    X_DOT = 1  # Velocity
    
    def f(self, t, x, args):
        pos = x[:, self.X]
        vel = x[:, self.X_DOT]
        return jnp.array([
            vel,
            -self.delta * vel - self.alpha * pos - self.beta * pos**3
        ])
    
    def g(self, t, x, args):
        return jnp.array([
            [0],
            [1]
        ])


class LienardSystem(eqx.Module):
    """Generalized Liénard equation - encompasses many nonlinear oscillators.
    
    Challenge: The nonlinear damping function creates complex phase portraits
    with multiple regions of attraction/repulsion. Energy-based Lyapunov
    functions are difficult due to the state-dependent damping.
    """
    n_dims: int = 2
    n_controls: int = 1
    a: float = 0.5
    b: float = 1.0
    c: float = 1.0
    
    # State indices
    X = 0  # Position
    Y = 1  # Transformed velocity variable
    
    def f(self, t, x, args):
        pos = x[:, self.X]
        y = x[:, self.Y]
        # Nonlinear damping function: f(x) = a*x^2 - b
        f_x = self.a * pos**2 - self.b
        return jnp.array([
            y - f_x,
            -self.c * pos
        ])
    
    def g(self, t, x, args):
        return jnp.array([
            [0],
            [1]
        ])


class RayleighOscillator(eqx.Module):
    """Rayleigh oscillator - velocity-dependent nonlinear damping.
    
    Challenge: The velocity-cubed term creates strong nonlinear damping
    that changes sign, leading to self-sustained oscillations. Finding
    a Lyapunov function that handles the velocity nonlinearity is difficult.
    """
    n_dims: int = 2
    n_controls: int = 1
    epsilon: float = 1.0  # Nonlinearity strength
    omega0: float = 1.0   # Natural frequency
    
    # State indices
    X = 0  # Position
    X_DOT = 1  # Velocity
    
    def f(self, t, x, args):
        pos = x[:, self.X]
        vel = x[:, self.X_DOT]
        return jnp.array([
            vel,
            -self.omega0**2 * pos + self.epsilon * vel * (1 - vel**2/3)
        ])
    
    def g(self, t, x, args):
        return jnp.array([
            [0],
            [1]
        ])


class NonlinearMassSpringDamper(eqx.Module):
    """Mass-spring-damper with nonlinear spring and damping.
    
    Challenge: Combined nonlinearities in both restoring force and damping
    create complex energy landscapes. The system can exhibit jump phenomena
    and multiple steady states.
    """
    n_dims: int = 2
    n_controls: int = 1
    m: float = 1.0      # Mass
    k1: float = 1.0     # Linear spring coefficient
    k3: float = 0.5     # Cubic spring coefficient
    c1: float = 0.1     # Linear damping
    c3: float = 0.05    # Cubic damping
    
    # State indices
    X = 0  # Position
    X_DOT = 1  # Velocity
    
    def f(self, t, x, args):
        pos = x[:, self.X]
        vel = x[:, self.X_DOT]
        # Nonlinear spring force: k1*x + k3*x^3
        # Nonlinear damping: c1*v + c3*v^3
        return jnp.array([
            vel,
            -(self.k1 * pos + self.k3 * pos**3) / self.m - 
            (self.c1 * vel + self.c3 * vel**3) / self.m
        ])
    
    def g(self, t, x, args):
        return jnp.array([
            [0],
            [1/self.m]
        ])
    
class NonlinearMassSpringDamperModified(eqx.Module):
    """Mass-spring-damper with nonlinear spring and damping.
    
    Challenge: Combined nonlinearities in both restoring force and damping
    create complex energy landscapes. The system can exhibit jump phenomena
    and multiple steady states.
    """
    n_dims: int = 2
    n_controls: int = 1
    m: float = 1.0      # Mass
    k: float = 1.0     # Linear spring coefficient
    b: float = 0.01
    
    # State indices
    X = 0  # Position
    X_DOT = 1  # Velocity
    
    def f(self, t, x, args):
        pos = x[:, self.X]
        vel = x[:, self.X_DOT]
        # Nonlinear spring force: k1*x + k3*x^3
        # Nonlinear damping: c1*v + c3*v^3
        return jnp.array([
            vel,
            (- self.b * vel - self.k * pos ** 3) / (self.m * 1 + pos ** 2)
        ])
    
    def g(self, t, x, args):
        pos = x[:, self.X]
        return jnp.array([
            [jnp.zeros_like(pos)],
            [1 / (self.m * 1 + pos ** 2)]
        ])


class JerkSystem(eqx.Module):
    """Simplified jerk equation - third-order system reduced to 2D.
    
    Challenge: The jerk (derivative of acceleration) creates highly
    nonlinear dynamics with potential for chaotic behavior. The x^2
    term creates strong nonlinearity that's difficult to compensate.
    """
    n_dims: int = 2
    n_controls: int = 1
    a: float = 0.6  # Nonlinearity parameter
    
    # State indices (using phase space reduction)
    X = 0  # Position
    Y = 1  # Velocity-like variable
    
    def f(self, t, x, args):
        pos = x[:, self.X]
        y = x[:, self.Y]
        return jnp.array([
            y,
            -self.a * y - pos**2 + 1
        ])
    
    def g(self, t, x, args):
        return jnp.array([
            [0],
            [1]
        ])


class NegativeDampingOscillator(eqx.Module):
    """Oscillator with regions of negative damping.
    
    Challenge: The state-dependent damping coefficient can become negative,
    injecting energy into the system. This creates instability that must
    be actively controlled.
    """
    n_dims: int = 2
    n_controls: int = 1
    omega0: float = 1.0  # Natural frequency
    alpha: float = 0.5   # Damping nonlinearity
    beta: float = 2.0    # Damping offset
    
    # State indices
    X = 0  # Position
    X_DOT = 1  # Velocity
    
    def f(self, t, x, args):
        pos = x[:, self.X]
        vel = x[:, self.X_DOT]
        # Damping coefficient: alpha * x^2 - beta (can be negative)
        damping = self.alpha * pos**2 - self.beta
        return jnp.array([
            vel,
            -self.omega0**2 * pos - damping * vel
        ])
    
    def g(self, t, x, args):
        return jnp.array([
            [0],
            [1]
        ])


class FitzHughNagumo(eqx.Module):
    """FitzHugh-Nagumo model - simplified neuron dynamics.
    
    Challenge: Exhibits excitable dynamics with fast-slow time scales.
    The cubic nonlinearity creates a characteristic N-shaped nullcline
    leading to relaxation oscillations.
    """
    n_dims: int = 2
    n_controls: int = 1
    a: float = 0.7     # Recovery variable parameter
    b: float = 0.8     # Recovery variable parameter
    tau: float = 12.5  # Time scale separation
    
    # State indices
    V = 0  # Voltage-like variable
    W = 1  # Recovery variable
    
    def f(self, t, x, args):
        v = x[:, self.V]
        w = x[:, self.W]
        return jnp.array([
            v - v**3/3 - w,
            (v + self.a - self.b * w) / self.tau
        ])
    
    def g(self, t, x, args):
        return jnp.array([
            [1],
            [0]
        ])


class TunnelDiodeOscillator(eqx.Module):
    """Tunnel diode oscillator circuit model.
    
    Challenge: The tunnel diode characteristic creates a region of negative
    resistance, leading to complex oscillatory behavior. The nonlinearity
    is non-polynomial, making analysis difficult.
    """
    n_dims: int = 2
    n_controls: int = 1
    C: float = 1.0   # Capacitance
    L: float = 1.0   # Inductance
    G: float = 0.5   # Conductance
    Ip: float = 1.0  # Peak current
    Vp: float = 1.0  # Peak voltage
    
    # State indices
    V = 0  # Voltage across capacitor
    I = 1  # Current through inductor
    
    def f(self, t, x, args):
        v = x[:, self.V]
        i = x[:, self.I]
        # Tunnel diode characteristic (simplified)
        h_v = self.Ip * v / self.Vp * jnp.exp(1 - v / self.Vp)
        return jnp.array([
            (i - h_v - self.G * v) / self.C,
            -v / self.L
        ])
    
    def g(self, t, x, args):
        return jnp.array([
            [0],
            [1/self.L]
        ])


class MorrisLecar(eqx.Module):
    """Morris-Lecar neuron model (reduced 2D version).
    
    Challenge: Exhibits complex bifurcations between resting, oscillatory,
    and excitable states. The hyperbolic tangent nonlinearities create
    sharp transitions in the dynamics.
    """
    n_dims: int = 2
    n_controls: int = 1
    C: float = 20.0    # Membrane capacitance
    gL: float = 2.0    # Leak conductance
    gCa: float = 4.4   # Calcium conductance
    gK: float = 8.0    # Potassium conductance
    VL: float = -60.0  # Leak potential
    VCa: float = 120.0 # Calcium potential
    VK: float = -84.0  # Potassium potential
    V1: float = -1.2   # Potential at which M_ss = 0.5
    V2: float = 18.0   # Reciprocal of slope of M_ss
    V3: float = 2.0    # Potential at which N_ss = 0.5
    V4: float = 30.0   # Reciprocal of slope of N_ss
    phi: float = 0.04  # Recovery variable rate
    
    # State indices
    V = 0  # Membrane potential
    N = 1  # Potassium channel activation
    
    def f(self, t, x, args):
        v = x[:, self.V]
        n = x[:, self.N]
        
        # Steady-state functions
        m_inf = 0.5 * (1 + jnp.tanh((v - self.V1) / self.V2))
        n_inf = 0.5 * (1 + jnp.tanh((v - self.V3) / self.V4))
        tau_n = 1 / (self.phi * jnp.cosh((v - self.V3) / (2 * self.V4)))
        
        # Currents
        i_Ca = self.gCa * m_inf * (v - self.VCa)
        i_K = self.gK * n * (v - self.VK)
        i_L = self.gL * (v - self.VL)
        
        return jnp.array([
            (-i_Ca - i_K - i_L) / self.C,
            (n_inf - n) / tau_n
        ])
    
    def g(self, t, x, args):
        return jnp.array([
            [1/self.C],
            [0]
        ])


class BistableRelay(eqx.Module):
    """Bistable relay system with hysteresis.
    
    Challenge: The relay nonlinearity creates discontinuous dynamics
    with hysteresis. Standard smooth Lyapunov functions fail at the
    switching boundaries.
    """
    n_dims: int = 2
    n_controls: int = 1
    a: float = 1.0     # System parameter
    b: float = 0.5     # Damping
    h: float = 0.3     # Hysteresis width
    
    # State indices
    X = 0  # Position
    X_DOT = 1  # Velocity
    
    def f(self, t, x, args):
        pos = x[:, self.X]
        vel = x[:, self.X_DOT]
        
        # Smooth approximation of relay with hysteresis
        relay = jnp.tanh(10 * pos) - 0.5 * jnp.tanh(10 * (pos - self.h)) - 0.5 * jnp.tanh(10 * (pos + self.h))
        
        return jnp.array([
            vel,
            -self.a * relay - self.b * vel
        ])
    
    def g(self, t, x, args):
        return jnp.array([
            [0],
            [1]
        ])


class QuinticOscillator(eqx.Module):
    """Oscillator with quintic nonlinearity - triple-well potential.
    
    Challenge: Three potential wells create complex global dynamics.
    The system can exhibit chaotic transients between wells and
    fractal basin boundaries.
    """
    n_dims: int = 2
    n_controls: int = 1
    a: float = 1.0   # Quadratic coefficient
    b: float = 1.0   # Quartic coefficient  
    c: float = 0.2   # Damping
    
    # State indices
    X = 0  # Position
    X_DOT = 1  # Velocity
    
    def f(self, t, x, args):
        pos = x[:, self.X]
        vel = x[:, self.X_DOT]
        # Potential: V(x) = -ax^2/2 + bx^4/4 + x^6/6
        # Force: F = ax - bx^3 - x^5
        return jnp.array([
            vel,
            self.a * pos - self.b * pos**3 - pos**5 - self.c * vel
        ])
    
    def g(self, t, x, args):
        return jnp.array([
            [0],
            [1]
        ])


class SaturatingOscillator(eqx.Module):
    """Oscillator with saturating nonlinearity.
    
    Challenge: The saturation creates bounded but complex dynamics.
    The smooth transition between linear and saturated regions makes
    Lyapunov design challenging as the effective gain varies.
    """
    n_dims: int = 2
    n_controls: int = 1
    k: float = 1.0      # Spring constant
    sat_level: float = 2.0  # Saturation level
    alpha: float = 5.0   # Saturation sharpness
    c: float = 0.1      # Damping
    
    # State indices
    X = 0  # Position
    X_DOT = 1  # Velocity
    
    def f(self, t, x, args):
        pos = x[:, self.X]
        vel = x[:, self.X_DOT]
        
        # Smooth saturation function
        sat_force = self.sat_level * jnp.tanh(self.alpha * pos / self.sat_level)
        
        return jnp.array([
            vel,
            -self.k * sat_force - self.c * vel
        ])
    
    def g(self, t, x, args):
        return jnp.array([
            [0],
            [1]
        ])


class CoupledNonlinearOscillator(eqx.Module):
    """Two-mode coupled oscillator with nonlinear coupling.
    
    Challenge: The nonlinear coupling creates energy transfer between
    modes that depends on amplitude. This leads to complex modal
    interactions and potential internal resonances.
    """
    n_dims: int = 2
    n_controls: int = 1
    omega1: float = 1.0   # First mode frequency
    omega2: float = 1.6   # Second mode frequency (near resonance)
    gamma: float = 0.3    # Nonlinear coupling strength
    delta: float = 0.05   # Damping
    
    # State indices (modal coordinates)
    Q1 = 0  # First mode
    Q2 = 1  # Second mode
    
    def f(self, t, x, args):
        q1 = x[:, self.Q1]
        q2 = x[:, self.Q2]
        
        # Using averaged equations (slowly varying amplitudes)
        return jnp.array([
            -self.delta * q1 - self.omega1**2 * q1 - self.gamma * q1 * q2**2,
            -self.delta * q2 - self.omega2**2 * q2 - self.gamma * q2 * q1**2
        ])
    
    def g(self, t, x, args):
        return jnp.array([
            [1],
            [0]
        ])


class NonlinearBeam(eqx.Module):
    """Nonlinear beam equation (spatial mode reduction).
    
    Challenge: Geometric nonlinearity from large deflections creates
    hardening behavior. The system exhibits jump phenomena and can
    have multiple stable periodic orbits.
    """
    n_dims: int = 2
    n_controls: int = 1
    EI: float = 1.0     # Flexural rigidity
    mu: float = 1.0     # Mass per unit length
    c: float = 0.05     # Structural damping
    alpha: float = 0.8  # Geometric nonlinearity coefficient
    
    # State indices (first mode approximation)
    W = 0  # Deflection
    W_DOT = 1  # Deflection velocity
    
    def f(self, t, x, args):
        w = x[:, self.W]
        w_dot = x[:, self.W_DOT]
        
        # Nonlinear restoring force from geometric stiffening
        return jnp.array([
            w_dot,
            -(self.EI / self.mu) * w * (1 + self.alpha * w**2) - self.c * w_dot
        ])
    
    def g(self, t, x, args):
        return jnp.array([
            [0],
            [1/self.mu]
        ])


class PredatorPrey(eqx.Module):
    """Predator-prey dynamics (Lotka-Volterra with saturation).
    
    Challenge: The system has a non-trivial equilibrium that's a center
    in the linear approximation. The saturation terms create limit cycles
    that are difficult to stabilize without destroying the equilibrium.
    """
    n_dims: int = 2
    n_controls: int = 1
    r: float = 1.0      # Prey growth rate
    K: float = 10.0     # Prey carrying capacity
    a: float = 0.1      # Predation rate
    e: float = 0.075    # Conversion efficiency
    m: float = 0.05     # Predator mortality
    
    # State indices
    PREY = 0  # Prey population
    PRED = 1  # Predator population
    
    def f(self, t, x, args):
        prey = x[:, self.PREY]
        pred = x[:, self.PRED]
        
        # Logistic growth for prey, Type II functional response
        prey_growth = self.r * prey * (1 - prey / self.K)
        predation = self.a * prey * pred / (1 + self.a * prey)
        
        return jnp.array([
            prey_growth - predation,
            self.e * predation - self.m * pred
        ])
    
    def g(self, t, x, args):
        # Control affects prey population (e.g., harvesting/stocking)
        return jnp.array([
            [1],
            [0]
        ])