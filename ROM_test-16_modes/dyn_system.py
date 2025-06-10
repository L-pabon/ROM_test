import numpy as np
import torch
from abc import ABC, abstractmethod 
import jax
import jax.numpy as jnp
from typing import Union, Tuple, Callable
from functools import partial
Array = Union[jnp.ndarray]

import diffrax as dfx

class System(ABC):
    def __init__(self, dt: float, n_x: int, n_u: int, *, integrator: str = "euler"):
        self.dt = float(dt)
        self.n_x = int(n_x)
        self.n_u = int(n_u)

        integrator = integrator.lower()
        if integrator not in {"euler", "rk4", "dopri5"}:
            raise ValueError(f"Unsupported integrator '{integrator}'")
        self.integrator = integrator

    @abstractmethod
    def continuous_dynamics(self, x: Union[np.ndarray, jnp.ndarray], u: Union[np.ndarray, jnp.ndarray]) -> Union[np.ndarray, jnp.ndarray]:
        """
        Compute the continuous time dynamics dx/dt = f(x,u) at several state, action pairs
        
        Args:
            x: State vector of shape (N, n_x)
            u: Control input vector of shape (N, n_u)
            
        Returns:
            dx/dt: State derivative vector of shape (N, n_x)
        """
        pass

    # ------------------------------------------------------------------
    # Fixed‑step integrators (Euler / RK4)
    # ------------------------------------------------------------------
    def _euler_step(self, x: Array, u: Array) -> Array:
        return x + self.dt * self.continuous_dynamics(x, u)

    def _rk4_step(self, x: Array, u: Array) -> Array:
        k1 = self.continuous_dynamics(x, u)
        k2 = self.continuous_dynamics(x + 0.5 * self.dt * k1, u)
        k3 = self.continuous_dynamics(x + 0.5 * self.dt * k2, u)
        k4 = self.continuous_dynamics(x + self.dt * k3, u)
        return x + (self.dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    # ------------------------------------------------------------------
    # Adaptive Diffrax step (Dormand–Prince 5)
    # ------------------------------------------------------------------
    def _dopri5_single(self, x: Array, u: Array) -> Array:
        """One adaptive step over :math:`[0, dt]` using Diffrax.

        The solver is JIT‑friendly and fully differentiable.  Internally the
        step size is adjusted to keep the local error below Diffrax defaults
        (``rtol=atol=1e‑5``); you can tweak via ``solver=…`` or ``stepsize_controller``
        if needed.
        """

        def ode(t, y, args):
            # Autonomuous system ⇒ *t* unused.
            return self.continuous_dynamics(y, u)

        # solver = dfx.Dopri5()  # classical Fehlberg(4) / Dormand–Prince(5)
        solver = dfx.Kvaerno5()  # more robust, but slower
        saveat = dfx.SaveAt(t1=self.dt)  # only final state needed
        stepsize_controller = dfx.PIDController(rtol=1e-6, atol=1e-6)

        sol = dfx.diffeqsolve(
            ode,
            solver,
            t0=0.0,
            t1=self.dt,
            dt0=self.dt / 10.0,  # conservative first guess
            y0=x,
            saveat=saveat,
            stepsize_controller=stepsize_controller,
            max_steps=8_192,     # safety against endless loops during AD
        )
        return sol.ys[0]
    
    # ------------------------------------------------------------------
    # Public discrete map f_d(x_k, u_k)
    # ------------------------------------------------------------------
    def discrete_dynamics(self, x: Array, u: Array) -> Array:
        """Compute :math:`x_{k+1}` from *x_k*, *u_k* using the selected scheme.

        Works for both single states ``(n_x,)`` and batches ``(B, n_x)``.
        """

        # Choose kernel for a *single* sample --------------------------------
        if self.integrator == "euler":
            step_fn: Callable[[Array, Array], Array] = self._euler_step
        elif self.integrator == "rk4":
            step_fn = self._rk4_step
        else:  # "dopri5"
            step_fn = self._dopri5_single

        # Vectorise if necessary --------------------------------------------
        if x.ndim == 1:  # (n_x,)
            return step_fn(x, u)
        else:            # (B, n_x)
            return jax.vmap(step_fn, in_axes=(0, 0))(x, u)
        
    def multistep_dynamics(self, x: Union[np.ndarray, jnp.ndarray], controls: Union[np.ndarray, jnp.ndarray]) -> Union[np.ndarray,jnp.ndarray]:
        """Repeatedly apply the discrete dynamics to get an (N+1 x n_x) array of states."""
        N, n_u = controls.shape
        states = [x]
        for i in range(N):
            curr_state = states[-1][None,:]
            curr_control = controls[i][None,:]
            next_state = self.discrete_dynamics(curr_state, curr_control).squeeze()

            # Convert back to numpy array from tensor if the starting state indicates that user expects numpy output
            # Note: user should ensure x is tensor if multistep_dynamics is to be used in backpropagation/loss
            if isinstance(x, np.ndarray) and isinstance(next_state, torch.Tensor):
                next_state = next_state.detach().cpu().numpy()

            states.append(next_state)

        if isinstance(x, np.ndarray):
            states = np.array(states)
        elif isinstance(x, torch.Tensor):
            states = torch.stack(states).to(x.device, dtype=torch.float32)
        else:
            # jnp for gusto
            states = jnp.array(states)
            
        return states
    
    def dynamics_jac(self, x: Union[np.ndarray, jnp.ndarray], u: Union[np.ndarray, jnp.ndarray], continuous=False) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Get linearized discrete dynamics matrices A_list, B_list, d_list around several state, action pairs.
        Can be overridden for analytical derivatives.
        """
        if continuous:
            f = partial(self.continuous_dynamics)
        else:
            f = partial(self.discrete_dynamics)
        
        if len(x.shape) == 2:
            # Compute Jacobians using JAX automatic differentiation
            A, B = jax.vmap(jax.jacfwd(f, argnums=(0, 1)))(x, u)
            
            # Compute affine terms
            d = jax.vmap(f)(x, u) - jnp.einsum('ijk,ik->ij', A, x) - jnp.einsum('ijk,ik->ij', B, u)
        else:
            A, B = jax.jacfwd(f, argnums=(0, 1))(x, u)
            d = f(x, u) - A @ x - B @ u
        
        return A, B, d
    
    def update_dynamics_using_obs(self, curr_state, obs):
        """Potentially update the dynamics using current state and latest observation."""
        pass

class LTISys(System):
    def __init__(self, A, B, C, *sup_args):
        self.A = A
        self.B = B
        self.C = C
        super(LTISys, self).__init__(*sup_args)
    
    def continuous_dynamics(self, x, u):
        # A is n_x x n_x, x.T is n_x x N -> n_x x N
        return (self.A @ x.T + self.B @ u.T + self.C).T

class SimpleTerrainBicycle(System):
    """
    A single-patch “terrain” bicycle model that can be auto‐differentiated by JAX.
    Controls: u = (vc, delta)
    State: x = (x, y, psi, vxb, vyb, omega)
    Tire forces come from:
       Fx = Cm * vc - Co
       Fyr = Cy * alpha_r
       Fyf = Cy * alpha_f
    with alpha_r, alpha_f computed from slip geometry.
    """

    def __init__(self, m, Iz, a, b, Cy, Cm, Co, *sup_args):
        super().__init__(*sup_args)
        self.m = m
        self.Iz = Iz
        self.a = a
        self.b = b
        self.Cy = Cy
        self.Cm = Cm
        self.Co = Co

    def continuous_dynamics(self, x, u):
        """
        x = [px, py, psi, vxb, vyb, omega]
        u = [vc, delta]
        Returns dx/dt as a jnp.array([...]).
        """
        px, py, psi, vxb, vyb, omega = x.T
        vc, delta = u.T

        # Use JAX-friendly clip:
        vxb_safe = jnp.maximum(vxb, 1e-2)  # Prevent division by zero

        # Slip angles
        alpha_r = jnp.arctan2(vyb - self.b * omega, vxb_safe)
        alpha_f = jnp.arctan2(vyb + self.a * omega, vxb_safe) - delta

        # Tire forces
        Fx  = self.Cm * vc - self.Co
        Fyr = self.Cy * alpha_r
        Fyf = self.Cy * alpha_f

        # Kinematic relationships
        dx   = vxb * jnp.cos(psi) - vyb * jnp.sin(psi)
        dy   = vxb * jnp.sin(psi) + vyb * jnp.cos(psi)
        dpsi = omega

        # EOM: mass and moment of inertia
        dvxb  = (1.0 / self.m) * (Fx - Fyf * jnp.sin(delta) + self.m * vyb * omega)
        dvyb  = (1.0 / self.m) * (Fyr + Fyf * jnp.cos(delta) - self.m * vxb * omega)
        domega = (1.0 / self.Iz) * (Fyf * self.a * jnp.cos(delta) - Fyr * self.b)

        return jnp.array([dx, dy, dpsi, dvxb, dvyb, domega])

def test_linearization(system, x0, u0, step=1):
    # Taylor centering point is x0, u0
    # Amount to move from centering point is step
    num_states = len(x0)
    num_actions = len(u0)
    
    # Move slightly away from Taylor center point
    x = x0 + step * np.random.rand(num_states)
    u = u0 + step * np.random.rand(num_actions)
    
    # Compute nonlinear step
    next_state = system.discrete_dynamics(x[None,:], u[None,:]).squeeze()

    # Compare to linearized prediction about Taylor center
    A, B, C = system.dynamics_jac(x0[None,:], u0[None,:])
    A, B, C = A.squeeze(), B.squeeze(), C.squeeze()
    if not isinstance(A, np.ndarray):
        A, B, C = A.cpu().detach().numpy(), B.cpu().detach().numpy(), C.cpu().detach().numpy()

    lin_next_state = A @ x + B @ u + C

    return next_state, lin_next_state

if __name__ == '__main__':
    #### Initialize system ####
    dt = 0.05 # s
    n_x = 6
    n_u = 2
    rk4 = False

    # Fill in these values with something reasonable for testing
    m = 1.2 # kg
    l = 0.176 # m
    w = 0.183 # m
    a = l/2 # m
    b = l/2 # m
    Iz = m/12 * (l**2 + w**2) # kg * m^2
    Cy = -10 # N/rad
    Cm = 1 # kg/s
    Co = 0.1 # N

    system = SimpleTerrainBicycle(m, Iz, a, b, Cy, Cm, Co, dt, n_x, n_u, rk4)

    #### Test system ####
    x0 = np.random.rand(n_x)
    u0 = np.random.rand(n_u)
    next_state, lin_next_state = test_linearization(system, x0, u0, step=0.1)
    print('next_state', next_state, 'lin_next_state', lin_next_state)