import jax.numpy as jnp
from typing import Optional, Tuple, Any
import time
from gusto import GuSTO, GuSTOConfig
import numpy as np


class MPCPolicy():
    """
    Model Predictive Control policy implementation using GuSTO optimizer.
    
    This policy uses a GuSTO-based MPC controller to compute optimal control actions
    based on the current state observation and a reference trajectory.
    """
    
    def __init__(self,
                 model,
                 config: GuSTOConfig,
                 U: Optional[Any] = None,
                 dU: Optional[Any] = None,
                 init_guess_type='shift'):
        """
        Initialize the MPC policy.
        
        Args:
            model: Model object representing the dynamical system
            config: GuSTO configuration parameters
            z_ref: Reference trajectory for the MPC controller
            U: Control constraints Polyhedron object
            dU: Control rate constraints Polyhedron object
            smoothing_func: Optional function to smooth observations
        """
        super().__init__()
        
        self.model = model
        self.dt = model.dt
        self.config = config
        self.U = U
        self.dU = dU

        # Extract dimensions
        self.n_x = model.n_x     # state dimension
        self.n_u = model.n_u     # control dimension
        self.n_z = self.config.H.shape[0] # performance dimension
        
        # MPC parameters
        self.N = config.N

        # Initialize warm start variables
        self.x_prev = jnp.zeros(self.n_x)
        self.u_prev = jnp.zeros(self.n_u)

        # For the slew rate cost
        self.last_applied_u = None

        # What type of initial guess to use (shift, dyn_feasible, zeros)
        self.init_guess_type = init_guess_type
        
    def reset(self, x0: jnp.ndarray, obs, z_ref: jnp.ndarray, start_with_solve=True):
        """
        Reset the policy with a new goal state.
        
        Args:
            x0: Initial state of the system
            obs: Initial observation
            z_ref: Reference trajectory for the MPC controller
        """
        self.z_ref = z_ref

        # Initialize GuSTO with zeros as initial guess
        u_init = jnp.zeros((self.N, self.n_u))
        x_init = self.model.multistep_dynamics(x0, u_init)
        z_ref_win = self.z_ref[0:self.N+1]
        
        self.gusto = GuSTO(
            self.model, 
            self.config,
            x0,
            u_init,
            x_init,
            z=z_ref_win,
            zf=z_ref_win[-1],
            U=self.U,
            dU=self.dU,
            start_with_solve=start_with_solve,
            solver='CLARABEL'
        )

        # Get initial solution x_opt (N+1 x n), u_opt (N x m)
        if start_with_solve:
            x_opt, u_opt, _, _ = self.gusto.get_solution()
            self.x_prev = x_opt
            self.u_prev = u_opt
        else:
            self.x_prev = jnp.zeros(self.n_x)
            self.u_prev = jnp.zeros(self.n_u)
    
        # Reset where index into reference
        self.t_idx = 0

    def compute_control(self, state: jnp.ndarray) -> Tuple[jnp.ndarray, dict]:
        """
        Compute the control action for the current observation.
        
        Args:
            obs: Current observation (could be full or partial state)
            
        Returns:
            u: Optimal control action
            info: Dictionary containing additional information
        """
        t_start = time.time()
        
        # Get reference trajectory for current MPC window
        max_ind = min(self.t_idx + self.N + 1, len(self.z_ref))
        z_ref_win = self.z_ref[self.t_idx:max_ind]
        # Pad if not length N+1
        if len(z_ref_win) < self.N + 1:
            k = self.N + 1 - len(z_ref_win)
            last_z = jnp.tile(self.z_ref[-1], (k, 1))
            z_ref_win = jnp.concatenate([z_ref_win, last_z])

        # Initialize next MPC problem
        if self.init_guess_type == 'shift':
            # We shift x_prev by one step and then re-insert 'state' as x_init[0].
            # This helps the solver solve from the correct initial state.
            x_init = jnp.concatenate([self.x_prev[1:], 
                                    self.model.discrete_dynamics(self.x_prev[-1], self.u_prev[-1])[None, :]], axis=0)
            x_init = x_init.at[0].set(state)  # Force the first predicted state to match the real current state

            u_init = jnp.concatenate([self.u_prev[1:], self.u_prev[-1:]], axis=0)

        elif self.init_guess_type == 'dyn_feasible':
            u_init = jnp.concatenate([self.u_prev[1:], self.u_prev[-1:]], axis=0)
            x_init = self.model.multistep_dynamics(state, u_init)
        
        else:
            u_init = jnp.zeros((self.N, self.n_u))
            x_init = self.model.multistep_dynamics(state, u_init)
        
        # Update LOCP parameter with the previously applied control
        if self.last_applied_u is not None:
            self.gusto.locp.u0_prev.value = np.asarray(self.last_applied_u)
        
        # Solve MPC problem
        self.gusto.solve(state, u_init, x_init, z=z_ref_win, zf=z_ref_win[-1])
        x_opt, u_opt, z_opt, solve_time = self.gusto.get_solution()

        # Store solution for warm start
        self.x_prev = x_opt
        self.u_prev = u_opt
        
        # Increment time index
        self.t_idx += 1
        
        # Prepare info dictionary
        info = {
            'solve_time': solve_time,
            'total_time': time.time() - t_start,
            'predicted_trajectory': z_opt,
            'control_trajectory': u_opt
        }
        
        self.last_applied_u = u_opt[0]

        return u_opt[0], info