"""
GuSTO (Guaranteed Sequential Trajectory Optimization) implementation, adopted from original code.
"""

import time
import jax
import jax.numpy as jnp
from functools import partial
from locp import LOCP
from dataclasses import dataclass, asdict
import numpy as np

@dataclass
class GuSTOConfig:
    """
    GuSTOConfig class for storing GuSTO parameters.
    """
    Qz: jnp.ndarray                     # positive semi-definite performance variable weighting matrix
    Qzf: jnp.ndarray                    # positive semi-definite terminal performance variable weighting matrix
    R: jnp.ndarray                      # positive definite control weighting matrix
    x_char: jnp.ndarray                 # characteristic quantities for x, for scaling
    f_char: jnp.ndarray                 # characteristic quantities for f, for scaling
    N: int = 8                          # integer optimization horizon
    epsilon: float = 0.01               # constraint violation threshold
    max_gusto_iters: int = 500          # maximum number of GuSTO iterations
    delta0: float = 1e4                 # trust region
    omega0: float = 1                   # slack variable weighting
    rho: float = 0.1                    # model compute_accuracy
    beta_fail: float = 0.5              # trust region update TODO: specify
    beta_succ: float = 2                # trust region update TODO: specify
    gamma_fail: float = 5               # cost function penalty term TODO: specify
    omega_max: float = 1e10             # cost function penalty term TODO: specify
    convg_thresh: float = 5.00          # convergence threshold
    verbose: int = 0                    # verbosity level (0, 1, 2)
    warm_start: bool = True             # warm start the solver
    H: jnp.ndarray = None               # performance mapping matrix
    R_du: jnp.ndarray = None  # control rate weighting matrix


class GuSTO:
    """
    GuSTO class for solving trajectory optimization problems via SQP.

    :model: TemplateModel object describing dynamics
    :config: GuSTOConfig object with parameters for GuSTO (see above)
    :x0: initial condition (n_x,)
    :u_init: control initial guess (N, n_u)
    :x_init: state initial guess (N+1, n_x)
    :z: (optional) desired tracking trajectory for objective function (N+1, n_z)
    :u: (optional) desired control for objective function (N, n_u)
    :zf: (optional) terminal target state (n_z,), defaults to 0 if Qzf provided
    :U: (optional) control constraint (Polyhedron object)
    :X: (optional) state constraint (Polyhedron object)
    :Xf: (optional) terminalstate constraint (Polyhedron object)
    :dU: (optional) u_k - u_{k-1} slew rate constraint (Polyhedron object)
    :kwargs: Keyword arguments for the solver and what solver to use, e.g.
    (https://osqp.org/docs/interfaces/solver_settings.html)
    """
    def __init__(self, model, config, x0, u_init, x_init,
                z=None, u=None, zf=None, U=None, X=None, Xf=None, dU=None,
                start_with_solve=True, **kwargs):
        self.model = model
        self.dt = self.model.dt

        # Extract configuration parameters
        self._extract_config(config)

        # Problem dimensions
        self.n_x = x0.shape[0]
        print(self.n_x)
        self.n_u = self.R.shape[0]
        self.n_z = self.Qz.shape[0]

        # Constraints - State, control, final state
        self.U = U
        self.X = X
        self.Xf = Xf
        self.dU = dU

        # Characteristic quantities
        self.x_scale = 1. / jnp.abs(self.x_char)
        self.f_scale = 1. / jnp.abs(self.f_char)

        # Problem parameters
        self.x_k = None  # previous state
        self.u_k = None  # previous input
        self.locp_solve_time = None  # time spent in LOCP solve

        # LOCP problem
        if self.verbose == 2:
            locp_verbose = True
        else:
            locp_verbose = False
        
        # Assert that there is a performance mapping matrix
        assert self.H is not None, 'Performance mapping matrix H must be provided'
        self.nonlinear_perf_mapping = False

        # Initialize LOCP
        self.locp = LOCP(self.N, self.H, self.Qz, self.R, Qzf=self.Qzf,
                         U=self.U, X=self.X, Xf=self.Xf, dU=self.dU,
                         verbose=locp_verbose, warm_start=self.warm_start, x_char=self.x_char,
                         nonlinear_perf_mapping=self.nonlinear_perf_mapping, R_du=self.R_du, **kwargs)

        # First SCP solve
        if start_with_solve:
            print('First solve may take a while due to factorization and caching.')
            self.solve(x0, u_init, x_init, z, zf, u)
    
    def performance_mapping(self, x):
        """
        Default performance mapping is linear.
        """
        return jnp.matmul(self.H, x.T).T

    def solve(self, x0, u_init, x_init, z=None, zf=None, u=None):
        """
        Solve the GuSTO problem with the given initial conditions and desired trajectories.

        :x0: initial condition jnp.array
        :u_init: control initial guess (N, n_u)
        :x_init: state initial guess (N+1, n_x)
        :z: (optional) desired tracking trajectory for objective function (N+1, n_z)
        :zf: (optional) desired terminal state for objective function (n_z,)
        :u: (optional) desired control for objective function (N, n_z)
        """

        # Timing information to be stored
        t0 = time.time()
        t_locp = 0.0

        itr = 0
        self.x_k = x_init
        self.u_k = u_init

        # Grab Jacobians for first solve
        A_d, B_d, d_d = self._get_dynamics_linearizations(self.x_k[:-1], self.u_k)

        if self.nonlinear_perf_mapping:
            H_d, c_d = self._get_perf_mapping_linearizations(self.x_k)
        else:
            H_d, c_d = None, None

        t_jac = time.time()
        if self.verbose >= 2:
            print('DEBUG: Jacobians computed in {:.4f} seconds'.format(t_jac - t0))

        new_solution = True
        Jstar_prev = jnp.inf
        delta_prev = jnp.inf
        omega_prev = jnp.inf

        converged = False

        delta = self.delta0
        omega = self.omega0

        if self.verbose >= 1:
            print('|   J   | TR_viol |  rho_k  |  X_viol |   x-x_k |  delta  |  omega |')
            print('--------------------------------------------------------------------')

        while self._is_valid_iteration(itr) and not converged and omega <= self.omega_max:
            rho_k = -1
            max_violation = -1
            dsol = -1
            delta_cur = delta  # just for printing
            omega_cur = omega  # just for printing

            # -----------------------------------------------------------
            # put this one-off probe right before locp.update(...)
            # -----------------------------------------------------------
            for j, A in enumerate(A_d):
                if (~jnp.isfinite(A)).any():
                    bad_step = j          # horizon index where things blow up
                    break
            else:
                bad_step = None

            if bad_step is not None:
                # dump the state/control that produced the invalid Jacobian
                x_bad = self.x_k[bad_step]
                if u is not None:
                    # batch vs. single
                    u_bad = u[bad_step] if getattr(u, "ndim", 1) == 2 else u
                else:
                    u_bad = None
                print(f"⚠️  bad Jacobian at step {bad_step}")
                print("state =", x_bad)
                print("input =", u_bad)


                # Optionally enter the debugger
                import pdb; pdb.set_trace()
                
            # Update the LOCP with new parameters and solve
            if new_solution:
                self.locp.update(A_d, B_d, d_d, x0, self.x_k, delta, omega, z=z, zf=zf, u=u, Hd=H_d, cd=c_d)
                new_solution = False
            else:
                self.locp.update(A_d, B_d, d_d, x0, self.x_k, delta, omega, z=z, zf=zf, u=u, Hd=H_d, cd=c_d, full=False)

            if self.verbose >= 2:
                print('DEBUG: Routines pre-solve computed in {:.4f} seconds'.format(time.time() - t0))

            # Solve the LOCP
            Jstar, success, stats = self.locp.solve()

            if not success:
                print('Iteration {} of problem cannot be solved, see solver status for more information'.format(itr))
                self.xopt = jnp.copy(self.x_k)
                self.uopt = jnp.copy(self.u_k)
                if self.nonlinear_perf_mapping:
                    self.zopt = self.performance_mapping(self.xopt.T).T
                else:
                    self.zopt = jnp.transpose(self.H @ self.xopt.T)
                return

            t_locp += stats.solve_time
            x_next, u_next, _ = self.locp.get_solution()

            # Check if trust region is satisfied
            e_tr, tr_satisfied = self._is_in_trust_region(self.x_k, x_next, delta)

            if tr_satisfied:
                rho_k = self._compute_accuracy(self.x_k, self.u_k, x_next, u_next, Jstar)

                # Nudge Gusto out of first iteration since it gets stuck
                if rho_k > self.rho and itr != 1:
                    delta = self.beta_fail * delta
                else:
                    """
                    First modification to GuSTO: if delta and omega are constant for two solves in a row,
                    yet the reported cost of the optimizer increases, decrease delta
                    """
                    if delta_prev == delta and omega_prev == omega and Jstar_prev <= Jstar:
                        delta = self.beta_fail * delta
                    delta_prev = delta
                    Jstar_prev = Jstar
                    omega_prev = omega

                    """
                    Second modification to GuSTO: remove delta increases for good model accuracy
                    """
                    # if rho_k < self.rho0:
                    #     delta = np.minimum(self.beta_succ * delta, self.delta0)
                    # else:
                    #     delta = delta

                    # Computes g2
                    max_violation, X_satisfied = self._state_constraints_violated(x_next)

                    """
                    Third modification to GuSTO: remove decreases of omega for satisifed X (creates oscillations)
                    """
                    # if X_satisfied:
                    #     omega = self.omega0
                    # else:
                    #     omega = self.gamma_fail * omega

                    if not X_satisfied:
                        omega = self.gamma_fail * omega

                    # Check for convergence
                    dsol, converged = self._is_converged(self.x_k, x_next, u_next)

                    # Optional: Enforce state constraints are satisfied upon convergence
                    if not X_satisfied:
                        converged = False

                    # Record that a new solution as been found
                    new_solution = True

            else:
                omega = self.gamma_fail * omega

            if self.verbose >= 2:
                print('DEBUG: Trust region + LOCP computed in {:.4f} seconds'.format(time.time() - t0))

            itr += 1

            if self.verbose >= 1:
                if rho_k < 0.0:
                    print('{:.2e}, {:.2e}, {}, {}, {}, {:.2e}, {:.2e}, {}'.format(
                        Jstar, e_tr, '-' * 8, '-' * 8, '-' * 8, delta_cur, omega_cur, itr))
                elif max_violation < 0.0:
                    print('{:.2e}, {:.2e}, {:.2e}, {}, {}, {:.2e}, {:.2e}, {}'.format(
                        Jstar, e_tr, rho_k, '-' * 8, '-' * 8, delta_cur, omega_cur, itr))
                else:
                    print('{:.2e}, {:.2e}, {:.2e}, {:.2e}, {:.2e}, {:.2e}, {:.2e}, {}'.format(
                        Jstar, e_tr, rho_k, max_violation, dsol, delta_cur, omega_cur, itr))

            # If valid solution, update and recompute dynamics
            if new_solution:
                self.x_k = x_next.copy()
                self.u_k = u_next.copy()
                if self.max_gusto_iters >= 1:
                    A_d, B_d, d_d = self._get_dynamics_linearizations(self.x_k[:-1], self.u_k)
                    
                    if self.nonlinear_perf_mapping:
                        H_d, c_d = self._get_perf_mapping_linearizations(self.x_k)
                    else:
                        H_d, c_d = None, None

        t_gusto = time.time() - t0
        if omega > self.omega_max:
            print('omega > omega_max, solution did not converge')
        if not self._is_valid_iteration(itr-1):
            print('Max iterations, solution did not converge')
        else:
            print('Solved in {} iterations/{:.3f} seconds, with {:.3f} s from LOCP solve'.format(itr, t_gusto, t_locp))

        # Save optimal solution
        self.xopt = jnp.copy(self.x_k)
        self.uopt = jnp.copy(self.u_k)
        if self.nonlinear_perf_mapping:
            self.zopt = self.performance_mapping(self.xopt.T).T
        else:
            self.zopt = jnp.transpose(self.H @ self.xopt.T)
        self.locp_solve_time = t_locp

    def get_solution(self):
        return self.xopt, self.uopt, self.zopt, self.locp_solve_time

    def _extract_config(self, config):
        """
        Extract configuration parameters from GuSTOConfig object.
        """
        for key, value in asdict(config).items():
            setattr(self, key, value)

    @partial(jax.jit, static_argnums=(0,))
    def _is_converged(self, x_k, x, u):
        """
        Sequential problem has converged when current and previous state input pairs are close.
        """
        dx = (1. / self.n_x) * jnp.sum(jnp.linalg.norm(jnp.multiply(self.x_scale, x - x_k), axis=1))
        dsol = (1. / self.N) * dx
        converged = jnp.where(dsol <= self.convg_thresh, True, False)
        return dsol, converged

    def _is_valid_iteration(self, itr):
        """
        Is the current iteration within the limits.
        """
        return jnp.less_equal(itr, self.max_gusto_iters)

    @partial(jax.jit, static_argnums=(0,))
    def _is_in_trust_region(self, x_k, x, delta):
        """
        Check if the new state is within the trust region of the previous state.
        """
        max_diff = jnp.max(jnp.linalg.norm(jnp.multiply(self.x_scale, x - x_k), ord=jnp.inf, axis=1))
        def outside_region(_):
            return max_diff, False
        def inside_region(_):
            return 0.0, True
        return jax.lax.cond(max_diff - delta > self.epsilon, outside_region, inside_region, operand=None)

    @partial(jax.jit, static_argnums=(0,))
    def _state_constraints_violated(self, x):
        """
        For GuSTO, state constraints get enforced as penalties, not as strict constraints. Computes whether the state
        constraints are within a user-chosen tolerance epsilon.
        """
        def compute_violation(x_row):
            return self.X.get_constraint_violation(x_row)

        if self.X is not None:
            # Vectorize the constraint violation computation
            violations = jax.vmap(compute_violation)(x)
            max_violation = jnp.max(violations)
        else:
            max_violation = 0.0

        def outside_threshold(_):
            return max_violation, False
        def inside_threshold(_):
            return max_violation, True

        return jax.lax.cond(max_violation > self.epsilon, outside_threshold, inside_threshold, operand=None)

    def _compute_accuracy(self, x_k, u_k, x, u, J):
        Ak_list, Bk_list, _ = self.model.dynamics_jac(x_k[:-1], u_k, continuous=True)
        fk_list = self.model.continuous_dynamics(x_k[:-1], u_k)
        f_list = self.model.continuous_dynamics(x[:-1], u)
        f_approx = fk_list + jnp.einsum("nij,nj->ni", Ak_list, (x - x_k)[:-1]) + jnp.einsum("nij,nj->ni", Bk_list, (u - u_k))
        error = self.dt * jnp.linalg.norm(self.f_scale * (f_list - f_approx), ord=2, axis=1).sum()
        approx = self.dt * jnp.linalg.norm(self.f_scale * f_approx, ord=2, axis=1).sum()
        
        rho_k = error / (J + approx)
        return rho_k

    @partial(jax.vmap, in_axes=(None, 0, 0))
    def _perform_dynamics_linearization(self, x, u):
        """
        Obtain the affine dynamics of each point along trajectory in a list.
        """
        f = partial(self.model.discrete_dynamics)
        A, B = jax.jacfwd(f, argnums=(0, 1))(x, u)
        d = f(x, u) - A @ x - B @ u
        return A, B, d

    @partial(jax.jit, static_argnums=(0,))
    @partial(jax.vmap, in_axes=(None, 0))
    def _perform_perf_mapping_linearization(self, x):
        """
        Obtain the affine performance mappings at each point along trajectory in a list.
        """
        g = self.performance_mapping
        H = jax.jacfwd(g)(x) 
        c = g(x) - H @ x
        return H, c

    def _get_dynamics_linearizations(self, x, u):
        """
        Wrapper method that calls self.model.get_dynamics_linearizations if it exists,
        otherwise it calls the local method _perform_dynamics_linearization.
        """
        if hasattr(self.model, 'dynamics_jac') and callable(getattr(self.model, 'dynamics_jac')):
            return self.model.dynamics_jac(x, u)
        else:
            return self._perform_dynamics_linearization(x, u)
        
    def _get_perf_mapping_linearizations(self, x):
        """
        Wrapper method that calls self.model.get_perf_mapping_linearizations if it exists,
        otherwise it calls the local method _perform_perf_mapping_linearization.
        """
        return self._perform_perf_mapping_linearization(x)
