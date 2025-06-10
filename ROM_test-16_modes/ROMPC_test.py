# ----------------------------------------------------------
# ROMPC_test.py
# ----------------------------------------------------------
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import pickle
import random

# Import the GuSTO config and MPC policy
from gusto import GuSTOConfig
from mpc_policy import MPCPolicy

# Import the generic System base class from dyn_system.py
#    Adjust import path to match your own project structure
from dyn_system import System

np.set_printoptions(threshold=np.inf)

class ROMTest(System):
    def __init__(self, M, Kss, Bp, Bh, Bt, n_s,
                 dt, n_x, n_u, rk4=False):
        super().__init__(dt, n_x, n_u, rk4)
        self.M = M
        self.Muu = M[:3,:3]
        self.Mrr = M[3:6,3:6]
        self.Mss = M[6:,6:]
        self.Kss = Kss
        self.Bp = Bp
        self.Bh = Bh
        self.Bt = Bt
        self.ns = n_s
        
    def _build_coupling_matrices_single(self, q_s, qdot_r):
        Bq = jnp.einsum('bgij,b->gij', self.Bp, q_s)        # (n_s,3,3)
        B_qq = jnp.einsum('bij,b->ij', Bq, q_s)             # (3,3)
        B_hq = jnp.einsum('bij,b->ij', self.Bh, q_s)        # (3,3)
        B_tq = jnp.einsum('gbi,b->gi', self.Bt, q_s)        # (n_s,3)
        B_qo = jnp.einsum('gij,j->gi', Bq, qdot_r)          # (n_s,3)
        B_ho = jnp.einsum('gij,j->gi', self.Bh, qdot_r)     # (n_s,3)
        B_to = jnp.einsum('bgi,i->bg', self.Bt, qdot_r)     # (n_s,n_s)

        return B_qq, B_hq, B_tq, B_qo, B_ho, B_to

    def _continuous_single(self, x, u):
        """
        x = [q_u, q_r, q_s, \dot{q}_u, dot{q}_r, dot{q}_s]
        u = [F_s] (treat control vector u (length = n_s) as *modal force*)
        q_u - vector of length 3
        q_r - vector of length 3
        q_s - vector of length n_s
        Returns dx/dt as a jnp.array([...]).
        """
        # Extract the state variables
        q = x[:3 + 3 + self.ns]
        qdot = x[3 + 3 + self.ns:]
        _, _, q_s = q[:3], q[3:6], q[6:]
        qdot_u, qdot_r, qdot_s = qdot[:3], qdot[3:6], qdot[6:]

        # Compute the derivatives using the ROM equations
        qddot_u = np.zeros(3)

        B_qq, B_hq, B_tq, B_qo, B_ho, B_to = self.build_coupling_matrices(q_s, qdot_r)
        c = (self.Mrr + B_qq + 2.0 * B_hq) @ qdot_r + B_tq.T @ qdot_s
        A = jnp.block([
            [self.Mrr + B_qq + 2.0 * B_hq, B_tq.T],
            [B_tq, self.Mss]
        ])
        # A @ [ qddot_r; qddot_s ] = [ rhs_r; rhs_s ]
        rhs_r = jnp.cross(c, qdot_r) - 2.0 * (B_qo + B_ho).T @ qdot_s
        rhs_s = -self.Kss @ q_s + (B_qo + B_ho) @ qdot_r - 2.0 * B_to @ qdot_s + u
        b = jnp.concatenate([rhs_r, rhs_s], axis=0)
        sol = jnp.linalg.solve(A, b)
        qddot_r = sol[:3]
        qddot_s = sol[3:]
        
        return jnp.concatenate([qdot_u, qdot_r, qdot_s, qddot_u, qddot_r, qddot_s])
    
    def build_coupling_matrices(self, q_s, qdot_r):
        """
        Works for either
            q_s     … (n_s,)                and qdot_r … (3,)      OR
            q_s     … (B, n_s) (batched)    and qdot_r … (B, 3)

        Returns the same tuple as before; when batched, every item has the
        extra leading dimension B.
        """
        if q_s.ndim == 1:                       # single sample
            return self._build_coupling_matrices_single(q_s, qdot_r)

        # batched: vmap across the leading dimension
        return jax.vmap(self._build_coupling_matrices_single,
                        in_axes=(0, 0))(q_s, qdot_r)

    def continuous_dynamics(self, x, u):
        """
        Time-continuous ROM dynamics.
        Accepts either one state/control pair:
            x … (nx,)   u … (nu,)
        or a batch:
            x … (B, nx) u … (B, nu)

        Returns dx/dt with the matching leading dimension.
        """
        if x.ndim == 1:                         # single sample
            return self._continuous_single(x, u)

        # batched
        return jax.vmap(self._continuous_single,
                        in_axes=(0, 0))(x, u)

def run_mpc_demo():
    """
    Demonstration of using the ROM with GuSTO-based MPC.
    """
    # 1) Define the system parameters
    ns = 16 # Number of structural modes
    M = jnp.eye(3 + 3 + ns)  # Mass matrix
    Kss = jnp.diag(jnp.array([
        7.007485000038743E+02, 1.008845031465958E+03, 1.897528578716773E+03, 2.034552994880513E+03, 2.134497533861053E+03, 2.284194395105604E+03, 2.287774786383397E+03, 3.714994174716178E+03,
        4.048460248593216E+03, 4.417705807469577E+03, 4.770688673914491E+03, 5.355637059558911E+03, 6.021748583432235E+03, 6.135043946323395E+03, 6.349264650911677E+03, 6.607331764352199E+03,
    ]))
    # Load B matrices from file "B_matrices.pkl"
    with open("modalflf.pkl", "rb") as f:
        B_matrices = pickle.load(f)
    Bp = jnp.asarray(B_matrices['Bp'])
    Bh = jnp.asarray(B_matrices['Bh'])
    Bt = jnp.asarray(B_matrices['Bt'])
    # Msu = jnp.asarray(B_matrices['Msu'])

    # # Update Mass Matrix with Msu & Mus matrices
    # M = M.at[6:,:3].set(Msu[0])
    # M = M.at[:3,6:].set(jnp.transpose(Msu[0]))

    # 2) Create the system
    rom = ROMTest(M=M, Kss=Kss, Bp=Bp, Bh=Bh, Bt=Bt, n_s=ns,
                     dt=0.001, n_x=2*(3+3+ns), n_u=ns, rk4=True)

    # 3) Prepare a GuSTO config
    # ----- what the controller should look at (z = H x) -------------------
    H = jnp.zeros((2*ns, rom.n_x)) # should be 2*ns
    for i in range(ns):
        H = H.at[i,6+i].set(1.0)

    for i in range(ns):
        H = H.at[i+ns,6+ns+i].set(1.0)

    Qz  = jnp.diag(jnp.array([10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0,
                              10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0]))  # state cost --> needs extra entries for each mode added
    Qzf = jnp.diag(jnp.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                              1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]))  # final state cost
    R   = 1e-4 * jnp.eye(ns)    # control effort cost

    # characteristic scales (rough guesses)
    x_char = jnp.ones(rom.n_x)
    f_char = jnp.ones(rom.n_x)*1e5

    cfg = GuSTOConfig(
        Qz=Qz, Qzf=Qzf, R=R,
        x_char=x_char, f_char=f_char,
        N=15,
        H=H
    )

    # 4) Create the MPC policy
    mpc = MPCPolicy(model=rom, config=cfg)

    # 5) Set an initial state and define a simple reference path
    x0 = jnp.zeros(rom.n_x)                                                 # initial state    
    for i in range(ns):
        x0 = x0.at[6+i].set(random.randint(10,100)/1000)                         # initial q_s[i]

    T  = 250                                                                 # 4 s sim

    # two sinusoidal references
    t_grid = jnp.arange(T + cfg.N + 1) * rom.dt
    z_ref  = jnp.stack([0* t_grid, 0* t_grid, 0* t_grid, 0* t_grid, 0* t_grid, 0* t_grid, 0* t_grid, 0* t_grid, 0* t_grid, 0* t_grid, 0* t_grid,
                        0* t_grid, 0* t_grid, 0* t_grid, 0* t_grid, 0* t_grid, 0* t_grid, 0* t_grid, 0* t_grid, 0* t_grid, 0* t_grid, 0* t_grid,
                        0* t_grid, 0* t_grid, 0* t_grid, 0* t_grid, 0* t_grid, 0* t_grid, 0* t_grid, 0* t_grid, 0* t_grid, 0* t_grid,
    ], axis=1)                                       # (T+N+1, 2)

    # 6) Reset the MPC with the current state and reference
    print("Resetting MPC with initial state and reference trajectory...")
    print("Initial state:", x0.shape)
    mpc.reset(x0, obs=x0, z_ref=z_ref, start_with_solve=True)

    # 7) Simulate the closed-loop system
    states, ctrls = [], []
    x = x0
    for k in range(T):
        u, _ = mpc.compute_control(x)
        x    = rom.discrete_dynamics(x, u)
        states.append(x)
        ctrls.append(u)

    states = jnp.stack(states)      # (T, nx)
    ctrls  = jnp.stack(ctrls)       # (T, ns)

    # 8) Visualization: compare actual position vs. reference + input forces
    t_plot = jnp.arange(T) * rom.dt          # time vector

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(7, 7), sharex=True,  # share the same x-axis (time)
        gridspec_kw={"hspace": 0.35})       # vertical spacing
    
    fig, (ax3, ax4) = plt.subplots(
        2, 1, figsize=(7, 7), sharex=True,  # share the same x-axis (time)
        gridspec_kw={"hspace": 0.35})       # vertical spacing
    
    fig, (ax5, ax6) = plt.subplots(
        2, 1, figsize=(7, 7), sharex=True,  # share the same x-axis (time)
        gridspec_kw={"hspace": 0.35})       # vertical spacing
    
    fig, (ax7, ax8) = plt.subplots(
        2, 1, figsize=(7, 7), sharex=True,  # share the same x-axis (time)
        gridspec_kw={"hspace": 0.35})       # vertical spacing

    # ── 1) modal amplitudes [1-4] ──────────────────────────────────────────────
    ax1.plot(t_plot, states[:, 6], label=r'$q_{s,1}$ (actual)')
    ax1.plot(t_plot, z_ref[:T, 0], '--',  label=r'$q_{s,1}$ ref')
    ax1.plot(t_plot, states[:, 7], label=r'$q_{s,2}$ (actual)')
    ax1.plot(t_plot, z_ref[:T, 1], '--',  label=r'$q_{s,2}$ ref')
    ax1.plot(t_plot, states[:, 8], label=r'$q_{s,3}$ (actual)')
    ax1.plot(t_plot, z_ref[:T, 2], '--',  label=r'$q_{s,3}$ ref')
    ax1.plot(t_plot, states[:, 9], label=r'$q_{s,4}$ (actual)')
    ax1.plot(t_plot, z_ref[:T, 3], '--',  label=r'$q_{s,4}$ ref')
    ax1.set_ylabel('modal amplitude [m]')
    ax1.set_title('Structural-mode tracking with GuSTO MPC')
    ax1.grid(True);  ax1.legend(loc='upper right', fontsize=8)

    # ── 2) control forces [1-4] ────────────────────────────────────────────────
    ax2.plot(t_plot, ctrls[:, 0], label=r'$F_{s,1}$')
    ax2.plot(t_plot, ctrls[:, 1], label=r'$F_{s,2}$')
    ax2.plot(t_plot, ctrls[:, 2], label=r'$F_{s,3}$')
    ax2.plot(t_plot, ctrls[:, 3], label=r'$F_{s,4}$')
    ax2.set_xlabel('time [s]')
    ax2.set_ylabel('modal force [N]')
    ax2.set_title('Structural-mode control forces')
    ax2.grid(True);  ax2.legend()

    # ── 3) modal amplitudes [5-8] ──────────────────────────────────────────────
    ax3.plot(t_plot, states[:, 10], label=r'$q_{s,5}$ (actual)')
    ax3.plot(t_plot, z_ref[:T, 4], '--',  label=r'$q_{s,5}$ ref')
    ax3.plot(t_plot, states[:, 11], label=r'$q_{s,6}$ (actual)')
    ax3.plot(t_plot, z_ref[:T, 5], '--',  label=r'$q_{s,6}$ ref')
    ax3.plot(t_plot, states[:, 12], label=r'$q_{s,7}$ (actual)')
    ax3.plot(t_plot, z_ref[:T, 6], '--',  label=r'$q_{s,7}$ ref')
    ax3.plot(t_plot, states[:, 13], label=r'$q_{s,8}$ (actual)')
    ax3.plot(t_plot, z_ref[:T, 7], '--',  label=r'$q_{s,8}$ ref')
    ax3.set_ylabel('modal amplitude [m]')
    ax3.set_title('Structural-mode tracking with GuSTO MPC')
    ax3.grid(True);  ax3.legend(loc='upper right', fontsize=8)

    # ── 4) control forces [5-8] ────────────────────────────────────────────────
    ax4.plot(t_plot, ctrls[:, 4], label=r'$F_{s,5}$')
    ax4.plot(t_plot, ctrls[:, 5], label=r'$F_{s,6}$')
    ax4.plot(t_plot, ctrls[:, 6], label=r'$F_{s,7}$')
    ax4.plot(t_plot, ctrls[:, 7], label=r'$F_{s,8}$')
    ax4.set_xlabel('time [s]')
    ax4.set_ylabel('modal force [N]')
    ax4.set_title('Structural-mode control forces')
    ax4.grid(True);  ax4.legend()

    # ── 5) modal amplitudes [9-12] ──────────────────────────────────────────────
    ax5.plot(t_plot, states[:, 14], label=r'$q_{s,9}$ (actual)')
    ax5.plot(t_plot, z_ref[:T, 8], '--',  label=r'$q_{s,9}$ ref')
    ax5.plot(t_plot, states[:, 15], label=r'$q_{s,10}$ (actual)')
    ax5.plot(t_plot, z_ref[:T, 9], '--',  label=r'$q_{s,10}$ ref')
    ax5.plot(t_plot, states[:, 16], label=r'$q_{s,11}$ (actual)')
    ax5.plot(t_plot, z_ref[:T, 10], '--',  label=r'$q_{s,11}$ ref')
    ax5.plot(t_plot, states[:, 17], label=r'$q_{s,12}$ (actual)')
    ax5.plot(t_plot, z_ref[:T, 11], '--',  label=r'$q_{s,12}$ ref')
    ax5.set_ylabel('modal amplitude [m]')
    ax5.set_title('Structural-mode tracking with GuSTO MPC')
    ax5.grid(True);  ax5.legend(loc='upper right', fontsize=8)

    # ── 6) control forces [9-12] ────────────────────────────────────────────────
    ax6.plot(t_plot, ctrls[:, 8], label=r'$F_{s,9}$')
    ax6.plot(t_plot, ctrls[:, 9], label=r'$F_{s,10}$')
    ax6.plot(t_plot, ctrls[:, 10], label=r'$F_{s,11}$')
    ax6.plot(t_plot, ctrls[:, 11], label=r'$F_{s,12}$')
    ax6.set_xlabel('time [s]')
    ax6.set_ylabel('modal force [N]')
    ax6.set_title('Structural-mode control forces')
    ax6.grid(True);  ax6.legend()

    # ── 7) modal amplitudes [13-16] ──────────────────────────────────────────────
    ax7.plot(t_plot, states[:, 18], label=r'$q_{s,13}$ (actual)')
    ax7.plot(t_plot, z_ref[:T, 12], '--',  label=r'$q_{s,13}$ ref')
    ax7.plot(t_plot, states[:, 19], label=r'$q_{s,14}$ (actual)')
    ax7.plot(t_plot, z_ref[:T, 13], '--',  label=r'$q_{s,14}$ ref')
    ax7.plot(t_plot, states[:, 20], label=r'$q_{s,15}$ (actual)')
    ax7.plot(t_plot, z_ref[:T, 14], '--',  label=r'$q_{s,15}$ ref')
    ax7.plot(t_plot, states[:, 21], label=r'$q_{s,16}$ (actual)')
    ax7.plot(t_plot, z_ref[:T, 15], '--',  label=r'$q_{s,16}$ ref')
    ax7.set_ylabel('modal amplitude [m]')
    ax7.set_title('Structural-mode tracking with GuSTO MPC')
    ax7.grid(True);  ax7.legend(loc='upper right', fontsize=8)

    # ── 8) control forces [13-16] ────────────────────────────────────────────────
    ax8.plot(t_plot, ctrls[:, 12], label=r'$F_{s,13}$')
    ax8.plot(t_plot, ctrls[:, 13], label=r'$F_{s,14}$')
    ax8.plot(t_plot, ctrls[:, 14], label=r'$F_{s,15}$')
    ax8.plot(t_plot, ctrls[:, 15], label=r'$F_{s,16}$')
    ax8.set_xlabel('time [s]')
    ax8.set_ylabel('modal force [N]')
    ax8.set_title('Structural-mode control forces')
    ax8.grid(True);  ax8.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_mpc_demo()