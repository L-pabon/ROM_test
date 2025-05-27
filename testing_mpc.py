# ----------------------------------------------------------
# testing_mpc.py
# ----------------------------------------------------------
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import pickle

# Import the GuSTO config and MPC policy
from gusto import GuSTOConfig
from mpc_policy import MPCPolicy

# Import the generic System base class from dyn_system.py
#    Adjust import path to match your own project structure
from dyn_system import System

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
    ns = 8 # Number of structural modes
    M = jnp.eye(3 + 3 + ns)  # Mass matrix
    Kss = jnp.diag(jnp.array([
        1.8548E+04, 3.3252E+04, 1.4848E+05, 1.9298E+05,
        4.1335E+05, 5.2723E+05, 1.0249E+06, 1.3429E+06
    ]))
    # Load B matrices from file "B_matrices.pkl"
    with open("B_matrices.pkl", "rb") as f:
        B_matrices = pickle.load(f)
    Bp = jnp.asarray(B_matrices['Bp'])
    Bh = jnp.asarray(B_matrices['Bh'])
    Bt = jnp.asarray(B_matrices['Bt'])

    # 2) Create the system
    rom = ROMTest(M=M, Kss=Kss, Bp=Bp, Bh=Bh, Bt=Bt, n_s=ns,
                     dt=0.002, n_x=2*(3+3+ns), n_u=ns, rk4=True)

    # 3) Prepare a GuSTO config
    # ----- what the controller should look at (z = H x) -------------------
    H = jnp.zeros((4, rom.n_x))
    H = H.at[0, 6].set(1.0)   # q_s1
    H = H.at[1, 7].set(1.0)   # q_s2
    H = H.at[2, 6+ns].set(1.0)  # q̇_s1
    H = H.at[3, 7+ns].set(1.0)  # q̇_s2

    Qz  = jnp.diag(jnp.array([10.0, 10.0, 10.0, 10.0]))  # state cost
    Qzf = jnp.diag(jnp.array([10.0, 10.0, 1.0, 1.0]))  # final state cost
    R   = 1e-7 * jnp.eye(ns)    # control effort cost

    # characteristic scales (rough guesses)
    x_char = jnp.ones(rom.n_x)
    f_char = jnp.ones(rom.n_x)*1e5

    cfg = GuSTOConfig(
        Qz=Qz, Qzf=Qzf, R=R,
        x_char=x_char, f_char=f_char,
        N=60,
        H=H
    )

    # 4) Create the MPC policy
    mpc = MPCPolicy(model=rom, config=cfg)

    # 5) Set an initial state and define a simple reference path
    x0 = jnp.zeros(rom.n_x)                   # initial state    
    x0 = x0.at[6].set(0.1)                    # initial q_s[0]
    x0 = x0.at[7].set(0.02)                   # initial q_s[1]
    T  = 200                                  # 4 s sim

    # two sinusoidal references
    t_grid = jnp.arange(T + cfg.N + 1) * rom.dt
    z_ref  = jnp.stack([0*jnp.sin(2.0 * jnp.pi * 0.5 * t_grid),
                        0*jnp.sin(2.0 * jnp.pi * 0.5 * t_grid),
                        0*jnp.sin(2.0 * jnp.pi * 0.5 * t_grid),
                        0*jnp.sin(2.0 * jnp.pi * 0.5 * t_grid)
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

    # ── 1) modal amplitudes ──────────────────────────────────────────────
    ax1.plot(t_plot, states[:, 6], label=r'$q_{s,1}$ (actual)')
    ax1.plot(t_plot, z_ref[:T, 0], '--',  label=r'$q_{s,1}$ ref')
    ax1.plot(t_plot, states[:, 7], label=r'$q_{s,2}$ (actual)')
    ax1.plot(t_plot, z_ref[:T, 1], '--',  label=r'$q_{s,2}$ ref')
    ax1.set_ylabel('modal amplitude [m]')
    ax1.set_title('Structural-mode tracking with GuSTO MPC')
    ax1.grid(True);  ax1.legend()

    # ── 2) control forces ────────────────────────────────────────────────
    ax2.plot(t_plot, ctrls[:, 0], label=r'$F_{s,1}$')
    ax2.plot(t_plot, ctrls[:, 1], label=r'$F_{s,2}$')
    ax2.set_xlabel('time [s]')
    ax2.set_ylabel('modal force [N]')
    ax2.set_title('Structural-mode control forces')
    ax2.grid(True);  ax2.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_mpc_demo()