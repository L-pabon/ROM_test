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
                 dt, n_x, n_u, integrator='rk4'):
        super().__init__(dt, n_x, n_u, integrator='rk4')
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

import matplotlib.pyplot as plt


def plot_modal_tracking(states, z_ref, ctrls, dt, modal_start_idx: int = 6):
    """Compact visualization of 16 structural modes + their control forces.

    Parameters
    ----------
    states : array-like, shape (T, nx)
        Closed‑loop state trajectory.
    z_ref  : array-like, shape (T, 16)
        Desired reference for each structural mode.
    ctrls  : array-like, shape (T, 16)
        Control forces applied to each mode.
    dt : float
        Sampling time [s].
    modal_start_idx : int, default=6
        Column index of q_{s,1} in *states*.
    """
    import numpy as np

    # Convert to NumPy (works for JAX, PyTorch, etc.)
    states_np = np.asarray(states)
    z_ref_np = np.asarray(z_ref)
    ctrls_np = np.asarray(ctrls)

    T = states_np.shape[0]
    t = np.arange(T) * dt

    # ── Figure 1: 16 modal amplitudes (4×4 grid) ────────────────────────────
    fig, axes = plt.subplots(4, 4, figsize=(14, 10), sharex=True)
    for i in range(16):
        r, c = divmod(i, 4)
        ax = axes[r, c]
        ax.plot(t, states_np[:, modal_start_idx + i], label=f"$q_{{s,{i+1}}}$")
        ax.plot(t, z_ref_np[:T, i], "--", label="ref", linewidth=1)
        ax.set_title(f"Mode {i+1}", fontsize=10)
        ax.grid(True, linewidth=0.3, alpha=0.6)

        if r == 3:
            ax.set_xlabel("time [s]")
        if c == 0:
            continue
            # ax.set_ylabel("amplitude [m]")

    # Single legend outside the axes grid to avoid clutter
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncol=2, fontsize=9, frameon=False)
    fig.suptitle("Structural‑mode tracking with GuSTO MPC", fontsize=14, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    # ── Figure 2: control forces (all actuators in one plot) ────────────────
    fig2, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t, ctrls_np)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("modal force")
    ax.set_title("Control forces (all 16 actuators)", fontsize=14)
    ax.grid(True)

    force_labels = [f"$F_{{s,{i+1}}}$" for i in range(16)]
    ax.legend(force_labels, ncol=4, fontsize=8, frameon=False)

    # ── Figure 3: rigid-body motion ───────────────────────
    # Infer number of structural modes and build index slices
    n_s = z_ref_np.shape[1]

    idx_q_u    = slice(0, 3)                               # q_u  (0‥2)
    idx_q_r    = slice(3, 6)                               # q_r  (3‥5)
    idx_qdot_u = slice(modal_start_idx + n_s,
                       modal_start_idx + n_s + 3)          # q̇_u
    idx_qdot_r = slice(modal_start_idx + n_s + 3,
                       modal_start_idx + n_s + 6)          # q̇_r

    fig3, axes = plt.subplots(2, 2, figsize=(12, 6), sharex=True)

    # ── q_u ────────────────────────────────────────────────────
    for i in range(3):
        axes[0, 0].plot(t, states_np[:, idx_q_u.start + i],
                        label=fr"$q_{{u,{i+1}}}$")
    axes[0, 0].set_title(r"Translation $q_u$")
    axes[0, 0].grid(True, linewidth=0.3, alpha=0.6)
    axes[0, 0].legend(fontsize=8, frameon=False)

    # ── q_r ────────────────────────────────────────────────────
    for i in range(3):
        axes[0, 1].plot(t, states_np[:, idx_q_r.start + i],
                        label=fr"$q_{{r,{i+1}}}$")
    axes[0, 1].set_title(r"Rotation $q_r$")
    axes[0, 1].grid(True, linewidth=0.3, alpha=0.6)
    axes[0, 1].legend(fontsize=8, frameon=False)

    # ── q̇_u ───────────────────────────────────────────────────
    for i in range(3):
        axes[1, 0].plot(t, states_np[:, idx_qdot_u.start + i],
                        label=fr"$\dot{{q}}_{{u,{i+1}}}$")
    axes[1, 0].set_title(r"Linear velocity $\dot{q}_u$")
    axes[1, 0].set_xlabel("time [s]")
    axes[1, 0].grid(True, linewidth=0.3, alpha=0.6)
    axes[1, 0].legend(fontsize=8, frameon=False)

    # ── q̇_r ───────────────────────────────────────────────────
    for i in range(3):
        axes[1, 1].plot(t, states_np[:, idx_qdot_r.start + i],
                        label=fr"$\dot{{q}}_{{r,{i+1}}}$")
    axes[1, 1].set_title(r"Angular velocity $\dot{q}_r$")
    axes[1, 1].set_xlabel("time [s]")
    axes[1, 1].grid(True, linewidth=0.3, alpha=0.6)
    axes[1, 1].legend(fontsize=8, frameon=False)

    fig3.suptitle("Rigid-body states of the", fontsize=14, y=0.98)
    fig3.tight_layout(rect=[0, 0, 1, 0.94])

    plt.show()


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
                     dt=5e-4, n_x=2*(3+3+ns), n_u=ns, integrator="dopri5")

    # 3) Prepare a GuSTO config
    # ----- what the controller should look at (z = H x) -------------------
    H = jnp.zeros((2*ns, rom.n_x)) # should be 2*ns
    start_qs  = 6               # start index for q_s
    start_qds = 6 + ns + 6      # start index for \dot{q}_s
    for i in range(ns):
        H = H.at[i,           start_qs  + i].set(1.0)      # q_s
        H = H.at[i + ns,      start_qds + i].set(1.0)      # \dot{q}_s

    Qz  = jnp.diag(jnp.array([10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0,
                              10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0]))  # state cost --> needs extra entries for each mode added
    Qzf = jnp.diag(jnp.array([10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0,
                              10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0])) 
    R   = 1e-4 * jnp.eye(ns)    # control effort cost

    # characteristic scales (rough guesses)
    x_char = jnp.ones(rom.n_x)*1e-2
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
    x0 = jnp.zeros(rom.n_x)                                 # initial state    
    for i in range(ns):
        x0 = x0.at[6+i].set(random.randint(10,100)/1000)    # initial q_s[i]

    T  = 500

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
        print(f"Step {k+1}/{T} ... ", end="")
        u, _ = mpc.compute_control(x)
        # print(f"Control:", u)
        x    = rom.discrete_dynamics(x, u)
        # print(f"State: {x}")
        states.append(x)
        ctrls.append(u)

    states = jnp.stack(states)      # (T, nx)
    ctrls  = jnp.stack(ctrls)       # (T, ns)

    # 8) Visualization: compare actual position vs. reference + input forces
    plot_modal_tracking(states, z_ref[:T], ctrls, rom.dt)


if __name__ == "__main__":
    run_mpc_demo()