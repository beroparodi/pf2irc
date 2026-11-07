import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scipy.optimize import brentq
plt.rcParams.update({"font.size": 9})
Ms = [0.2, 0.4, 0.6, 0.8, 1.0]
cmap = plt.get_cmap("Blues"); cols = [cmap(0.35 + 0.6*i/(len(Ms)-1)) for i in range(len(Ms))]
etab = 1.5*np.pi; K = 1 + 1.5*np.pi
def R_of(M, t):                       # units G = v = 1; halts at eta = 3pi/2, R = M
    if t >= M*K: return M, "halted"
    eta = brentq(lambda e: M*(e - np.sin(e)) - t, 0, etab)
    return M*(1 - np.cos(eta)), ("expanding" if eta < np.pi else "falling")
fig, ax = plt.subplots(1, 3, figsize=(11.5, 3.7), gridspec_kw=dict(width_ratios=[1.2, 1, 1.1]))
# (a) radius vs time
T = np.linspace(0, 7, 800)
for M, c in zip(Ms, cols):
    ax[0].plot(T, [R_of(M, t)[0] for t in T], c=c, lw=2, label=f"M = {M}")
    ax[0].plot([np.pi*M], [2*M], "o", c=c, ms=4)
ax[0].plot(T, T/K, "--", c="0.35", lw=1.2, label=r"front $R_b=vt/(1+3\pi/2)$")
ax[0].axvline(4.0, c="0.6", lw=0.8, ls=":"); ax[0].text(3.95, 0.05, "snapshot (b)", fontsize=8, color="0.4", ha="right")
ax[0].set_xlabel(r"time $t$  [units $GM_1/v^3$]"); ax[0].set_ylabel(r"radius $R$  [units $GM_1/v^2$]")
ax[0].set_title("(a) each shell: cycloid, turnaround (dot), halt at the front", fontsize=9)
ax[0].set_ylim(0, 2.2); ax[0].legend(fontsize=7.5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=3)
# (b) snapshot of nested spheres
t0 = 4.0
Mg = np.linspace(0.125, 2.0, 16)
for M in Mg[::-1]:
    R, st = R_of(M, t0)
    style = dict(halted=dict(ls="-", lw=1.6), falling=dict(ls="--", lw=1.2), expanding=dict(ls=":", lw=1.2))[st]
    ax[1].add_patch(Circle((0, 0), R, fill=False, color=cmap(0.3 + 0.32*M), **style))
Rb = t0/K
ax[1].add_patch(Circle((0, 0), Rb, fill=True, color="0.85", zorder=0))
ax[1].set_xlim(-4.1, 4.1); ax[1].set_ylim(-4.1, 4.1); ax[1].set_aspect("equal"); ax[1].axis("off")
ax[1].set_title(f"(b) nested top hats at t = {t0}: one sphere per mass,\nM = 0.1 to 2 (inner = lighter)", fontsize=9)
from matplotlib.lines import Line2D
ax[1].legend([Line2D([], [], c="0.3", ls="-", lw=1.6), Line2D([], [], c="0.3", ls="--"), Line2D([], [], c="0.3", ls=":")],
             ["halted (inside the front, grey)", "falling back", "still expanding"], fontsize=7.5, frameon=False,
             loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=3, handlelength=2)
ax[1].text(0, -0.08, "front", ha="center", fontsize=7.5, color="0.3")
# (c) mean overdensity vs time
eta = np.linspace(1e-3, etab, 600)
for M, c in zip(Ms, cols):
    t = M*(eta - np.sin(eta)); d = 4.5*(eta - np.sin(eta))**2/(1 - np.cos(eta))**3
    ax[2].loglog(t, d, c=c, lw=2)
    ax[2].plot([np.pi*M], [9*np.pi**2/16], "o", c=c, ms=4); ax[2].plot([K*M], [4.5*K**2], "s", c=c, ms=4)
ax[2].axhline(1, c="0.6", lw=0.8, ls=":")
ax[2].text(0.012, 1.15, "background", fontsize=7.5, color="0.4")
ax[2].text(0.012, 6.6, r"turnaround: $1+\langle\delta\rangle=5.55$ (dots)", fontsize=7.5, color="0.3")
ax[2].text(0.012, 190, r"halting: $1+\langle\delta\rangle=147$ (squares)", fontsize=7.5, color="0.3")
ax[2].set_xlabel(r"time $t$"); ax[2].set_ylabel(r"$1+\langle\delta\rangle$ inside the shell")
ax[2].set_title("(c) same history, later for larger M ($t\\propto M$)", fontsize=9)
ax[2].set_xlim(1e-2, 8); ax[2].set_ylim(0.8, 400)
fig.tight_layout(); fig.savefig("nested_tophats.png", dpi=170)
