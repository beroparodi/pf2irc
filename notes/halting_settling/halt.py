"""Why a shell does not halt by itself: orbit of one shell (mass M, energy -v^2/2) falling into the settled galaxy.
Units G = M = v = 1: the shell halts (A3) at R_b = GM/v^2 = 1. The galaxy inside R_b is taken as a singular
isothermal sphere frozen at the moment of arrival, M(<R) = R, so Phi = ln R - 1 inside and -1/R outside.
The angular momentum is received at turnaround (R_ta = 2), as in the paper: seed spin j = 2 sqrt2 lambda,
or the angular momentum of a circular orbit at R_b, j = 1 (the f = 1 run of Appendix H)."""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
plt.rcParams.update({"font.size": 9})
lam = 0.035; j_seed = 2*np.sqrt(2)*lam
Phi = lambda R: np.where(R <= 1, np.log(np.maximum(R, 1e-300)) - 1, -1/R)
dPhi = lambda R: 1/R if R <= 1 else 1/R**2
def turning(j):
    E = -0.5 + j**2/8                      # rotation added at turnaround, R_ta = 2
    f = lambda R: E - Phi(R) - j**2/(2*R**2)
    Rp = brentq(f, 1e-12 if j > 0 else 1e-300, 1.0) if j > 0 else 0.0
    Ra = 2.0
    return E, Rp, Ra
def orbit(j, tmax=40):
    # radial cycloid up to turnaround (t = pi), then with angular momentum j in the frozen potential
    eta = np.linspace(0, np.pi, 200); t1 = eta - np.sin(eta); R1 = 1 - np.cos(eta)
    rhs = lambda t, y: [y[1], -dPhi(y[0]) + j**2/y[0]**3]
    sol = solve_ivp(rhs, [np.pi, tmax], [2.0, 0.0], max_step=0.01, rtol=1e-9, atol=1e-12)
    return np.concatenate([t1, sol.t]), np.concatenate([R1, sol.y[0]])
out = []
for lab, j in (("seed spin", j_seed), ("circular at R_b (f = 1)", 1.0)):
    E, Rp, Ra = turning(j)
    out.append(f"{lab}: j = {j:.4f} GM/v, E = {E:.4f} v^2, pericentre {Rp:.4f} R_b (= R_b/{1/Rp:.0f}), apocentre {Ra:.1f} R_b")
out.append("circular orbit at R_b: j = 1, E = 1/2 + Phi(1) = -0.5 = the seed energy; arrival at R_b with j = 0: |Rdot| = 1 = v")
print("\n".join(out)); open("halt_out.txt", "w").write("\n".join(out) + "\n")

fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.9))
R = np.logspace(-3, np.log10(3), 800)
cols = {"seed": "#1f4e79", "f1": "#c0504d"}
for key, j, lab in (("seed", j_seed, rf"seed spin, $j={j_seed:.2f}\,GM/v$"), ("f1", 1.0, r"$j=GM/v$ (circular at $R_b$)")):
    E, Rp, Ra = turning(j)
    ax[0].semilogx(R, Phi(R) + j**2/(2*R**2), c=cols[key], lw=2, label=lab)
    ax[0].plot([Rp, Ra], [E, E], c=cols[key], lw=1, ls="--")
    ax[0].plot([Rp], [E], "o", c=cols[key], ms=4)
ax[0].semilogx(R, Phi(R), c="0.55", lw=1.2, ls=":", label=r"$j=0$ (radial)")
ax[0].plot([1], [-0.5], "s", c="k", ms=5)
ax[0].annotate("circular orbit at $R_b$:\n$E=-v^2/2$, $j=GM/v$", xy=(1, -0.5), xytext=(0.09, 0.25), fontsize=7.5,
               arrowprops=dict(arrowstyle="->", color="0.3", lw=0.8))
ax[0].axvline(1, c="0.6", lw=0.8, ymax=0.55); ax[0].text(1.05, -3.05, "$R_b$", fontsize=8, color="0.4")
ax[0].set_xlim(1e-3, 3); ax[0].set_ylim(-3.2, 0.6)
ax[0].set_xlabel(r"radius $R$  [units $R_b=GM/v^2$]"); ax[0].set_ylabel(r"effective potential  [units $v^2$]")
ax[0].set_title("(a) effective potential and energy of the shell (dashed)", fontsize=9)
ax[0].legend(fontsize=7.5, frameon=False, loc="lower left")
for key, j in (("seed", j_seed), ("f1", 1.0)):
    t, Rt = orbit(j)
    ax[1].plot(t, Rt, c=cols[key], lw=1.4)
eta = np.linspace(0, 1.5*np.pi, 200)
ax[1].plot(eta - np.sin(eta), 1 - np.cos(eta), c="k", lw=2)
ax[1].plot([1 + 1.5*np.pi, 40], [1, 1], c="k", lw=2, label="model: halts at $R_b$ (A3)")
ax[1].set_xlim(0, 40); ax[1].set_ylim(0, 2.2)
ax[1].set_xlabel(r"time $t$  [units $GM/v^3$]"); ax[1].set_ylabel(r"radius $R$  [units $R_b$]")
ax[1].set_title("(b) the shell keeps oscillating; the model halts it", fontsize=9)
ax[1].legend(fontsize=7.5, frameon=False, loc="upper right")
fig.tight_layout(); fig.savefig("shell_halt.png", dpi=170)
