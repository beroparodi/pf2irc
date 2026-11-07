"""Nested top hats at two times: snapshot of the spheres and the density profile (units G = v = 1)."""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.lines import Line2D
from scipy.optimize import brentq
plt.rcParams.update({"font.size": 9})
cmap = plt.get_cmap("Blues")
K = 1 + 1.5*np.pi; etab = 1.5*np.pi; MSEED = 4.0          # the seed: shells M = 0 ... 4
def state(M, t):
    if t >= M*K: return M, "halted"
    eta = brentq(lambda e: M*(e - np.sin(e)) - t, 0, etab)
    return M*(1 - np.cos(eta)), ("expanding" if eta < np.pi else "falling")
xi_of = lambda e: (1 - np.cos(e))/(e - np.sin(e))
D_of = lambda e: 1 - np.sin(e)*(e - np.sin(e))/(1 - np.cos(e))**2
def density(R, t):
    """rho(R, t): halted matter (M = R, a singular isothermal sphere; the small core is omitted), the infalling
    shells (local density from the cycloids, Eq. 19 of the paper) out to the edge of the seed, background beyond."""
    Rb = t/K
    if R <= Rb: return 1/(4*np.pi*R**2), "halted"
    eta = brentq(lambda e: xi_of(e) - R/t, 1e-4, etab)
    M = t/(eta - np.sin(eta))
    if M > MSEED: return 1/(6*np.pi*t**2), "background"
    return 1/(4*np.pi*R**2)/((1 - np.cos(eta))*D_of(eta)), ("expanding" if eta < np.pi else "falling")
LS = dict(halted="-", falling="--", expanding=":", background="-")
times = [4.0, 8.0]
fig, ax = plt.subplots(2, 2, figsize=(10, 8.6), gridspec_kw=dict(width_ratios=[1, 1.25]))
Mg = np.linspace(0.25, MSEED, 16)
for row, t in enumerate(times):
    a = ax[row, 0]
    Rb = t/K
    a.add_patch(Circle((0, 0), Rb, fill=True, color="0.85", zorder=0))
    for M in Mg[::-1]:
        R, st = state(M, t)
        a.add_patch(Circle((0, 0), R, fill=False, color=cmap(0.3 + 0.65*M/MSEED), ls=LS[st],
                           lw=1.6 if st == "halted" else 1.2))
    a.set_xlim(-8.2, 8.2); a.set_ylim(-8.2, 8.2); a.set_aspect("equal"); a.axis("off")
    a.set_title(f"t = {t:g}: nested spheres, M = 0.25 to 4 (inner = lighter)", fontsize=9)
    a.text(0, -Rb - 0.15, "front", ha="center", va="top", fontsize=7.5, color="0.3")
    # density profile
    b = ax[row, 1]
    R = np.logspace(-1.5, np.log10(40), 700)
    rho, st = zip(*[density(r, t) for r in R]); rho = np.array(rho); st = np.array(st)
    for s_ in ("halted", "falling", "expanding", "background"):
        m = st == s_
        if m.any():
            b.loglog(np.where(m, R, np.nan), np.where(m, rho, np.nan), ls=LS[s_],
                     c=("0.55" if s_ == "background" else cmap(0.85)), lw=2 if s_ != "background" else 1.2)
    for r, lab in ((Rb, "front $R_b$"), (2*t/np.pi, "turnaround")):
        b.axvline(r, ymax=0.55, c="0.6", lw=0.8)
        b.text(r*1.07, 0.25, lab, rotation=90, fontsize=7.5, color="0.4", va="bottom")
    b.set_xlim(10**-1.5, 40); b.set_ylim(1e-4, 1e2)
    b.set_xlabel(r"radius $R$  [units $GM_1/v^2$]"); b.set_ylabel(r"density $\rho$  [units $v^6/G^3M_1^2$]")
    b.set_title(f"t = {t:g}: density profile", fontsize=9)
for r in range(2):
    ax[r, 1].legend([Line2D([], [], c=cmap(0.85), ls="-", lw=2), Line2D([], [], c=cmap(0.85), ls="--", lw=2),
                     Line2D([], [], c=cmap(0.85), ls=":", lw=2), Line2D([], [], c="0.55", lw=1.2)],
                    ["halted (singular isothermal, core omitted)", "falling back", "still expanding",
                     "background beyond the seed"], fontsize=7.5, frameon=False, loc="upper right")
fig.tight_layout(); fig.savefig("nested_tophats_2.png", dpi=170)
for t in times: print(t, "front", round(t/K, 3), "turnaround", round(2*t/np.pi, 3), "jump", round(density(t/K*0.999, t)[0]/density(t/K*1.001, t)[0], 2))
