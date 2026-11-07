"""Fig. G.2: DDO 161, the coupling of v, r_c and n in the free-exponent fits (App. G).
P18 (n = 2) and the fit with v, r_c, n free are those of the paper (sparc79_fits.csv, fit_exponent_results.csv);
the third fit has v fixed at the flat speed of P18 (parodi2018_B3.csv) and r_c, n free, as in n_vflat.py.
Fits use the paper's errors; the figure shows the SPARC errors. Run from code/: python3 fig_ddo161.py"""
import csv, os, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.optimize import least_squares
import fit_exponent as fe
SP = "sparc_data"; name = "DDO161"
d = np.loadtxt(os.path.join(SP, "rotmod", name+"_rotmod.dat")); R, V, eV = d.T[:3]
E = np.sqrt(eV**2+(0.03*V)**2+1.0)
f2 = {r["name"]: r for r in csv.DictReader(open(os.path.join(SP, "sparc79_fits.csv")))}[name]
fn = {r["name"]: r for r in csv.DictReader(open("fit_exponent_results.csv"))}[name]
B2 = list(csv.DictReader(open(os.path.join(SP, "parodi2018_B2.csv")))); B3 = list(csv.DictReader(open(os.path.join(SP, "parodi2018_B3.csv"))))
vf = {b2["name"]: float(b3["vflat"]) for b2, b3 in zip(B2, B3) if b3["vflat"]}[name]
res = lambda p: (fe.vc(R, vf, np.exp(p[0]), p[1])-V)/E
s = min((least_squares(res, [np.log(r0), n0], bounds=([np.log(0.01), fe.N_MIN], [np.log(100), fe.N_MAX]))
         for r0 in (1, 3, 10) for n0 in (1.0, 2.0)), key=lambda s: s.cost)
chi = 2*s.cost; sn = np.sqrt(np.linalg.inv(s.jac.T@s.jac)[1, 1])*np.sqrt(max(1, chi/(len(R)-2)))
fits = [("P18, $n=2$", float(f2["v"]), float(f2["rc"]), 2.0, "C3", "-"),
        (rf"$v,r_c,n$ free: $n={float(fn['n']):.2f}$", float(fn["v"]), float(fn["rc"]), float(fn["n"]), "C0", "--"),
        (rf"$v=V_{{\rm flat}}$, $n={s.x[1]:.2f}$", vf, np.exp(s.x[0]), s.x[1], "C2", "-.")]
rr = np.linspace(0.05, 25, 500)
plt.rcParams.update({"font.size": 7.5})
fig, (a, b) = plt.subplots(2, 1, figsize=(3.5, 3.6), sharex=True, gridspec_kw=dict(height_ratios=[3, 1.1], hspace=0.06))
a.errorbar(R, V, eV, fmt="o", color="k", ms=2.5, capsize=1.5, lw=0.7, label="DDO 161 (SPARC)", zorder=5)
for lab, v, rc, n, c, ls in fits:
    a.plot(rr, fe.vc(rr, v, rc, n), color=c, ls=ls, lw=1.2, label=rf"{lab}, $v={v:.1f}$")
    b.plot(R, V-fe.vc(R, v, rc, n), "o", color=c, ms=2.2, mfc="none" if c != "C3" else c)
a.axvline(R.max(), color="0.6", lw=0.5, ls=":"); a.text(R.max()+0.3, 75, "last point", color="0.4", fontsize=6.5)
a.set_ylim(0, 79); a.set_ylabel(r"$V$ [km s$^{-1}$]"); a.legend(fontsize=6.3, loc="lower right", frameon=False)
b.axhline(0, color="k", lw=0.5); b.set_ylim(-7, 7); b.set_xlim(0, 25)
b.set_xlabel("$R$ [kpc]"); b.set_ylabel(r"res. [km s$^{-1}$]")
fig.savefig("fig_ddo161.png", dpi=300, bbox_inches="tight")
print(f"v fixed at V_flat = {vf}: r_c = {np.exp(s.x[0]):.2f}, n = {s.x[1]:.2f} +- {sn:.2f}")
for lab, v, rc, n, c, ls in fits:
    print(f"{lab:30s} v={v:.1f} rc={rc:.2f} n={n:.2f}: V(20 kpc) = {fe.vc(np.array([20.0]), v, rc, n)[0]:.1f}")
