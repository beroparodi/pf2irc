"""Compactness and the clock (Sect. 6.4): does v/r_c correlate with the central surface brightness mu0 of the disc
also where dark matter dominates at r_c, and for the dark matter alone?

    python3 compactness_check.py      -> compactness_check_out.txt

mu0: 3.6 micron, face-on, as in make_figures.py; C: compactness parameter of Parodi (2018, Table B.1), HSB = C > 1.
Baryon share at r_c: baryon_share_out.txt (SPARC mass models, gas x1, disc 0.5, bulge 0.7).
Dark matter alone: V_DM^2 = V^2 - V_gas|V_gas| - Y_d V_disk^2 - 0.7 V_bul^2, fitted with the P18 profile, for
Y_d = 0.3, 0.5, 0.7, 1.0 (points with V_DM^2 <= 0 dropped; galaxies with fewer than 4 points left, or a fitted
r_c outside 0.01-50 kpc, skipped). This uses a decomposition, unlike the total-matter results of the paper.
"""
import csv, os, numpy as np
from scipy import stats as st
from scipy.optimize import least_squares
SP = "sparc_data"
ex = np.exp
def m(x): return x*(1-ex(-x))**2 - 0.5*x*(1+x)*ex(-2*x) + 0.25*(1-ex(-2*x))
vn = lambda x: np.sqrt(m(x)/x)
F = list(csv.DictReader(open(os.path.join(SP, "sparc79_fits.csv"))))
B1L = list(csv.DictReader(open(os.path.join(SP, "parodi2018_B1.csv"))))   # same order as sparc79_fits.csv
SH = {l.split()[0]: float(l.split()[1]) for l in open("baryon_share_out.txt").readlines()[1:]}
def mu0_faceon(name, Rd):   # as in make_figures.py
    d = np.loadtxt(os.path.join(SP, "rotmod", name+"_rotmod.dat"), ndmin=2); R, SB = d[:, 0], d[:, 6]; ok = SB > 0
    msk = ok & (R <= 2.5*Rd)
    if msk.sum() < 2: msk = ok.copy(); msk[4:] = False
    return 3.24 + 21.572 - 2.5*np.polyfit(R[msk], np.log(SB[msk]), 1)[1]/np.log(10)
def fit_dm(name, Yd):
    """P18 fit to the dark matter alone: V_DM^2 = V^2 - V_gas|V_gas| - Yd V_disk^2 - 0.7 V_bul^2."""
    d = np.loadtxt(os.path.join(SP, "rotmod", name+"_rotmod.dat"))
    R, V, e0, Vg, Vd, Vb = d.T[:6]
    e = np.sqrt(e0**2+(0.03*V)**2+1.0)
    v2 = V**2 - Vg*np.abs(Vg) - Yd*Vd**2 - 0.7*Vb**2
    ok = v2 > 0
    if ok.sum() < 4: return None
    Vdm = np.sqrt(v2[ok]); edm = V[ok]*e[ok]/Vdm; R = R[ok]
    res = lambda p: (np.exp(p[0])*vn(R/np.exp(p[1]))-Vdm)/edm
    s = min((least_squares(res, [np.log(Vdm.max()), np.log(r)]) for r in [0.3, 1, 3, 10]), key=lambda s: s.cost)
    v, rc = np.exp(s.x)
    if rc > 50 or rc < 0.01: return None
    return v/rc, int((~ok).sum())
names = [f["name"] for f in F]
q = np.array([float(f["v"])/float(f["rc"]) for f in F]); lq = np.log10(q)
mu = np.array([mu0_faceon(n, float(b["Rd"])) for n, b in zip(names, B1L)])
C = np.array([float(b["C"]) for b in B1L])
sh = np.array([SH[n] for n in names])
out = []
def rep(s): print(s); out.append(s)
def corr(lbl, msk, y=lq, x=mu):
    p = st.pearsonr(x[msk], y[msk]); s = st.spearmanr(x[msk], y[msk])
    hs, ls = msk & (C > 1), msk & (C <= 1)
    rep(f"{lbl:46s} N={msk.sum():2d}  Pearson r={p[0]:+.2f} (p={p[1]:.1e})  Spearman {s[0]:+.2f} (p={s[1]:.1e})  "
        f"median v/rc HSB {np.median(10**y[hs]):.0f} ({hs.sum()}) LSB {np.median(10**y[ls]):.0f} ({ls.sum()})")
allm = np.ones(len(q), bool)
rep("log(v/r_c) of the total-matter fit vs mu0 [3.6 um, face-on]:")
corr("all 79", allm)
corr("dark matter dominates at r_c (share < 1/2)", sh < 0.5)
corr("dark matter strongly dominates (share < 0.3)", sh < 0.3)
corr("share < 0.2", sh < 0.2)
corr("without NGC1705, NGC4214, UGC07690", ~np.isin(names, ["NGC1705", "NGC4214", "UGC07690"]))
p = st.pearsonr(mu, sh); rep(f"baryon share vs mu0: Pearson r={p[0]:+.2f} (p={p[1]:.1e}); median share HSB {np.median(sh[C>1]):.2f}, LSB {np.median(sh[C<=1]):.2f}")
X = np.column_stack([np.ones_like(mu), mu, sh]); b, *_ = np.linalg.lstsq(X, lq, rcond=None)
r_part = st.pearsonr(mu - np.polyval(np.polyfit(sh, mu, 1), sh), lq - np.polyval(np.polyfit(sh, lq, 1), sh))
rep(f"partial correlation log(v/r_c)-mu0 at fixed baryon share: r={r_part[0]:+.2f} (p={r_part[1]:.1e}); "
    f"fit log(v/r_c) = {b[0]:.2f} {b[1]:+.3f} mu0 {b[2]:+.2f} share")
rep("")
rep("v/r_c of the dark matter alone (baryons subtracted, gas x1, bulge 0.7, disc Y_d) vs mu0:")
for Yd in (0.3, 0.5, 0.7, 1.0):
    R_ = [fit_dm(n, Yd) for n in names]
    ok = np.array([r is not None for r in R_]); qd = np.array([r[0] if r else np.nan for r in R_])
    lqd = np.log10(qd)
    p = st.pearsonr(mu[ok], lqd[ok]); s = st.spearmanr(mu[ok], lqd[ok])
    hs, ls = ok & (C > 1), ok & (C <= 1)
    pr = st.pearsonr(lq[ok], lqd[ok])
    rep(f"  Y_d={Yd}: N={ok.sum()}  Pearson r={p[0]:+.2f} (p={p[1]:.1e})  Spearman {s[0]:+.2f} (p={s[1]:.1e})  "
        f"median DM v/rc HSB {np.nanmedian(qd[hs]):.0f} LSB {np.nanmedian(qd[ls]):.0f}; DM vs total log(v/rc) r={pr[0]:+.2f}")
    ok2 = ok & (sh < 0.5)
    p2 = st.pearsonr(mu[ok2], lqd[ok2])
    rep(f"          and dark matter dominant at r_c: N={ok2.sum()} Pearson r={p2[0]:+.2f} (p={p2[1]:.1e})")
open("compactness_check_out.txt", "w").write("\n".join(out)+"\n")
rep("")
ext = np.isin(names, ["NGC1705", "NGC4214", "UGC07690"])
corr("share < 1/2, without the three extremes", (sh < 0.5) & ~ext)
corr("share < 0.3, without the three extremes", (sh < 0.3) & ~ext)
for lbl, msk in [("all", allm), ("share<0.5", sh < 0.5), ("share<0.3", sh < 0.3), ("share<0.2", sh < 0.2)]:
    rep(f"mu0 spread {lbl:10s}: 16/50/84% = {np.percentile(mu[msk],[16,50,84]).round(2)}, std {mu[msk].std():.2f} mag")
rep("shares of the extremes: " + ", ".join(f"{n} {SH[n]:.2f}" for n in ["NGC1705", "NGC4214", "UGC07690", "NGC2915"]))
open("compactness_check_out.txt", "w").write("\n".join(out)+"\n")
