"""Robustness of the free exponent n (Sect. 7, App. G) against the coupling of n, r_c and v.

With v, r_c and n free, a slowly rising curve can take a small n together with a large r_c and a v above the
observed outer level (e.g. DDO 161: n = 1.41, v = 72 km/s against a flat 66-67 km/s). Here v is tied to an
independent flat speed, measured along the outer curve, and only r_c and n are fitted:
  (a) V_flat of P18 (Parodi 2018, Table B3; 54 galaxies), v fixed;
  (b) V_flat of SPARC (Lelli et al. 2016, BTFR table; 34 galaxies), v fixed;
  (c) as (b), but v free with a Gaussian prior from the SPARC error of log V_flat.
The counts use the definitions of the paper: constrained = sigma_n < 1 and n not at a bound; core = |n-2| <= sigma_n;
cuspier = (n-2)/sigma_n < -2 (sigma_n floored at 0.1); NFW reached = cuspier and |n-1| <= sigma_n; firmly NFW-like = n <= 1.
Also: the local density slope of a P18 core, 2x u'/u - 2 (P18, Eq. 40: about -1.5x), at the median innermost radius
of the 79 curves (Sect. 7). Errors as in the paper. Run from code/: python3 n_vflat.py  (output n_vflat_out.txt, n_vflat_results.csv)"""
import csv, os, numpy as np
from scipy.optimize import least_squares
import fit_exponent as fe
SP = "sparc_data"
out = open("n_vflat_out.txt", "w")
def P(*a): print(*a); print(*a, file=out)
G = {g["name"]: g for g in fe.load()}
FR = {r["name"]: r for r in csv.DictReader(open("fit_exponent_results.csv"))}
B2 = list(csv.DictReader(open(os.path.join(SP, "parodi2018_B2.csv"))))
B3 = list(csv.DictReader(open(os.path.join(SP, "parodi2018_B3.csv"))))
VF18 = {b2["name"].replace("UGC0786", "UGC07866"): float(b3["vflat"]) for b2, b3 in zip(B2, B3) if b3["vflat"]}
VFS = {}
for l_ in open(os.path.join(SP, "BTFR_Lelli2016a.mrt")).read().split("\n"):
    p_ = l_.split()
    if len(p_) == 11:
        try:
            if float(p_[6]) > 0: VFS[p_[0]] = (10**float(p_[6]), float(p_[7]))
        except ValueError: pass
VFS = {k: v for k, v in VFS.items() if k in G}

def fit_vfixed(g, v):
    res = lambda p: (fe.vc(g["R"], v, np.exp(p[0]), p[1]) - g["V"]) / g["e"]
    lo, hi = [np.log(0.01), fe.N_MIN], [np.log(100), fe.N_MAX]
    best = min((least_squares(res, [np.log(r0), n0], bounds=(lo, hi)) for r0 in (0.3, 1, 3, 10) for n0 in (1.0, 2.0, 3.0)),
               key=lambda s: s.cost)
    chi2, dof = 2*best.cost, max(len(g["R"])-2, 1)
    try: sn = np.sqrt(np.linalg.inv(best.jac.T@best.jac)[1, 1])*np.sqrt(max(1.0, chi2/dof))
    except np.linalg.LinAlgError: sn = np.nan
    return chi2, np.exp(best.x[0]), best.x[1], sn

def fit_vprior(g, v0, elog):
    sl = elog*np.log(10)                                   # error of ln v
    res = lambda p: np.concatenate([(fe.vc(g["R"], np.exp(p[0]), np.exp(p[1]), p[2]) - g["V"]) / g["e"],
                                    [(p[0]-np.log(v0))/sl]])
    lo, hi = [-5, np.log(0.01), fe.N_MIN], [10, np.log(100), fe.N_MAX]
    best = min((least_squares(res, [np.log(v0), np.log(r0), n0], bounds=(lo, hi)) for r0 in (0.3, 1, 3, 10) for n0 in (1.0, 2.0, 3.0)),
               key=lambda s: s.cost)
    r = best.fun[:-1]; chi2, dof = np.sum(r**2), max(len(g["R"])-2, 1)
    try: sn = np.sqrt(np.linalg.inv(best.jac.T@best.jac)[2, 2])*np.sqrt(max(1.0, chi2/dof))
    except np.linalg.LinAlgError: sn = np.nan
    return chi2, np.exp(best.x[0]), np.exp(best.x[1]), best.x[2], sn

def counts(n, s):
    ok = np.isfinite(s) & (s < 1) & (n > fe.N_MIN+0.01) & (n < fe.N_MAX-0.01)
    n, s = n[ok], s[ok]
    return dict(N=int(ok.sum()), median=np.median(n) if ok.sum() else np.nan, core=int((abs(n-2) <= s).sum()),
                cuspier=int(((n-2)/np.maximum(s, 0.1) < -2).sum()), nfw=int(((abs(n-1) <= s) & ((n-2)/np.maximum(s, 0.1) < -2)).sum()), firm=int((n <= 1).sum()),
                above=int(((n-2)/np.maximum(s, 0.1) > 2).sum()))
def show(lbl, c):
    P(f"  {lbl:38s} constrained {c['N']:2d}; median n {c['median']:.2f}; core {c['core']:2d}; cuspier >2sig {c['cuspier']:2d}; "
      f"NFW reached {c['nfw']:2d} (n<=1: {c['firm']}); above 2 by >2sig {c['above']}")

u_ = lambda x: 1+(x-1)*np.exp(-x); du_ = lambda x: (2-x)*np.exp(-x)
loc = lambda x, n=2.0: n*x*du_(x)/u_(x)-2
fits79 = {r["name"]: r for r in csv.DictReader(open(os.path.join(SP, "sparc79_fits.csv")))}
xin = np.array([G[k]["R"][0]/float(fits79[k]["rc"]) for k in G])
P("Local density slope of P18, 2x u'/u - 2 (asymptotic slope 0 at x -> 0):")
P("  x = 0.1, 0.15, 0.2, 0.3, 0.5: " + ", ".join(f"{loc(x):+.2f}" for x in (0.1, 0.15, 0.2, 0.3, 0.5)))
P(f"  innermost point of the 79 curves: x_in median {np.median(xin):.2f} (16-84%: {np.percentile(xin, 16):.2f}-{np.percentile(xin, 84):.2f});"
  f" local slope there {loc(np.median(xin)):+.2f} (n = 2), {loc(np.median(xin), 1.6):+.2f} (n = 1.6)\n")
rows = []
P("Free exponent n with v tied to an independent flat speed\n")
allfree = np.array([[float(FR[k]["n"]), float(FR[k]["sigma_n"]) if FR[k]["sigma_n"] not in ("", "nan") else np.nan] for k in G])
P("All 79 (paper, v free):"); show("v, r_c, n free", counts(allfree[:, 0], allfree[:, 1]))
for lbl, VF, mode in [("(a) V_flat of P18, v fixed", VF18, "fix"), ("(b) V_flat of SPARC, v fixed", {k: v[0] for k, v in VFS.items()}, "fix"),
                      ("(c) V_flat of SPARC, v with prior", VFS, "prior")]:
    names = [k for k in G if k in VF]
    nf = np.array([float(FR[k]["n"]) for k in names]); sf = np.array([float(FR[k]["sigma_n"]) if FR[k]["sigma_n"] not in ("", "nan") else np.nan for k in names])
    vfree = np.array([float(FR[k]["v"]) for k in names])
    nt, st, vt, ct = [], [], [], []
    for k in names:
        g = G[k]
        if mode == "fix":
            c2, rc, n_, s_ = fit_vfixed(g, VF[k]); v_ = VF[k]
        else:
            c2, v_, rc, n_, s_ = fit_vprior(g, *VF[k])
        nt.append(n_); st.append(s_); vt.append(v_); ct.append(c2)
        rows.append(dict(sample=lbl[:3], name=k, vflat=f"{(VF[k] if mode=='fix' else VF[k][0]):.1f}", v_free=f"{float(FR[k]['v']):.1f}",
                         n_free=FR[k]["n"], sigma_free=FR[k]["sigma_n"], v_tied=f"{v_:.1f}", n_tied=f"{n_:.3f}", sigma_tied=f"{s_:.3f}",
                         chi2_free=FR[k]["chi2"], chi2_tied=f"{c2:.2f}", chi2_n2=FR[k]["chi2_n2"]))
    nt, st, vt = map(np.array, (nt, st, vt))
    P(f"\n{lbl}: {len(names)} galaxies")
    P(f"  v_free / V_flat: median {np.median(vfree/np.array([VF[k] if mode=='fix' else VF[k][0] for k in names])):.3f}, "
      f"16-84% {np.percentile(vfree/np.array([VF[k] if mode=='fix' else VF[k][0] for k in names]),[16,84]).round(3)}")
    show("same galaxies, v free (paper)", counts(nf, sf))
    show("v tied to V_flat", counts(nt, st))
    okb = np.isfinite(sf) & (sf < 1) & np.isfinite(st) & (st < 1)
    P(f"  galaxies constrained in both: {okb.sum()}; median change of n (tied - free): {np.median(nt[okb]-nf[okb]):+.2f}; "
      f"n rises in {(nt[okb] > nf[okb]).sum()} of {okb.sum()}")
with open("n_vflat_results.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

# one common n for the same galaxies, v free (as in the paper) and v tied
from scipy.optimize import minimize_scalar
def fit_vfixed_n(g, v, n):
    res = lambda p: (fe.vc(g["R"], v, np.exp(p[0]), n) - g["V"]) / g["e"]
    return 2*min((least_squares(res, [np.log(r0)]) for r0 in (0.3, 1, 3, 10)), key=lambda s: s.cost).cost
P("\nOne common n (r_c free per galaxy):")
for lbl, VF in [("(a) P18 V_flat", VF18), ("(b) SPARC V_flat", {k: v[0] for k, v in VFS.items()})]:
    names = [k for k in G if k in VF]
    for mode in ("v free", "v tied"):
        tot = (lambda n: sum(fe.fit_fixed_n(G[k], n)[0] for k in names)) if mode == "v free" else \
              (lambda n: sum(fit_vfixed_n(G[k], VF[k], n) for k in names))
        r = minimize_scalar(tot, bounds=(1.0, 3.0), method="bounded", options={"xatol": 0.01})
        P(f"  {lbl}, {len(names)} galaxies, {mode}: best common n = {r.x:.2f}; chi2 there {r.fun:.1f}, at n = 2 {tot(2.0):.1f}")
out.close()
