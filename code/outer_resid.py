"""Stacked outer residuals: a direct look at the mass excess c (second-round referee, minor 21; Sect. 2.2).

    python3 outer_resid.py      -> outer_resid_out.txt

For P18 (c = 1/4, curve overshoots v by 1.8% at x = 5.5) and for the pseudo-isothermal sphere (c = -pi/2, curve
approaches its asymptote from below), each fitted to the 79 curves with the same errors, the residuals
(V_obs - V_model)/v_model,asymptote are stacked in bins of x = R/r_c. A wrong sign of c would leave a systematic
pattern at x = 3-10, where the two curve shapes differ most for a given asymptote.
"""
import sys, numpy as np
sys.argv = [sys.argv[0], "0"]
import robustness as R
from scipy.optimize import least_squares
vpi = lambda x: np.sqrt(1 - np.arctan(x)/x)
bins = [0, 1, 2, 3, 5, 7, 10, 20]
out = []
for lab, fun in (("P18", R.vn), ("pseudo-isothermal", vpi)):
    xs, rs, ws = [], [], []
    for g in R.GAL:
        Rr, V, e = g["R"], g["V"], g["e"]
        res = lambda p: (np.exp(p[0])*fun(Rr/np.exp(p[1])) - V)/e
        s = min((least_squares(res, [np.log(V.max()), np.log(r)]) for r in [0.3, 1, 3, 10]), key=lambda s: s.cost)
        v, rc = np.exp(s.x)
        xs += list(Rr/rc); rs += list((V - v*fun(Rr/rc))/v); ws += list((v/e)**2)
    xs, rs, ws = map(np.array, (xs, rs, ws))
    line = [lab]
    for lo, hi in zip(bins[:-1], bins[1:]):
        k = (xs >= lo) & (xs < hi)
        m = np.sum(rs[k]*ws[k])/np.sum(ws[k]); err = 1/np.sqrt(np.sum(ws[k]))
        sd = rs[k].std()/np.sqrt(k.sum())
        line.append(f"x {lo}-{hi}: {100*m:+.2f}% (+-{100*sd:.2f}, N={k.sum()})")
    out.append("; ".join(line))
open("outer_resid_out.txt", "w").write("\n".join(out) + "\n")
print("\n".join(out))
