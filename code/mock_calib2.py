"""Mock calibration of the slope b of r_c ~ v^b with three noise models (second-round referee point M4; Appendix F).

    python3 mock_calib2.py NMOCK SEED BTRUE SCATTER NOISE   -> mock2_<NOISE>_<BTRUE>_<SCATTER>_<SEED>.npy

As mock_calib.py, except for the noise added to each model curve:
  white069 : independent Gaussian noise, errors x sqrt(chi2_nu) = x0.69 (the first calibration);
  white1   : independent Gaussian noise with the full (floor-enlarged) errors, i.e. not scaled down;
  resid    : the galaxy's own residual pattern about its P18 fit, V_obs - V_model, cyclically shifted by a
             random number of points, randomly reversed and randomly sign-flipped. This keeps the amplitude and
             the correlation along the curve of the real deviations (beam smearing, non-circular motions, warps).
Stored per mock: recovered b, plain std of log r_c, Pearson r, robust spread (1.4826 MAD), Spearman rho.
"""
import sys, numpy as np
N, SEED, BT, SI, NOISE = int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]), sys.argv[5]
sys.argv = [sys.argv[0], "0"]
import robustness as R
from scipy.stats import spearmanr
rng = np.random.default_rng(SEED)
L10 = np.log(10)
RES = [g["V"] - g["v"]*R.vn(g["R"]/g["rc"]) for g in R.GAL]

def noise(i, g):
    n = len(g["R"])
    if NOISE == "white069": return rng.normal(0, 1, n)*g["e"]*np.sqrt(R.chi2/R.dof)
    if NOISE == "white1":   return rng.normal(0, 1, n)*g["e"]
    if NOISE == "resid":
        r = np.roll(RES[i], rng.integers(n))
        if rng.random() < 0.5: r = r[::-1]
        return r*rng.choice([-1, 1])
    raise ValueError(NOISE)

def one():
    a0 = np.median(R.lrc); lrt = a0 + BT*R.xc + rng.normal(0, SI, len(R.xc))
    vs, rcs, sxs, sys_, cs = [], [], [], [], []
    for i, (g, lr) in enumerate(zip(R.GAL, lrt)):
        for _try in range(20):                 # a rare draw makes the refit fail (non-finite); draw again
            Vm = g["v"]*R.vn(g["R"]/10**lr) + noise(i, g)
            try:
                v_, rc_, slv, slr, cvr, _ = R.fit(g["R"], Vm, g["e"]); break
            except ValueError:
                continue
        vs.append(v_); rcs.append(rc_); sxs.append(slv/L10); sys_.append(slr/L10); cs.append(cvr)
    x = np.log10(vs); y = np.log10(rcs); sx = np.minimum(sxs, 0.5); sy = np.minimum(sys_, 0.5)
    p = R.ml_y(x-np.median(x), y, sx, sy, np.array(cs)*sx*sy)
    return p[1], y.std(), np.corrcoef(x, y)[0, 1], 1.4826*np.median(np.abs(y-np.median(y))), spearmanr(x, y)[0]

M = np.array([one() for _ in range(N)])
np.save(f"mock2_{NOISE}_{BT:.1f}_{SI:.2f}_{SEED}.npy", M)
print(f"{NOISE} b={BT} scatter={SI} seed={SEED}: N={N} median b {np.median(M[:,0]):.2f}; "
      f"P(b<=0.97) {(M[:,0]<=0.97).mean():.4f}; MAD {np.median(M[:,3]):.2f}; Spearman {np.median(M[:,4]):.2f}; "
      f"P(rho<=0.54) {(M[:,4]<=0.54).mean():.4f}", flush=True)
