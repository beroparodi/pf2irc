"""Mock calibration of the slope b of r_c ~ v^b (referee point M4; Appendix F).

    python3 mock_calib.py NMOCK SEED BTRUE SCATTER      -> mock_calib_<BTRUE>_<SCATTER>_<SEED>.npy

Each mock keeps every galaxy's fitted v and its observed radii and errors, draws log r_c from
log r_c = median + BTRUE (log v - median) + N(0, SCATTER), draws the curve from the IRC with Gaussian noise
of the observed size (errors x sqrt(chi2_nu)), refits v and r_c, and applies the paper's estimator
(maximum likelihood with intrinsic scatter in log r_c, fit errors only).
Stored per mock: recovered b, spread (std) of the refitted log r_c, correlation of log v and log r_c,
and their robust versions (1.4826 x median absolute deviation; Spearman rank correlation), since a few refits of
still-rising mock curves run off along the v-r_c degeneracy and dominate the plain std and correlation.
"""
import sys, numpy as np
N, SEED, BT, SI = int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])
sys.argv = [sys.argv[0], "0"]
import robustness as R
from scipy.stats import spearmanr
rng = np.random.default_rng(SEED)
L10 = np.log(10)

def one():
    a0 = np.median(R.lrc); lrt = a0 + BT*R.xc + rng.normal(0, SI, len(R.xc))
    vs, rcs, sxs, sys_, cs = [], [], [], [], []
    for g, lr in zip(R.GAL, lrt):
        v, rc = g["v"], 10**lr
        Vm = v*R.vn(g["R"]/rc) + rng.normal(0, 1, len(g["R"]))*g["e"]*np.sqrt(R.chi2/R.dof)
        v_, rc_, slv, slr, cvr, _ = R.fit(g["R"], Vm, g["e"])
        vs.append(v_); rcs.append(rc_); sxs.append(slv/L10); sys_.append(slr/L10); cs.append(cvr)
    x = np.log10(vs); y = np.log10(rcs); sx = np.minimum(sxs, 0.5); sy = np.minimum(sys_, 0.5)
    p = R.ml_y(x-np.median(x), y, sx, sy, np.array(cs)*sx*sy)
    mad = 1.4826*np.median(np.abs(y-np.median(y)))
    return p[1], y.std(), np.corrcoef(x, y)[0, 1], mad, spearmanr(x, y)[0]

M = np.array([one() for _ in range(N)])
np.save(f"mock_calib_{BT:.1f}_{SI:.2f}_{SEED}.npy", M)
print(f"b={BT} scatter={SI} seed={SEED}: N={N} median b {np.median(M[:,0]):.2f} "
      f"(16-84% {np.percentile(M[:,0],16):.2f}-{np.percentile(M[:,0],84):.2f}); spread {np.median(M[:,1]):.2f}; "
      f"corr {np.median(M[:,2]):.2f}; MAD spread {np.median(M[:,3]):.2f}; Spearman {np.median(M[:,4]):.2f}; "
      f"frac b<=0.97 {(M[:,0]<=0.97).mean():.4f}", flush=True)
