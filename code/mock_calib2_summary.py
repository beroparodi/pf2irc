"""Summary of the mock calibration with three noise models (Appendix F, Table F.2; second-round referee M4).

    python3 mock_calib2_summary.py      -> mock_calib2_summary.txt

Reads mock2_<NOISE>_<BTRUE>_<SCATTER>_<SEED>.npy written by mock_calib2.py and gives, per set, the median
recovered slope (16-84%), the robust spread of log r_c, the Spearman rank correlation, and the fractions of
samples as extreme as the data (b <= 0.97; rho <= 0.54) with Clopper-Pearson 95% intervals.
"""
import glob, numpy as np
from scipy.stats import beta
def cp(k, n):                                   # Clopper-Pearson 95% interval
    lo = 0.0 if k == 0 else beta.ppf(0.025, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(0.975, k + 1, n - k)
    return lo, hi
rows = ["noise     b_true scatter    N  median b (16-84%)   spread  rank   p_b [95%]                p_rho [95%]"]
order = {"white069": 0, "white1": 1, "resid": 2}
files = sorted(glob.glob("mock2_*.npy"), key=lambda f: (order[f.split("_")[1]], -float(f.split("_")[2])))
for f in files:
    _, noise, bt, si, seed = f[:-4].split("_")
    M = np.load(f); n = len(M)
    kb = int((M[:, 0] <= 0.97).sum()); kr = int((M[:, 4] <= 0.54).sum())
    lb, hb = cp(kb, n); lr, hr = cp(kr, n)
    rows.append(f"{noise:9s} {float(bt):5.1f} {float(si):6.2f} {n:5d}  {np.median(M[:,0]):.2f} ({np.percentile(M[:,0],16):.2f}-{np.percentile(M[:,0],84):.2f})"
                f"    {np.median(M[:,3]):.2f}   {np.median(M[:,4]):.2f}   {kb/n:.3f} [{lb:.3f}-{hb:.3f}]    {kr/n:.3f} [{lr:.3f}-{hr:.3f}]")
open("mock_calib2_summary.txt", "w").write("\n".join(rows) + "\n"); print("\n".join(rows))
