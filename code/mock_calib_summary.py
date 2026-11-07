"""Summary of the mock calibration (Appendix F; referee point M4).  python3 mock_calib_summary.py
Reads mock_calib_<b>_<scatter>_<seed>.npy written by mock_calib.py and compares with the data:
ML slope 0.97 (fit errors only, the estimator applied to the mocks), robust spread of log r_c
(1.4826 MAD) 0.32 dex and Spearman rank correlation of log v and log r_c 0.54."""
import numpy as np, glob
B, MAD, SP = 0.97, 0.324, 0.54
print(f"{'true b':>6} {'scatter':>7} {'N':>5} {'median b (16-84%)':>22} {'MAD':>5} {'Spearman':>8} {'P(b<=0.97)':>10} {'P(rho<=0.54)':>12}")
for f in sorted(glob.glob('mock_calib_*.npy')):
    _, _, bt, si, seed = f[:-4].split('_')
    M = np.load(f)
    rob = M.shape[1] >= 5
    print(f"{bt:>6} {si:>7} {len(M):5d} {np.median(M[:,0]):8.2f} ({np.percentile(M[:,0],16):.2f}-{np.percentile(M[:,0],84):.2f})"
          f" {np.median(M[:,3]) if rob else np.nan:5.2f} {np.median(M[:,4]) if rob else np.nan:8.2f}"
          f" {(M[:,0]<=B).mean():10.3f} {((M[:,4]<=SP).mean() if rob else np.nan):12.3f}")
