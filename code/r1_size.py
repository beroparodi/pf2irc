"""Stellar size R1 (radius where Sigma_* = 1 Msun/pc^2) of the 79 SPARC dwarfs
against the fitted core radius r_c and the disc scale length R_d (Sect. 6.5).

R1 is the threshold size of Trujillo et al. (2020) and Brook et al. (2026,
arXiv:2610.35366). Sigma_* = 0.5 SBdisk + 0.7 SBbul from the 3.6 micron
profiles in the SPARC rotmod files (L/pc^2 -> Msun/pc^2). Where the profile
reaches the threshold, R1 is interpolated log-linearly; otherwise the outer
half of the profile is extrapolated as an exponential.
Run from code/:  python3 r1_size.py   (output r1_size_out.txt)
"""
import csv, collections
import numpy as np

UPS_D, UPS_B, SIG1 = 0.5, 0.7, 1.0
fits = {r["name"]: r for r in csv.DictReader(open("sparc_data/sparc79_fits.csv"))}
# R_d from Parodi (2018), Table B1; its rows are in the order of Table B2,
# which carries the SPARC names (as in make_table.py)
B1 = list(csv.DictReader(open("sparc_data/parodi2018_B1.csv")))
B2 = list(csv.DictReader(open("sparc_data/parodi2018_B2.csv")))
disc = {b2["name"].replace("UGC0786", "UGC07866"): b1 for b1, b2 in zip(B1, B2)}
assert set(disc) == set(fits), set(fits) ^ set(disc)

rows = []
for g, r in fits.items():
    v, rc, Rd = float(r["v"]), float(r["rc"]), float(disc[g]["Rd"])
    d = np.loadtxt(f"sparc_data/rotmod/{g}_rotmod.dat")
    R, S = d[:, 0], UPS_D * d[:, 6] + UPS_B * d[:, 7]
    R, S = R[S > 0], S[S > 0]
    below = np.where(S < SIG1)[0]
    if len(below) and below[0] > 0:
        i = below[0]
        R1 = R[i-1] + (R[i] - R[i-1]) * np.log(S[i-1] / SIG1) / np.log(S[i-1] / S[i])
        how = "data"
    else:
        m = R >= 0.5 * R.max()
        p = np.polyfit(R[m], np.log(S[m]), 1)
        R1 = (np.log(SIG1) - p[1]) / p[0]
        how = "extrapolated"
    rows.append((g, v, rc, Rd, R1, how))

out = open("r1_size_out.txt", "w")
def P(*a):
    print(*a); print(*a, file=out)

P(f"galaxies: {len(rows)}; R1 from", dict(collections.Counter(r[5] for r in rows)))
P(f"Upsilon_disc = {UPS_D}, Upsilon_bulge = {UPS_B}, threshold {SIG1} Msun/pc^2\n")
P(f"{'galaxy':12s} {'v':>7s} {'r_c':>6s} {'R_d':>6s} {'R1':>6s}  R1 from")
for g, v, rc, Rd, R1, how in rows:
    P(f"{g:12s} {v:7.1f} {rc:6.2f} {Rd:6.2f} {R1:6.2f}  {how}")

def s(x): return np.std(np.log10(x), ddof=1)
for lab, sel in [("all 79", [True] * len(rows)), ("R1 from the data only", [r[5] == "data" for r in rows])]:
    sel = np.array(sel)
    v, rc, Rd, R1 = (np.array([r[k] for r in rows])[sel] for k in (1, 2, 3, 4))
    P(f"\n{lab}: N = {sel.sum()}")
    for nm, x in [("R1/r_c", R1 / rc), ("R_d/r_c", Rd / rc), ("R1/R_d", R1 / Rd)]:
        P(f"  {nm:8s} median {np.median(x):5.2f}   scatter of log {s(x):.3f} dex")
    for nm, y in [("r_c", rc), ("R_d", Rd), ("R1", R1)]:
        p = np.polyfit(np.log10(v), np.log10(y), 1)
        res = np.log10(y) - np.polyval(p, np.log10(v))
        P(f"  {nm:4s} against v: slope {p[0]:.2f}, scatter {np.std(res, ddof=2):.3f} dex")
out.close()
