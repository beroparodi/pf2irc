"""Baryon share of V^2 at r_c for the 79 dwarfs (referee minor point 24; Table E.1 and Appendix G).

    python3 baryon_share.py      -> baryon_share_out.txt

V_bar^2 = V_gas|V_gas| + 0.5 V_disk^2 + 0.7 V_bul^2 (3.6 micron), interpolated linearly in the SPARC rotmod
file at R = r_c (taken from the innermost point if r_c lies inside it); divided by the IRC model
V^2(r_c) = v^2 u(1)^2... evaluated as v^2 m(1)/1 with m(x) = int_0^x u^2 dt, u = 1 - (1-x) e^-x.
Galaxies with a share above 1/2 are baryon-dominated at r_c and are marked in Table E.1.
"""
import csv, numpy as np
from scipy.integrate import quad
u = lambda x: 1 - (1 - x) * np.exp(-x)
m1 = quad(lambda t: u(t)**2, 0, 1)[0]
out = []
for f in csv.DictReader(open("sparc_data/sparc79_fits.csv")):
    d = np.loadtxt(f"sparc_data/rotmod/{f['name']}_rotmod.dat")
    R, Vg, Vd, Vb = d[:, 0], d[:, 3], d[:, 4], d[:, 5]
    vb2 = Vg * np.abs(Vg) + 0.5 * Vd**2 + 0.7 * Vb**2
    v, rc = float(f["v"]), float(f["rc"])
    share = np.interp(rc, R, vb2) / (v**2 * m1)
    out.append((f["name"], share))
s = np.array([o[1] for o in out])
with open("baryon_share_out.txt", "w") as fo:
    fo.write(f"baryon share of V^2 at r_c: 16/50/84% = {np.percentile(s,[16,50,84]).round(2)}; "
             f"above 1/2: {(s>0.5).sum()} of {len(s)}\n")
    for n, x in out: fo.write(f"{n:10s} {x:.2f}\n")
print(open("baryon_share_out.txt").read())
