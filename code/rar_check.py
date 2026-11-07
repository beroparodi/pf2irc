"""The radial acceleration relation and MOND against the P18 fits of the 79 dwarfs (referee point M5; Sect. 6.6).

    python3 rar_check.py

(1) Per galaxy, the asymptotic mass excess the RAR implies, c_RAR = G M_b / (2 v^2 r_c), with M_b the baryonic
    mass inside the last measured radius (Upsilon_disk = 0.5 at 3.6 micron, bulge 0.7) or from the baryonic
    Tully-Fisher relation, M_b = v^4/(G g_dagger). P18 uses one c = 1/4 for all galaxies.
(2) The asymptotic excess for different interpolating functions nu(y), g = g_b nu(g_b/g_dagger):
    RAR (McGaugh et al. 2016) and 'simple': nu ~ y^-1/2 + 1/2  -> excess M_b/2; 'standard': no constant term.
(3) The MOND central surface density relation (Milgrom 2016; Lelli et al. 2016): Sigma_dyn = Sigma_M S(Sigma_b/Sigma_M),
    S(y) = int_0^y nu(y') dy', Sigma_M = g_dagger/(2 pi G). Sigma_dyn of each galaxy from its IRC fit with Toomre's
    formula Sigma_dyn = (1/2 pi G) int_0^inf V^2/R^2 dR; Sigma_b from the central stellar surface brightness (SB0,
    Upsilon = 0.5), as in Lelli et al. (2016).
(4) The peak of the total acceleration g = (v^2/r_c) m(x)/x^2 of P18 against g_dagger, the threshold below which
    stellar feedback ejects most of the gas (Grudic et al. 2020, MNRAS 496, L127). A common c in Eq. (29) gives
    v^2/r_c = 2 c g_dagger, so g_max = 0.96 c g_dagger.
"""
import csv, numpy as np
from scipy.integrate import quad

G = 4.30091e-3                       # pc (km/s)^2 / Msun
GD = 1.2e-10 * 3.0857e16 / 1e6       # g_dagger in (km/s)^2/pc  (1.2e-10 m/s^2 = 3.70 (km/s)^2/pc)
SIG_M = GD / (2 * np.pi * G)         # Msun/pc^2
U_D, U_B = 0.5, 0.7

fits = list(csv.DictReader(open("sparc_data/sparc79_fits.csv")))
u = lambda x: 1 - (1 - x) * np.exp(-x)
m = lambda x: quad(lambda t: u(t)**2, 0, x)[0]
I_toomre = quad(lambda x: m(x) / x**3, 1e-6, 50)[0] + quad(lambda x: (x + 0.25) / x**3, 50, np.inf)[0]

nu = lambda y: 1 / (1 - np.exp(-np.sqrt(y)))
S = lambda y: quad(nu, 0, y, limit=200)[0]

rows = []
for f in fits:
    d = np.loadtxt(f"sparc_data/rotmod/{f['name']}_rotmod.dat")
    R, Vg, Vd, Vb = d[-1, 0], d[-1, 3], d[-1, 4], d[-1, 5]
    Vbar2 = Vg * abs(Vg) + U_D * Vd**2 + U_B * Vb**2
    v, rc = float(f["v"]), float(f["rc"]) * 1e3           # km/s, pc
    Mb = R * 1e3 * Vbar2 / G                              # Msun
    Mb_btf = v**4 / (G * GD)
    c_last = G * Mb / (2 * v**2 * rc)
    c_btf = G * Mb_btf / (2 * v**2 * rc)
    sig_tot = v**2 / (np.pi * G * rc)                     # rho(0) r_c, Msun/pc^2
    sig_dyn = v**2 / (2 * np.pi * G * rc) * I_toomre
    sig_b = U_D * float(f["SB0"]) if f["SB0"] not in ("", "nan") else np.nan
    sig_mond = SIG_M * S(sig_b / SIG_M) if np.isfinite(sig_b) and sig_b > 0 else np.nan
    rows.append((f["name"], v, rc, c_last, c_btf, sig_tot, sig_dyn, sig_b, sig_mond))

A = np.array([r[1:] for r in rows], float)
v, rc, cl, cb, st, sd, sb, sm = A.T
lv = np.log10(v)
def pc(a): a = a[np.isfinite(a)]; return np.percentile(a, [16, 50, 84])
print(f"Sigma_M = g_dagger/(2 pi G) = {SIG_M:.0f} Msun/pc^2; Toomre integral = {I_toomre:.3f} "
      f"(Sigma_dyn = {I_toomre/2:.3f} rho(0) r_c)")
print(f"(1) c_RAR from M_b at the last point: 16/50/84% = {pc(cl).round(2)}; slope dlog c/dlog v = "
      f"{np.polyfit(lv, np.log10(cl), 1)[0]:+.2f}; scatter {np.log10(cl).std():.2f} dex")
print(f"    c_RAR from the BTFR:              16/50/84% = {pc(cb).round(2)}; slope {np.polyfit(lv, np.log10(cb), 1)[0]:+.2f}; "
      f"scatter {np.log10(cb).std():.2f} dex")
print(f"    P18 uses c = 0.25 for every galaxy")
for name, fn in [("RAR/McGaugh", lambda y: 1/(1-np.exp(-np.sqrt(y)))),
                 ("simple", lambda y: 0.5 + np.sqrt(0.25 + 1/y)),
                 ("standard", lambda y: np.sqrt(0.5 + np.sqrt(0.25 + 1/y**2)))]:
    y = 1e-6
    print(f"(2) {name:12s}: nu(y) - y^-1/2 at y = 1e-6: {fn(y) - y**-0.5:+.3f}")
k = np.isfinite(sm) & np.isfinite(sb) & (sb > 0)
res = np.log10(sd[k] / sm[k])
print(f"(3) CSDR, N = {k.sum()}: log Sigma_dyn - log Sigma_MOND: median {np.median(res):+.2f}, scatter {res.std():.2f} dex; "
      f"Sigma_b0 16/50/84% = {pc(sb[k]).round(0)} Msun/pc^2")
print(f"    total Sigma = rho(0) r_c against Sigma_b0: slope dlog Sigma/dlog Sigma_b = "
      f"{np.polyfit(np.log10(sb[k]), np.log10(st[k]), 1)[0]:+.2f}, corr {np.corrcoef(np.log10(sb[k]), np.log10(st[k]))[0,1]:+.2f}")
print(f"    MOND prediction against Sigma_b0 over the sample: slope "
      f"{np.polyfit(np.log10(sb[k]), np.log10(sm[k]), 1)[0]:+.2f}")

from scipy.optimize import minimize_scalar
pk = minimize_scalar(lambda x: -m(x) / x**2, bounds=(0.2, 5), method="bounded")
gfac, xpk = -pk.fun, pk.x
gmax = gfac * v**2 / rc / GD                                # g_max / g_dagger
print(f"(4) g = GM/R^2 peaks at x = {xpk:.2f} with g_max = {gfac:.3f} v^2/r_c (m(1) = {m(1):.3f}); "
      f"common c = 1/4: g_max = {gfac*2*0.25:.2f} g_dagger")
print(f"    79 dwarfs: g_max/g_dagger 16/50/84% = {pc(gmax).round(2)}; above g_dagger: {(gmax > 1).sum()} of {len(gmax)}; "
      f"g_max ~ v^{np.polyfit(lv, np.log10(gmax), 1)[0]:.2f}")
