"""Systematic band of the core ratio y = R_b/r_c and of the epoch (referee point M1, M6).

    python3 y_band.py

The core equation (Sect. 4.3) in the general form for a profile with mass excess c and core constant C_E:
    c ln y + A y = c (1 - s 3 lam^2 - eps) - C_E,   A = s 3 lam^2 + eps + k ell,   ell = Lambda R_b^2 / (3 v^2)
Variants:
  k   : how the Lambda term is booked. k = 2/3: the P18 density is kept and v_c^2 = v^2 - Lambda R^2/3 (paper);
        k = 1: the settled galaxy keeps v_c = v (density adjusted); k = 0: Lambda ignored in the balance.
  s   : how the spin energy is booked. s = 1: the rotational energy at turnaround comes on top of E = -v^2/2
        (paper, Eq. 16); s = 0: it is part of E = -v^2/2 (spin drops out).
  eps : a net energy of the settled galaxy relative to the balance, in units of v^2 M / 2; +-0.009 is the
        unrelaxed outer layer of Table 1 / Appendix D.
  n   : profile exponent (n = 2: P18; n = 1.76: best common exponent; n = 1: NFW-like cusp, c = 0, C_E = -3/4).
The front is the explicit growth law of Eq. (11) (as in the recipe of Sect. 5.3); the epoch is the redshift at
which the model v/r_c equals the observed median, 28.7 km/s/kpc. Cosmology as in the paper (flat, Om = 0.3, H0 = 70).
"""
import numpy as np
from scipy.special import lambertw
from scipy.integrate import quad
from scipy.optimize import brentq

H0, Om, OL = 0.07, 0.3, 0.7                    # H0 in km/s/kpc
L3, CL, K = OL * H0**2, 9.640, 1 + 1.5 * np.pi
GYR = 0.977792                                 # kpc/(km/s) in Gyr
VRC_OBS = 28.7
LAM = 0.035

def t_of_z(z):                                 # in kpc/(km/s)
    a = 1 / (1 + z)
    return 2 / (3 * H0 * np.sqrt(OL)) * np.arcsinh(np.sqrt(OL / Om) * a**1.5)

u = lambda x: 1 - (1 - x) * np.exp(-x)
def c_CE(n, X=60):
    c = quad(lambda x: u(x)**n - 1, 0, 200, limit=400)[0]
    m = lambda x: quad(lambda t: u(t)**n, 0, x, limit=200)[0]
    I = quad(lambda x: m(x) * u(x)**n / x, 1e-9, X, limit=400)[0]
    return c, I - X - c * np.log(X)
PROF = {2.0: (0.25, -0.91737), 1.76: c_CE(1.76), 1.0: (0.0, -0.75),
        "piso": (-np.pi / 2, -np.pi / 2 * (1 - np.log(2)))}   # pseudo-isothermal sphere, m = x - arctan x

def y_of(ell, lam=LAM, k=2/3, s=1, eps=0.0, n=2.0):
    c, CE = PROF[n]
    A = s * 3 * lam**2 + eps + k * ell
    Bp = c * (1 - s * 3 * lam**2 - eps) - CE
    if c == 0:                                 # n = 1: linear, A y = -C_E
        return -CE / A if A > 0 else np.nan
    if A == 0:
        return np.exp(Bp / c)
    arg = (A / c) * np.exp(Bp / c)
    if arg < -1 / np.e:
        return np.nan                          # no solution (net energy loss too large, Sect. 5.6)
    if c < 0:                                  # pseudo-isothermal: the large root (branch -1); the other is y ~ 2
        return (c / A) * lambertw(arg, -1).real
    return (c / A) * lambertw(arg).real        # principal branch: the small-y root for A < 0

def vrc_model(z, v=60.0, **kw):                # v/r_c at the end of growth (independent of v)
    t = t_of_z(z)
    l0 = L3 * t**2 / K**2
    Rb = v * t / (K * (1 + CL * l0))
    y = y_of(L3 * Rb**2 / v**2, **kw)
    return y * v / Rb, y

VRC_OBS_N = {2.0: 28.7, 1.76: 25.35}          # median v/r_c of the fits with that exponent (fit_exponent.py)
def z_end(**kw):
    vobs = VRC_OBS_N.get(kw.get("n", 2.0))
    if vobs is None: return np.nan                             # n = 1 fits the curves poorly (Sect. 7)
    f = lambda z: vrc_model(z, **kw)[0] - vobs
    zg = np.linspace(1e-4, 6, 601); fg = np.array([f(z) for z in zg])
    if not np.isfinite(fg[0]): return np.nan
    if fg[0] >= 0: return 0.0                                  # v/r_c falls with time; 0 = growth ends today
    i = np.where(np.isfinite(fg) & (fg > 0))[0]
    return brentq(f, zg[i[0]-1], zg[i[0]]) if len(i) else np.nan

variants = [
    ("paper: k=2/3, spin on top, n=2", {}),
    ("Lambda booked with k=1", {"k": 1.0}),
    ("spin inside E=-v^2/2 (s=0)", {"s": 0}),
    ("unrelaxed layer eps=+0.009", {"eps": 0.009}),
    ("unrelaxed layer eps=-0.009", {"eps": -0.009}),
    ("profile n=1.76", {"n": 1.76}),
    ("profile n=1 (NFW-like cusp)", {"n": 1.0}),
    ("pseudo-isothermal sphere", {"n": "piso"}),
    ("lambda=0.025", {"lam": 0.025}),
    ("lambda=0.050", {"lam": 0.050}),
]
print(f"{'variant':34s} {'y(z=2)':>7s} {'y(z=0.5)':>8s} {'v/rc z=2':>8s} {'v/rc z=3':>8s} {'z_end':>6s}")
for lab, kw in variants:
    print(f"{lab:34s} {vrc_model(2, **kw)[1]:7.1f} {vrc_model(0.5, **kw)[1]:8.1f} "
          f"{vrc_model(2, **kw)[0]:8.1f} {vrc_model(3, **kw)[0]:8.1f} {z_end(**kw):6.2f}")

# envelope over the booking choices, the unrelaxed layer and the profile exponent, at median spin.
# The unrelaxed layer enters with either sign (Table 2): eps = -0.009, 0, +0.009. A net loss of 0.009 has no
# P18 solution at early epochs (Sect. 5.4); such combinations are counted separately.
for EPS in ((0.0, 0.009), (-0.009, 0.0, 0.009)):
    ys, zs, v2, v3, nos2 = [], [], [], [], 0
    for k in (2/3, 1.0):
        for s_ in (0, 1):
            for eps in EPS:
                for n in (2.0, 1.76):
                    kw = dict(k=k, s=s_, eps=eps, n=n)
                    ys.append(vrc_model(0.5, **kw)[1]); zs.append(z_end(**kw))
                    a2, a3 = vrc_model(2, **kw)[0], vrc_model(3, **kw)[0]
                    if np.isfinite(a2): v2.append(a2)
                    else: nos2 += 1
                    if np.isfinite(a3): v3.append(a3)
    ys, zs = np.array(ys), np.array(zs)
    print(f"envelope (k = 2/3 or 1, s = 0 or 1, eps in {EPS}, n = 2 or 1.76; {len(ys)} combinations): "
          f"y(z=0.5) = {np.nanmin(ys):.0f}-{np.nanmax(ys):.0f}, z_end = {np.nanmin(zs):.2f}-{np.nanmax(zs):.2f}; "
          f"v/r_c at z=2: {min(v2):.0f}-{max(v2):.0f} ({nos2} without P18 solution), at z=3: {min(v3):.0f}-{max(v3):.0f}")
print(f"nominal v/r_c at z = 2 and 3: {vrc_model(2)[0]:.0f}, {vrc_model(3)[0]:.0f}")
for f in (0.5, 2/3, 1.5, 2.0):                 # the factor-two uncertainty of the absolute core
    z = brentq(lambda z: f * vrc_model(z)[0] - VRC_OBS, 1e-4, 10) if f * vrc_model(1e-4)[0] < VRC_OBS else 0.0
    print(f"  y x {f:.2f}: z_end = {z:.2f}")
