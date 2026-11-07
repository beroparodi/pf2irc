"""Mass excess c of a cored profile, and what it does to the core ratio and the epoch
(Sects. 2.3 and 4.3).

The solutions of u'' + 2u' + u = 1 with u(0) = 0 (finite central density) are
    u_a = 1 + (a x - 1) e^{-x},   u_a'(0) = 1 + a,   a = 1: P18.
Their mass excess is c(a) = int_0^inf (u_a^2 - 1) dx = a^2/4 + 3a/2 - 3/2, and the first-order
part of the departure from the SIS, 2(u_a - 1), carries the mass 2(a - 1): only P18 conserves
mass at first order, and its excess is purely second order, int (u - 1)^2 dx = 1/4.

For each profile: c and the core constant C_E (-G U/(v^4 r_c) -> y + c ln y + C_E), the core
ratio y from c ln y + A y = c(1 - 3 lambda^2) - C_E, and the end of growth that reproduces the
observed median v/r_c = 28.7 km/s/kpc (nominal choices of Table 1: 2/3 Lambda term, spin on top,
epsilon = 0, lambda = 0.035). Units: kpc, km/s; time in kpc/(km/s) = 0.9778 Gyr.
"""
import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.special import lambertw
from scipy.optimize import brentq
H0, Om, OL = 0.07, 0.3, 0.7
L3, K, CL, LAM, VRC_OBS = OL*H0**2, 1 + 1.5*np.pi, 9.640, 0.035, 28.7

def t_of_z(z):
    a = 1/(1 + z)
    return 2/(3*H0*np.sqrt(OL))*np.arcsinh(np.sqrt(OL/Om)*a**1.5)

def c_CE(mp, X=4000.0):
    """Mass excess c and core constant C_E for a profile with m'(x) = mp(x)."""
    x = np.concatenate([np.linspace(0, 1e-3, 200)[:-1], np.geomspace(1e-3, X, 400000)])
    d = mp(x); m = cumulative_trapezoid(d, x, initial=0)
    c = m[-1] - x[-1]
    I = cumulative_trapezoid(np.where(x > 0, m*d/np.maximum(x, 1e-300), 0), x, initial=0)
    far = x > 200
    return c, np.median(I[far] - x[far] - c*np.log(x[far]))

def y_core(c, CE, ell, lam=LAM):
    A = 3*lam**2 + 2*ell/3
    if abs(c) < 1e-6:
        return -CE/A                                    # linear case, c = 0
    return (c/A)*lambertw((A/c)*np.exp((c*(1 - 3*lam**2) - CE)/c)).real

def vrc(c, CE, z):
    """v/r_c [km/s/kpc] for growth ending at z, and the core ratio y."""
    t = t_of_z(z); l0 = L3*t**2/K**2; ell = l0/(1 + CL*l0)**2
    y = y_core(c, CE, ell)
    return y*(1 + CL*l0)*K/t, y

def z_end(c, CE, obs=VRC_OBS):
    f = lambda z: vrc(c, CE, z)[0] - obs
    return brentq(f, 0, 20) if f(0) < 0 else None      # None: no epoch up to today

u_a = lambda a: (lambda x: (1 + (a*x - 1)*np.exp(-x))**2)

if __name__ == "__main__":
    print("a       c (num)  c (formula)  C_E      y(z=2)  y(z=0.5)  z_end   v/rc if growth ends today")
    for cc in (0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30):
        a = -3 + np.sqrt(9 + 4*(1.5 + cc))                # inverse of c(a)
        c, CE = c_CE(u_a(a))
        ze = z_end(c, CE)
        print(f"{a:.4f}  {c:7.4f}  {a*a/4 + 1.5*a - 1.5:7.4f}     {CE:7.4f}  {vrc(c, CE, 2)[1]:6.1f}  "
              f"{vrc(c, CE, 0.5)[1]:7.1f}   {'none' if ze is None else f'{ze:.2f}':>5}   {vrc(c, CE, 0.0)[0]:.1f}")
    # how much the curve changes: v_c of u_a against P18 at the same v and r_c
    x = np.linspace(1e-3, 15, 20000)
    def vc(a):
        m = cumulative_trapezoid(u_a(a)(x), x, initial=0) + (1 + a)**2*x[0]**3/3
        return np.sqrt(m/x)
    for a in (0.975, 1.025):
        print(f"a = {a}: max |v_c/v_c(P18) - 1| for x <= 15 at fixed v, r_c: {np.max(np.abs(vc(a)/vc(1.0) - 1)):.4f}")
    from scipy.integrate import quad
    for a in (0.975, 1.0, 1.025):
        first = 2*quad(lambda s: (a*s - 1)*np.exp(-s), 0, np.inf)[0]
        second = quad(lambda s: ((a*s - 1)*np.exp(-s))**2, 0, np.inf)[0]
        print(f"a = {a}: first-order mass 2 int(u_a - 1) = {first:+.4f} (= 2(a-1)), "
              f"second order int(u_a - 1)^2 = {second:.4f}, sum = c = {first + second:.4f}")
    # threshold: the c for which growth ending today gives the observed median v/r_c
    def g(cc):
        a = -3 + np.sqrt(9 + 4*(1.5 + cc)); c, CE = c_CE(u_a(a)); return vrc(c, CE, 0.0)[0] - VRC_OBS
    print(f"smallest c that reaches the observed median v/r_c by today: {brentq(g, 0.0, 0.15):.3f}")
