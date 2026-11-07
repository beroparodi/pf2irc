"""Accuracy of the first-order Lambda lag of the front, Eq. (11) (second-round referee, minor 19).

    python3 lag_check.py      -> lag_check_out.txt

Units G = M = v = 1 for one shell: Rdot^2 = 2/R - 1 + q R^2 with q = (Lambda/3)(GM/v^3)^2; the shell halts on
its way back at R = GM/v^2 (the y -> infinity limit of Eq. 9; the halting condition involves only masses).
For each epoch the shell halting at t is found exactly (quadrature of dt = dR/Rdot), giving the exact front
R_b = GM/v^2, and compared with R_b = xi_b v t/(1 + c_Lambda ell_0).
"""
import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
# units G = M = v = 1: Rdot^2 = 2/R - 1 + q R^2 ; halt at R = 1 on the way back
def tau(q):
    f = lambda R: 2/R - 1 + q*R*R
    Rta = brentq(f, 1.0, q**(-1/3)) if q > 0 else 2.0
    # time 0 -> Rta: substitute R = Rta (1 - s^2), dR = -2 Rta s ds
    g = lambda s: 2*Rta*s/np.sqrt(f(Rta*(1-s*s))) if s > 0 else 2*Rta/np.sqrt(-(-2/Rta**2 + 2*q*Rta)*Rta)
    up = quad(g, 0, 1, limit=200)[0]
    s1 = np.sqrt(1 - 1/Rta)
    down = quad(g, 0, s1, limit=200)[0]
    return up + down
t0 = tau(0); out = [f"tau(0) = {t0:.10f}, expected 1 + 3 pi/2 = {1+1.5*np.pi:.10f}"]
H0, Om, OL = 0.07, 0.3, 0.7; L3 = OL*H0**2; K = 1+1.5*np.pi; CL = 9.640
def t_of_z(z):
    a = 1/(1+z); return 2/(3*H0*np.sqrt(OL))*np.arcsinh(np.sqrt(OL/Om)*a**1.5)
v = 60.0
for z in (3, 2, 1, 0.5, 0.25, 0.0):
    t = t_of_z(z)
    # exact: find M with t = tau(q) G M / v^3, q = L3 (GM/v^3)^2 ; R_b = GM/v^2
    T = lambda m: tau(L3*m*m)*m - t          # m = GM/v^3 in time units
    m = brentq(T, 1e-3, t/t0)
    Rb_exact = m*v
    l0 = L3*t*t/K**2
    Rb_first = v*t/(K*(1+CL*l0))
    out.append(f"z={z}: l0={l0:.4f}  Rb exact {Rb_exact:.3f}  first-order {Rb_first:.3f}  ratio {Rb_first/Rb_exact:.4f}")

open("lag_check_out.txt", "w").write("\n".join(out) + "\n"); print("\n".join(out))
