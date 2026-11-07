"""
Which orbits turn the singular isothermal sphere (SIS) into P18?
Units G = v = r_c = 1.  rho = u^2/(4 pi x^2), u = 1 + (x-1) e^-x,  M = m(x).
1. SIS -> P18 as a redistribution of mass (exact).
2. Circular orbits: required j(M) = GM/v_c.
(Distribution functions: p18_df.py, p18_om.py.)
"""
import numpy as np
from scipy.integrate import quad, cumulative_trapezoid
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq
out=open("p18_inverse_out.txt","w")
def P(*a):
    s=" ".join(str(x) for x in a); print(s); out.write(s+"\n")
u=lambda x: 1+(x-1)*np.exp(-x)
m=lambda x: x*(-np.expm1(-x))**2+0.25*(1-(1+2*x+2*x*x)*np.exp(-2*x))
# ---------- 1. redistribution
m1=m(1.0)
P("1. SIS -> P18 as redistribution of mass (units v^2 r_c/G)")
P(f"   u-1=(x-1)e^-x: rho_P18 = rho_SIS exactly at x=1; below inside, above outside; max ratio u^2={u(2.0)**2:.4f} at x=2")
P(f"   mass removed inside r_c: 1-m(1) = {1-m1:.4f};  added outside: {1-m1+0.25:.4f};  net excess c = 0.25")
xs=np.array([0.1,0.3,0.5,1,2,3,5,10])
P("   x      m(x)     x-m(x)   (mass still missing inside x)")
for x in xs: P(f"   {x:5.1f} {m(x):9.5f} {x-m(x):+9.5f}")
xz=brentq(lambda x: x-m(x),2,10); P(f"   M_P18 = M_SIS again at x = {xz:.3f}")
# ---------- 2. circular orbits
P("\n2. Circular orbits: j(M) = GM/v_c(R(M)); in units of the SIS value GM/v: sqrt(x/m)")
lam=0.035; jseed=2*np.sqrt(2)*lam
P(f"   seed spin gives j = {jseed:.3f} GM/v (lambda = 0.035)")
P("   x      v_c/v    j/(GM/v)   j/j_seed")
for x in xs:
    r=np.sqrt(x/m(x)); P(f"   {x:5.1f} {1/r:8.4f} {r:9.3f} {r/jseed:9.1f}")
out.close()
