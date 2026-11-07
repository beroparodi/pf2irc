# Test particles falling from rest at r=2 R_b into a flattened logarithmic (flat-curve) potential
# Phi = v^2/2 ln(rc^2 + R^2 + z^2/q(t)^2), units v=1, R_b=1 (so GM/v = 1 = j of a circular orbit at R_b)
import numpy as np
from scipy.integrate import solve_ivp
rc=0.01
def run(lam_deg, qfun, tmax=40):
    lam=np.radians(lam_deg)
    y0=[2*np.cos(lam),2*np.sin(lam),0,0]   # meridional plane: L_z=0 stays 0 in axisymmetry
    def f(t,y):
        x,z,vx,vz=y; q=qfun(t); D=rc**2+x*x+z*z/q**2
        return [vx,vz,-x/D,-z/q**2/D]
    s=solve_ivp(f,(0,tmax),y0,rtol=1e-9,atol=1e-11,dense_output=True,max_step=0.01)
    t=s.t; x,z,vx,vz=s.y; j=x*vz-z*vx; r=np.hypot(x,z)
    return t,r,j
def qconst(q): return lambda t: q
def qgrow(qmin,T): return lambda t: 1-(1-qmin)*min(t/T,1)
tff=None
for lab,qf in [("q=0.9",qconst(0.9)),("q=0.8",qconst(0.8)),("q=0.7",qconst(0.7)),
               ("q:1->0.7 over 10",qgrow(0.7,10)),("q:1->0.7 over 30",qgrow(0.7,30))]:
    for lam in [20,45,70]:
        t,r,j=run(lam,qf)
        # first pericentre passage: first local min of r
        i=np.where((r[1:-1]<r[:-2])&(r[1:-1]<r[2:]))[0]
        jfirst=abs(j[i[0]+1]) if len(i) else np.nan
        print(f"{lab:18s} lat={lam:2d}  |j| at 1st peri={jfirst:5.3f}  max|j| (t<40)={np.max(abs(j)):5.3f}  t_first_peri={t[i[0]+1] if len(i) else np.nan:4.2f}")
