# Energy change and apocentres for the growing-flattening runs (same set-up as axi_test.py)
import numpy as np
from scipy.integrate import solve_ivp
rc=0.01
def run(lam_deg,qmin,T,tmax=60):
    lam=np.radians(lam_deg); y0=[2*np.cos(lam),2*np.sin(lam),0,0]
    qf=lambda t: 1-(1-qmin)*min(t/T,1)
    def f(t,y):
        x,z,vx,vz=y; q=qf(t); D=rc**2+x*x+z*z/q**2
        return [vx,vz,-x/D,-z/q**2/D]
    s=solve_ivp(f,(0,tmax),y0,rtol=1e-9,atol=1e-11,max_step=0.01)
    x,z,vx,vz=s.y; q=np.array([qf(t) for t in s.t])
    E=0.5*(vx**2+vz**2)+0.5*np.log(rc**2+x*x+z*z/q**2)
    r=np.hypot(x,z); late=s.t>tmax-15
    return E[0],E[-1],r[late].min(),r[late].max()
for qmin,T in [(0.8,3),(0.8,10),(0.7,3),(0.7,10)]:
    for lam in [20,45,70]:
        E0,E1,rmin,rmax=run(lam,qmin,T)
        print(f"q 1->{qmin} over {T:2d}: lat={lam:2d}  dE={E1-E0:+.3f} v^2  late r range {rmin:.2f}-{rmax:.2f} R_b")
