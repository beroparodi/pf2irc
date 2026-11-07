# How close is the isotropic P18 to an isothermal (Maxwellian, sigma^2 = v^2/2) system?
import numpy as np
from p18core import u,m,rho
from scipy.integrate import cumulative_trapezoid
d=np.load("p18_om.npz"); Es,fI,fS=d["Es"],d["fI"],d["fS"]
lnr=np.linspace(np.log(1e-5),np.log(1e9),200001); r=np.exp(lnr)
Phi=np.concatenate([[0],cumulative_trapezoid(m(r)/r,lnr)])+(2/3)*r[0]**2
Ec=Phi+0.5*m(r)/r
out=open("p18_iso_compare_out.txt","w")
def P(s): print(s); out.write(s+"\n")
P("f_P18(E) / f_iso(E), f_iso = A exp(-2E/v^2) with A fixed by the far field (= SIS):")
P("  E-Phi(0)   R_circ/r_c   ratio")
for E in [1e-4,0.01,0.05,0.1,0.2,0.3,0.5,0.7,1,1.5,2,3,5,8]:
    i=np.argmin(abs(Es-E)); P(f"  {Es[i]:8.4f}  {np.interp(Es[i],Ec,r):9.3f}   {fI[i]/fS[i]:.4f}")
i=np.argmax(fI/fS); j=np.argmin(fI/fS)
P(f"  max {np.max(fI/fS):.4f} at R_circ = {np.interp(Es[i],Ec,r):.2f} r_c;  min {np.min(fI/fS):.4f} at R_circ = {np.interp(Es[j],Ec,r):.2f} r_c")
P("\nrho vs Phi: an isothermal sphere has rho = rho(0) exp(-2 Phi/v^2) at every radius")
P("  x     Phi(x)    rho/rho0    exp(-2 Phi)   ratio")
for x in [0.1,0.3,0.5,1,2,3,5,10,30]:
    ph=np.interp(x,r,Phi); q=rho(x)*np.pi
    P(f"  {x:5.1f} {ph:8.4f}  {q:9.4e}  {np.exp(-2*ph):9.4e}  {q/np.exp(-2*ph):.4f}")
out.close()
