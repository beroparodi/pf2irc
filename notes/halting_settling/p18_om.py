"""
Osipkov-Merritt DFs of P18 with analytic second derivatives (no numerical noise).
f(Q) = 1/(2 sqrt2 pi^2) int_Q^inf d2 rho_Q/dPhi2 dPhi/sqrt(Phi-Q),  rho_Q = (1+r^2/r_a^2) rho.
Positivity is required (a) for all Q up to the circular energy at R_b = 34-200 r_c, (b) globally.
"""
import numpy as np
from p18core import u,m,rho,cu
from scipy.integrate import cumulative_trapezoid
out=open("p18_om_out.txt","w")
def P(s): print(s); out.write(s+"\n")
lnr=np.linspace(np.log(1e-5),np.log(1e9),200001); r=np.exp(lnr)
Phi=np.concatenate([[0],cumulative_trapezoid(m(r)/r,lnr)])+(2/3)*r[0]**2
cdu=np.array([n*cu[n] for n in range(1,len(cu))]); cddu=np.array([n*cdu[n] for n in range(1,len(cdu))])
def du(x): return np.where(x<0.05,np.polyval(cdu[::-1],x),(2-x)*np.exp(-x))
def ddu(x): return np.where(x<0.05,np.polyval(cddu[::-1],x),(x-3)*np.exp(-x))
U,U1,U2,M=u(r),du(r),ddu(r),m(r)
# w = u/r with series-safe forms: w' = (u' r - u)/r^2, w'' = u''/r - 2u'/r^2 + 2u/r^3
# u' r - u and its pieces cancel at small r: use series for (u'r-u)/r^2 = sum (n-1) cu_n x^(n-2)
cw1=np.array([(n-1)*cu[n] for n in range(2,len(cu))])          # coefficients of x^(n-2)
cw2=np.array([(n-1)*(n-2)*cu[n] for n in range(3,len(cu))])     # w'' coefficients of x^(n-3)
small=r<0.05
W=np.where(small,np.polyval(cu[1:][::-1],r),U/r)
W1=np.where(small,np.polyval(cw1[::-1],r),(U1*r-U)/r**2)
W2=np.where(small,np.polyval(cw2[::-1],r),U2/r-2*U1/r**2+2*U/r**3)
rho0=W*W/(4*np.pi); rho1=W*W1/(2*np.pi); rho2=(W1*W1+W*W2)/(2*np.pi)
dPhi=M/r**2; d2Phi=U*U/r**2-2*M/r**3   # Phi'' = m'/r^2 - 2m/r^3
def G2(ra):
    a=0 if ra is None else 1/ra**2
    q0=rho0*(1+a*r*r); q1=rho1*(1+a*r*r)+rho0*2*a*r; q2=rho2*(1+a*r*r)+4*a*r*rho1+2*a*rho0
    # d/dPhi = (1/Phi') d/dr ;  d2/dPhi2 = (q2 Phi' - q1 Phi'')/Phi'^3
    return (q2*dPhi-q1*d2Phi)/dPhi**3
def f_of(ra,Es):
    g=G2(ra); res=[]
    for E in Es:
        tmax=np.sqrt(Phi[-1]-E)
        t=np.concatenate([np.linspace(0,1,4001),np.linspace(1,tmax,20001)[1:]])
        res.append(np.trapezoid(2*np.interp(E+t*t,Phi,g),t))
    return np.array(res)/(2*np.sqrt(2)*np.pi**2)
C=Phi[-1]-np.log(r[-1])+0.25/r[-1]
fS=lambda E: np.exp(-2*(E-C))*np.sqrt(np.pi/2)/(2*np.sqrt(2)*np.pi**3)
Ec=Phi+0.5*m(r)/r
E_of_R=lambda R: np.interp(R,r,Ec)
Es=np.concatenate([np.geomspace(1e-6,1,200),np.linspace(1.02,14,300)])
fI=f_of(None,Es); P(f"isotropic: min f/f_SIS = {np.min(fI/fS(Es)):.4f}")
for ra in [0.3,0.5,0.7,1.0,1.5,2,3,5,10]:
    f=f_of(ra,Es); q=f/fS(Es)
    neg=Es[q<0]
    s=f"Q-Phi0 in [{neg.min():.3g}, {neg.max():.3g}] (R_circ {np.interp(neg.min(),Ec,r):.3g}-{np.interp(neg.max(),Ec,r):.3g} r_c)" if len(neg) else "none"
    P(f"r_a = {ra:5.1f} r_c: min f/f_SIS = {q.min():+.3e};  negative at {s}")
def ok(ra,Emax):
    k=Es<=Emax; f=f_of(ra,Es[k]); return np.all(f>0)
for Rb in [34,200,1e6]:
    Emax=E_of_R(Rb); lo,hi=0.05,20
    if not ok(hi,Emax): P(f"R up to {Rb}: no r_a <= 20 works"); continue
    for k in range(22):
        mid=np.sqrt(lo*hi)
        if ok(mid,Emax): hi=mid
        else: lo=mid
    P(f"positivity for energies up to the circular energy at {Rb:g} r_c: r_a >= {hi:.3f} r_c "
      f"(beta = {1/(1+hi**2):.2f} at r_c, {4/(4+hi**2):.2f} at 2 r_c, {25/(25+hi**2):.2f} at 5 r_c)")
np.savez("p18_om.npz",Es=Es,fI=fI,fS=fS(Es))
out.close()
