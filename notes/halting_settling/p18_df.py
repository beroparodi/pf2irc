"""
Distribution functions of the P18 profile in its own potential (units G = v = r_c = 1).
 - isotropic f(E) (Eddington); Osipkov-Merritt models are in p18_om.py
 - check by rebuilding rho from f
 - cumulative energy distribution M(<E) of the isotropic P18 vs the isotropic SIS
"""
import numpy as np
from p18core import u,m,rho,cu
from scipy.integrate import cumulative_trapezoid
out=open("p18_df_out.txt","w")
def P(s): print(s); out.write(s+"\n")
lnr=np.linspace(np.log(1e-5),np.log(1e9),300001); r=np.exp(lnr)
dPhi=m(r)/r**2
Phi=np.concatenate([[0],cumulative_trapezoid(dPhi*r,lnr)])+ (2/3)*r[0]**2  # Phi(0)=0
C=Phi[-1]-np.log(r[-1])+0.25/r[-1]
# derivative of u (series below 0.05)
cdu=np.array([n*cu[n] for n in range(1,len(cu))])
def du(x): return np.where(x<0.05,np.polyval(cdu[::-1],x),(2-x)*np.exp(-x))
def rho_q(ra):   # augmented density for OM: (1 + r^2/ra^2) rho
    return rho(r)*(1+(r/ra)**2 if ra else 1)
def drho_dr(ra):
    d=u(r)*(du(r)*r-u(r))/(2*np.pi*r**3)
    if ra: d=d*(1+(r/ra)**2)+rho(r)*2*r/ra**2
    return d
def f_of(ra,Es):
    G1=drho_dr(ra)/dPhi                     # d rho/d Phi
    G2=np.gradient(G1,lnr)/r/dPhi           # d2 rho/d Phi2
    res=[]
    for E in Es:
        tmax=np.sqrt(Phi[-1]-E); t=np.concatenate([np.linspace(0,1,4001),np.linspace(1,tmax,20001)[1:]]) if tmax>1 else np.linspace(0,tmax,4001)
        res.append(np.trapezoid(2*np.interp(E+t*t,Phi,G2),t))
    return np.array(res)/(2*np.sqrt(2)*np.pi**2)
Es=np.concatenate([np.geomspace(1e-7,1,240),np.linspace(1.02,18,425)])
fI=f_of(None,Es)
fS=lambda E: np.exp(-2*(E-C))*np.sqrt(np.pi/2)/(2*np.sqrt(2)*np.pi**3)   # isotropic SIS aligned far out
P(f"Isotropic P18: min f/f_SIS = {np.min(fI/fS(Es)):.4f} -> f positive everywhere: {np.all(fI>0)}")
# rebuild rho
lf=np.log(fI)
def rho_rebuilt(ph):
    t=np.concatenate([np.linspace(0,1,3001),np.linspace(1,np.sqrt(Es[-1]-ph),6001)[1:]])
    E=ph+t*t; return 4*np.pi*np.trapezoid(np.exp(np.interp(E,Es,lf))*np.sqrt(2)*2*t*t,t)
chk=[rho_rebuilt(np.interp(x,r,Phi))/rho(x) for x in [1e-3,0.01,0.1,1,3,10,100]]
P("rebuilt/true rho at r = 1e-3,0.01,0.1,1,3,10,100: "+" ".join(f"{c:.4f}" for c in chk))
# cumulative energy distribution: isotropic P18 vs SIS
def g_of_E(E,Ph,rr):  # density of states 16 pi^2 int r^2 sqrt(2(E-Phi)) dr
    k=Ph<E; return 16*np.pi**2*np.trapezoid(rr[k]**2*np.sqrt(2*np.maximum(E-Ph[k],0)),rr[k])
sel=slice(0,None,20)
NI=np.array([fI[i]*g_of_E(E,Phi[sel],r[sel]) for i,E in enumerate(Es)])
MI=cumulative_trapezoid(NI,Es,initial=0)
# SIS: N(E) analytic: f*g, g = 16 pi^2 r_E^3 sqrt2 Gamma(3/2)/3^1.5, r_E = exp(E-C)
from math import gamma
NS=lambda E: fS(E)*16*np.pi**2*np.exp(3*(E-C))*np.sqrt(2)*gamma(1.5)/3**1.5
MS=lambda E: NS(E)          # dM/dE = N ∝ e^{E}, so M(<E) = N(E)
vc2=m(r)/r; Ec=Phi+0.5*vc2; Rc=lambda E: np.interp(E,Ec,r)
P("\nIsotropic P18 vs isotropic SIS (potentials matched far out): mass with energy below E")
P("  E-Phi(0)  R_circ(E)  M_P18(<E)  M_SIS(<E)  ratio   [M_P18(<R_c), M_SIS(<R_c) for comparison]")
for E in [0.1,0.3,0.5,1,1.5,2,3,4,6]:
    i=np.argmin(abs(Es-E)); R=Rc(Es[i])
    P(f"  {Es[i]:7.3f}  {R:8.3f}  {MI[i]:9.4f}  {MS(Es[i]):9.4f}  {MI[i]/MS(Es[i]):6.3f}   [{m(R):.4f}, {R:.4f}]")
np.savez("p18_df.npz",Es=Es,fI=fI,fS=fS(Es),MI=MI,MS=MS(Es),Rc=Rc(Es))
out.close()
