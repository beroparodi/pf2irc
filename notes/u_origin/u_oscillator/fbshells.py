"""
Toy model: self-similar collisionless shells (as code/shells/shells.py, seed of the paper) plus a central
baryonic component whose size jumps repeatedly between compact and expanded ("bursts": gas blown out and
falling back). The baryons are prescribed, not evolved: Plummer sphere of mass Mb = fb * xi_f * t and radius
a = a_c * t (compact) or a_e * t (expanded); a cycle lasts Pfrac * t (uniform in ln t), compact for a fraction
`duty` of it. Everything scales with t, so the run stays self-similar. Units G = v = 1.
"""
import numpy as np, time
from scipy.optimize import brentq
def run(N=2000, Mmin=1e-4, Mmax=1.0, lam=0.035, Tend=np.pi, soft=2e-4, eta_dt=0.03, snaps=None,
        fb=0.0, xi_f=0.2, a_c=0.002, a_e=0.05, Pfrac=0.05, duty=0.5):
    M=np.geomspace(Mmin,Mmax,N); dm=np.diff(np.concatenate([[0.0],M])); Mf=M-0.5*dm
    Rta=2*Mf; tta=np.pi*Mf; j=np.sqrt(2)*lam*Rta
    t=tta[0]*1.0001
    def eta_of(Mi,t):
        g=lambda e: Mi*(e-np.sin(e))-t
        return brentq(g,1e-9,2*np.pi-1e-9) if t<2*np.pi*Mi else 2*np.pi-1e-9
    eta=np.array([eta_of(Mi,t) for Mi in Mf])
    r=Mf*(1-np.cos(eta)); vr=np.sin(eta)/(1-np.cos(eta))
    L=np.where(t>=tta,j,0.0)
    snaps=sorted(snaps or [Tend]); out={}; si=0
    def bary(t):
        if fb<=0: return 0.0,1.0
        ph=(np.log(t)/Pfrac)%1.0
        return fb*xi_f*t, (a_c if ph<duty else a_e)*t
    def acc(r,L,t):
        o=np.argsort(r); cm=np.cumsum(dm[o]); Menc=np.empty_like(cm); Menc[o]=cm-0.5*dm[o]
        rs=np.sqrt(r*r+(soft*t)**2)
        a=-Menc*r/rs**3 + L*L/np.maximum(r,1e-12)**3
        Mb,ab=bary(t)
        if Mb>0: a-= Mb*r/(r*r+ab*ab)**1.5
        return a,Menc
    a,Menc=acc(r,L,t); nstep=0; t0=time.time()
    while t<Tend:
        act=t>=tta-1e-12
        Mb,ab=bary(t)
        tdyn=np.sqrt((r[act]**2+(soft*t)**2)**1.5/np.maximum(Menc[act]+Mb,1e-12))
        dt=eta_dt*min(tdyn.min(),(r[act]/np.maximum(np.abs(vr[act]),1e-9)).min())
        dt=min(dt,1e-3*t,snaps[si]-t+1e-15)
        if fb>0:   # do not step across a switch of the baryon radius
            ph=np.log(t)/Pfrac; nxt=(np.floor(ph)+ (duty if ph%1<duty else 1.0))*Pfrac
            dt=min(dt,np.exp(nxt)-t+1e-12)
        vr+=0.5*dt*a; r+=dt*vr
        neg=r<0; r[neg]=-r[neg]; vr[neg]=-vr[neg]
        t+=dt
        newL=(t>=tta)&(L==0); L[newL]=j[newL]
        a,Menc=acc(r,L,t); vr+=0.5*dt*a; nstep+=1
        if t>=snaps[si]-1e-14:
            out[float(snaps[si])]=dict(r=r.copy(),dm=dm,t=t); si+=1
            if si>=len(snaps): break
    return out,nstep,time.time()-t0
