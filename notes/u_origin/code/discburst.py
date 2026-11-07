"""
Spherical dark matter answering a gas disc that is filled slowly and emptied suddenly (Sect. 7.2 of the note).

Dark matter: N particles in 3D, initially an isotropic singular isothermal sphere (SIS) with v = 1
(rho = 1/(4 pi r^2), Maxwellian velocities with sigma = 1/sqrt2), truncated at R_out. The force is spherical:
a = -[M_dm(<r) + M_d(t) mu(r/R_d)] x / (r^2+eps^2)^{3/2}, with M_dm(<r) from the particles (fine log grid).
Gas disc: exponential, scale R_d = 1, entering through its spherically enclosed mass M_d mu(r), mu = 1-(1+x)e^-x.
Schedule ('cycles'): relax T0; build the disc up to M_lo over T_fill; then nb bursts, each a linear fill from
M_lo to M_lo+dM over T_fill (slow: adiabatic contraction) followed by an instantaneous drop back to M_lo
(impulsive); then M_lo is kept to the end. 'adiabatic': disc grown once to M_lo over T_fill and kept.
'none': no disc (reference). All runs last T0 + n_cyc*T_fill + T_end. Units G = v = R_d = 1.
"""
import numpy as np, time, pickle
mu=lambda x: 1-(1+x)*np.exp(-x)

def run(name, N=200000, R_out=100.0, eps=0.02, dt=0.004, T0=10.0, T_fill=20.0, n_cyc=8, T_end=30.0,
        M_lo=0.5, dM=0.6, mode="cycles", seed=1, nsnap=15, nb=None):
    nb=n_cyc-1 if nb is None else nb
    rng=np.random.default_rng(seed)
    r=rng.uniform(0,R_out,N); c=rng.uniform(-1,1,N); ph=rng.uniform(0,2*np.pi,N); s=np.sqrt(1-c*c)
    X=np.stack([r*s*np.cos(ph),r*s*np.sin(ph),r*c],1); V=rng.normal(0,1/np.sqrt(2),(N,3))
    mp=R_out/N                                   # SIS with v = 1: M(<r) = r
    lmin,lmax,nbin=np.log(1e-3),np.log(3*R_out),6000; dl=(lmax-lmin)/nbin
    edges=np.exp(lmin+dl*np.arange(nbin+1))
    def Mdm(rr):                                 # enclosed dark mass, linear within log bins
        k=np.clip(((np.log(np.maximum(rr,1e-3))-lmin)/dl).astype(np.int64),0,nbin-1)
        h=np.bincount(k,minlength=nbin); cm=np.concatenate([[0.0],np.cumsum(h)])*mp
        f=np.clip((rr-edges[k])/(edges[k+1]-edges[k]),0,1)
        out=cm[k]+f*h[k]*mp; out[rr<1e-3]=0.0
        return out
    Ttot=T0+n_cyc*T_fill+T_end
    def Md(t):
        if mode=="none" or t<T0: return 0.0
        tt=t-T0
        if mode=="adiabatic": return M_lo*min(tt/T_fill,1.0)
        if tt<T_fill: return M_lo*tt/T_fill                       # first build-up to M_lo
        tt-=T_fill; k=int(tt//T_fill)
        if k>=min(nb,n_cyc-1): return M_lo
        return M_lo+dM*((tt%T_fill)/T_fill)                       # slow fill, sudden drop at cycle end
    def acc(X,t):
        rr=np.sqrt((X*X).sum(1)); M=Mdm(rr)+Md(t)*mu(rr)
        return -X*(M/(rr*rr+eps*eps)**1.5)[:,None]
    snapt=np.linspace(Ttot-T_end*0.6,Ttot,nsnap); si=0; snaps=[]
    t=0.0; A=acc(X,t); t0=time.time(); nstep=int(round(Ttot/dt))
    for i in range(nstep):
        V+=0.5*dt*A; X+=dt*V; t+=dt
        A=acc(X,t); V+=0.5*dt*A
        if si<nsnap and t>=snapt[si]-1e-9:
            rr=np.sqrt((X*X).sum(1)); sel=rr<20
            vr=(X[sel]*V[sel]).sum(1)/rr[sel]; v2=(V[sel]**2).sum(1)
            snaps.append(dict(t=t,r=rr[sel],vr=vr,vt2=v2-vr**2)); si+=1
        if i%2500==0: print(name,f"t={t:.1f} Md={Md(t):.3f} wall={time.time()-t0:.0f}s",flush=True)
    pickle.dump(dict(name=name,mp=mp,snaps=snaps,M_lo=M_lo if mode!="none" else 0.0,dM=dM,mode=mode,nb=nb,
                     T_fill=T_fill,n_cyc=n_cyc,eps=eps,N=N,R_out=R_out),open(f"run_{name}.pkl","wb"))

RUNS={"none":dict(mode="none"),
      "adiab":dict(mode="adiabatic"),
      "cyc03":dict(mode="cycles",dM=0.3),
      "cyc06":dict(mode="cycles",dM=0.6),
      "big3":dict(mode="cycles",dM=1.5,nb=3),
      "strong7":dict(mode="cycles",dM=3.0)}
