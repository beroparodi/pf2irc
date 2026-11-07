"""
Test particles in the P18 potential whose exponential sphere (the part carrying the
mass excess, M_e = v^2 r_c/4G, rho ∝ exp(-2R/r_c)) is made triaxial, static or tumbling,
growing over T_g.  Units: G = v = r_c = 1;  R_b = y = 34 (example galaxy, 84/2.44 kpc).
Potential:  Phi = Phi_1(r) + Phi_e(m),  m^2 = X^2 + Y^2/p^2 + Z^2/q^2 in the body frame
(rotating about z at Omega_p).  dPhi_1/dr = (1-e^-r)^2/r ;  dPhi_e/dm = M_e(m)/m^2.
Particles: early-accreted, nearly radial matter (App. H of the paper): apocentres r_a
log-uniform in 0.5..34, j = 0.05 j_circ(r_a), random orientation, random phase.
Diagnostics after t = 300 (= 12 Gyr for r_c/v = 40 Myr):
 - time-averaged density of the population, log slope at 0.1-0.5 r_c (cusp of radial
   orbits: -2; isotropic population at fixed energies: ~0, a core)
 - anisotropy beta inside r_c, median pericentre, |j|/j_circ(r_a), energy change.
"""
import numpy as np, sys
rng=np.random.default_rng(1)
Y=34.0
def dphi1(r): return (-np.expm1(-r))**2/np.maximum(r,1e-12)
def Me(m):
    return 0.25*(1-(1+2*m+2*m*m)*np.exp(-2*m))
def dphie_over_m(m):  # (dPhi_e/dm)/m, regular at 0 (-> 2/3 * rho_e0*4pi/... ) use series
    small=m<1e-3
    out=np.empty_like(m)
    mm=m[~small]; out[~small]=Me(mm)/mm**3
    out[small]=1/3.0   # Me ~ (1/3) m^3 * ... : 0.25*(4/3 m^3)=m^3/3
    return out
def accel(x,t,p,q,Om):
    # inertial -> body
    c,s=np.cos(Om*t),np.sin(Om*t)
    X= c*x[:,0]+s*x[:,1]; Yb=-s*x[:,0]+c*x[:,1]; Z=x[:,2]
    r=np.sqrt((x**2).sum(1))
    a=-(dphi1(r)/np.maximum(r,1e-12))[:,None]*x
    m=np.sqrt(X*X+Yb*Yb/p**2+Z*Z/q**2)
    g=dphie_over_m(m)
    aX=-g*X; aY=-g*Yb/p**2; aZ=-g*Z/q**2
    a[:,0]+= c*aX-s*aY; a[:,1]+= s*aX+c*aY; a[:,2]+=aZ
    return a
# potential (spherical) for energies
from scipy.integrate import quad
def Phi1(r):  # Phi_1(r) - Phi_1(Y*4) reference, numeric
    return -quad(dphi1,r,200)[0]
rr=np.logspace(-4,np.log10(200),2000)
P1=np.array([Phi1(r) for r in rr])
def Phi_e_sph(m): return -quad(lambda s: Me(s)/s**2,m,200)[0]-Me(200)/200
Pe=np.array([Phi_e_sph(r) for r in rr])
def Phi_spher(r): return np.interp(r,rr,P1+Pe)
def jcirc(r): # circular orbit in spherical P18: v_c^2 = m(x)/x
    m=r*(-np.expm1(-r))**2+Me(r); return np.sqrt(m/r)*r
def setup(N):
    ra=np.exp(rng.uniform(np.log(0.5),np.log(Y),N))
    u=rng.normal(size=(N,3)); u/=np.linalg.norm(u,axis=1)[:,None]
    w=rng.normal(size=(N,3)); w-= (w*u).sum(1)[:,None]*u; w/=np.linalg.norm(w,axis=1)[:,None]
    x=ra[:,None]*u; vv=(0.05*jcirc(ra)/ra)[:,None]*w
    return x,vv,ra
def run(p,q,Om,Tg,N=400,tend=300.,dt=0.004,seedphase=True):
    x,vv,ra=setup(N)
    # random phase: pre-integrate each particle in the spherical potential for a random time
    if seedphase:
        tph=rng.uniform(0,1,N)*2*np.pi*ra  # ~ up to one period
        tt=0.0; steps=int(tph.max()/dt)+1
        for k in range(steps):
            act=(tt<tph)[:,None]
            a=accel(x,0,1,1,0); v2=vv+0.5*dt*a*act; x=x+dt*v2*act
            a=accel(x,0,1,1,0); vv=v2+0.5*dt*a*act; tt+=dt
    j0=np.linalg.norm(np.cross(x,vv),axis=1)
    E0=0.5*(vv**2).sum(1)+Phi_spher(np.linalg.norm(x,axis=1))
    nst=int(tend/dt); rs=[]; vr=[]; vt=[]
    peri=np.full(N,np.inf); t=0.0
    a=accel(x,t,1,1,Om)
    for k in range(nst):
        f=min(t/Tg,1.0) if Tg>0 else 1.0
        pp=1-(1-p)*f; qq=1-(1-q)*f
        vv+=0.5*dt*a; x+=dt*vv; t+=dt
        f=min(t/Tg,1.0) if Tg>0 else 1.0
        pp=1-(1-p)*f; qq=1-(1-q)*f
        a=accel(x,t,pp,qq,Om); vv+=0.5*dt*a
        if t>tend-100 and k%25==0:
            r=np.linalg.norm(x,axis=1); rs.append(r.copy())
            vrad=(x*vv).sum(1)/r; vr.append(vrad); vt.append(np.sqrt(np.maximum((vv**2).sum(1)-vrad**2,0)))
            peri=np.minimum(peri,r)
    rs=np.concatenate(rs); vr=np.concatenate(vr); vt=np.concatenate(vt)
    # time-averaged density
    edges=np.logspace(-2,np.log10(40),31); h,_=np.histogram(rs,edges)
    vol=4/3*np.pi*(edges[1:]**3-edges[:-1]**3); rho=h/vol; rc_=np.sqrt(edges[1:]*edges[:-1])
    sel=(rc_>0.1)&(rc_<0.5)&(h>0)
    slope=np.polyfit(np.log(rc_[sel]),np.log(rho[sel]),1)[0]
    sel2=(rc_>3)&(rc_<15)&(h>0); slope_out=np.polyfit(np.log(rc_[sel2]),np.log(rho[sel2]),1)[0]
    ins=rs<1; beta=1-np.mean(vt[ins]**2)/(2*np.mean(vr[ins]**2))
    j1=np.linalg.norm(np.cross(x,vv),axis=1)
    E1=0.5*(vv**2).sum(1)+Phi_spher(np.linalg.norm(x,axis=1))
    return dict(slope=slope,slope_out=slope_out,beta=beta,peri=np.median(peri),
                jr0=np.median(j0/jcirc(ra)),jr1=np.median(j1/jcirc(ra)),
                jr1_in=np.median((j1/jcirc(ra))[ra<3]),jr1_out=np.median((j1/jcirc(ra))[ra>15]),
                dE=np.median(E1-E0),dEabs=np.median(abs(E1-E0)),rho=(rc_,rho))
if __name__=="__main__":
    cases=[("spherical",1,1,0,0),
           ("triax 0.9/0.8 static, grow 20",0.9,0.8,0,20),
           ("triax 0.8/0.6 static, grow 20",0.8,0.6,0,20),
           ("triax 0.8/0.6 tumbling 0.3, grow 20",0.8,0.6,0.3,20),
           ("triax 0.8/0.6 tumbling 0.1, grow 20",0.8,0.6,0.1,20),
           ("triax 0.9/0.8 tumbling 0.3, grow 20",0.9,0.8,0.3,20)]
    import pickle; res={}
    for lab,p,q,Om,Tg in cases:
        d=run(p,q,Om,Tg); res[lab]=d
        print(f"{lab:38s} slope(0.1-0.5)={d['slope']:+.2f} slope(3-15)={d['slope_out']:+.2f} beta(<rc)={d['beta']:+.2f} "
              f"peri_med={d['peri']:.3f} j/jc: {d['jr0']:.3f}->{d['jr1']:.3f} (ra<3: {d['jr1_in']:.3f}, ra>15: {d['jr1_out']:.3f}) "
              f"dE med={d['dE']:+.3f} |dE| med={d['dEabs']:.3f}",flush=True)
    pickle.dump(res,open("triax_core.pkl","wb"))
