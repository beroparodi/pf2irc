"""
From an inclined spiral on (shrinking) spheres to a logarithmic spiral in the disc.
A parcel of gas moves in the P18 potential (units G = v = r_c = 1) and feels a drag
towards a gas flow that rotates about z with angular speed f*Omega(r),
    a = -grad Phi - gamma (v - f Omega(r) zhat x r),   gamma = kappa * Omega(r),
Omega(r) = v_c(r)/r.  Outputs: analytic predictions (flat part, core) and numerical
trajectories with measured pitch angle and inclination.
"""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
from p18core import m
def vc(r): return np.sqrt(m(r)/r)
def Om(r): return vc(r)/r
# ---------- analytic: flat part, steady self-similar inflow v_r=-A v, v_phi=B v
def flat(kappa,f):
    B=brentq(lambda B: B**3+(kappa**2-1)*B-kappa**2*f,1e-6,1.0)
    A=kappa*(B-f)/B
    return A,B,np.degrees(np.arctan(A/B)),kappa*f/(2*B*B)
# ---------- analytic: harmonic core, zeta'' = -w^2 zeta - gamma(zeta' - i f w zeta)
w=2/np.sqrt(3)
def core(kappa,f):
    g=kappa*w
    roots=np.roots([1,g,w*w-1j*g*f*w])
    s=[z for z in roots if z.imag>0][0]   # prograde
    return np.degrees(np.arctan(-s.real/s.imag)), s
# ---------- numerics
def rhs(t,y,kappa,f):
    x=y[:3]; v=y[3:]; r=np.linalg.norm(x)
    O=Om(r); g=-(m(r)/r**3)*x
    flow=f*O*np.array([-x[1],x[0],0.0])
    return np.concatenate([v,g-kappa*O*(v-flow)])
def run(kappa,f,r0=60.,inc=40.,tmax=None):
    i=np.radians(inc)
    x0=np.array([r0,0,0.]); n_t=np.array([0,np.cos(i),np.sin(i)])   # tangential direction in inclined plane
    A,B,_,_=flat(kappa,f)                 # start on the steady inflow of the flat part
    v0=-A*vc(r0)*x0/r0+B*vc(r0)*n_t
    ev=lambda t,y,*a: np.linalg.norm(y[:3])-0.05; ev.terminal=True
    s=solve_ivp(rhs,(0,tmax or 4000),np.concatenate([x0,v0]),args=(kappa,f),rtol=1e-9,atol=1e-11,
                max_step=0.5,events=ev,dense_output=False)
    return s
def diagnostics(s,f):
    x=s.y[:3].T; v=s.y[3:].T; r=np.linalg.norm(x,axis=1)
    R=np.hypot(x[:,0],x[:,1]); phi=np.unwrap(np.arctan2(x[:,1],x[:,0]))
    L=np.cross(x,v); inc=np.degrees(np.arccos(L[:,2]/np.linalg.norm(L,axis=1)))
    vr=(x*v).sum(1)/r
    vt=np.linalg.norm(L,axis=1)/r          # tangential speed in the orbital plane
    pitch=np.degrees(np.arctan(-vr/vt))     # pitch in the orbital plane
    return r,R,phi,inc,pitch
if __name__=="__main__":
    out=open("spiral_model_out.txt","w")
    def P(x): print(x); out.write(x+"\n")
    P("Flat part (v_c = v): steady inflow  v_r = -A v, v_phi = B v;  pitch p: tan p = A/B;")
    P("  inclination decays as tan(i/2) ~ exp(-q phi), q = kappa f/(2 B^2) per radian")
    P("  kappa   f     B       A      pitch(deg)   q      | core pitch (deg)")
    cases=[(0.1,0.0),(0.1,0.5),(0.2,0.0),(0.2,0.5),(0.3,0.5),(0.5,0.5),(0.5,0.8)]
    for k,f in cases:
        A,B,p,q=flat(k,f); pc,_=core(k,f)
        P(f"  {k:4.2f}  {f:4.2f}  {B:6.4f}  {A:6.4f}   {p:6.2f}     {q:6.4f}  | {pc:6.2f}")
    P("  check f=0: sin p = kappa ->  p = asin(kappa): "+", ".join(f"{np.degrees(np.arcsin(k)):.2f}" for k in [0.1,0.2]))
    P("\nNumerical trajectories (start r=60 r_c, inclination 40 deg, on the steady inflow v_r=-A v_c, v_t=B v_c):")
    res={}
    for k,f in [(0.2,0.5),(0.3,0.5),(0.2,0.0)]:
        s=run(k,f); r,R,phi,inc,pitch=diagnostics(s,f); res[(k,f)]=s
        A,B,pf,q=flat(k,f); pc,_=core(k,f)
        def at(rr):
            j=np.argmin(abs(r-rr)); return pitch[j],inc[j],phi[j]
        def pav(lo,hi):   # mean over all passages through the radial range
            k2=(r>lo)&(r<hi); return np.mean(pitch[k2]) if k2.any() else np.nan
        P(f" kappa={k}, f={f}: predicted pitch flat {pf:.2f}, core {pc:.2f} deg")
        P(f"   measured mean pitch at r=20-30: {pav(20,30):.2f}, 8-12: {pav(8,12):.2f}, 3-5: {pav(3,5):.2f}, 1-2: {pav(1,2):.2f}, 0.2-0.5: {pav(0.2,0.5):.2f}")
        # inclination decay rate per radian, fit between r=40 and 10
        k2=(r<40)&(r>10)&(inc>0.5)
        if k2.sum()>10:
            slope=np.polyfit(phi[k2],np.log(np.tan(np.radians(inc[k2])/2)),1)[0]
            P(f"   inclination: {inc[0]:.1f} deg at start, {inc[np.argmin(abs(r-10))]:.2f} at r=10, {inc[np.argmin(abs(r-1))]:.3f} at r=1;"
              f" d ln tan(i/2)/d phi = {slope:.4f} (predicted {-q:.4f})")
    P("\nSmall-kappa formula for any rotation curve: tan p = kappa (1-f)/(1+beta_s), beta_s = dln v_c/dln R")
    for x in [0.3,1,2,3,5.5,10,30]:
        h=1e-4; bs=(np.log(vc(x*(1+h)))-np.log(vc(x*(1-h))))/(2*h)
        P(f"   x={x:5.1f}: beta_s={bs:+.3f};  kappa=0.2,f=0.5: p={np.degrees(np.arctan(0.2*0.5/(1+bs))):.2f} deg")
    import pickle; pickle.dump({k:(v.t,v.y) for k,v in res.items()},open("spiral_runs.pkl","wb"))
    out.close()
