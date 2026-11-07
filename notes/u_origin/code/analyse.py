"""Analysis of the disc-burst runs (Sect. 7.2): dark-matter density against the reference run, inner slope,
displaced dark matter, P18 fit to the total (dark matter + retained disc), anisotropy. Output analyse_out.txt."""
import numpy as np, pickle, sys, warnings
from scipy.optimize import least_squares
warnings.filterwarnings("ignore")
e=np.exp
mu=lambda x: 1-(1+x)*e(-x)
def m18(x): x=np.asarray(x,float); return x*(1-e(-x))**2-0.5*x*(1+x)*e(-2*x)+0.25*(1-e(-2*x))
names=sys.argv[1:] or ["adiab","cyc03","cyc06","big3","strong7"]
R=np.geomspace(0.03,15,70); Rb=np.geomspace(0.03,15,31); Rc=np.sqrt(Rb[1:]*Rb[:-1]); vol=Rb[1:]**3-Rb[:-1]**3
def stack(n):
    d=pickle.load(open(f"run_{n}.pkl","rb")); M=[];H=[];B=[]
    for s in d["snaps"]:
        r=np.sort(s["r"]); M.append(np.searchsorted(r,R)*d["mp"])
        h,_=np.histogram(s["r"],Rb); H.append(h)
        k=np.digitize(s["r"],Rb)-1
        B.append([1-s["vt2"][k==i].mean()/(2*(s["vr"][k==i]**2).mean()) if (k==i).sum()>50 else np.nan for i in range(len(Rc))])
    return d,np.mean(M,0),np.mean(H,0),np.nanmean(B,0)
out=open("analyse_out.txt","w")
def P(*a): print(*a); print(*a,file=out)
ref=stack("none"); res={}
k=(Rc>0.05)&(Rc<0.2); sref=np.polyfit(np.log(Rc[k]),np.log(ref[2][k]/vol[k]),1)[0]
P("radii in units of R_d; masses in v^2 R_d/G; reference = SIS run without disc ('none')")
P(f"reference: M_dm(<R)/R at R = 0.1, 0.3, 1, 3, 10: {(np.interp([0.1,0.3,1,3,10],R,ref[1])/np.array([0.1,0.3,1,3,10])).round(3)}; "
  f"inner slope {sref:+.2f}")
for n in names:
    d,M,H,B=stack(n); Mlo=d["M_lo"]
    F=ref[1]-M; rho=H/np.maximum(ref[2],1)
    Mt=M+Mlo*mu(R); V2=Mt/R; sel=(R>=0.05)&(R<=10)
    f=lambda p: np.log(e(2*p[0])*m18(R[sel]/e(p[1]))/(R[sel]/e(p[1])))-np.log(V2[sel])
    s=least_squares(f,[0.0,0.0]); v,rc=e(s.x); rms=np.sqrt(np.mean(s.fun**2))/2
    slope=np.polyfit(np.log(Rc[k]),np.log(H[k]/vol[k]),1)[0]
    ip=np.argmax(F); iz=np.where(F[ip:]<0)[0]; zero=R[ip+iz[0]] if len(iz) else np.nan
    res[n]=dict(R=R,F=F,rho=rho,Rc=Rc,B=B,V2=V2,v=v,rc=rc,Mlo=Mlo,slope=slope)
    P(f"\n{n}: mode {d['mode']}, M_lo = {Mlo}, dM = {d['dM']}, bursts = {d['nb'] if d['mode']=='cycles' else 0}")
    P(f"  inner slope (R = 0.05-0.2): {slope:+.2f}")
    P(f"  DM density / reference at R = 0.1, 0.3, 1, 2, 3, 5: {np.interp([0.1,0.3,1,2,3,5],Rc,rho).round(2)}")
    P(f"  displaced DM F: maximum {F[ip]:+.3f} at R = {R[ip]:.2f}; F(1) = {np.interp(1,R,F):+.3f}; F(10) = {np.interp(10,R,F):+.3f}")
    P(f"  total: P18 fit v = {v:.3f}, r_c = {rc:.3f} R_d, rms(log V) = {rms:.4f}")
    P(f"  beta at R = 0.3, 1, 2, 3, 5: {np.interp([0.3,1,2,3,5],Rc,B).round(2)}  (reference {np.interp([0.3,1,2,3,5],Rc,ref[3]).round(2)})")
pickle.dump(dict(res=res,ref=dict(R=R,M=ref[1],B=ref[3],Rc=Rc)),open("analysis.pkl","wb"))
