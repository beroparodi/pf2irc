"""Displaced mass F(xi) = M_base(<xi) - M_run(<xi) over the full range, and a fit of the P18 template
A*(x - m(x)), x = xi/xi_s, compared with the pseudo-isothermal template A*arctan(x) (no excess)."""
import pickle, numpy as np, sys
from scipy.optimize import least_squares
from p18core import m
out=open("analyse2_out.txt","w")
def P(s): print(s); out.write(s+"\n")
xs=np.geomspace(1.5e-3,1.2,300)
def cumM(name):
    d=pickle.load(open(f'run_{name}.pkl','rb')); cs=[]
    for t,s in d.items():
        xi=s['r']/s['t']; w=s['dm']/s['t']; o=np.argsort(xi)
        cs.append(np.interp(xs,xi[o],np.cumsum(w[o])))
    return np.mean(cs,axis=0)
Mb=cumM("base")
P(f"total mass /t within xi=1.2: base {Mb[-1]:.4f}")
names=sys.argv[1:] or ["fb05","fb10","fb20","fb05fast"]
res={}
for n in names:
    try: M=cumM(n)
    except FileNotFoundError: P(f"{n}: missing"); continue
    F=Mb-M; i=np.argmax(F); xm,Fm=xs[i],F[i]
    neg=np.where((xs>xm)&(F<0))[0]; xz=xs[neg[0]] if len(neg) else np.nan
    Fend=F[-1]
    def fitT(T):
        r=lambda p: (p[0]*T(xs/p[1])-F)/Fm
        k=(xs>0.003)&(xs<1.0)
        s=least_squares(lambda p: r(p)[k],[Fm/0.52,xm],bounds=([0,1e-3],[10,2]))
        return s.x, np.sqrt(np.mean(s.fun**2))
    (A1,s1),e1=fitT(lambda x: x-m(x))
    (A2,s2),e2=fitT(lambda x: np.arctan(x))
    res[n]=dict(F=F,xm=xm,Fm=Fm,p18=(A1,s1),pis=(A2,s2))
    P(f"{n:9s} peak F={Fm:.3e} at xi={xm:.3f}; zero at xi={xz:.3f} (zero/peak {xz/xm:.2f}); F(1.2)/peak={Fend/Fm:+.2f};"
      f" fit P18: xi_s={s1:.3f} rms={e1:.3f}; fit arctan: rms={e2:.3f}")
P("P18 itself: zero/peak = 3.29, F(inf)/peak = -0.48")
pickle.dump(dict(xs=xs,Mb=Mb,res=res),open("analyse2.pkl","wb"))
out.close()
