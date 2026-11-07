"""Stacked self-similar DM profiles mu(xi) = (G/v^2) dM/dR for the toy runs; comparison with the run without
baryons; displaced mass F(xi) = M_base(<xi) - M_run(<xi) (the analogue of x - m(x) for P18)."""
import pickle, numpy as np, sys
from scipy.optimize import least_squares
out=open("analyse_out.txt","w")
def P(s): print(s); out.write(s+"\n")
u2=lambda x:(1+(x-1)*np.exp(-x))**2
def stack(name,nb=36,lo=1e-3,hi=0.5):
    d=pickle.load(open(f'run_{name}.pkl','rb'))
    e=np.geomspace(lo,hi,nb+1); acc=np.zeros(nb); cum=[]
    xs=np.geomspace(lo,hi,200)
    for t,s in d.items():
        xi=s['r']/s['t']; w=s['dm']/s['t']
        acc+=np.histogram(xi,bins=e,weights=w)[0]
        o=np.argsort(xi); cum.append(np.interp(xs,xi[o],np.cumsum(w[o])))
    mu=acc/len(d)/np.diff(e); x=np.sqrt(e[1:]*e[:-1])
    return x,mu,xs,np.mean(cum,axis=0)
names=sys.argv[1:] or ["base","fb05","fb10","fb05fast","fb10far","fb20"]
PAR={"base":dict(fb=0),"fb05":dict(fb=0.05),"fb10":dict(fb=0.10),"fb05fast":dict(fb=0.05),"fb10far":dict(fb=0.10,a_e=0.2),"fb20":dict(fb=0.20),"fb05ae01":dict(fb=0.05,a_e=0.01),"fb05ae02":dict(fb=0.05,a_e=0.02),"fb02ae01":dict(fb=0.02,a_e=0.01)}
def Mbary(xs,n):   # time-averaged enclosed baryon mass /t (Plummer, compact a_c=0.002 and expanded a_e half the time each)
    p=dict(fb=0,a_c=0.002,a_e=0.05,duty=0.5); p.update(PAR[n]); Mtot=p["fb"]*0.2
    pl=lambda a: xs**3/(xs**2+a**2)**1.5
    return Mtot*(p["duty"]*pl(p["a_c"])+(1-p["duty"])*pl(p["a_e"]))
res={}
for n in names:
    try: res[n]=stack(n)
    except FileNotFoundError: P(f"{n}: missing"); continue
xb,mub,xs,Mb=res["base"]
def slope(x,mu,a,b):
    k=(x>a)&(x<b)&(mu>0); return np.polyfit(np.log(x[k]),np.log(mu[k]),1)[0]-2
P("run       rho slope 0.002-0.006  0.006-0.02 | displaced mass F=M_base-M_run: max (xi) | zero (xi) | ratio zero/max | F at 0.1  | sign changes of mu/mu_base-1 (0.002-0.15)")
for n,(x,mu,xs_,M) in res.items():
    s1,s2=slope(x,mu,0.002,0.006),slope(x,mu,0.006,0.02)
    if n=="base":
        P(f"{n:9s} {s1:+.2f} {s2:+.2f}"); continue
    F=Mb-M; k=(xs>0.0015)&(xs<0.3)
    i=np.argmax(F[k]); xmax=xs[k][i]; Fmax=F[k][i]
    after=(xs>xmax)&(xs<0.3)&(F<0)
    xz=xs[after][0] if after.any() else np.nan
    F01=np.interp(0.1,xs,F)
    rr=mu/mub-1; kk=(x>0.002)&(x<0.15)&np.isfinite(rr)
    sc=np.sum(np.diff(np.sign(rr[kk]))!=0)
    P(f"{n:9s} {s1:+.2f} {s2:+.2f} | {Fmax:.2e} ({xmax:.4f}) | {xz:.4f} | {xz/xmax:5.2f} | {F01:+.2e} | {sc}")
    Ft=Mb-(M+Mbary(xs,n))
    P(f"          total matter (DM + time-averaged baryons): F at xi=0.003,0.01,0.03,0.1,0.2: "+" ".join(f"{np.interp(q,xs,Ft):+.2e}" for q in [0.003,0.01,0.03,0.1,0.2]))
P("P18 reference: F = x-m peaks at x=1 (0.52), zero at 3.29 (ratio 3.29), F(inf) = -c = -0.25 (ratio to peak -0.48)")
pickle.dump(res,open("profiles.pkl","wb"))
out.close()
