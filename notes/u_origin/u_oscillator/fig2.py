import pickle, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from p18core import m
xs=np.geomspace(1.5e-3,1.2,300)
def cum(name):
    d=pickle.load(open(f'run_{name}.pkl','rb')); cs=[]
    for t,s in d.items():
        xi=s['r']/s['t']; w=s['dm']/s['t']; o=np.argsort(xi); cs.append(np.interp(xs,xi[o],np.cumsum(w[o])))
    return np.mean(cs,axis=0)
Mb=cum("base")
runs=[("fb05","bursts, gas to 0.05 vt",'C1'),("fb10far","f_b=0.10, gas to 0.2 vt",'C4'),
      ("fb05ae02","confined: gas to 0.02 vt (~3 r_c)",'C2'),("fb05ae01","confined: gas to 0.01 vt (~2 r_c)",'C0'),
      ("fb02ae01","weak, confined (f_b=0.02, 0.01 vt)",'C7')]
fig,ax=plt.subplots(1,2,figsize=(11.5,4.2))
for n,l,c in runs:
    F=Mb-cum(n); ax[0].semilogx(xs,F,c=c,label=l)
ax[0].axhline(0,c="0.5",lw=0.8)
ax[0].axvline(0.006,c="0.6",ls=":"); ax[0].text(0.0065,0.07,r"$r_c\approx R_b/35$",fontsize=8,color="0.4")
ax[0].axvline(0.21,c="0.6",ls="--"); ax[0].text(0.22,0.07,"front",fontsize=8,color="0.4")
ax[0].set_xlabel(r"$\xi=R/vt$"); ax[0].set_ylabel(r"displaced DM mass $F=M_{\rm no\,bursts}(<\xi)-M(<\xi)$  [$v^3t/G$]")
ax[0].set_title("(a) where the displaced dark matter goes",fontsize=10); ax[0].legend(fontsize=7.5)
xx=np.logspace(-1.5,1.6,300); ax[1].semilogx(xx,(xx-m(xx))/0.5196,"k--",lw=1.5,label=r"P18: $(x-m)/0.52$, $x=R/r_c$")
for n,l,c in runs[:4]:
    F=Mb-cum(n); k=(xs>0.0015)&(xs<0.3); i=np.argmax(F[k]); xm=xs[k][i]; Fm=F[k][i]
    ax[1].semilogx(xs/xm,F/Fm,c=c,label=l)
ax[1].axhline(0,c="0.5",lw=0.8); ax[1].set_xlim(0.02,40)
ax[1].set_xlabel("radius / radius of maximum displacement"); ax[1].set_ylabel("displaced mass / its maximum")
ax[1].set_title("(b) same shape as P18, but the peak sits at the front",fontsize=10); ax[1].legend(fontsize=7.5)
fig.tight_layout(); fig.savefig("fbshells2.png",dpi=150)
