import pickle, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from p18core import m
res=pickle.load(open("profiles.pkl","rb"))
xb,mub,xs,Mb=res["base"]
lab={"base":"no baryons","fb05":"f_b=0.05","fb10":"f_b=0.10","fb20":"f_b=0.20","fb05fast":"f_b=0.05, fast cycles","fb10far":"f_b=0.10, far blow-out","fb05ae02":"f_b=0.05, gas to 0.02 vt","fb05ae01":"f_b=0.05, gas to 0.01 vt","fb02ae01":"f_b=0.02, gas to 0.01 vt"}
fig,ax=plt.subplots(1,3,figsize=(13,4))
for i,(n,(x,mu,xs_,M)) in enumerate(res.items()):
    c="k" if n=="base" else f"C{i}"
    ax[0].loglog(x,mu,c=c,lw=1.3 if n=="base" else 1,label=lab.get(n,n))
    if n!="base":
        ax[1].semilogx(x,mu/mub,c=c,label=lab.get(n,n))
        F=Mb-M; k=(xs>0.0015)&(xs<0.3); i0=np.argmax(F[k]); xm=xs[k][i0]; Fm=F[k][i0]
        ax[2].semilogx(xs/xm,F/Fm,c=c,label=lab.get(n,n))
xx=np.logspace(-1.5,1.5,300); ax[2].semilogx(xx,(xx-m(xx))/0.5196,"k--",lw=1.5,label="P18: (x-m)/0.52")
ax[0].set_xlabel(r"$\xi=R/vt$"); ax[0].set_ylabel(r"$\mu=(G/v^2)\,\partial M/\partial R$"); ax[0].set_title("(a) stacked DM profiles",fontsize=10); ax[0].legend(fontsize=7)
ax[1].axhline(1,c="0.5",lw=0.8); ax[1].set_xlabel(r"$\xi$"); ax[1].set_ylabel(r"$\mu/\mu_{\rm no\ baryons}$"); ax[1].set_title("(b) change caused by the bursts",fontsize=10); ax[1].set_ylim(0,2)
ax[2].axhline(0,c="0.5",lw=0.8); ax[2].set_xlabel(r"radius / radius of maximum displacement"); ax[2].set_ylabel("displaced mass / its maximum"); ax[2].set_title("(c) displaced mass, compared with P18",fontsize=10); ax[2].legend(fontsize=7); ax[2].set_xlim(0.03,30)
fig.tight_layout(); fig.savefig("fbshells.png",dpi=150)
