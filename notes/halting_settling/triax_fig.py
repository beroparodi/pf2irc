import pickle, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
res=pickle.load(open("triax_core.pkl","rb"))
fig,ax=plt.subplots(figsize=(6,4.2))
for lab,d in res.items():
    r,rho=d["rho"]; ok=rho>0
    ax.loglog(r[ok],rho[ok]/np.interp(1,r[ok],rho[ok]),label=f"{lab} (slope {d['slope']:+.2f})")
rr=np.logspace(-2,1.5,100); ax.loglog(rr,0.5*rr**-2,"k:",lw=0.8,label=r"$\propto r^{-2}$")
ax.axvspan(0.1,0.5,color="0.9",zorder=0)
ax.set_xlabel(r"$r/r_c$"); ax.set_ylabel(r"time-averaged density (norm. at $r_c$)")
ax.set_xlim(0.01,40); ax.legend(fontsize=6.5,loc="lower left"); fig.tight_layout(); fig.savefig("triax_core.png",dpi=160)
