# Figure 2: isotropic f(E) of P18 vs isothermal; OM f(Q) for several r_a. Also checks the
# central law f ~ 0.0105/(E - Phi0) and lists where OM goes negative.
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import importlib.util, sys
spec=importlib.util.spec_from_file_location("om","p18_om.py")
src=open("p18_om.py").read().split("C=Phi[-1]")[0]   # reuse grid and G2/f_of without running the scans
src="\n".join(l for l in src.split("\n") if not l.startswith(("out=open","def P(")))   # do not overwrite p18_om_out.txt
ns={}; exec(src,ns)
f_of=ns["f_of"]; Phi=ns["Phi"]; r=ns["r"]; m=ns["m"]
C=Phi[-1]-np.log(r[-1])+0.25/r[-1]
fS=lambda E: np.exp(-2*(E-C))*np.sqrt(np.pi/2)/(2*np.sqrt(2)*np.pi**3)
Ec=Phi+0.5*m(r)/r
Es=np.concatenate([np.geomspace(1e-5,1,120),np.linspace(1.02,8,140)])
fI=f_of(None,Es)
pred=(1/np.pi)*(1.5)**1.5/(4*np.sqrt(2)*np.pi**2)
with open("p18_df_fig_out.txt","w") as o:
    for E in [1e-5,1e-4,1e-3,1e-2]:
        i=np.argmin(abs(Es-E)); s=f"f(E)*(E-Phi0) at E-Phi0={Es[i]:.1e}: {fI[i]*Es[i]:.5f}  (predicted {pred:.5f})"; print(s); o.write(s+"\n")
fig,ax=plt.subplots(1,2,figsize=(10,4))
a=ax[0]; a.loglog(Es,fI,"k",label="P18, isotropic")
a.loglog(Es,fS(Es),"C0--",label=r"isothermal, $\propto e^{-2E/v^2}$ (SIS far field)")
a.loglog(Es,pred/Es,"C3:",label=r"$0.0105/(E-\Phi_0)$ (central law)")
a.set_xlabel(r"$E-\Phi(0)$  [$v^2$]"); a.set_ylabel(r"$f(E)$  [$G^{-3/2}v^{-3}r_c^{-3/2}\cdot$ units]")
a.set_ylabel(r"$f(E)$ (units $G=v=r_c=1$)"); a.legend(fontsize=8); a.set_title("(a) isotropic distribution function",fontsize=10)
a.set_ylim(1e-6,1e4)
a=ax[1]
for ra,c in [(0.5,"C3"),(0.7,"C1"),(0.865,"k"),(2.0,"C0")]:
    f=f_of(ra,Es); q=f/fS(Es)
    a.semilogx([np.interp(E,Ec,r) for E in Es],q,c=c,label=fr"$r_a={ra}\,r_c$")
a.axhline(0,c="0.5",lw=0.8); a.set_ylim(-5,8); a.set_xlim(0.03,300)
a.set_xlabel(r"$R_{\rm circ}(Q)/r_c$"); a.set_ylabel(r"$f(Q)\,/\,f_{\rm iso}(Q)$")
a.set_title("(b) Osipkov–Merritt: radial outside $r_a$",fontsize=10); a.legend(fontsize=8)
fig.tight_layout(); fig.savefig("p18_df.png",dpi=150)
