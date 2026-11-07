import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from p18core import u,m
from scipy.integrate import cumulative_trapezoid
x=np.logspace(-2,2,600)
fig,ax=plt.subplots(2,2,figsize=(10,7.4))
a=ax[0,0]; a.semilogx(x,u(x)**2,"k"); a.axhline(1,c="0.5",lw=0.8)
a.fill_between(x,u(x)**2,1,where=x<1,color="C3",alpha=0.3,label="removed: 0.52 $v^2r_c/G$")
a.fill_between(x,u(x)**2,1,where=x>1,color="C0",alpha=0.3,label="added: 0.77 $v^2r_c/G$")
a.axvline(1,c="0.6",ls=":"); a.set_xlabel("$x=R/r_c$"); a.set_ylabel(r"$\rho_{\rm P18}/\rho_{\rm SIS}=u^2$")
a.set_title("(a) P18 = SIS with matter moved out of the core",fontsize=10); a.legend(fontsize=8,loc="lower right")
a=ax[0,1]; js=np.sqrt(x/m(x)); a.loglog(x,js,"k",label="circular orbits: $j=GM/v_c$")
a.axhline(1,c="0.5",lw=0.8,label="SIS circular: $GM/v$"); a.axhline(0.099,c="C1",ls="--",label=r"seed spin ($\lambda=0.035$)")
a.set_xlabel("$x=R/r_c$"); a.set_ylabel("specific angular momentum $j$  [$GM/v$]")
a.set_title("(b) angular momentum of each mass shell",fontsize=10); a.legend(fontsize=8); a.set_ylim(0.05,30)
a=ax[1,0]; ra=0.865
for r_a,ls,lab in [(0.865,"-",r"$r_a=0.87\,r_c$ (most radial possible)"),(2,"--",r"$r_a=2\,r_c$"),(5,":",r"$r_a=5\,r_c$")]:
    a.semilogx(x,x**2/(x**2+r_a**2),"k",ls=ls,label=lab)
a.axvline(1,c="0.6",ls=":"); a.set_xlabel("$x=R/r_c$"); a.set_ylabel(r"anisotropy $\beta$ (1 radial, 0 isotropic)")
a.set_title("(c) how radial the orbits may be (Osipkov–Merritt)",fontsize=10); a.legend(fontsize=8,loc="upper left")
d=np.load("p18_df.npz"); Es,MI,MS,Rc=d["Es"],d["MI"],d["MS"],d["Rc"]
k=(Rc>0.05)&(Rc<200)
a=ax[1,1]; a.semilogx(Rc[k],MI[k]/MS[k],"k"); a.axhline(1,c="0.5",lw=0.8); a.axvline(1,c="0.6",ls=":")
a.set_xlabel(r"$R_{\rm circ}(E)/r_c$ (radius of the circular orbit with energy $E$)")
a.set_ylabel(r"$M_{\rm P18}(<E)\,/\,M_{\rm SIS}(<E)$")
a.set_title("(d) bound mass, isotropic P18 vs isotropic SIS",fontsize=10)
fig.tight_layout(); fig.savefig("p18_inverse.png",dpi=150)
