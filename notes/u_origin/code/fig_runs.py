"""Fig. 2 of the note: the disc-burst runs."""
import numpy as np, pickle, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
e=np.exp
def m18(x): return x*(1-e(-x))**2-0.5*x*(1+x)*e(-2*x)+0.25*(1-e(-2*x))
u=lambda x: 1+(x-1)*e(-x)
A=pickle.load(open("analysis.pkl","rb")); res=A["res"]
col={"adiab":"C0","cyc03":"C1","cyc06":"C3","big3":"C2","strong7":"C4"}
lab={"adiab":"disc grown slowly, kept","cyc03":r"7 bursts, $\Delta M=0.3$","cyc06":r"7 bursts, $\Delta M=0.6$",
     "big3":r"3 bursts, $\Delta M=1.5$","strong7":r"7 bursts, $\Delta M=3$"}
fig,ax=plt.subplots(1,4,figsize=(17,4.3)); xx=np.geomspace(0.03,15,300)
for n,r in res.items():
    ax[0].semilogx(r["Rc"],r["rho"],col[n],label=lab[n])
    ax[1].semilogx(r["R"],np.sqrt(r["V2"]),col[n]); x=r["R"]/r["rc"]; ax[1].semilogx(r["R"],r["v"]*np.sqrt(m18(x)/x),col[n],ls=":")
    ax[2].semilogx(r["R"],r["F"],col[n]); ax[3].semilogx(r["Rc"],r["B"],col[n])
ax[0].semilogx(xx,u(xx)**2,"k--",lw=0.8,label=r"P18 $u^2$, $r_c=R_d$"); ax[0].axhline(1,color="gray",lw=0.5)
ax[0].set_ylim(0,1.6); ax[0].set_xlabel(r"$R/R_d$"); ax[0].set_ylabel(r"$\rho_{\rm DM}/\rho_{\rm ref}$"); ax[0].legend(fontsize=7)
ax[0].set_title("(a) dark matter against the SIS run")
ax[1].set_xlabel(r"$R/R_d$"); ax[1].set_ylabel(r"$V/v$ (dark matter + retained disc)"); ax[1].set_title("(b) total curve; dotted: P18 fit")
ax[2].semilogx(xx,xx-m18(xx),"k--",lw=0.8,label=r"P18 $x-m$, $r_c=R_d$"); ax[2].axhline(0,color="gray",lw=0.5)
ax[2].set_xlabel(r"$R/R_d$"); ax[2].set_ylabel(r"$F=M_{\rm ref}-M_{\rm DM}$"); ax[2].legend(fontsize=7); ax[2].set_title("(c) displaced dark matter")
b=1-xx*(2-xx)*e(-xx)/u(xx)-m18(xx)/xx
ax[3].semilogx(A["ref"]["Rc"],A["ref"]["B"],"k",lw=0.8,label="reference"); ax[3].semilogx(xx,b,"k--",lw=0.8,label=r"P18 at $\sigma_r^2=v^2/2$")
ax[3].axhline(0,color="gray",lw=0.5); ax[3].set_ylim(-0.3,0.5); ax[3].set_xlabel(r"$R/R_d$"); ax[3].set_ylabel(r"$\beta$")
ax[3].legend(fontsize=7); ax[3].set_title("(d) anisotropy")
plt.tight_layout(); plt.savefig("../fig_runs.png",dpi=130)
