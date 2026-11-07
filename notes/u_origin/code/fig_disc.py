"""Fig. 1 of the note: the disc representation of u, the split of the displaced mass, the Jeans anisotropy."""
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid
e=np.exp
x=np.linspace(0,10,4001)
u=1+(x-1)*e(-x); mu=1-(1+x)*e(-x); mu1=x*e(-x); mu2=(1-x)*e(-x)
m=x*(1-e(-x))**2-0.5*x*(1+x)*e(-2*x)+0.25*(1-e(-2*x))
fig,ax=plt.subplots(1,3,figsize=(15,4.3))
a=ax[0]
a.plot(x,mu,"C0",label=r"$\mu$: disc mass inside $x$")
a.plot(x,mu1,"C1",label=r"$\mu'=xe^{-x}$: annulus mass")
a.plot(x,mu2,"C2",label=r"$\mu''=(1-x)e^{-x}$")
a.plot(x,u,"k",lw=2,label=r"$u=1-\mu''$")
a.axhline(1,color="gray",lw=0.5); a.axhline(0,color="gray",lw=0.5); a.axvline(1,color="gray",lw=0.5,ls=":"); a.axvline(2,color="gray",lw=0.5,ls=":")
a.set_xlim(0,8); a.set_xlabel(r"$x=R/r_c$ $(=R/R_d)$"); a.set_title(r"(a) P18's $u$ from an exponential disc, $R_d=r_c$"); a.legend(fontsize=8)
a=ax[1]
I=cumulative_trapezoid(mu2**2,x,initial=0)
a.plot(x,x-m,"k",lw=2,label=r"$F=x-m$ (P18)")
a.plot(x,2*mu1,"C1",label=r"$2\mu'$: first order, conserves mass")
a.plot(x,-I,"C3",label=r"$-\int_0^x\mu''^2$: second order, $\to -c=-1/4$")
a.axhline(0,color="gray",lw=0.5); a.set_xlim(0,10); a.set_xlabel(r"$x$"); a.set_ylabel(r"mass in $v^2r_c/G$")
a.set_title("(b) displaced mass, exact split"); a.legend(fontsize=8)
a=ax[2]
xx=np.geomspace(0.01,60,2000); uu=1+(xx-1)*e(-xx); mm=xx*(1-e(-xx))**2-0.5*xx*(1+xx)*e(-2*xx)+0.25*(1-e(-2*xx))
b=1-xx*(2-xx)*e(-xx)/uu-mm/xx
a.semilogx(xx,b,"k",lw=2,label=r"$\beta=1-xu'/u-m/x$")
a.axhline(0,color="gray",lw=0.5); a.set_xlabel(r"$x$"); a.set_ylabel(r"$\beta=1-\sigma_t^2/\sigma_r^2$")
a.set_title(r"(c) anisotropy P18 needs at $\sigma_r^2=v^2/2$"); a.legend(fontsize=8)
plt.tight_layout(); plt.savefig("../fig_disc.png",dpi=140)
