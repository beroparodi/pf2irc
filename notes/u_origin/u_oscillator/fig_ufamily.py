# Fig. 1: the family u_b(x) = 1 + (b x - 1) e^{-x}, and P18 against the non-singular isothermal sphere
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
x=np.linspace(0,8,600)
fig,ax=plt.subplots(1,2,figsize=(11,4))
for b,c in [(0,"C7"),(0.5,"C0"),(np.sqrt(15)-3,"C2"),(1,"k"),(2,"C3")]:
    lab=fr"$b={b:.3f}$" if abs(b-(np.sqrt(15)-3))<1e-9 else fr"$b={b:g}$"
    if b==1: lab+=" (P18)"
    ax[0].plot(x,1+(b*x-1)*np.exp(-x),c=c,lw=2 if b==1 else 1.2,label=lab)
ax[0].axhline(1,c="0.5",lw=0.8); ax[0].set_xlabel(r"$x$"); ax[0].set_ylabel(r"$u_b(x)$")
ax[0].set_title(r"(a) $u_b=1+(bx-1)e^{-x}$: overshoot for $b>0$",fontsize=10); ax[0].legend(fontsize=8)
# Emden: psi'' + 2 psi'/xi = e^{-psi}; rho/rho_SIS = e^{-psi} xi^2/2; xi = sqrt8 * x (same central density)
def f(s,y): return [y[1], np.exp(-y[0])-2*y[1]/s]
s0=1e-4; sol=solve_ivp(f,(s0,400),[s0**2/6,s0/3],rtol=1e-10,atol=1e-12,dense_output=True)
X=np.logspace(-2,2,800); xi=np.sqrt(8)*X; psi=sol.sol(xi)[0]
ax[1].semilogx(X,np.exp(-psi)*xi**2/2,c="C1",label="isothermal sphere (same $\\rho(0)$)")
ax[1].semilogx(X,(1+(X-1)*np.exp(-X))**2,c="k",lw=2,label=r"P18, $u^2$")
ax[1].axhline(1,c="0.5",lw=0.8); ax[1].set_xlabel(r"$x=R/r_c$"); ax[1].set_ylabel(r"$\rho/\rho_{\rm SIS}$")
ax[1].set_title("(b) critically damped in $x$ vs underdamped in $\\ln x$",fontsize=10); ax[1].legend(fontsize=8); ax[1].set_ylim(0,1.8)
fig.tight_layout(); fig.savefig("fig_ufamily.png",dpi=150)
