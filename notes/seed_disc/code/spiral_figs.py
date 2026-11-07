import numpy as np, pickle, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa
from spiral_model import flat, core, vc, diagnostics
# ---------------- Fig 1: loxodrome and its stereographic image
alpha=np.radians(75)                      # angle with the meridians
y=np.linspace(-3.2,3.2,3000)              # Mercator ordinate
lam=np.tan(alpha)*y; psi=2*np.arctan(np.exp(y))-np.pi/2
X,Yc,Z=np.cos(psi)*np.cos(lam),np.cos(psi)*np.sin(lam),np.sin(psi)
fig=plt.figure(figsize=(10,4.6))
a=fig.add_subplot(1,2,1,projection="3d")
u,v=np.mgrid[0:2*np.pi:40j,-np.pi/2:np.pi/2:20j]
a.plot_wireframe(np.cos(v)*np.cos(u),np.cos(v)*np.sin(u),np.sin(v),color="0.85",lw=0.4)
a.plot(X,Yc,Z,"C3",lw=1.2); a.set_box_aspect((1,1,1)); a.set_axis_off()
a.set_title(r"(a) loxodrome, $75^\circ$ to the meridians",fontsize=10)
rho=np.tan(np.pi/4-psi/2)                 # stereographic from the south pole
b=fig.add_subplot(1,2,2)
b.plot(rho*np.cos(lam),rho*np.sin(lam),"C3",lw=1.0)
t=np.linspace(0,2*np.pi,200); b.plot(np.cos(t),np.sin(t),"0.6",lw=0.8,ls="--",label="image of the equator")
for L in np.linspace(0,2*np.pi,12,endpoint=False): b.plot([0,3*np.cos(L)],[0,3*np.sin(L)],"0.85",lw=0.5)
b.set_xlim(-2.6,2.6); b.set_ylim(-2.6,2.6); b.set_aspect("equal"); b.legend(fontsize=8,loc="lower right")
b.set_title(r"(b) stereographic image: $\rho=e^{-\lambda\cot\alpha}$, pitch $15^\circ$",fontsize=10)
fig.tight_layout(); fig.savefig("fig_loxodrome.png",dpi=150)
# ---------------- Fig 2: dynamics
runs=pickle.load(open("spiral_runs.pkl","rb"))
class S: pass
def get(key):
    t,yy=runs[key]; s=S(); s.t=t; s.y=yy; return s
k,f=0.2,0.5; s=get((k,f)); x=s.y[:3].T
r,R,phi,inc,pitch=diagnostics(s,f)
A,B,pf,q=flat(k,f)
fig=plt.figure(figsize=(12,7.6))
a=fig.add_subplot(2,3,1,projection="3d")
a.plot(x[:,0],x[:,1],x[:,2],"k",lw=0.5); a.set_box_aspect((1,1,0.6))
a.set_xlabel("x"); a.set_ylabel("y"); a.set_zlabel("z"); a.set_title(r"(a) trajectory, $\kappa=0.2$, $f=0.5$",fontsize=10)
a.tick_params(labelsize=7)
b=fig.add_subplot(2,3,2)
b.plot(x[:,0],x[:,1],"k",lw=0.5)
# fitted log spiral in the flat part (r = 3..30): ln r = c0 - tan(p) * phi
kk=(r>3)&(r<30); cfit=np.polyfit(phi[kk],np.log(r[kk]),1)
ph=np.linspace(phi[kk].min(),phi[kk].max(),2000); rr=np.exp(np.polyval(cfit,ph))
b.plot(rr*np.cos(ph),rr*np.sin(ph),"C3--",lw=0.8,label=fr"log spiral fit, pitch {np.degrees(np.arctan(-cfit[0])):.1f}$^\circ$")
b.set_aspect("equal"); b.set_xlim(-45,45); b.set_ylim(-45,45); b.legend(fontsize=8,loc="lower left")
b.set_xlabel(r"$x/r_c$"); b.set_ylabel(r"$y/r_c$"); b.set_title("(b) face-on",fontsize=10)
c=fig.add_subplot(2,3,3)
lat=np.degrees(np.arcsin(x[:,2]/r))
c.plot(phi/(2*np.pi),lat,"k",lw=0.6,label=r"latitude of $\hat r$")
i0=np.radians(inc[0]); env=np.degrees(2*np.arctan(np.tan(i0/2)*np.exp(-q*phi)))
c.plot(phi/(2*np.pi),env,"C3--",lw=0.9,label=r"$\pm i$, $\tan(i/2)\propto e^{-q\varphi}$")
c.plot(phi/(2*np.pi),-env,"C3--",lw=0.9)
c.set_xlabel("turns about the z-axis"); c.set_ylabel("latitude [deg]"); c.legend(fontsize=8)
c.set_title("(c) the path on the sphere of directions",fontsize=10)
d=fig.add_subplot(2,3,4)
for key,col in [((0.2,0.5),"k"),((0.3,0.5),"C0"),((0.2,0.0),"C1")]:
    ss=get(key); r2,R2,phi2,inc2,p2=diagnostics(ss,key[1]); A2,B2,pf2,q2=flat(*key)
    d.semilogy(phi2/(2*np.pi),np.tan(np.radians(inc2)/2),c=col,lw=1,label=fr"$\kappa={key[0]}$, $f={key[1]}$")
    d.semilogy(phi2/(2*np.pi),np.tan(np.radians(inc2[0])/2)*np.exp(-q2*phi2),c=col,ls=":",lw=1)
d.set_xlabel("turns"); d.set_ylabel(r"$\tan(i/2)$"); d.legend(fontsize=8); d.set_title("(d) inclination of the orbit (dotted: prediction)",fontsize=10)
e=fig.add_subplot(2,3,5)
bins=np.logspace(np.log10(0.15),np.log10(40),25); xc=np.sqrt(bins[1:]*bins[:-1])
xs=np.logspace(np.log10(0.15),np.log10(40),300)
h=1e-4; bs=(np.log(vc(xs*(1+h)))-np.log(vc(xs*(1-h))))/(2*h)
for key,col in [((0.2,0.5),"k"),((0.3,0.5),"C0"),((0.2,0.0),"C1")]:
    ss=get(key); r2,R2,phi2,inc2,p2=diagnostics(ss,key[1])
    med=[np.median(p2[(r2>bins[i])&(r2<bins[i+1])]) if ((r2>bins[i])&(r2<bins[i+1])).any() else np.nan for i in range(len(xc))]
    e.semilogx(xc,med,"o",c=col,ms=3)
    kap,ff=key; e.semilogx(xs,np.degrees(np.arctan(kap*(1-ff)/(1+bs))),c=col,lw=1,label=fr"$\kappa={kap}$, $f={ff}$")
e.axvline(1,c="0.7",ls=":"); e.set_xlabel(r"$r/r_c$"); e.set_ylabel("pitch angle [deg]")
e.set_title(r"(e) pitch: $\tan p=\kappa(1-f)/(1+{\rm d}\ln v_c/{\rm d}\ln R)$",fontsize=10); e.legend(fontsize=8)
g=fig.add_subplot(2,3,6)
for key,col,lab in [((0.2,0.5),"k","f=0.5: settles into the disc"),((0.2,0.0),"C1","f=0: stays in its tilted plane")]:
    ss=get(key); xx=ss.y[:3].T; g.plot(xx[:,1],xx[:,2],c=col,lw=0.5,label=lab)
g.set_xlabel(r"$y/r_c$"); g.set_ylabel(r"$z/r_c$"); g.legend(fontsize=8,loc="lower right"); g.set_title("(f) side view",fontsize=10)
g.set_xlim(-50,50); g.set_ylim(-45,45); g.set_aspect("equal")
fig.tight_layout(); fig.savefig("fig_spiral_model.png",dpi=150)
print("fit pitch",np.degrees(np.arctan(-cfit[0])), "pred",pf)
