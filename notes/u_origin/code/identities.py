"""
Checks of every formula and number of Sects. 2-6 of the note (output identities_out.txt).
Units: x = R/r_c; masses in v^2 r_c/G; energies per unit mass in v^2.
"""
import numpy as np
from scipy.integrate import quad, solve_ivp
from scipy.optimize import brentq, minimize_scalar
from scipy.special import i0, i1, k0, k1
e=np.exp
u   =lambda x: 1+(x-1)*e(-x)                    # P18
ub  =lambda x,b: 1+(b*x-1)*e(-x)                # regular family
mu  =lambda x: 1-(1+x)*e(-x)                    # exponential disc, R_d = r_c: enclosed-mass fraction
mu1 =lambda x: x*e(-x)                          # mu'  = mass per unit radius (annulus mass)
mu2 =lambda x: (1-x)*e(-x)                      # mu''
mu3 =lambda x: (x-2)*e(-x)                      # mu'''
def m(x): return x*(1-e(-x))**2-0.5*x*(1+x)*e(-2*x)+0.25*(1-e(-2*x))   # int_0^x u^2
out=open("identities_out.txt","w")
def P(*a): print(*a); print(*a,file=out)
xs=np.linspace(1e-4,15,3001); h=1e-5
num=lambda f,x:(f(x+h)-f(x-h))/(2*h)

P("== Sect. 2: the ODE and the regular family")
up=num(u,xs); upp=(u(xs+h)-2*u(xs)+u(xs-h))/h**2
P(f"max |u''+2u'+u-1| = {np.abs(upp+2*up+u(xs)-1).max():.1e};  u(0) = {u(0.0)}, u'(0) = {num(u,1e-6):.6f}")
P(f"max |m' - u^2| = {np.abs(num(m,xs)-u(xs)**2).max():.1e}")
cb=lambda b: quad(lambda x: ub(x,b)**2-1,0,np.inf)[0]
for b in (0.5,1.0,2.0): P(f"b = {b}: c(b) numerical {cb(b):.6f}, formula b^2/4+3b/2-3/2 = {b*b/4+1.5*b-1.5:.6f}")
P(f"c(b) = 0 at b = {brentq(cb,0.5,1.5):.6f}  (sqrt15 - 3 = {np.sqrt(15)-3:.6f})")
P(f"P18 maximum: u(2) = {u(2):.6f} = 1+e^-2, u^2 = {u(2)**2:.4f}")
P(f"centre: (u - 2x + 1.5x^2)/x^3 at x = 1e-3: {(u(1e-3)-2e-3+1.5e-6)/1e-9:.3f} (finite)")
P(f"displaced mass F = x - m: F(1) = {1-m(1):.4f}; zero at x = {brentq(lambda x:x-m(x),2,5):.3f}; F(40) = {40-m(40):.4f}")
P(f"added outside r_c: int_1^oo (u^2-1) = {quad(lambda x:u(x)**2-1,1,np.inf)[0]:.4f}")

P("\n== Sect. 2: the non-singular isothermal sphere with the same central density (x = xi/sqrt8)")
sol=solve_ivp(lambda s,y:[y[1],np.exp(-y[0])-2*y[1]/s],[1e-6,400],[1e-12/6,1e-6/3],rtol=1e-10,atol=1e-12,dense_output=True)
xi=np.geomspace(0.05,400,200000); ratio=xi**2*np.exp(-sol.sol(xi)[0])/2
i=np.argmax(ratio); j=i+np.argmin(ratio[i:np.searchsorted(xi,200)])
P(f"first maximum rho/rho_SIS = {ratio[i]:.3f} at x = {xi[i]/np.sqrt(8):.2f}; first minimum {ratio[j]:.3f} at x = {xi[j]/np.sqrt(8):.2f}")
P(f"linearised roots of d_ss + d_s + 2 d = 0: {np.roots([1,1,2])}")

P("\n== Sect. 3: the disc representation")
P(f"max |mu' - numerical| = {np.abs(mu1(xs)-num(mu,xs)).max():.1e};  max |mu'' - numerical| = {np.abs(mu2(xs)-num(mu1,xs)).max():.1e}")
P(f"max |u - (1 - mu'')| = {np.abs(u(xs)-(1-mu2(xs))).max():.1e};  max |u - (mu + 2mu')| = {np.abs(u(xs)-(mu(xs)+2*mu1(xs))).max():.1e}")
for b in (0.0,0.5,2.0):
    P(f"b = {b}: max |u_b - 1 - (-mu'' + (b-1) mu')| = {np.abs(ub(xs,b)-1-(-mu2(xs)+(b-1)*mu1(xs))).max():.1e}")
P(f"(D+1) e^-x = {np.abs(-e(-xs)+e(-xs)).max()};  max |(D+1) mu' - e^-x| = {np.abs(mu2(xs)+mu1(xs)-e(-xs)).max():.1e}")
P(f"annulus mass mu' peaks at x = {minimize_scalar(lambda x:-mu1(x),bounds=(0.1,5),method='bounded').x:.4f}; "
  f"mu'' most negative at x = {minimize_scalar(mu2,bounds=(0.5,6),method='bounded').x:.4f}; mu'''(2) = {mu3(2.0)}")

P("\n== Sect. 4: mass conservation at first order and the excess at second order")
for b in (0.5,1.0,2.0):
    P(f"b = {b}: int (u_b - 1) = {quad(lambda x:ub(x,b)-1,0,np.inf)[0]:+.6f} (b-1 = {b-1:+.1f});  "
      f"int (u_b-1)^2 = {quad(lambda x:(ub(x,b)-1)**2,0,np.inf)[0]:.6f} (b^2/4-b/2+1/2 = {b*b/4-b/2+0.5:.6f})")
F2=lambda x: 2*mu1(x)-quad(lambda t:mu2(t)**2,0,x)[0]
P(f"max |F - (2mu' - int_0^x mu''^2)| = {max(abs((x-m(x))-F2(x)) for x in np.linspace(0,15,151)):.1e}")
P(f"at x = 1: 2mu' = {2*mu1(1):.4f}, int_0^1 mu''^2 = {quad(lambda t:mu2(t)**2,0,1)[0]:.4f}, F = {1-m(1):.4f}")

P("\n== Sect. 5: first-order reading")
P(f"xi/r_c = 2 x e^-x: maximum {2*mu1(1):.3f} at x = 1;  eps/v^2 = 2 e^-x: eps(0) = 2, eps(1) = {2*e(-1):.3f}")
P(f"total int eps dm_SIS = int_0^oo 2 e^-x dx = {quad(lambda x:2*e(-x),0,np.inf)[0]:.3f} v^2 (v^2 r_c/G)")
P(f"|u-1| at x = 0.5, 1, 1.5, 2, 3: {np.abs(u(np.array([0.5,1,1.5,2,3]))-1).round(3)}")
v,rc,G=73.0,2.4,4.301e-6; Msc=v*v*rc/G
P(f"median dwarf (v = 73 km/s, r_c = 2.4 kpc): v^2 r_c/G = {Msc:.2e} Msun; 2 v^2 * that = {2*Msc*1.989e33*(v*1e5)**2:.1e} erg")

P("\n== Sect. 6: Jeans equation with sigma_r^2 = v^2/2: beta = 1 - x u'/u - m/x")
beta=lambda x: 1-x*(2-x)*e(-x)/u(x)-m(x)/x
for x in (0.01,0.1,0.3,0.5,1,1.5,2,2.5,3,4,6,10,20,50): P(f"  x = {x:5}: beta = {beta(x):+.3f}")
xx=np.linspace(0.01,60,60000); bb=beta(xx)
P(f"max beta = {bb.max():.3f} at x = {xx[bb.argmax()]:.2f}; min beta = {bb.min():.3f} at x = {xx[bb.argmin()]:.2f}")

P("\n== Sect. 7: the disc in the STH collapse")
xi_b=1/(1+1.5*np.pi)
P(f"xi_b = 1/(1+3pi/2) = {xi_b:.4f}; orbital period at r_c / t = 2 pi xi_b / y = {2*np.pi*xi_b/34:.3f} (y = 34)")
for lam,y in [(0.025,39),(0.035,34),(0.05,28)]:
    P(f"  lambda = {lam}: lambda y = {lam*y:.2f}; R_d/r_c (MMW, f_j=1) = {lam*y/np.sqrt(2):.2f}; seed disc x1.06-1.4: "
      f"{1.06*lam*y/np.sqrt(2):.2f}-{1.4*lam*y/np.sqrt(2):.2f};  v_rot/sigma(DM) <= (2 sqrt2 lambda v)/(v/sqrt2) = 4 lambda = {4*lam:.3f}")
fb=0.0224/(0.0224+0.120); P(f"cosmic baryon fraction Omega_b/Omega_m = {fb:.3f}")
y=34; Mall=fb*y; Mret=0.017*y
P(f"all baryons inside the front: f_b y = {Mall:.2f}; retained (1.7%): 0.017 y = {Mret:.2f}; P18 deficit at r_c: {1-m(1):.3f}")
vd2=lambda R: 2*(R/2)**2*(i0(R/2)*k0(R/2)-i1(R/2)*k1(R/2))            # thin exponential disc: V^2 in G M_d/R_d
s=minimize_scalar(lambda R:-vd2(R),bounds=(0.5,5),method='bounded')
P(f"thin exponential disc: V_max^2 = {-s.fun:.4f} G M_d/R_d at R = {s.x:.2f} R_d; all baryons (R_d = r_c): V_max/v = {np.sqrt(-s.fun*Mall):.2f}; "
  f"retained: {np.sqrt(-s.fun*Mret):.2f}")
P(f"lifting the expelled gas from r_c to the front: ({Mall:.2f} - {Mret:.2f}) x ln y = {(Mall-Mret)*np.log(y):.1f} v^2 (v^2 r_c/G), "
  f"against 2 for the dark matter")
P("shielding: dark mass M_dm = r, disc M_d mu(r/R_d) -> ratio M_d mu(r)/r:")
for r in (0.01,0.1,0.3,1.0,3.0): P(f"  r/R_d = {r}: M_d mu/M_dm = {mu(r)/r:.4f} M_d   (small-r limit r/2 = {r/2:.4f})")
out.close()
