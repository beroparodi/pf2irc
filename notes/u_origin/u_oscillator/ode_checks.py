"""Checks of the statements on u(x) = 1 + (x-1) e^{-x} and its family u_b = 1 + (b x - 1) e^{-x}."""
import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
out=open("ode_checks_out.txt","w")
def P(s): print(s); out.write(s+"\n")
u =lambda x,b=1.0: 1+(b*x-1)*np.exp(-x)
du=lambda x,b=1.0: (b+1-b*x)*np.exp(-x)
d2u=lambda x,b=1.0: (b*x-2*b-1)*np.exp(-x)
xs=np.linspace(0,30,3001)
for b in [0.0,0.5,1.0,2.0]:
    r=d2u(xs,b)+2*du(xs,b)+u(xs,b)-1
    P(f"b={b}: max |u''+2u'+u-1| = {np.max(abs(r)):.1e};  u(0)={u(0,b):.1f}, u'(0)={du(0,b):.1f}")
c_num=lambda b: quad(lambda x: u(x,b)**2-1,0,60)[0]
for b in [0.0,0.5,0.873,1.0,1.5]:
    P(f"c({b}) numeric {c_num(b):+.6f}   formula b^2/4+3b/2-3/2 = {b*b/4+1.5*b-1.5:+.6f}")
P(f"threshold c=0: numeric b = {brentq(c_num,0.5,1.0):.6f}; sqrt(15)-3 = {np.sqrt(15)-3:.6f}")
m=lambda x: quad(lambda s: u(s)**2,0,x)[0]
P(f"P18: max of u at x=2: u = {u(2):.6f} = 1+e^-2 = {1+np.exp(-2):.6f};  u^2 = {u(2)**2:.4f}")
P(f"F = x - m(x): F(1) = {1-m(1):.4f};  zero at x = {brentq(lambda x: x-m(x),2,10):.3f};  F(40) = {40-m(40):+.4f} (-c = -0.25)")
P(f"added outside r_c: int_1^inf (u^2-1) = {quad(lambda x: u(x)**2-1,1,60)[0]:.4f} (= c + 1 - m(1))")
for b in [0.0,1.0]:
    a1=-(b+0.5)/(b+1)
    P(f"b={b}: centre u = (b+1)x - (b+1/2)x^2 + ...: rho/rho(0) ~ 1 + {2*a1:+.3f} x  (check at x=1e-4: {(u(1e-4,b)/((b+1)*1e-4))**2:.6f} vs {1+2*a1*1e-4:.6f})")
# Emden: linearised Lane-Emden about the SIS, delta_ss + delta_s + 2 delta = 0
r=np.roots([1,1,2]); P(f"isothermal sphere about the SIS: roots in ln(xi) = {r[0]:.4f}, {r[1]:.4f}  (-1/2 +- i sqrt7/2 = {np.sqrt(7)/2:.4f})")
out.close()
