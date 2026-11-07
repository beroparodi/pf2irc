# P18 functions accurate at all x (series below x = 0.05). Units G = v = r_c = 1.
import numpy as np
from math import factorial
N=16
cu=np.zeros(N+1)
for n in range(1,N+1): cu[n]=(-1)**(n-1)*(n+1)/factorial(n)   # u = sum cu_n x^n
cu2=np.convolve(cu,cu)[:N+1]                                      # u^2
cm=np.zeros(N+2); cm[1:]=cu2/np.arange(1,N+2)                     # m = int u^2
def _poly(c,x): return np.polyval(c[::-1],x)
def u(x):
    x=np.asarray(x,float); return np.where(x<0.05,_poly(cu,x),1+(x-1)*np.exp(-x))
def m(x):
    x=np.asarray(x,float)
    big=x*(-np.expm1(-x))**2+0.25*(1-(1+2*x+2*x*x)*np.exp(-2*x))
    return np.where(x<0.05,_poly(cm,x),big)
def rho(x): x=np.asarray(x,float); return u(x)**2/(4*np.pi*x*x)
if __name__=="__main__":
    for x in [0.01,0.049,0.051,0.5]:
        h=1e-6; print(x,u(x),m(x),(m(x+h)-m(x-h))/(2*h)-u(x)**2)
