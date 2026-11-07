"""Circular speed of the excess mass beyond the accretion front, for comparison with stacked weak lensing
(Sect. 8.3). Self-similar growth without Lambda (Sect. 5.1): at time t the shell at cycloid phase eta encloses
M = v^3 t / [G (eta - sin eta)] at xi = R/vt = (1 - cos eta)/(eta - sin eta). Lensing measures the mass in excess
of the cosmic mean; the Einstein-de Sitter background, rho = 1/(6 pi G t^2), contributes (2/9) xi^3 in these units.
Also: the front today if accretion continued until today, R_b = xi_b v t0/(1 + c_Lambda l0), Eq. (8).
python3 lensing_check.py   (output lensing_check_out.txt)"""
import numpy as np
out = open("lensing_check_out.txt", "w")
def P(*a): print(*a); print(*a, file=out)
eta = np.linspace(1e-3, 1.5*np.pi, 400000)
xi = (1-np.cos(eta))/(eta-np.sin(eta)); M = 1/(eta-np.sin(eta)); dM = M-(2/9)*xi**3
xb = 1/(1+1.5*np.pi)
P("R/R_b   v_c/v (excess mass over the cosmic mean)   v_c/v (total)")
for f in (1, 1.5, 2, 3, 4, 5, 6, 8):
    i = np.argmin(abs(xi-f*xb)); P(f"{f:5.1f}   {np.sqrt(dM[i]/xi[i]):.3f}   {np.sqrt(M[i]/xi[i]):.3f}")
# the front today (H0 = 70, Omega_m = 0.3, flat); 1 km/s * 1 Gyr = 1.0227 kpc
H0 = 70/977.8                                # 1/Gyr
OL = 0.7; t0 = 2/(3*H0*np.sqrt(OL))*np.arcsinh(np.sqrt(OL/0.3))
L3 = OL*H0**2                                # Lambda/3 in 1/Gyr^2
l0 = L3*xb**2*t0**2; cL = 9.640
P(f"\nt0 = {t0:.2f} Gyr, l0(t0) = {l0:.4f}, lag factor 1 + c_L l0 = {1+cL*l0:.3f}")
for v in (60, 75, 100, 150):
    Rb = xb*v*t0*1.0227/(1+cL*l0)
    f = 1000/Rb; i = np.argmin(abs(xi-f*xb))
    P(f"v = {v:3d} km/s: R_b(today) = {Rb:5.0f} kpc; 1 Mpc = {f:.1f} R_b; v_c(1 Mpc)/v = {np.sqrt(dM[i]/xi[i]):.2f}")
out.close()
