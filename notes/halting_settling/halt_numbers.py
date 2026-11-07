"""Numbers for the section 'Why the shells do not halt by themselves' of the tophats note (units G = M = v = 1)."""
import numpy as np
from scipy.optimize import brentq
from scipy.integrate import quad
Phi = lambda R: np.log(R) - 1 if R <= 1 else -1/R
def peri(j, E):
    return brentq(lambda R: E - Phi(R) - j**2/(2*R**2), 1e-12, 1.0)
def radial_period(j, E, Rp, Ra):
    # T_r = 2 int dR / sqrt(2(E-Phi) - j^2/R^2), substitution to remove endpoint singularities
    f = lambda R: 2*(E - Phi(R)) - j**2/R**2
    g = lambda th: (Ra - Rp)/2*np.sin(th)/np.sqrt(max(f(Rp + (Ra - Rp)/2*(1 - np.cos(th))), 1e-300))
    return 2*quad(g, 0, np.pi, limit=400)[0]
out = []
out.append("(2) circular orbit in the settled galaxy (SIS, flat speed v): E(R) = 1/2 + ln(R/R_b) - 1; equals -1/2 only at R = R_b")
for R in (0.5, 1.0, 2.0):
    Ec = 0.5 + np.log(R) - 1 if R <= 1 else -0.5/R          # outside R_b: Kepler, v_c^2 = GM/R
    out.append(f"    R = {R} R_b: E_circ = {Ec:+.3f} v^2" + ("" if R <= 1 else "  (outside the galaxy: Kepler, -GM/2R)"))
out.append("(4) seed spin lambda -> j = 2 sqrt2 lambda GM/v, energy E = -1/2 + j^2/8, pericentre, apocentre 2 R_b, radial period")
for lam in (0.021, 0.035, 0.058, 0.07, 1/(2*np.sqrt(2))):
    j = 2*np.sqrt(2)*lam; E = -0.5 + j**2/8; Rp = peri(j, E)
    Tr = radial_period(j, E, Rp, 2.0)
    out.append(f"    lambda = {lam:.3f}: j = {j:.3f} GM/v, E = {E:+.4f} v^2, pericentre {Rp:.4f} R_b (R_b/{1/Rp:.1f}), "
               f"radial period {Tr:.2f} GM/v^3")
# epicycles at R_b inside SIS: kappa = sqrt2 Omega; core: omega = 2/sqrt3 v/r_c
v, rc, Rb, M = 60.0, 2.44, 84.2, 7.1e10   # example galaxy, km/s, kpc, Msun
GYR = 0.977792
Om = v/Rb; kap = np.sqrt(2)*Om; w = 2/np.sqrt(3)*v/rc
out.append(f"example galaxy (v = 60 km/s, R_b = 84.2 kpc, r_c = 2.44 kpc, growth ends z = 0.5, t = 8.43 Gyr):")
out.append(f"    j_circ = v R_b = {v*Rb:.0f} kpc km/s; seed j (lambda=0.035) = {0.099*v*Rb:.0f} kpc km/s")
out.append(f"    orbital period at R_b 2pi/Omega = {2*np.pi/Om*GYR:.2f} Gyr; epicyclic (radial) period 2pi/kappa = {2*np.pi/kap*GYR:.2f} Gyr")
out.append(f"    core oscillator omega = (2/sqrt3) v/r_c = {w:.2f} km/s/kpc; period 2pi/omega = {2*np.pi/w*GYR*1000:.0f} Myr")
out.append(f"    radial period of the seed-spin orbit = 6.48 GM/v^3 = {6.48*Rb/v*GYR:.1f} Gyr for the outermost shell (GM/v^3 = R_b/v = {Rb/v*GYR:.2f} Gyr)")
jc = 0.099; Ecj = 0.5 + np.log(jc) - 1
out.append(f"    circular orbit with the seed's j = 0.099 lies at R = j = 0.099 R_b with E = {Ecj:+.2f} v^2; the shell's E = -0.50 exceeds it by {-0.5-Ecj:.2f} v^2")
# shock temperature for gas falling at v
mp, k, mu = 1.6726e-24, 1.3807e-16, 0.59
for vin in (60.0, 100.0):
    T = 3/16*mu*mp*(vin*1e5)**2/k
    out.append(f"    strong accretion shock at infall speed {vin:.0f} km/s: T = 3 mu m_p v^2/(16 k) = {T:.2e} K")
# torque estimate: tangential force eps GM/R^2 over a dynamical time R/v changes j by ~ eps GM/v
out.append("(6) torque estimate: Delta j ~ eps (GM/R^2) R (R/v) = eps GM/v; raising j from 0.1 to 1 GM/v needs eps ~ 1")
print("\n".join(out)); open("halt_numbers_out.txt", "w").write("\n".join(out) + "\n")
