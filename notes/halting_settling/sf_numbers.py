"""Numbers for the section 'Can star formation and supernovae halt the shells?' of the tophats note."""
import numpy as np
G = 6.674e-8; mp = 1.6726e-24; X = 0.76; Mpc = 3.0857e24
H0 = 70e5/Mpc; Om, Ob = 0.3, 0.048
rho_c0 = 3*H0**2/(8*np.pi*G)
out = [f"rho_crit,0 = {rho_c0:.3e} g/cm^3 (H0 = 70)"]
for z in (0.5, 1.0, 2.0):
    rho_m = Om*rho_c0*(1 + z)**3
    mean_in = 147*rho_m                 # mean density inside the front (paper Sect. 3.3)
    loc = mean_in/3                     # local density at the front for M ~ R (SIS): rho = (1/3) mean
    fb = Ob/Om
    nH_loc = X*fb*loc/mp; nH_mean = X*fb*mean_in/mp
    out.append(f"z = {z}: rho_m = {rho_m:.2e}; inside front mean {mean_in:.2e}, at front {loc:.2e} g/cm^3; "
               f"gas (cosmic f_b = {fb:.2f}) n_H at front {nH_loc:.1e}, mean inside {nH_mean:.1e} cm^-3")
# SN energy per unit stellar mass: 1e51 erg per 100 Msun
Msun = 1.989e33
e_sn = 1e51/(100*Msun); out.append(f"SN energy per unit stellar mass = {e_sn:.2e} erg/g = ({np.sqrt(e_sn)/1e5:.0f} km/s)^2")
v = 60.0
out.append(f"epsilon_fb = 2 f_c (710/v)^2 Delta M*/M = {2*(710/v)**2:.0f} f_c Delta M*/M at v = {v:.0f} km/s")
out.append("paper (robustness_results.txt): feedback per unit f_c median 0.65 (16-84%: 0.24-4.0); cooling -0.085 (-0.18 to -0.03); "
           "scalings with v: feedback ~ v^-2.5, cooling ~ v^-0.6")
# indirect effect on dark matter at the front: potential change from moving a gas fraction f of the mass
for f in (0.017, 0.16):
    out.append(f"moving a mass fraction {f} changes Phi at the front by at most ~{f:.3f} v^2 per event (vs 1/2 v^2 to be turned)")
print("\n".join(out)); open("sf_numbers_out.txt", "w").write("\n".join(out) + "\n")
