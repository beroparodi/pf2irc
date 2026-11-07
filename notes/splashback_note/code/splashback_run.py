"""1D collisionless shell run of the pf2irc seed for the splashback note.
Seed: N shells, masses 1e-4..1 (units G = v = 1), common energy -1/2, common bang time;
each shell receives j = sqrt2 lambda v R_ta at turnaround (lambda = 0.035); shells may cross.
This is the run 'seed035' of the main paper (Appendix H; code/shells/driver2.py there), with the same
settings: 16 snapshots between 0.7 and 1 times the turnaround time of the outermost shell,
saved to run_splash_<N>.pkl; the analysis is done by splashback_fig.py.
python3 splashback_run.py 3000      (about 30 minutes)

Uses shells.py of the main paper as corrected on 26 Oct 2026: the initial cycloid of each shell now uses the
same mass as the force on it (M - dm/2); before, the outer shells started with too much energy.
"""
import sys, pickle, numpy as np, shells
N = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
T = np.pi; snaps = list(np.linspace(0.7*T, T, 16))
out, n, tt = shells.run(N=N, Mmin=1e-4, Mmax=1.0, soft=2e-4, eta_dt=0.03, jmode='seed', lam=0.035, Tend=T, snaps=snaps)
pickle.dump(out, open(f"run_splash_{N}.pkl", "wb"))
print(f"N = {N}: {n} steps, {tt:.0f} s; snapshots written to run_splash_{N}.pkl")
