# Controls for triax_core.py: (1) isotropic population with the same apocentre energies in the
# spherical potential (what a fully scattered population would give); (2) dt halved and
# (3) another random seed for the strongest static case.
import numpy as np, importlib.util
spec=importlib.util.spec_from_file_location("tc","triax_core.py"); tc=importlib.util.module_from_spec(spec); spec.loader.exec_module(tc)
# (1) isotropic: same energies as radial orbits from r_a, but velocity direction random
orig=tc.setup
def setup_iso(N):
    x,vv,ra=orig(N)
    E=tc.Phi_spher(ra)            # energy of a particle at rest at r_a (j small)
    r=np.linalg.norm(x,axis=1)
    # place each particle at a random radius inside r_a with random direction, speed from E
    rr=ra*tc.rng.uniform(0,1,N)**(1/3)
    u=tc.rng.normal(size=(N,3)); u/=np.linalg.norm(u,axis=1)[:,None]
    sp=np.sqrt(np.maximum(2*(E-tc.Phi_spher(rr)),0))
    w=tc.rng.normal(size=(N,3)); w/=np.linalg.norm(w,axis=1)[:,None]
    return rr[:,None]*u, sp[:,None]*w, ra
tc.setup=setup_iso
d=tc.run(1,1,0,0,seedphase=False); print(f"isotropic control (spherical)      slope(0.1-0.5)={d['slope']:+.2f} beta(<rc)={d['beta']:+.2f}",flush=True)
tc.setup=orig
d=tc.run(0.8,0.6,0,20,dt=0.002); print(f"triax 0.8/0.6 static, dt/2         slope(0.1-0.5)={d['slope']:+.2f} beta(<rc)={d['beta']:+.2f} j/jc(ra<3)={d['jr1_in']:.3f}",flush=True)
tc.rng=np.random.default_rng(7)
d=tc.run(0.8,0.6,0,20); print(f"triax 0.8/0.6 static, other seed   slope(0.1-0.5)={d['slope']:+.2f} beta(<rc)={d['beta']:+.2f} j/jc(ra<3)={d['jr1_in']:.3f}",flush=True)
tc.rng=np.random.default_rng(7)
d=tc.run(1,1,0,0); print(f"spherical, other seed              slope(0.1-0.5)={d['slope']:+.2f} beta(<rc)={d['beta']:+.2f}",flush=True)
