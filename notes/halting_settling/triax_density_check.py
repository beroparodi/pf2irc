# Density implied by the triaxial potential (Laplacian / 4 pi G): positivity and axis ratios
import numpy as np, importlib.util
spec=importlib.util.spec_from_file_location("tc","triax_core.py"); tc=importlib.util.module_from_spec(spec); spec.loader.exec_module(tc)
def lap(x,p,q,h=1e-4):
    out=0
    for k in range(3):
        e=np.zeros(3); e[k]=h
        out+=(tc.accel((x+e)[None],0,p,q,0)[0][k]-tc.accel((x-e)[None],0,p,q,0)[0][k])/(2*h)
    return -out/(4*np.pi)
for p,q in [(0.9,0.8),(0.8,0.6)]:
    print(f"potential axis ratios of the exponential sphere {p}/{q}:")
    for r in [0.05,0.3,1,3,10]:
        rx,ry,rz=[lap(np.array(v)*r,p,q) for v in ([1,0,0],[0,1,0],[0,0,1])]
        rs=lap(np.array([1,0,0])*r,1,1)
        print(f"  r={r:5.2f}  rho_x/rho_sph={rx/rs:5.2f} rho_y={ry/rs:5.2f} rho_z={rz/rs:5.2f}  min>0: {min(rx,ry,rz)>0}")
