import pickle, numpy as np, fbshells, sys
from multiprocessing import Pool
RUNS={"base":dict(N=2000),"fb05":dict(N=2000,fb=0.05),"fb10":dict(N=2000,fb=0.10),
      "fb05fast":dict(N=2000,fb=0.05,Pfrac=0.01),"fb10far":dict(N=2000,fb=0.10,a_e=0.2),"fb20":dict(N=2000,fb=0.20),
      "fb05ae01":dict(N=2000,fb=0.05,a_e=0.01),"fb05ae02":dict(N=2000,fb=0.05,a_e=0.02),
      "fb02ae01":dict(N=2000,fb=0.02,a_e=0.01)}
def one(name):
    T=np.pi; snaps=list(np.linspace(0.7*T,T,16))
    out,n,tt=fbshells.run(Tend=T,snaps=snaps,**RUNS[name])
    pickle.dump(out,open(f'run_{name}.pkl','wb'))
    return f"{name} {RUNS[name]} steps {n} time {tt:.0f}s"
if __name__=="__main__":
    names=sys.argv[1:] or list(RUNS)
    with Pool(2) as p:
        for r in p.imap_unordered(one,names): print(r,flush=True)
