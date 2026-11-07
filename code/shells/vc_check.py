import pickle, numpy as np, sys
XB=1/(1+1.5*np.pi)
for n in sys.argv[1:]:
    out=pickle.load(open(f'run_{n}.pkl','rb'))
    a={f:[] for f in (0.1,0.3,0.6,1.0)}; fo=[]; mi=[]
    for t,s in out.items():
        r,dm,M=s['r'],s['dm'],s['M']; t=s['t']
        for f in a: a[f].append(dm[r<f*XB*t].sum()/(f*XB*t))
        orb=t>2*np.pi*M; fo.append(dm[orb&(r>XB*t)].sum()/dm[orb].sum()); mi.append(dm[r<XB*t].sum()/(XB*t))
        i=np.argmax(M)
    print(n,' '.join(f'vc2/v2({f}Rb)={np.mean(v):.2f}' for f,v in a.items()),'| orbiting outside R_b %.3f | M(<R_b)/M_h %.3f'%(np.mean(fo),np.mean(mi)), '| outer shell r at last snap %.3f'%r[i])
