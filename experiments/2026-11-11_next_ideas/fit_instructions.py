import pandas as pd, numpy as np
from scipy.optimize import nnls
d=pd.read_csv('experiments/2026-11-11_next_ideas/eval/idea3_mapping.csv')
prof=pd.read_csv('experiments/2026-11-10_kc_p_tradeoff/analysis/report/profile/profile_summary.csv')
for name,lab,T in (('fem2d_permuted','fem2d_permuted_r128c64_rs_p{p}_t256_f4',256),('diag_dd0.05','diag_dd0.05_r128c64_rs_p{p}_t256_f2',256)):
    nb=len(np.load(f'experiments/2026-11-11_next_ideas/blocks/{name}__128x64.npz')['nnz'])
    m=d[(d.input==name)&(d['shape']=='128x64')].set_index(['mode','p'])
    rows=[]
    for p in (8,16,32,64,128):
        if ('now',p) not in m.index: continue
        r=m.loc[('now',p)]; y=float(prof[prof.label==lab.format(p=p)].warp_inst_executed.iloc[0])
        rows.append((p,y,r.passes*r.trips,r.passes*nb*T/32,r.passes*r.warp_iters))
    t=pd.DataFrame(rows,columns=['p','inst','pass_trips','pass_ctawarps','pass_items'])
    print(name,'blocks',nb); print(t.to_string(index=False))
    X=np.c_[np.ones(len(t)),t.pass_trips,t.pass_ctawarps]; c,_=nnls(X,t.inst.values)
    pred=X@c; print('NNLS c0=%.0fM per_trip=%.2f per_pass_ctawarp=%.1f  max rel err %.3f'%(c[0]/1e6,c[1],c[2],np.max(abs(pred-t.inst)/t.inst)))
    for mode in ('sorted','sorted_skip'):
        out=[]
        for p in t.p:
            r=m.loc[(mode,p)]; base=m.loc[('now',p)]
            now=c[0]+c[1]*base.passes*base.trips+c[2]*base.passes*nb*T/32
            new=c[0]+c[1]*r.passes*r.trips+c[2]*r.passes*nb*T/32
            out.append(f'P{p}:{new/now:.3f}')
        print(' ',mode,' '.join(out))
