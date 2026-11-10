import pandas as pd, numpy as np
d=pd.read_csv('experiments/2026-11-11_next_ideas/eval/idea3_mapping.csv')
i1=pd.read_csv('experiments/2026-11-11_next_ideas/eval/idea1_active_columns.csv').set_index(['input','shape'])
meas={('fem2d_permuted','128x64',32):1666.7,('fem2d_permuted','256x64',64):1733.1,('roadnet','128x64',32):5924.2,('diag_dd0.05','128x64',64):441.9}
EMPTY_ITEM=44; STEPA_PER_ELEM=25
rows=[]
for (inp,shape,p),m in meas.items():
    g=d[(d.input==inp)&(d['shape']==shape)&(d.p==p)].set_index('mode')
    passes=g.loc['now','passes']
    saved_rows=passes*(g.loc['now','warp_iters']-g.loc['sorted_skip','warp_iters'])*EMPTY_ITEM/1e6
    cols=i1.loc[(inp,shape),'x_sectors_now_M']*32/(512*4)*1e6
    act=i1.loc[(inp,shape),'x_sectors_compact_M']*32/(512*4)*1e6
    stepA=cols*512/32*STEPA_PER_ELEM/1e6
    rows.append(dict(input=inp,shape=shape,P=p,measured_inst_M=m,empty_row_frac=g.loc['now','empty_row_frac'],
        warp_items_now_M=passes*g.loc['now','warp_iters']/1e6, warp_items_skip_M=passes*g.loc['sorted_skip','warp_iters']/1e6,
        save_row_skip_M=saved_rows, stepA_now_M=stepA, save_vec4_M=stepA*0.75, save_compact_M=stepA*(1-act/cols),
        save_vec4_compact_M=stepA-stepA*(act/cols)/4))
t=pd.DataFrame(rows)
for c in ('save_row_skip_M','save_vec4_M','save_compact_M','save_vec4_compact_M'):
    t[c.replace('save','pct')]=100*t[c]/t.measured_inst_M
pd.set_option('display.width',250)
print(t.round(1).to_string(index=False))
t.to_csv('experiments/2026-11-11_next_ideas/eval/instruction_budget.csv',index=False)
