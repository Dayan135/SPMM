"""Plot and analyze the completed FPT sweep (local, no CUDA required)."""
import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm


def gm(values):
    return float(np.exp(np.mean(np.log(values))))


def save_csv(path, rows):
    if rows:
        with path.open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader(); w.writerows(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, default=Path('tests/torch_fast_spdmm/benchmark/results/full_fpt_20261009/aggregate/summary.csv'))
    p.add_argument('--out', type=Path, default=Path('experiments/2026-10-09_full_fpt/plots'))
    a = p.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(a.input.open()))
    valid = [r for r in rows if r['correctness'] == 'passed' and r['median_ms']]
    boo = [r for r in valid if r['algorithm'] == 'boo' and r['scope'] == 'full_call']
    for r in boo:
        r['time'] = float(r['median_ms'])
    # Restore only explicitly recorded duplicate labels for factorial comparisons.
    # These are copied observations, never additional measurements or replicates.
    reused = 0
    for d in rows:
        if not d['duplicate_of']: continue
        for r in list(boo):
            if (r['task_index'],r['matrix'],r['features'],r['labeling']) == (d['task_index'],d['matrix'],d['features'],d['duplicate_of']):
                boo.append(dict(r, labeling=d['labeling'], k=d['labeling'].rsplit('k',1)[1], reused=True)); reused += 1
    direct = defaultdict(list)
    for r in valid:
        if r['algorithm']=='cusparse_direct' and r['scope']=='full_call' and not r['config'].startswith('direct_perm_'):
            direct[(r['task_index'],r['matrix'],r['features'],r['labeling'])].append(r)
    def baselines(r):
        key = (r['task_index'],r['matrix'],r['features'],r['labeling'])
        if key not in direct and r.get('reused'):
            key = (*key[:3],r['labeling'].replace('k64','k32'))
        return direct[key]
    matrices = sorted({r['matrix'] for r in boo}, key=lambda m:(m.startswith('generated/'),m))
    fs = [32,128,512]; methods=['identity','hem','graclus','metis']
    def label(m): return m.replace('generated/','').replace('_n1048576',' · 1M').replace('_n262144',' · 262k')
    plt.rcParams.update({'font.size':10, 'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':180})
    def finish(fig,name):
        fig.savefig(a.out/f'{name}.png',bbox_inches='tight')
        fig.savefig(a.out/f'{name}.pdf',bbox_inches='tight');plt.close(fig)
    winners=[]; heat=np.zeros((len(matrices),3)); annotations={}
    for i,m in enumerate(matrices):
        for j,f in enumerate(fs):
            cell=[r for r in boo if r['matrix']==m and int(r['features'])==f]
            b=min(cell,key=lambda r:r['time']); ds=baselines(b); dt=min(float(d['median_ms']) for d in ds)
            pooled=[d for k,v in direct.items() if k[1:3]==(m,str(f)) for d in v]
            pooled_dt=min(float(d['median_ms']) for d in pooled)
            ratio=dt/b['time'];heat[i,j]=np.log2(ratio)
            annotations[i,j]=f'{ratio:.2f}'+('*' if len(ds)<4 else '')
            winners.append(dict(matrix=m,features=f,cluster=b['cluster'],k=b['k'],fpt=b['features_per_thread'],boo_ms=b['time'],direct_ms=dt,speedup=ratio,direct_algorithms=len(ds),pooled_direct_ms=pooled_dt,pooled_speedup=pooled_dt/b['time']))
    save_csv(a.out/'best_configs.csv',winners)
    fig,ax=plt.subplots(figsize=(7,12))
    im=ax.imshow(heat,cmap='RdYlGn',norm=TwoSlopeNorm(vmin=-1.6,vcenter=0,vmax=1.6),aspect='auto')
    ax.set_xticks(range(3),fs);ax.set_yticks(range(len(matrices)),[label(m) for m in matrices]);ax.set_xlabel('Dense feature width F')
    for (i,j),s in annotations.items():ax.text(j,i,s,ha='center',va='center',fontsize=9)
    ax.axhline(5.5,color='black',lw=1.3)
    ax.set_title('Best BOO vs same-cell original cuSPARSE\nSpeedup > 1 favors BOO; * fewer than 4 direct algorithms')
    fig.colorbar(im,ax=ax,shrink=.5,label='log₂(cuSPARSE time / BOO time)');finish(fig,'01_speedup_overview')
    paired=[]
    groups=defaultdict(dict)
    for r in boo:groups[(r['matrix'],r['features'],r['cluster'],r['k'])][int(r['features_per_thread'])]=r
    for key, variants in groups.items():
        if set(variants)!={1,2,4}: continue
        for fpt in (2,4):
            b=variants[1];v=variants[fpt]
            paired.append(dict(matrix=key[0],features=int(key[1]),cluster=key[2],k=int(key[3]),fpt=fpt,speedup=b['time']/v['time'],nnz_per_block=float(b['stored_entries'])*float(b['compressed_fraction'])/max(1,float(b['num_blocks'])),remainder_fraction=float(b['remainder_nnz'])/float(b['stored_entries']),reused=bool(b.get('reused'))))
    save_csv(a.out/'paired_fpt.csv',paired)
    fig,axes=plt.subplots(1,2,figsize=(13,5),sharey=True)
    statistics=[]
    for ax,real in zip(axes,(True,False)):
        data=[]; ticks=[]
        for f in fs:
            for fpt in (2,4):
                vals=[r['speedup'] for r in paired if not r['reused'] and r['matrix'].startswith('generated/') != real and r['features']==f and r['fpt']==fpt]
                data.append(vals);ticks.append(f'F{f}\nFPT{fpt}')
                statistics.append(dict(family='real' if real else 'generated',features=f,fpt=fpt,cells=len(vals),geomean_speedup=gm(vals),min_speedup=min(vals),max_speedup=max(vals),regressions_over_5pct=sum(v<1/1.05 for v in vals)))
        ax.boxplot(data,tick_labels=ticks,showfliers=True);ax.axhline(1,color='black',ls='--');ax.set_yscale('log',base=2)
        ax.set_title('Real matrices' if real else 'Generated matrices');ax.set_ylabel('FPT1 time / variant time')
        ax.grid(axis='y',alpha=.2)
    fig.suptitle('Paired FPT gains at fixed matrix, clustering and k\nBoxes: quartiles and median; whiskers: 1.5 IQR; duplicate labels excluded')
    fig.tight_layout();finish(fig,'02_paired_fpt');save_csv(a.out/'fpt_statistics.csv',statistics)
    fig,ax=plt.subplots(figsize=(9,12)); ch=np.zeros((len(matrices),4))
    for i,m in enumerate(matrices):
        vals=[min(r['time'] for r in boo if r['matrix']==m and r['features']=='512' and r['cluster']==c) for c in methods]
        ch[i,:]=np.log2(vals[0]/np.array(vals))
    im=ax.imshow(ch,cmap='RdYlGn',norm=TwoSlopeNorm(vmin=min(-1,float(ch.min())),vcenter=0,vmax=max(1,float(ch.max()))),aspect='auto')
    ax.set_xticks(range(4),methods);ax.set_yticks(range(len(matrices)),[label(m) for m in matrices])
    for i in range(len(matrices)):
        for j in range(4):ax.text(j,i,f'{2**ch[i,j]:.2f}',ha='center',va='center')
    ax.set_title('Clustering at F=512: identity time / method time\nEach method chooses its fastest k and FPT; cross-allocation comparison')
    fig.colorbar(im,ax=ax,shrink=.5,label='log₂ speedup over identity');finish(fig,'03_clustering')
    fig,ax=plt.subplots(figsize=(9,12));kh=np.zeros((len(matrices),4)); krows=[]
    for i,m in enumerate(matrices):
        for j,c in enumerate(methods):
            best={k:min(r['time'] for r in boo if r['matrix']==m and r['features']=='512' and r['cluster']==c and r['k']==str(k)) for k in (32,64)}
            ratio=best[32]/best[64];kh[i,j]=np.log2(ratio)
            krows.append(dict(matrix=m,cluster=c,k32_ms=best[32],k64_ms=best[64],k32_over_k64=ratio))
    im=ax.imshow(kh,cmap='RdYlGn',norm=TwoSlopeNorm(vmin=min(-1,float(kh.min())),vcenter=0,vmax=max(1,float(kh.max()))),aspect='auto')
    ax.set_xticks(range(4),methods);ax.set_yticks(range(len(matrices)),[label(m) for m in matrices])
    for i in range(len(matrices)):
        for j in range(4):ax.text(j,i,f'{2**kh[i,j]:.2f}',ha='center',va='center')
    ax.set_title('k choice at F=512, best FPT for each k\nk32 time / k64 time: >1 favors k64; <1 favors k32')
    fig.colorbar(im,ax=ax,shrink=.5,label='log₂(k32 time / k64 time)');finish(fig,'04_group_size');save_csv(a.out/'group_size.csv',krows)
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    for real,color,name in [(False,'#aaa','Generated'),(True,'#0072B2','Real')]:
        subset=[r for r in paired if r['features']==512 and r['fpt']==4 and not r['reused'] and r['matrix'].startswith('generated/') != real]
        for ax,xfield in zip(axes,['nnz_per_block','remainder_fraction']):
            ax.scatter([r[xfield] for r in subset],[r['speedup'] for r in subset],s=25,alpha=.65,c=color,label=name)
    axes[0].set_xscale('log');axes[0].set_xlabel('Mean stored entries per BOO block')
    axes[1].set_xlabel('Fraction of entries in CSR remainder')
    for ax in axes:ax.axhline(1,color='black',ls='--');ax.set_ylabel('FPT1 / FPT4 time');ax.legend()
    fig.suptitle('Structure vs paired FPT4 benefit at F=512\nAssociation only; each point is a matrix / clustering / k configuration')
    fig.tight_layout();finish(fig,'05_structure')
    scaling=[]
    for small in winners:
        if small['features'] != 512 or not small['matrix'].endswith('_n262144'): continue
        large=next(r for r in winners if r['matrix']==small['matrix'].replace('_n262144','_n1048576') and r['features']==512)
        scaling.append(dict(profile=small['matrix'].replace('generated/','').replace('_n262144',''),boo_time_growth=large['boo_ms']/small['boo_ms'],direct_time_growth=large['pooled_direct_ms']/small['pooled_direct_ms']))
    fig,ax=plt.subplots(figsize=(11,5));x=np.arange(len(scaling))
    ax.bar(x-.18,[r['boo_time_growth'] for r in scaling],.36,label='Best BOO')
    ax.bar(x+.18,[r['direct_time_growth'] for r in scaling],.36,label='Fastest observed original cuSPARSE')
    ax.axhline(4,color='black',ls='--',label='4× matrix dimension')
    ax.set_xticks(x,[r['profile'] for r in scaling],rotation=35,ha='right')
    ax.set_ylabel('1M runtime / 262k runtime');ax.legend(fontsize=8)
    ax.set_title('Size scaling at F=512; best configuration selected separately at each size\nNoise density is fixed: noisy inputs can grow faster than 4× in entry count')
    finish(fig,'06_size_scaling');save_csv(a.out/'size_scaling.csv',scaling)
    lines=['# Full FPT sweep analysis','',
        'Complete-call timings. All 104 GPU tasks completed. 1,836 measured BOO combinations plus 36 explicitly reused duplicate-label combinations. '
        'No extra measurements are inferred for those duplicates. Selection is post-hoc; allocations were not independently replicated.','',
        '## Plots','',*[f'- [{n}](./{n}.png) (also PDF)' for n in ['01_speedup_overview','02_paired_fpt','03_clustering','04_group_size','05_structure','06_size_scaling']],'',
        '## Baseline sensitivity','',
        '| Comparison | Winning matrix / F cells | Total |','|---|---:|---:|',
        f"| Best BOO vs same-cell direct | {sum(r['speedup']>1 for r in winners)} | {len(winners)} |",
        f"| Same-cell direct, all four algorithms available | {sum(r['speedup']>1 and r['direct_algorithms']==4 for r in winners)} | {sum(r['direct_algorithms']==4 for r in winners)} |",
        f"| Best BOO vs fastest direct observed across all label shards | {sum(r['pooled_speedup']>1 for r in winners)} | {len(winners)} |",'',
        'The pooled comparison is a sensitivity check across allocations, not a paired measurement. '
        'Missing algorithms can make apparent wins optimistic. See best_configs.csv for both ratios.','',
        '## Paired FPT behavior','',
        '| Family | F | FPT | Cells | Geomean speedup | Worst speedup | >5% regressions |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for r in statistics:lines.append(f"| {r['family']} | {r['features']} | {r['fpt']} | {r['cells']} | {r['geomean_speedup']:.3f} | {r['min_speedup']:.3f} | {r['regressions_over_5pct']} |")
    lines+=['','Ratios below one are regressions. Geomeans weight configurations equally, not by runtime or application frequency.', '',
            '## Best configuration at F=512','',
            '| Matrix | Method | k | FPT | Same-cell speedup | Pooled-baseline speedup |',
            '|---|---|---:|---:|---:|---:|']
    for r in winners:
        if r['features']==512:lines.append(f"| {label(r['matrix'])} | {r['cluster']} | {r['k']} | {r['fpt']} | {r['speedup']:.3f} | {r['pooled_speedup']:.3f} |")
    (a.out/'README.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))


if __name__=='__main__':main()
