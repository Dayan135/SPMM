"""Aggregate the three Phase B jobs; diagnostic arms stay separate."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics as st
import math

from report_phase_a import read, write, key, values, stats, cell


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('results',type=Path);p.add_argument('--out',type=Path,default=Path('.'))
    a=p.parse_args()
    raw=[];samples=[];validation=[];metadata=[]
    groups=defaultdict(lambda:defaultdict(list))
    for j in (1,2,3):
        root=a.results/f'phase_b_job{j}'
        m=json.loads((root/'metadata.json').read_text());metadata.append(m)
        assert m['status']=='complete' and m['completed_cells']==84 and m['completed_validation_cells']==84
        for name,collection in [('summary',raw),('samples',samples),('validation',validation)]:
            rows=read(root/f'{name}.csv')
            for r in rows:
                assert int(r['replicate'])==j
                r.update(slurm_job_id=m['slurm_job_id'],node=m['node'])
                if name=='summary':groups[key(r)][r['config']].append(r)
                collection.append(r)
    assert len(groups)==84 and len(raw)==84*3*40 and len(samples)==len(raw)*20
    assert len(validation)==len(raw)
    for r in validation:
        assert r['correctness']==('diagnostic_not_correct' if 'diagnostic' in r['family'] else 'passed')
    assert len({m['slurm_job_id'] for m in metadata})==3
    for field in ('git_commit','source_sha256','extension_sha256'):
        assert len({m[field] for m in metadata})==1,field
    selected=[];diagnostics=[];paired=defaultdict(list)
    for workload,configs in sorted(groups.items(),key=lambda item:(item[0][1],item[0][2],item[0][0])):
        assert len(configs)==40 and all(len(v)==3 for v in configs.values())
        for field in ('matrix_sha256','x_sha256','rows','cols','source_sha256'):
            assert len({r[field] for v in configs.values() for r in v})==1,(workload,field)
        best={family:min((v for v in configs.values() if v[0]['family']==family),key=lambda v:st.median(values(v)))
              for family in ('direct_cpp','pbr_cpp','boo_cpp','rowexcl_cpp')}
        row=dict(profile=workload[0],requested_size=workload[1],features=workload[2])
        for family,rows in best.items():
            row[family+'_config']=rows[0]['config'];stats(row,family+'_ms',values(rows))
            if family!='direct_cpp':
                row[family+'_compressed_fraction']=rows[0]['compressed_fraction']
                row[family+'_execution_path']=rows[0]['execution_path']
                ratio=[d/t for d,t in zip(values(best['direct_cpp']),values(rows))]
                stats(row,family+'_speedup_vs_direct',ratio)
                paired[(workload[1],family+'_speedup_vs_direct')].append(ratio)
        for baseline in ('pbr_cpp','boo_cpp'):
            ratio=[b/t for b,t in zip(values(best[baseline]),values(best['rowexcl_cpp']))]
            stats(row,'rowexcl_speedup_vs_'+baseline,ratio)
            paired[(workload[1],'rowexcl_speedup_vs_'+baseline)].append(ratio)
        # Matched grouping avoids interpreting HEM selection as kernel benefit.
        target=best['rowexcl_cpp'][0]['k']
        same=next(v for v in configs.values() if v[0]['family']=='boo_cpp' and v[0]['k']==target)
        row['same_k_boo_config']=same[0]['config']
        ratio=[b/t for b,t in zip(values(same),values(best['rowexcl_cpp']))]
        stats(row,'rowexcl_speedup_vs_same_k_boo',ratio)
        paired[(workload[1],'rowexcl_speedup_vs_same_k_boo')].append(ratio)
        selected.append(row)
        # Diagnostic paired with each selected real baseline's exact config.
        for family in ('pbr_cpp','boo_cpp'):
            name=best[family][0]['config'].removesuffix('_cpp')+'_diag'
            d=configs[name]
            r=dict(profile=workload[0],requested_size=workload[1],features=workload[2],family=family,
                   baseline_config=best[family][0]['config'],diagnostic_config=name,
                   correctness='diagnostic_not_correct',interpretation='invalid output; block-only no-zero/no-atomic diagnostic')
            stats(r,'real_ms',values(best[family]));stats(r,'diagnostic_ms',values(d))
            stats(r,'real_over_diagnostic',[b/t for b,t in zip(values(best[family]),values(d))])
            diagnostics.append(r)
    geomeans=[]
    for (size,comparison),ratios in sorted(paired.items()):
        assert len(ratios)==28
        means=[math.exp(st.mean(math.log(r[j]) for r in ratios)) for j in range(3)]
        r=dict(requested_size=size,comparison=comparison,workloads=28);stats(r,'speedup',means);geomeans.append(r)
    prefix=a.out/'BOO_ROWEXCL_phase_b_'
    for name,rows in [('all_configs.csv',raw),('samples.csv.gz',samples),('validation.csv.gz',validation),
                      ('workloads.csv',selected),('diagnostics.csv',diagnostics),('geomeans.csv',geomeans)]:
        write(Path(str(prefix)+name),rows)
    Path(str(prefix)+'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    m=metadata[0]
    lines=['# Row-exclusive BOO: Phase B','',
        f"Measured source `{m['git_commit']}`; source SHA256 `{m['source_sha256']}`; extension SHA256 `{m['extension_sha256']}`.",'',
        'Uses the symmetric harness merged from `c2ccde5`, with a separate row-exclusive registry adapter. The original bench_spmm.py is unchanged by this adapter. All real arms use persistent GPU formats, X, output and workspaces. C++ CUDA events bracket execution; PBR/BOO include their required Y zeroing and stream joins. Row-exclusive needs neither zeroing nor atomics. Its 12 sweep entries share one output buffer and run serially, with no rebinding/allocation in samples. Diagnostics omit zeroing, atomics and remainder work, produce incorrect outputs, and are never ranked as valid methods.','',
        'Sweep: PBR block/min_nnz = 2/2, 2/3, 4/2, 4/4, 4/8, 8/2, 8/8, 8/12; shared-memory tier 2048; block kernel with CSR path where compressed_fraction=0. BOO uses legacy HEM targets 8/16/32/64, 256 threads, p=64, rowseg, remainder threshold 4. Row-exclusive uses the same BOO objects, 128 threads, slabs 16/32/64, 32-row cap, requested nnz cap 2048 and 8 KiB staging budget. Direct cuSPARSE sweeps DEFAULT and CSR_ALG1/2/3, FP32/int32, alpha=1/beta=0, preprocessing and buffers outside timing. All configurations are retained.','',
        'Five warmups, 20 individually timed calls per configuration, seeded rotating order. All 84 workloads pass independent SciPy FP64 validation at atol=rtol=1e-4 in all three jobs. Diagnostic outputs are explicitly excluded. Best configs minimize the median of three job medians, keeping the chosen config fixed across jobs. Tables report median [min,max] across jobs, with paired speedups; >1 favors the named numerator comparison. Differences under 1.3× are not interpreted as wins.','',
        '| Replicate | Job | Node | GPU |','|---|---|---|---|']
    for i,m in enumerate(metadata,1):lines.append(f"| {i} | {m['slurm_job_id']} | {m['node']} | {m['gpu']} |")
    lines+=['','## Geometric means','','| N | Comparison | Median [range] |','|---:|---|---:|']
    for r in geomeans:lines.append(f"| {r['requested_size']} | {r['comparison']} | {cell(r,'speedup',3)}× |")
    lines+=['','## Every workload','','Times in ms; each method shows its best fixed configuration in the linked CSV. PBR path and compressed fraction are retained there for every result.','',
        '| Profile | N | F | Direct | PBR | BOO | Row-exclusive | Row-exclusive/direct speedup |',
        '|---|---:|---:|---|---|---|---|---|']
    for r in selected:
        lines.append(f"| {r['profile']} | {r['requested_size']} | {r['features']} | "+' | '.join(cell(r,f+'_ms') for f in ('direct_cpp','pbr_cpp','boo_cpp','rowexcl_cpp'))+f" | {cell(r,'rowexcl_cpp_speedup_vs_direct',3)}× |")
    lines+=['','## Data','',
        '- [Every configuration](BOO_ROWEXCL_phase_b_all_configs.csv), [selected workloads and speedups](BOO_ROWEXCL_phase_b_workloads.csv), [geometric means](BOO_ROWEXCL_phase_b_geomeans.csv).',
        '- [Paired invalid-output diagnostics](BOO_ROWEXCL_phase_b_diagnostics.csv); all diagnostic sweep entries remain in the configuration CSV.',
        '- [Raw samples](BOO_ROWEXCL_phase_b_samples.csv.gz), [validation](BOO_ROWEXCL_phase_b_validation.csv.gz), [provenance](BOO_ROWEXCL_phase_b_metadata.json).','']
    (a.out/'BOO_ROWEXCL_PHASE_B.md').write_text('\n'.join(lines))
    print(f'Phase B validated: {len(raw)} configs, {len(samples)} samples, {sum(r["correctness"]=="passed" for r in validation)} reference checks')


if __name__=='__main__':main()
