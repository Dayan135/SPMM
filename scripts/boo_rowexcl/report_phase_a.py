"""Offline Phase A aggregation; no CUDA imports or runtime changes."""
import argparse
from collections import defaultdict
import csv
import gzip
import json
import math
from pathlib import Path
import statistics as st


def read(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def write(path, rows):
    columns = list(dict.fromkeys(k for row in rows for k in row))
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'wt', newline='') as f:
        w = csv.DictWriter(f, fieldnames=columns); w.writeheader(); w.writerows(rows)


def key(row):
    return row['profile'], int(row['requested_size']), int(row['features'])


def values(rows):
    return [float(r['median_ms']) for r in sorted(rows, key=lambda r: int(r['replicate']))]


def stats(row, prefix, data):
    row.update({prefix: st.median(data), prefix+'_min': min(data), prefix+'_max': max(data)})


def cell(row, name, precision=4):
    return f"{row[name]:.{precision}f} [{row[name+'_min']:.{precision}f}, {row[name+'_max']:.{precision}f}]"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('results', type=Path)
    p.add_argument('--out', type=Path, default=Path('.'))
    a = p.parse_args()
    groups = defaultdict(lambda: defaultdict(list))
    metadata, raw, samples, validation, distributions = [], [], [], [], []
    for j in (1,2,3):
        root = a.results/f'phase_a_job{j}'
        m = json.loads((root/'metadata.json').read_text())
        assert m['status']=='complete' and m['completed_cells']==84
        assert m['completed_validation_cells']==(210 if j==1 else 84)
        metadata.append(m)
        for name, collection in [('summary',raw),('samples',samples),('validation',validation),('cta_distribution',distributions)]:
            rows = read(root/f'{name}.csv')
            for r in rows:
                assert int(r['replicate'])==j
                r.update(slurm_job_id=m['slurm_job_id'],node=m['node'])
                if name=='summary':
                    assert r['correctness']=='passed'
                    groups[key(r)][r['config']].append(r)
                collection.append(r)
    assert len(groups)==84 and len(raw)==3*84*32 and len(samples)==len(raw)*20
    assert len(validation)==3*84*32+126*12
    assert all(r['correctness']=='passed' for r in validation)
    assert len({m['slurm_job_id'] for m in metadata})==3
    for field in ('git_commit','source_sha256','extension_sha256'):
        assert len({m[field] for m in metadata})==1, field
    # The histogram accounts for every stored nonzero once per feature slab.
    histograms = defaultdict(list)
    for r in distributions:
        histograms[(key(r), int(r['replicate']), int(r['k']), int(r['slab']), r['metric'])].append(r)
    assert len(histograms) == 84*3*4*3*2
    for (workload, replicate, target, slab, metric), histogram in histograms.items():
        config = groups[workload][f'rowexcl_cpp_k{target}_s{slab}']
        r = next(v for v in config if int(v['replicate']) == replicate)
        assert sum(int(h['row_units']) for h in histogram) == int(r['num_units'])
        assert sum(int(h['ctas']) for h in histogram) == int(r['grid_ctas'])
        if metric == 'nnz':
            assert sum(int(h['value'])*int(h['row_units']) for h in histogram) == int(r['nnz'])
            assert max(int(h['value']) for h in histogram) <= int(r['effective_nnz_cap'])
        else:
            fraction = sum(int(h['row_units']) for h in histogram if int(h['value']) > 1) / int(r['num_units'])
            assert abs(fraction-float(r['chunked_fraction'])) < 1e-12
    selected, matched, geomeans, ncu_selection = [], [], [], []
    paired = defaultdict(list)
    for k, configs in sorted(groups.items(), key=lambda item:(item[0][1],item[0][2],item[0][0])):
        assert len(configs)==32 and all(len(v)==3 for v in configs.values())
        rows = [r for v in configs.values() for r in v]
        for field in ('matrix_sha256','x_sha256','source_sha256','rows','cols'):
            assert len({r[field] for r in rows})==1, (k,field)
        best = {}
        for family in ('direct_cpp','rowexcl_cpp','boo_api','rowexcl_api'):
            choices = [v for v in configs.values() if v[0]['family']==family]
            best[family] = min(choices, key=lambda v: st.median(values(v)))
        out = dict(profile=k[0], requested_size=k[1], features=k[2], rows=rows[0]['rows'], cols=rows[0]['cols'])
        for family, group in best.items():
            out[family+'_config']=group[0]['config']
            stats(out,family+'_ms',values(group))
            if family!='direct_cpp':
                for field in ('k','slab','compressed_fraction','num_blocks'):
                    out[family+'_'+field]=group[0].get(field,'')
        for field in ('requested_nnz_cap','effective_nnz_cap','max_rows_per_cta',
                      'num_units','grid_ctas','cta_nnz_min','cta_nnz_median','cta_nnz_p90','cta_nnz_p99',
                      'cta_nnz_max','chunked_fraction','chunks_per_cta_max','smem_bytes',
                      'registers_per_thread','resident_ctas_per_sm','split_row_groups','zero_nnz_units'):
            out[field]=best['rowexcl_cpp'][0].get(field,'')
        cpp = [b/r for b,r in zip(values(best['direct_cpp']),values(best['rowexcl_cpp']))]
        api = [b/r for b,r in zip(values(best['boo_api']),values(best['rowexcl_api']))]
        matched_api = [b/r for b,r in zip(values(configs[f"boo_api_k{best['rowexcl_api'][0]['k']}"]),values(best['rowexcl_api']))]
        for name, data in [('cpp_speedup_vs_direct',cpp),('api_speedup_vs_best_boo',api),('api_speedup_vs_same_k_boo',matched_api)]:
            stats(out,name,data)
            for j,v in enumerate(data,1): paired[(k[1],name,j)].append(v)
        selected.append(out)
        for target in (8,16,32,64):
            baseline=configs[f'boo_api_k{target}']
            options=[v for v in configs.values() if v[0]['family']=='rowexcl_api' and int(v[0]['k'])==target]
            chosen=min(options,key=lambda v:st.median(values(v)))
            row=dict(profile=k[0],requested_size=k[1],features=k[2],k=target,slab=chosen[0]['slab'])
            stats(row,'boo_ms',values(baseline)); stats(row,'rowexcl_ms',values(chosen))
            stats(row,'speedup',[b/r for b,r in zip(values(baseline),values(chosen))])
            matched.append(row)
        if k[1:]==(262144,512) and k[0] in ('fem2d','fem2d_permuted','diag_b1.0'):
            ncu_selection.append(dict(profile=k[0],mode='rowexcl',k=int(best['rowexcl_cpp'][0]['k']),slab=int(best['rowexcl_cpp'][0]['slab'])))
            ncu_selection.append(dict(profile=k[0],mode='boo',k=int(best['rowexcl_cpp'][0]['k']),slab=32))
    for n in (16384,65536,262144):
        for comparison in ('cpp_speedup_vs_direct','api_speedup_vs_best_boo','api_speedup_vs_same_k_boo'):
            numbers=[]
            for j in (1,2,3):
                v=paired[n,comparison,j]; assert len(v)==28
                numbers.append(math.exp(st.mean(math.log(x) for x in v)))
            row=dict(requested_size=n,comparison=comparison,workloads=28); stats(row,'speedup',numbers)
            geomeans.append(row)
    for name,data in [('all_configs.csv',raw),('workloads.csv',selected),('matched_k.csv',matched),
                      ('geomeans.csv',geomeans),('samples.csv.gz',samples),('validation.csv.gz',validation),
                      ('cta_distribution.csv.gz',distributions)]:
        write(a.out/f'BOO_ROWEXCL_phase_a_{name}',data)
    (a.out/'BOO_ROWEXCL_phase_a_metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    (a.out/'BOO_ROWEXCL_phase_a_profile_selection.json').write_text(json.dumps(ncu_selection,indent=2)+'\n')
    lines=['# Row-exclusive BOO: Phase A', '',
           f"Measured source: `{metadata[0]['git_commit']}`; source SHA256 `{metadata[0]['source_sha256']}`; extension SHA256 `{metadata[0]['extension_sha256']}`.", '',
           f"{metadata[0]['gpu']}; PyTorch {metadata[0]['torch_version']}; PyTorch CUDA {metadata[0]['cuda_version']}; FP32 values, int32 indices, seed 0.", '',
           '| Replicate | Slurm job | Node | Timed / validated workloads |','|---|---|---|---|']
    for j,m in enumerate(metadata,1):
        lines.append(f"| {j} | {m['slurm_job_id']} | {m['node']} | {m['completed_cells']} / {m['completed_validation_cells']} |")
    lines += ['', 'The BOO format and HEM implementation are unchanged. HEM targets {8,16,32,64}, feasible_first=False, rowmajor blocks, small_block_policy=remainder, threshold=4 match the existing comparison. Each row-exclusive plan references the **same BOO object** as its corresponding current-BOO arm; values and packed coordinates are not copied or reencoded. Auxiliary tables index block rows, row slices and column-group chunks.', '',
              'A CTA owns up to 32 rows from one HEM group and one disjoint feature slab (16/32/64; API default 32), using 128 threads. It keeps sums in registers across all the group’s blocks, then processes the owned rows’ CSR remainders in the same CTA and writes Y with plain stores. Empty rows are scheduled and overwritten with zero. There is no Y memset, atomic update, temporary output, or separate remainder kernel.', '',
              'Balance: greedily split groups **by rows only**, with requested 2,048 stored nonzeros/CTA and at most 32 rows/CTA. The effective cap rises to the maximum single-row degree if necessary; no implementation can enforce a smaller hard cap on an indivisible row without splitting that row. Both requested/effective caps and exact per-CTA histograms are recorded. The shared-memory budget is 8 KiB per CTA, raised when needed to hold one entire column group (up to the default 48 KiB limit). The union of a group’s column groups is staged at once when it fits this budget; otherwise whole column groups are packed into successive chunks. Chunking frequency is structural, exact, and recorded outside timing.', '',
              'Timing: five warmups, then 20 individual CUDA-event samples in a seeded rotating order, with resident A and X. Preparation, transfers, independent SciPy FP64 validation (rtol=atol=1e-4), and plan construction are untimed. C++ row-exclusive timing encloses exactly its kernel launch(es) with Y preallocated. Direct cuSPARSE reuses descriptors/workspace/output and times only cusparseSpMM, alpha=1/beta=0, after bufferSize and optional preprocess. The full-API comparison calls both methods through Python: row-exclusive allocates uninitialized Y; current BOO allocates/zeros Y and joins its block/remainder streams. **No C++/API cross-scope speedup is reported.**', '',
              'Each workload selects the lowest median of three job medians per family, retaining the same configuration across jobs. All 32 configurations/workload and all samples remain in the CSVs. Times and speedups below are median [min,max] across jobs; speedups are paired within each job. >1 means row-exclusive is faster. Geometric means combine 28 workloads per size within each job, then report median/range of the three means. Best-vs-best API selection can choose different HEM targets; the matched-k results isolate execution on the identical BOO grouping.', '',
              'Validation covers the 84 requested workloads in all three jobs, plus all 126 additional workload cells at F={100,200,520} in replicate 1, across every HEM target and slab. GPU edge-case tests additionally poison Y to catch unwritten elements, force chunking and row splitting, and cover zero matrices, empty rows, rectangular inputs and pure CSR remainders.', '',
              '## Geometric means', '', '| N | Comparison | Speedup median [range] |','|---:|---|---:|']
    for r in geomeans: lines.append(f"| {r['requested_size']} | {r['comparison']} | {cell(r,'speedup',3)}× |")
    lines += ['', '## Per workload', '', '| Profile | N | F | Direct C++ ms | Row-exclusive C++ ms (config) | BOO API ms | Row-exclusive API ms (config) | C++ / API speedup |', '|---|---:|---:|---|---|---|---|---|']
    for r in selected:
        lines.append(f"| {r['profile']} | {r['requested_size']} | {r['features']} | {cell(r,'direct_cpp_ms')} | {cell(r,'rowexcl_cpp_ms')} ({r['rowexcl_cpp_config']}) | {cell(r,'boo_api_ms')} | {cell(r,'rowexcl_api_ms')} ({r['rowexcl_api_config']}) | {r['cpp_speedup_vs_direct']:.3f} / {r['api_speedup_vs_best_boo']:.3f} |")
    lines += ['', '## Robust-margin counts', '', 'These descriptive counts require at least a 1.3× margin in every job; they are not confidence intervals.', '', '| Comparison | ≥1.3× faster in all jobs | ≥1.3× slower in all jobs | Remaining |', '|---|---:|---:|---:|']
    for c in ('cpp_speedup_vs_direct','api_speedup_vs_best_boo','api_speedup_vs_same_k_boo'):
        fast=sum(r[c+'_min']>=1.3 for r in selected); slow=sum(r[c+'_max']<=1/1.3 for r in selected)
        lines.append(f'| {c} | {fast} | {slow} | {84-fast-slow} |')
    lines += ['', '## Artifacts', '',
              '- [Every configuration](BOO_ROWEXCL_phase_a_all_configs.csv), [selected workloads](BOO_ROWEXCL_phase_a_workloads.csv), [same-grouping API comparisons](BOO_ROWEXCL_phase_a_matched_k.csv), [geometric means](BOO_ROWEXCL_phase_a_geomeans.csv).',
              '- [Individual samples](BOO_ROWEXCL_phase_a_samples.csv.gz), [all correctness checks](BOO_ROWEXCL_phase_a_validation.csv.gz), [exact CTA nnz/chunk histograms](BOO_ROWEXCL_phase_a_cta_distribution.csv.gz), [metadata](BOO_ROWEXCL_phase_a_metadata.json).',
              '- Kernel registers, resident CTA bound, shared memory, chunking frequency, row splits and nnz quantiles appear on every row-exclusive result in the all-config CSV. CTA histograms include the feature-slab multiplicity.',
              '- Reproduce with the isolated `scripts/boo_rowexcl/job.sh`: validate, smoke, then three separate jobs with arguments 1, 2 and 3. No pip installation is used.', '']
    (a.out/'BOO_ROWEXCL_PHASE_A.md').write_text('\n'.join(lines))
    print(f'Phase A validated: {len(raw)} configurations, {len(samples)} samples, {len(validation)} reference checks.')


if __name__=='__main__':
    main()
