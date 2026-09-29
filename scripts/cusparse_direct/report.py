"""Offline report: python scripts/cusparse_direct/report.py RESULTS_ROOT."""
import argparse
from collections import defaultdict
import csv
import gzip
import json
import math
from pathlib import Path
import sqlite3
import statistics as st


def read_csv(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    columns = list(dict.fromkeys(k for r in rows for k in r))
    if not columns:  # A trace may contain no memcpy/memset activities.
        columns = ['profile', 'requested_size', 'features', 'slurm_job_id', 'node',
                   'kind', 'start_ns', 'end_ns', 'duration_ms', 'bytes', 'stream', 'correlation_id']
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def key(row):
    return row['profile'], int(row['requested_size']), int(row['features'])


def interval(values):
    return dict(median=st.median(values), low=min(values), high=max(values))


def fmt(row, field, digits=4):
    return f"{row[field]:.{digits}f} [{row[field+'_min']:.{digits}f}, {row[field+'_max']:.{digits}f}]"


def merged_duration(intervals):
    end = -1
    total = 0
    for lo, hi in sorted(intervals):
        total += max(0, hi-max(lo, end))
        end = max(end, hi)
    return total / 1e6


def profile_rows(root, expected):
    kernels, activities, summaries = [], [], []
    for path in sorted((root / 'profiles').glob('*.json')):
        meta = json.loads(path.read_text())
        k = key(meta)
        ref = expected[k][0]
        for field in ('matrix_sha256', 'x_sha256', 'source_sha256'):
            assert meta[field] == ref[field], (path, field)
        common = dict(profile=k[0], requested_size=k[1], features=k[2],
                      slurm_job_id=meta['slurm_job_id'], node=meta['node'])
        db = sqlite3.connect(path.with_suffix('.sqlite'))
        db.row_factory = sqlite3.Row
        tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        strings = dict(db.execute('SELECT id, value FROM StringIds'))
        cell_kernels = []
        for i, r in enumerate(db.execute('SELECT * FROM CUPTI_ACTIVITY_KIND_KERNEL ORDER BY start')):
            name = strings[r['demangledName']]
            # Explicit name-based classification; raw names retained for audit.
            is_spmm = 'spmm' in name.lower() or 'csrmm' in name.lower()
            row = dict(common, kernel_index=i, kernel_name=name, is_spmm=is_spmm,
                       duration_ms=(r['end']-r['start'])/1e6,
                       start_ns=r['start'], end_ns=r['end'], stream=r['streamId'],
                       correlation_id=r['correlationId'])
            kernels.append(row)
            cell_kernels.append(row)
        assert cell_kernels, path
        cell_activities = []
        for table in ('CUPTI_ACTIVITY_KIND_MEMSET', 'CUPTI_ACTIVITY_KIND_MEMCPY'):
            if table not in tables:
                continue
            for r in db.execute(f'SELECT * FROM {table} ORDER BY start'):
                row = dict(common, kind=table.removeprefix('CUPTI_ACTIVITY_KIND_'),
                           start_ns=r['start'], end_ns=r['end'],
                           duration_ms=(r['end']-r['start'])/1e6, bytes=r['bytes'],
                           stream=r['streamId'], correlation_id=r['correlationId'])
                activities.append(row)
                cell_activities.append(row)
        union = merged_duration([(r['start_ns'], r['end_ns']) for r in cell_kernels+cell_activities])
        spmm_union = merged_duration([(r['start_ns'], r['end_ns']) for r in cell_kernels if r['is_spmm']])
        summaries.append(dict(common, kernel_count=len(cell_kernels),
                              spmm_kernel_count=sum(r['is_spmm'] for r in cell_kernels),
                              all_kernel_ms=sum(r['duration_ms'] for r in cell_kernels),
                              spmm_kernel_ms=sum(r['duration_ms'] for r in cell_kernels if r['is_spmm']),
                              other_kernel_ms=sum(r['duration_ms'] for r in cell_kernels if not r['is_spmm']),
                              memory_activity_ms=sum(r['duration_ms'] for r in cell_activities),
                              gpu_activity_union_ms=union,
                              spmm_activity_union_ms=spmm_union,
                              outside_spmm_ms=meta['profiled_event_ms']-spmm_union,
                              outside_spmm_fraction=1-spmm_union/meta['profiled_event_ms'],
                              profiled_event_ms=meta['profiled_event_ms'],
                              event_minus_gpu_activity_ms=meta['profiled_event_ms']-union,
                              host_call_ms=meta['host_call_ms']))
        db.close()
    assert len(summaries) == 84 and {key(r) for r in summaries} == set(expected)
    return kernels, activities, summaries


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('results', type=Path)
    p.add_argument('--out', type=Path, default=Path('.'))
    a = p.parse_args()
    metas, raw, samples = [], [], []
    grouped = defaultdict(list)
    for job in range(1, 4):
        folder = a.results / f'job{job}'
        meta = json.loads((folder / 'metadata.json').read_text())
        assert meta['status'] == 'complete' and meta['completed_cells'] == 84
        metas.append(meta)
        rows = read_csv(folder / 'summary.csv')
        assert len(rows) == 84*17
        for row in rows:
            row.update(replicate=job, slurm_job_id=meta['slurm_job_id'], node=meta['node'])
            assert row['correctness'] in ('passed', 'not_supported')
            if row['algorithm'] in ('pbr', 'boo'):
                assert 0 <= float(row['compressed_fraction']) <= 1
            raw.append(row)
            grouped[key(row)].append(row)
        for row in read_csv(folder / 'samples.csv'):
            samples.append(dict(row, replicate=job, slurm_job_id=meta['slurm_job_id'], node=meta['node']))
        expected_samples = sum(int(r['repeats']) for r in rows)
        assert sum(s['replicate'] == job for s in samples) == expected_samples
    assert len({m['slurm_job_id'] for m in metas}) == 3
    for field in ('git_commit', 'source_sha256', 'extension_sha256'):
        assert len({m[field] for m in metas}) == 1, field
    assert len(grouped) == 84
    selected, paired = [], []
    for k, rows in sorted(grouped.items(), key=lambda item: (item[0][1], item[0][2], item[0][0])):
        for field in ('matrix_sha256', 'x_sha256', 'source_sha256', 'rows', 'cols'):
            assert len({r[field] for r in rows}) == 1, (k, field)
        by_config = defaultdict(list)
        for r in rows:
            by_config[r['config']].append(r)
        assert len(by_config) == 17 and all(len(v) == 3 for v in by_config.values())
        best = {}
        for family in ('cusparse_direct', 'cusparse_csr', 'pbr', 'boo'):
            options = [v for v in by_config.values() if v[0]['algorithm'] == family and
                       all(r['correctness'] == 'passed' for r in v)]
            assert options, (k, family)
            best[family] = sorted(min(options, key=lambda v: st.median(float(r['median_ms']) for r in v)),
                                  key=lambda r: r['replicate'])
        out = dict(profile=k[0], requested_size=k[1], features=k[2], rows=rows[0]['rows'],
                   cols=rows[0]['cols'], nnz=rows[0]['nnz'])
        direct = [float(r['median_ms']) for r in best['cusparse_direct']]
        for family, label in [('cusparse_direct', 'direct'), ('cusparse_csr', 'torch'), ('pbr', 'pbr'), ('boo', 'boo')]:
            values = [float(r['median_ms']) for r in best[family]]
            out[label+'_config'] = best[family][0]['config']
            out.update({label+'_ms': st.median(values), label+'_ms_min': min(values), label+'_ms_max': max(values)})
            ratios = [d/v for d, v in zip(direct, values)]
            out.update({label+'_speedup': st.median(ratios), label+'_speedup_min': min(ratios), label+'_speedup_max': max(ratios)})
            for j, ratio in enumerate(ratios, 1):
                paired.append(dict(profile=k[0], requested_size=k[1], features=k[2], replicate=j,
                                   method=label, speedup=ratio))
            if family in ('pbr', 'boo'):
                for field in ('compressed_fraction', 'execution_path', 'num_blocks'):
                    assert len({r[field] for r in best[family]}) == 1
                    out[label+'_'+field] = best[family][0][field]
        selected.append(out)
    geomeans = []
    for n in (16384, 65536, 262144):
        for method in ('torch', 'pbr', 'boo'):
            values = []
            for job in (1, 2, 3):
                ratios = [r['speedup'] for r in paired if r['requested_size'] == n and r['method'] == method and r['replicate'] == job]
                assert len(ratios) == 28
                values.append(math.exp(st.mean(math.log(v) for v in ratios)))
            geomeans.append(dict(requested_size=n, method=method, workloads=28,
                                 speedup=st.median(values), speedup_min=min(values), speedup_max=max(values)))
    kernels, activities, profile_summary = profile_rows(a.results, grouped)
    for suffix, rows in [('all_configs', raw), ('workloads', selected), ('geomeans', geomeans),
                         ('torch_kernels', kernels), ('torch_memory', activities), ('torch_profiles', profile_summary)]:
        write_csv(a.out / f'CUSPARSE_DIRECT_{suffix}.csv', rows)
    with gzip.open(a.out / 'CUSPARSE_DIRECT_samples.csv.gz', 'wt', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(samples[0])); w.writeheader(); w.writerows(samples)
    (a.out / 'CUSPARSE_DIRECT_metadata.json').write_text(json.dumps(metas, indent=2)+'\n')
    profile_metadata = [json.loads(p.read_text()) for p in sorted((a.results/'profiles').glob('*.json'))]
    (a.out / 'CUSPARSE_DIRECT_profiles_metadata.json').write_text(json.dumps(profile_metadata, indent=2)+'\n')
    profiler_version = (a.results / 'nsys_version.txt').read_text().strip()
    validation = (a.results.parent / 'correctness.log').read_text().splitlines()[-1]
    lines = ['# Direct cuSPARSE comparison', '',
             f"Measured source commit: `{metas[0]['git_commit']}`. Source SHA256: `{metas[0]['source_sha256']}`. Extension SHA256: `{metas[0]['extension_sha256']}`.", '',
             f"GPU: {metas[0]['gpu']}; PyTorch {metas[0]['torch_version']}; PyTorch CUDA {metas[0]['cuda_version']}. Runtime cuSPARSE version(s): " + ', '.join(sorted({r['cusparse_version'] for r in raw if r.get('cusparse_version')})) + '.', '',
             'cuSPARSE build-header version(s): ' + ', '.join(sorted({r['cusparse_header_version'] for r in raw if r.get('cusparse_header_version')})) + '. Runtime and header versions are recorded separately; the extension was built with the cluster conda CUDA 12.9 toolkit.', '',
             f'Validation: `{validation}`. Profiler: {profiler_version}.', '',
             '| Replicate | Slurm job | GPU node |', '|---|---|---|']
    lines += [f"| {i} | {m['slurm_job_id']} | {m['node']} |" for i, m in enumerate(metas, 1)]
    lines += ['', 'All 84 workloads use FP32 CSR values, int32 indices, seeded identical A and X across methods and jobs: 14 profiles × N={16,384; 65,536; 262,144} × F={128; 512}. Matrix and X hashes are checked by the report generator. Every supported result passed an independent SciPy FP64 reference at rtol=atol=1e-4.', '',
              'Five warmups, then 20 individual CUDA-event timings in seeded, interleaved configuration order. Each job reports its median; tables give the median and [minimum, maximum] of the three job medians. Conversion, staging, validation, bufferSize and preprocess are outside timing.', '',
              'Direct: DEFAULT, CSR_ALG1, CSR_ALG2, CSR_ALG3; non-transposed A/X, row-major X/Y, alpha=1, beta=0. Each algorithm owns a reusable sparse descriptor, dense descriptors, output and workspace. C++ CUDA events enclose only cusparseSpMM. bufferSize and optional preprocess are each called once per plan; statuses and workspace bytes are preserved in the all-config CSV. Internal initialization performed by cusparseSpMM is included. [NVIDIA SpMM API](https://docs.nvidia.com/cuda/archive/12.9.1/cusparse/index.html#cusparsespmm).', '',
              'PBR: kernel=block, csr_mode=0, smem=2048, (block size:min_nnz)={2:2, 2:3, 4:2, 4:4, 4:8, 8:2, 8:8, 8:12}. BOO: HEM target={8,16,32,64}, threads=256, p=64, rowseg, row-major nonzeros, small blocks below 4 sent to remainder, hem_feasible_first=False, launch_mode=legacy. These retain the earlier 84-workload sweep. Both use complete API calls, including allocating/zeroing Y and joining remainder work; torch.sparse.mm is also a complete call. Direct cuSPARSE reuses Y, so it has a deliberately narrower timing scope.', '',
              'For each workload and method, select the configuration with the lowest median of three job medians, then retain that same configuration in all three jobs. Selection is retrospective, not a production dispatcher. All four cuSPARSE algorithms and all PBR/BOO configurations remain in the raw CSV. Speedup is direct/method, paired within each job; >1 means faster than direct. The geometric means first combine the 28 workloads per size within each job, then report median and range across jobs. Do not read small differences (roughly below 1.3×) as robust wins.', '',
              '## Geometric means versus direct cuSPARSE', '', '| N | Method | Speedup, median [range] |', '|---:|---|---:|']
    lines += [f"| {r['requested_size']} | {r['method']} | {fmt(r, 'speedup', 3)}× |" for r in geomeans]
    lines += ['', 'Plain reading: the direct baseline removes PyTorch wrapper/setup costs from the baseline comparison. The following counts require at least a 1.3× margin in **every** job; the remaining workloads have smaller or inconsistent margins. These are measurements on three allocated GPUs, not confidence intervals or universal rankings.', '',
              '| Method | Faster than direct by ≥1.3× in every job | Slower by ≥1.3× in every job | Remaining |',
              '|---|---:|---:|---:|']
    for method in ('torch', 'pbr', 'boo'):
        faster = sum(r[method+'_speedup_min'] >= 1.3 for r in selected)
        slower = sum(r[method+'_speedup_max'] <= 1/1.3 for r in selected)
        lines.append(f'| {method} | {faster} | {slower} | {84-faster-slower} |')
    lines += ['', '## Per workload', '', 'Times in ms, median [range]. Full selected configuration, fractions, paths and speedup ranges are in [workload CSV](CUSPARSE_DIRECT_workloads.csv). `block+csr` means block kernel plus remainder; `csr_only` means no compressed blocks.', '',
              '| Profile | N | F | Direct algorithm | Direct ms | Torch ms | PBR ms (path; fraction) | BOO ms (fraction) | Speedup torch / PBR / BOO |',
              '|---|---:|---:|---|---|---|---|---|---|']
    for r in selected:
        lines.append(f"| {r['profile']} | {r['requested_size']} | {r['features']} | {r['direct_config'].removeprefix('direct_')} | {fmt(r,'direct_ms')} | {fmt(r,'torch_ms')} | {fmt(r,'pbr_ms')} ({r['pbr_execution_path']}; {float(r['pbr_compressed_fraction']):.3f}) | {fmt(r,'boo_ms')} ({float(r['boo_compressed_fraction']):.3f}) | {r['torch_speedup']:.3f} / {r['pbr_speedup']:.3f} / {r['boo_speedup']:.3f} |")
    lines += ['', '## One-call torch.sparse.mm traces', '',
              'Nsight Systems captures exactly one warmed call per workload in a separate process, after the timed sweep in replicate 1. Capture uses cudaProfilerApi; no kernel filter. [Every launched kernel and its duration](CUSPARSE_DIRECT_torch_kernels.csv), [memory activities](CUSPARSE_DIRECT_torch_memory.csv), and [per-call decomposition](CUSPARSE_DIRECT_torch_profiles.csv) are retained. Kernel names containing `spmm` or `csrmm` are classified as SpMM; all names remain auditable.', '',
              'The table below uses the same profiled call for CUDA-event time and trace durations. GPU activity is the union of kernel and memory-operation intervals; event minus that union includes launch gaps and instrumentation overhead, not a pure CPU cost. Kernel durations can overlap, so their sum need not equal GPU busy time. Profiling perturbs execution: these traces are diagnostics, not the timing used for speedups. No subtraction is made between a profiled kernel time and an unprofiled benchmark time.', '',
              '| Profile | N | F | Kernels | Profiled event ms | SpMM kernels ms | Other kernels ms | Memory ms | Event minus GPU busy ms | Outside SpMM |',
              '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in sorted(profile_summary, key=lambda r: (r['requested_size'],r['features'],r['profile'])):
        lines.append(f"| {r['profile']} | {r['requested_size']} | {r['features']} | {r['kernel_count']} | {r['profiled_event_ms']:.5f} | {r['spmm_kernel_ms']:.5f} | {r['other_kernel_ms']:.5f} | {r['memory_activity_ms']:.5f} | {r['event_minus_gpu_activity_ms']:.5f} | {r['outside_spmm_fraction']:.1%} |")
    lines += ['', '## Artifacts and reproduction', '',
              '- [All configurations, all three jobs](CUSPARSE_DIRECT_all_configs.csv), including compressed_fraction for every PBR/BOO result and direct preprocess status.',
              '- [All individual CUDA-event samples, gzip CSV](CUSPARSE_DIRECT_samples.csv.gz).',
              '- [Job metadata](CUSPARSE_DIRECT_metadata.json) and [geometric means](CUSPARSE_DIRECT_geomeans.csv).',
              '- [Per-workload profiler metadata](CUSPARSE_DIRECT_profiles_metadata.json), including source, matrix and X hashes.',
              '- Submission recipe: `sbatch scripts/cusparse_direct/job.sh validate`, then separate `sbatch scripts/cusparse_direct/job.sh 1`, `2`, `3` submissions. Set the snapshot path in the script for another checkout. Profiling runs after replicate 1.',
              '- Offline report: `python scripts/cusparse_direct/report.py <results-root>`. Raw Nsight SQLite/report files remain in `/home/dayanb/research/cusparse-direct-20260929/results/profiles`.', '']
    (a.out / 'CUSPARSE_DIRECT.md').write_text('\n'.join(lines))
    print(f'Wrote {len(raw)} configs, {len(samples)} samples, {len(kernels)} kernel events; all integrity checks passed.')


if __name__ == '__main__':
    main()
