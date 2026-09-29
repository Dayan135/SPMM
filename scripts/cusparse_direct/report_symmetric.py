"""Offline report for the symmetric-timing follow-up: python
scripts/cusparse_direct/report_symmetric.py RESULTS_ROOT.

Reads 3 replicate jobs from a --symmetric bench_spmm run (45 configs per
workload: the original 17 -- cusparse_csr, 8 pbr, 4 boo, 4 direct
cusparseSpMM_only -- plus 4 direct full_call, 16 pbr cpp_only/diagnostic,
8 boo cpp_only/diagnostic) and three sets of nsys profiles (torch/pbr/boo),
and writes CUSPARSE_DIRECT_SYMMETRIC.md + CSVs.

Two real comparison levels, reported separately:
  - api_level: every method's Python-callable entry point, full call
    (torch.sparse.mm, pbr_matmul, boo_matmul, and direct cuSPARSE routed
    through the same Python-side event loop instead of its own timer).
  - cpp_level: C++-only timing for direct/pbr/boo (PBRTimedSpMM/BOOTimedSpMM
    mirror DirectSpMM's timed_run() scope exactly). No cpp_level entry for
    torch -- that code belongs to PyTorch, not this repo.
Plus a diagnostic section (upper-bound only, output incorrect by
construction, never compared as if it were a real measurement).
"""
import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path
import statistics as st
import sys

sys.path.insert(0, str(Path(__file__).parent))
from report import read_csv, write_csv, key, interval, fmt, merged_duration  # noqa: E402


def profile_rows_for_method(root, expected, method):
    """Same shape as report.py's profile_rows, restricted to one method's
    {profile}_{n}_f{f}_{method}.json/.sqlite files."""
    import sqlite3
    kernels, activities, summaries = [], [], []
    for path in sorted((root / 'profiles').glob(f'*_{method}.json')):
        meta = json.loads(path.read_text())
        k = key(meta)
        ref = expected[k][0]
        for field in ('matrix_sha256', 'x_sha256', 'source_sha256'):
            assert meta[field] == ref[field], (path, field)
        common = dict(profile=k[0], requested_size=k[1], features=k[2], method=method,
                      slurm_job_id=meta['slurm_job_id'], node=meta['node'])
        db = sqlite3.connect(path.with_suffix('.sqlite'))
        db.row_factory = sqlite3.Row
        tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        strings = dict(db.execute('SELECT id, value FROM StringIds'))
        cell_kernels = []
        for i, r in enumerate(db.execute('SELECT * FROM CUPTI_ACTIVITY_KIND_KERNEL ORDER BY start')):
            name = strings[r['demangledName']]
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
    assert len(summaries) == 84 and {key(r) for r in summaries} == set(expected), \
        (method, len(summaries))
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
        assert meta.get('symmetric'), f'job{job} was not run with --symmetric'
        metas.append(meta)
        rows = read_csv(folder / 'summary.csv')
        for row in rows:
            row.update(replicate=job, slurm_job_id=meta['slurm_job_id'], node=meta['node'])
            assert row['correctness'] in ('passed', 'not_supported')
            if row['algorithm'] in ('pbr', 'boo'):
                assert 0 <= float(row['compressed_fraction']) <= 1
            raw.append(row)
            grouped[key(row)].append(row)
        for row in read_csv(folder / 'samples.csv'):
            samples.append(dict(row, replicate=job, slurm_job_id=meta['slurm_job_id'], node=meta['node']))
    assert len({m['slurm_job_id'] for m in metas}) == 3
    for field in ('git_commit', 'source_sha256', 'extension_sha256'):
        assert len({m[field] for m in metas}) == 1, field
    assert len(grouped) == 84

    def best_median(rows_for_config_list):
        """rows_for_config_list: list of (config_name -> [row, row, row]) with
        exactly one row per replicate. Returns the config with lowest median
        of its 3 replicate medians, and that config's 3 rows sorted by
        replicate."""
        best = None
        for cfg_rows in rows_for_config_list:
            if len(cfg_rows) != 3 or any(r['correctness'] != 'passed' for r in cfg_rows):
                continue
            med = st.median(float(r['median_ms']) for r in cfg_rows)
            if best is None or med < best[0]:
                best = (med, sorted(cfg_rows, key=lambda r: r['replicate']))
        return best[1] if best else None

    api_rows, cpp_rows, diag_rows = [], [], []
    api_paired, cpp_paired = [], []
    for k, rows in sorted(grouped.items(), key=lambda item: (item[0][1], item[0][2], item[0][0])):
        for field in ('matrix_sha256', 'x_sha256', 'source_sha256', 'rows', 'cols'):
            assert len({r[field] for r in rows}) == 1, (k, field)
        by_config = defaultdict(list)
        for r in rows:
            by_config[r['config']].append(r)

        def family_configs(algorithm, timing_scope):
            return [v for cfg, v in by_config.items()
                    if v[0]['algorithm'] == algorithm and v[0]['timing_scope'] == timing_scope]

        common = dict(profile=k[0], requested_size=k[1], features=k[2],
                      rows=rows[0]['rows'], cols=rows[0]['cols'], nnz=rows[0]['nnz'])

        # --- API level: direct(full_call) vs torch vs pbr(full_call) vs boo(full_call) ---
        api_best = {
            'direct': best_median(family_configs('cusparse_direct', 'full_call')),
            'torch':  best_median(family_configs('cusparse_csr', 'full_call')),
            'pbr':    best_median(family_configs('pbr', 'full_call')),
            'boo':    best_median(family_configs('boo', 'full_call')),
        }
        if all(api_best.values()):
            out = dict(common)
            direct_vals = [float(r['median_ms']) for r in api_best['direct']]
            for label in ('direct', 'torch', 'pbr', 'boo'):
                vals = [float(r['median_ms']) for r in api_best[label]]
                out[label+'_config'] = api_best[label][0]['config']
                out[label+'_ms'] = st.median(vals)
                if label != 'direct':
                    ratios = [d/v for d, v in zip(direct_vals, vals)]
                    out[label+'_speedup'] = st.median(ratios)
                    for j, ratio in enumerate(ratios, 1):
                        api_paired.append(dict(profile=k[0], requested_size=k[1], features=k[2],
                                               replicate=j, method=label, speedup=ratio))
                if label in ('pbr', 'boo'):
                    out[label+'_execution_path'] = api_best[label][0]['execution_path']
                    out[label+'_compressed_fraction'] = api_best[label][0]['compressed_fraction']
            api_rows.append(out)

        # --- C++ level: direct(cusparseSpMM_only) vs pbr(cpp_only) vs boo(cpp_only) ---
        cpp_best = {
            'direct': best_median(family_configs('cusparse_direct', 'cusparseSpMM_only')),
            'pbr':    best_median(family_configs('pbr', 'cpp_only')),
            'boo':    best_median(family_configs('boo', 'cpp_only')),
        }
        if all(cpp_best.values()):
            out = dict(common)
            direct_vals = [float(r['median_ms']) for r in cpp_best['direct']]
            for label in ('direct', 'pbr', 'boo'):
                vals = [float(r['median_ms']) for r in cpp_best[label]]
                out[label+'_config'] = cpp_best[label][0]['config']
                out[label+'_ms'] = st.median(vals)
                if label != 'direct':
                    ratios = [d/v for d, v in zip(direct_vals, vals)]
                    out[label+'_speedup'] = st.median(ratios)
                    for j, ratio in enumerate(ratios, 1):
                        cpp_paired.append(dict(profile=k[0], requested_size=k[1], features=k[2],
                                               replicate=j, method=label, speedup=ratio))
                if label in ('pbr', 'boo'):
                    out[label+'_execution_path'] = cpp_best[label][0]['execution_path']
                    out[label+'_compressed_fraction'] = cpp_best[label][0]['compressed_fraction']
            cpp_rows.append(out)

        # --- Diagnostic upper bound: pbr/boo diagnostic vs direct(cpp) ---
        diag_best = {
            'pbr': best_median(family_configs('pbr_diagnostic', 'diagnostic_upper_bound')),
            'boo': best_median(family_configs('boo_diagnostic', 'diagnostic_upper_bound')),
        }
        if cpp_best['direct'] and (diag_best['pbr'] or diag_best['boo']):
            out = dict(common)
            direct_vals = [float(r['median_ms']) for r in cpp_best['direct']]
            for label in ('pbr', 'boo'):
                if not diag_best[label]:
                    continue
                vals = [float(r['median_ms']) for r in diag_best[label]]
                out[label+'_diag_config'] = diag_best[label][0]['config']
                out[label+'_diag_ms'] = st.median(vals)
                ratios = [d/v for d, v in zip(direct_vals, vals)]
                out[label+'_diag_speedup_vs_direct_cpp'] = st.median(ratios)
            diag_rows.append(out)

    def geomeans(paired, methods):
        out = []
        for n in (16384, 65536, 262144):
            for method in methods:
                out_vals, n_workloads = [], 0
                for job in (1, 2, 3):
                    ratios = [r['speedup'] for r in paired
                             if r['requested_size'] == n and r['method'] == method and r['replicate'] == job]
                    if ratios:
                        n_workloads = len(ratios)
                        out_vals.append(math.exp(st.mean(math.log(v) for v in ratios)))
                if out_vals:
                    out.append(dict(requested_size=n, method=method, workloads=n_workloads,
                                    speedup=st.median(out_vals), speedup_min=min(out_vals), speedup_max=max(out_vals)))
        return out

    api_geomeans = geomeans(api_paired, ('torch', 'pbr', 'boo'))
    cpp_geomeans = geomeans(cpp_paired, ('pbr', 'boo'))

    profile_kernels, profile_activities, profile_summary = {}, {}, {}
    for method in ('torch', 'pbr', 'boo'):
        pk, pa, ps = profile_rows_for_method(a.results, grouped, method)
        profile_kernels[method] = pk
        profile_activities[method] = pa
        profile_summary[method] = ps

    for suffix, rows in [('all_configs', raw), ('api_level_workloads', api_rows),
                         ('cpp_level_workloads', cpp_rows), ('diagnostic_workloads', diag_rows),
                         ('api_level_geomeans', api_geomeans), ('cpp_level_geomeans', cpp_geomeans)]:
        write_csv(a.out / f'CUSPARSE_DIRECT_SYMMETRIC_{suffix}.csv', rows)
    for method in ('torch', 'pbr', 'boo'):
        write_csv(a.out / f'CUSPARSE_DIRECT_SYMMETRIC_profiles_{method}.csv', profile_summary[method])
    (a.out / 'CUSPARSE_DIRECT_SYMMETRIC_metadata.json').write_text(json.dumps(metas, indent=2)+'\n')

    lines = ['# Symmetric direct-cuSPARSE comparison', '',
             'Follow-up to [CUSPARSE_DIRECT.md](CUSPARSE_DIRECT.md), which compared direct '
             'cuSPARSE\'s C++-only cusparseSpMM timing against PBR/BOO/torch.sparse.mm\'s '
             'Python-side full-call timing -- two different measurement scopes, not an '
             'apples-to-apples comparison. This run gives every method both scopes.', '',
             f"Measured source commit: `{metas[0]['git_commit']}`. Source SHA256: `{metas[0]['source_sha256']}`.",
             '',
             '## API level (every method\'s Python-callable full call)', '',
             '| N | Method | Speedup vs direct, median [range] |', '|---:|---|---:|']
    lines += [f"| {r['requested_size']} | {r['method']} | {fmt(r, 'speedup', 3)}x |" for r in api_geomeans]
    lines += ['', '## C++ level (direct cusparseSpMM_only vs PBRTimedSpMM/BOOTimedSpMM cpp_only -- '
                   'no torch entry, that code belongs to PyTorch)', '',
             '| N | Method | Speedup vs direct, median [range] |', '|---:|---|---:|']
    lines += [f"| {r['requested_size']} | {r['method']} | {fmt(r, 'speedup', 3)}x |" for r in cpp_geomeans]
    lines += ['', '## Diagnostic: atomic/zero upper bound', '',
             '**Output is incorrect by construction (Y left unzeroed, Step-C atomicAdd replaced '
             'with a plain store) -- this is not a real timing, it bounds what removing the '
             'zero+atomic cost could gain if a future redesign eliminated the need for them '
             '(e.g. one-CTA-per-row-group ownership). For BOO specifically this is a stronger '
             'claim than for PBR: BOO\'s rowseg atomicAdd is not purely a synchronization tax -- '
             'a row\'s contribution is genuinely split across multiple blocks in its row-group, '
             'so the diagnostic silently drops real cross-block accumulation, not just atomic '
             'overhead. Never compare these numbers as if they were achievable today.**', '',
             '| Profile | N | F | PBR diag ms | PBR diag speedup vs direct(cpp) | BOO diag ms | BOO diag speedup vs direct(cpp) |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for r in diag_rows:
        pbr_s = f"{r['pbr_diag_speedup_vs_direct_cpp']:.3f}" if 'pbr_diag_speedup_vs_direct_cpp' in r else '-'
        boo_s = f"{r['boo_diag_speedup_vs_direct_cpp']:.3f}" if 'boo_diag_speedup_vs_direct_cpp' in r else '-'
        pbr_ms = f"{r['pbr_diag_ms']:.5f}" if 'pbr_diag_ms' in r else '-'
        boo_ms = f"{r['boo_diag_ms']:.5f}" if 'boo_diag_ms' in r else '-'
        lines.append(f"| {r['profile']} | {r['requested_size']} | {r['features']} | "
                     f"{pbr_ms} | {pbr_s} | {boo_ms} | {boo_s} |")
    lines += ['', '## GPU-busy time from nsys traces (one call per workload, per method)', '',
             '| Method | Profile | N | F | Profiled event ms | SpMM kernel ms | Outside SpMM |',
             '|---|---|---:|---:|---:|---:|---:|']
    for method in ('torch', 'pbr', 'boo'):
        for r in sorted(profile_summary[method], key=lambda r: (r['requested_size'], r['features'], r['profile'])):
            lines.append(f"| {method} | {r['profile']} | {r['requested_size']} | {r['features']} | "
                         f"{r['profiled_event_ms']:.5f} | {r['spmm_kernel_ms']:.5f} | {r['outside_spmm_fraction']:.1%} |")
    lines += ['', '## Artifacts', '',
             '- [All configurations, all three jobs](CUSPARSE_DIRECT_SYMMETRIC_all_configs.csv).',
             '- [API-level workloads](CUSPARSE_DIRECT_SYMMETRIC_api_level_workloads.csv), '
             '[C++-level workloads](CUSPARSE_DIRECT_SYMMETRIC_cpp_level_workloads.csv), '
             '[diagnostic](CUSPARSE_DIRECT_SYMMETRIC_diagnostic_workloads.csv).',
             '- [API-level geomeans](CUSPARSE_DIRECT_SYMMETRIC_api_level_geomeans.csv), '
             '[C++-level geomeans](CUSPARSE_DIRECT_SYMMETRIC_cpp_level_geomeans.csv).',
             '- Per-method GPU-busy traces: '
             '[torch](CUSPARSE_DIRECT_SYMMETRIC_profiles_torch.csv), '
             '[pbr](CUSPARSE_DIRECT_SYMMETRIC_profiles_pbr.csv), '
             '[boo](CUSPARSE_DIRECT_SYMMETRIC_profiles_boo.csv).',
             '- [Job metadata](CUSPARSE_DIRECT_SYMMETRIC_metadata.json).', '']
    (a.out / 'CUSPARSE_DIRECT_SYMMETRIC.md').write_text('\n'.join(lines))
    print(f'Wrote {len(raw)} configs; {len(api_rows)} api-level, {len(cpp_rows)} cpp-level, '
         f'{len(diag_rows)} diagnostic workload rows; all integrity checks passed.')


if __name__ == '__main__':
    main()
