"""Phase A: row-exclusive BOO and direct cuSPARSE at matching timing levels.

All construction/validation is outside timing. Fixed original HEM and BOO
settings; feature slabs are the only kernel sweep. Separate full Python API
and preallocated C++ kernel-launch timing categories are never mixed.
"""
import argparse
import csv
import json
from pathlib import Path
import statistics

import numpy as np
import torch
import torch_fast_spdmm as spmm
import torch_fast_spdmm_core as core

from benchmark.bench_4way import DEFAULT_PROFILES
from benchmark.bench_spmm import (DIRECT_ALGORITHMS, build_variants, digest_arrays,
                                   round_orders, source_metadata)
from benchmark.profile_pbr import build_matrix


def append(path, rows):
    if not rows:
        return
    fields = sorted({k for r in rows for k in r})
    if path.exists():
        with path.open() as f:
            existing = next(csv.reader(f))
        if set(fields) - set(existing):
            raise ValueError(f'CSV schema changed: {path}: {set(fields)-set(existing)}')
        fields = existing
    with path.open('a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if f.tell() == 0:
            writer.writeheader()
        writer.writerows(rows)


def validate_output(y, reference, tolerance, name):
    # SciPy created the FP64 reference on CPU; compare on GPU to avoid copying
    # every large output back to the host. Both construction and checks are untimed.
    difference = (y.to(torch.float64) - reference).abs()
    error = float(difference.max().item()) if difference.numel() else 0.
    if not bool(torch.all(difference <= tolerance).item()):
        raise AssertionError(f'{name}: mismatch against SciPy float64; max_abs_error={error}')
    return error


def make_variants(boos, plans, csr, x, timed):
    variants = []
    for config, boo in boos:
        k = config['k']
        if timed:
            variants.append((dict(config, config=f'boo_api_k{k}', family='boo_api',
                                   timing_scope='full_python_api'),
                             lambda b=boo: spmm.boo_matmul(b, x, threads=256, p=64,
                                 row_seg=True, csr_mode=0, launch_mode='legacy'), None))
        for slab, plan in plans[k].items():
            bound = plan.bind(x)
            common = dict(config, **plan.stats, kernel='row_exclusive', p=slab, launch_mode='row_exclusive',
                          grid_ctas=plan.stats['num_units']*((x.shape[1]+slab-1)//slab))
            variants.append((dict(common, config=f'rowexcl_cpp_k{k}_s{slab}', family='rowexcl_cpp',
                                   timing_scope='preallocated_cpp_kernel_launches'), bound.run, bound.timed_run))
            if timed:
                variants.append((dict(common, config=f'rowexcl_api_k{k}_s{slab}', family='rowexcl_api',
                                       timing_scope='full_python_api'),
                                 lambda p=plan: spmm.boo_rowexcl_matmul(p, x), None))
    if timed:
        for algorithm in DIRECT_ALGORITHMS:
            plan = core.DirectSpMM(csr.crow_indices(), csr.col_indices(), csr.values(), x, algorithm)
            config = dict(config=f'direct_{algorithm}', algorithm='cusparse_direct', family='direct_cpp',
                          direct_algorithm=algorithm, timing_scope='preallocated_cpp_cusparseSpMM',
                          **plan.info())
            if not config['supported']:
                raise RuntimeError(f'Direct cuSPARSE {algorithm} unsupported: {config["unsupported_reason"]}')
            variants.append((config, plan.run, plan.timed_run))
    return variants


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--replicate', type=int, default=1)
    p.add_argument('--sizes', type=int, nargs='+', default=[16384, 65536, 262144])
    p.add_argument('--features', type=int, nargs='+', default=[128, 512])
    p.add_argument('--validation-features', type=int, nargs='*', default=[100, 200, 520])
    p.add_argument('--profiles', nargs='+', default=DEFAULT_PROFILES)
    p.add_argument('--boo-k', type=int, nargs='+', default=[8, 16, 32, 64])
    p.add_argument('--slabs', type=int, nargs='+', default=[16, 32, 64])
    p.add_argument('--max-cta-nnz', type=int, default=2048)
    p.add_argument('--max-cta-rows', type=int, default=32)
    p.add_argument('--smem-bytes', type=int, default=8192)
    p.add_argument('--warmup', type=int, default=5)
    p.add_argument('--repeats', type=int, default=20)
    p.add_argument('--seed', type=int, default=0)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    if (a.out/'metadata.json').exists():
        p.error('Choose a fresh output directory')
    configs = [dict(algorithm='cusparse_csr', config='cusparse_csr')]
    configs += [dict(algorithm='boo', config=f'boo_hem_k{k}', cluster='hem', k=k,
                     threads=256, p=64, kernel='rowseg', small_block_policy='remainder',
                     small_block_threshold=4, hem_feasible_first=False, launch_mode='legacy') for k in a.boo_k]
    metadata = dict(source_metadata(Path(__file__).resolve().parents[3]),
                    arguments=dict(vars(a), out=str(a.out)), status='running', completed_cells=0,
                    completed_validation_cells=0,
                    timing='5 warmups / 20 rotated individual event samples by default; C++ and full-API scopes distinct')
    def save():
        (a.out/'metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
    save()
    try:
        with torch.inference_mode():
            for n in a.sizes:
                for profile in a.profiles:
                    print(f'PREPARE {profile} N={n}', flush=True)
                    mat = build_matrix(n, profile, seed=a.seed).astype(np.float32).tocsr()
                    mat.sum_duplicates(); mat.sort_indices()
                    mat.indptr = mat.indptr.astype(np.int32)
                    mat.indices = mat.indices.astype(np.int32)
                    matrix_hash = digest_arrays(np.asarray(mat.shape), mat.indptr, mat.indices, mat.data)
                    staged = build_variants(mat, configs, a.seed)
                    csr = staged[0][1]
                    boos = staged[1:]
                    plans = {}
                    for config, boo in boos:
                        print(f'  row-exclusive plans k={config["k"]}', flush=True)
                        plans[config['k']] = {slab: spmm.prepare_boo_rowexcl(boo, slab=slab,
                            max_cta_nnz=a.max_cta_nnz, max_cta_rows=a.max_cta_rows,
                            smem_bytes=a.smem_bytes) for slab in a.slabs}
                    extra = a.validation_features if a.replicate == 1 else []
                    widths = list(dict.fromkeys(a.features + extra))
                    for f in widths:
                        timed = f in a.features
                        print(f'  {"TIME+CHECK" if timed else "CHECK"} F={f}', flush=True)
                        x_cpu = torch.randn(mat.shape[1], f, generator=torch.Generator().manual_seed(a.seed))
                        x = x_cpu.cuda()
                        common = dict(profile=profile, requested_size=n, rows=mat.shape[0],
                                      cols=mat.shape[1], features=f, nnz=int(mat.nnz), dtype='float32',
                                      index_dtype='int32', replicate=a.replicate,
                                      matrix_sha256=matrix_hash, x_sha256=digest_arrays(x_cpu.numpy()),
                                      source_sha256=metadata['source_sha256'])
                        variants = make_variants(boos, plans, csr, x, timed)
                        ref_cpu = mat.astype(np.float64) @ x_cpu.numpy().astype(np.float64)
                        reference = torch.from_numpy(ref_cpu).cuda()
                        tolerance = 1e-4 + 1e-4*reference.abs()
                        del ref_cpu
                        validation = []
                        for config, call, _ in variants:
                            y = call()
                            error = validate_output(y, reference, tolerance, config['config'])
                            validation.append(dict(common, config=config['config'], family=config['family'],
                                                   max_abs_error=error, correctness='passed'))
                            del y
                        append(a.out/'validation.csv', validation)
                        del reference, tolerance
                        metadata['completed_validation_cells'] += 1
                        if timed:
                            for _, call, _ in variants:
                                for _ in range(a.warmup):
                                    call()
                            torch.cuda.synchronize()
                            start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
                            start.record(); end.record(); end.synchronize()
                            samples = []
                            for repeat, order in enumerate(round_orders(len(variants), a.repeats, a.seed), 1):
                                for position, i in enumerate(order):
                                    config, call, cpp_timer = variants[i]
                                    if cpp_timer is not None:
                                        ms = cpp_timer()
                                    else:
                                        start.record(); y = call(); end.record(); end.synchronize()
                                        ms = start.elapsed_time(end)
                                        del y
                                    if not np.isfinite(ms) or ms <= 0:
                                        raise AssertionError(f'Invalid timing {config["config"]}: {ms}')
                                    samples.append(dict(common, config=config['config'], family=config['family'],
                                                        repeat=repeat, position=position, time_ms=ms))
                            summary = []
                            for i, (config, _, _) in enumerate(variants):
                                times = [s['time_ms'] for s in samples if s['config']==config['config']]
                                summary.append(dict(common, **config, correctness='passed',
                                    max_abs_error=validation[i]['max_abs_error'], median_ms=statistics.median(times),
                                    min_ms=min(times), max_ms=max(times), p25_ms=float(np.quantile(times,.25)),
                                    p75_ms=float(np.quantile(times,.75)), warmup=a.warmup, repeats=a.repeats))
                            append(a.out/'summary.csv', summary)
                            append(a.out/'samples.csv', samples)
                            distributions = []
                            for k, slab_plans in plans.items():
                                for slab, plan in slab_plans.items():
                                    for metric, values in [('nnz', plan.unit_nnz), ('chunks', plan.unit_chunks)]:
                                        unique, counts = np.unique(values, return_counts=True)
                                        for value, count in zip(unique, counts):
                                            distributions.append(dict(common, k=k, slab=slab, metric=metric,
                                                value=int(value), row_units=int(count),
                                                ctas=int(count)*((f+slab-1)//slab)))
                            append(a.out/'cta_distribution.csv', distributions)
                            metadata['completed_cells'] += 1
                        save()
                        del variants, x, x_cpu
                    del plans, staged, boos, csr, mat
                    torch.cuda.empty_cache()
        metadata['status'] = 'complete'
    except BaseException as exc:
        metadata.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        save()
    print('BOO_ROWEXCL_PHASE_A_DONE', flush=True)


if __name__ == '__main__':
    main()
