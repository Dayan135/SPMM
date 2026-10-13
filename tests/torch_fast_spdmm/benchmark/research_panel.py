"""Saved-label synthetic timing and profiling panel for the next BOO experiments.

Run on the cluster from tests/torch_fast_spdmm. First use --phase cluster on
a CPU node (CUDA_VISIBLE_DEVICES=), then --phase benchmark or profile on GPU.
Profile exactly one workload/method under ncu or nsys using cudaProfilerApi.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import json
from pathlib import Path
import time

import numpy as np
import torch
import torch_fast_spdmm as spmm
import torch_fast_spdmm_core as core

from benchmark.bench_clustering import cluster_phase, feature_tile, load_matrix, saved_labels
from benchmark.bench_spmm import (DIRECT_ALGORITHMS, append_csv, digest_arrays, measure_cell,
                                  source_metadata)
from benchmark.panel_structure import boo_structure, csr_structure
from benchmark.profile_pbr import _MATRIX_PROFILES

PROFILES = ['fem2d', 'fem2d_permuted', 'diag_b0.3', 'diag_b1.0',
            'diag_skewed', 'pattern_row_skewed']


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase', choices=['cluster', 'benchmark', 'profile'], required=True)
    p.add_argument('--labels', type=Path, required=True)
    p.add_argument('--out', type=Path)
    p.add_argument('--sizes', nargs='+', type=int, default=[262144])
    p.add_argument('--profiles', nargs='+', choices=sorted(_MATRIX_PROFILES), default=PROFILES)
    p.add_argument('--features', nargs='+', type=int, default=[128, 512])
    p.add_argument('--methods', nargs='+', choices=['identity', 'hem', 'metis', 'graclus'],
                   default=['identity', 'hem'])
    p.add_argument('--groups', nargs='+', type=int, choices=[16, 32, 64], default=[32, 64])
    p.add_argument('--boo-fpt', nargs='+', type=int, choices=[1, 2, 4], default=[1],
                   help='Pair features-per-thread variants on identical saved labels; requires FPT build')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--warmup', type=int, default=5)
    p.add_argument('--repeats', type=int, default=20)
    p.add_argument('--profile-method', choices=['boo', 'torch', 'direct'], default='boo')
    p.add_argument('--direct-algorithm', choices=DIRECT_ALGORITHMS, default='DEFAULT')
    p.add_argument('--baseline-summary', type=Path,
                   help='Profile the best full-call direct algorithm for this exact A/X from a completed panel')
    a = p.parse_args(argv)
    if min(a.sizes + a.features) <= 0 or a.repeats < 1 or a.warmup < 0:
        p.error('sizes/features/repeats must be positive and warmup nonnegative')
    if not 0 <= a.seed < 2**32:
        p.error('seed must fit uint32')
    for values in (a.sizes, a.profiles, a.features, a.methods, a.groups, a.boo_fpt):
        if len(values) != len(set(values)):
            p.error('duplicate sweep entries are not allowed')
    if a.phase == 'benchmark' and 1 not in a.boo_fpt:
        p.error('--boo-fpt must include 1 as the paired baseline')
    if a.phase != 'cluster' and a.out is None:
        p.error('--out is required for benchmark/profile')
    if a.phase == 'profile' and any(len(v) != 1 for v in
                                    (a.sizes, a.profiles, a.features, a.methods, a.groups, a.boo_fpt)):
        p.error('profile requires exactly one size/profile/F/clustering method/group/FPT')
    return a


def validate_labels(rows, cols, shape, k):
    for labels, length in zip((rows, cols), shape):
        if labels.ndim != 1 or len(labels) != length or labels.dtype.kind not in 'iu':
            raise ValueError('saved labels have incorrect shape or dtype')
        if labels.size:
            if labels.min() < 0 or labels.max() >= length:
                raise ValueError('saved labels are out of range')
            counts = np.bincount(labels)
            if (counts == 0).any() or counts.max() > k:
                raise ValueError('saved labels have gaps or exceed the group cap')


def prepare_boo(a, mat, profile, size, method, k):
    matrix_sha = digest_arrays(mat.indptr, mat.indices, mat.data)
    rows, cols, cluster_ms, provenance = saved_labels(a, profile, size, method, k, matrix_sha)
    validate_labels(rows, cols, mat.shape, k)
    t0 = time.perf_counter()
    boo = spmm.csr_to_boo(mat, row_labels=rows, col_labels=cols,
                          nz_order='rowmajor', small_block_policy='remainder',
                          small_block_threshold=4)
    build_ms = 1000 * (time.perf_counter() - t0)
    info = spmm.boo_info(boo)
    config = dict(algorithm='boo', config=f'boo_{method}_k{k}', cluster=method, k=k,
                  threads=256, launch_mode='auto', hem_feasible_first=True,
                  small_block_policy='remainder', small_block_threshold=4,
                  nz_order='rowmajor', cluster_ms=cluster_ms, build_ms=build_ms,
                  labels_sha256=digest_arrays(rows, cols),
                  compressed_fraction=info.compressed_fraction, remainder_nnz=info.remainder_nnz,
                  num_blocks=info.num_blocks, **provenance)
    return config, boo


def matched_ratios(rows):
    """Replace legacy convenience ratios that mix full-call and C++ scopes."""
    for row in rows:
        scope = row['timing_scope']
        direct_scope = 'cusparseSpMM_only' if scope == 'cpp_only' else scope
        direct = [r['median_ms'] for r in rows if r['algorithm'] == 'cusparse_direct'
                  and r['timing_scope'] == direct_scope and r.get('median_ms')]
        torch_times = [r['median_ms'] for r in rows if r['algorithm'] == 'cusparse_csr'
                       and r['timing_scope'] == scope and r.get('median_ms')]
        row['speedup_vs_direct'] = min(direct) / row['median_ms'] if direct and row.get('median_ms') else None
        row['speedup_vs_cusparse'] = (min(torch_times) / row['median_ms']
                                      if torch_times and row.get('median_ms') else None)
        if row['algorithm'] == 'boo' and scope == 'cpp_only':
            row['launch_mode'] = 'preallocated_legacy_fork_join'


def select_direct_algorithm(path, matrix_sha, x_sha):
    with path.open() as f:
        rows = [r for r in csv.DictReader(f) if r['algorithm'] == 'cusparse_direct'
                and r['timing_scope'] == 'full_call' and r['correctness'] == 'passed'
                and r['matrix_sha256'] == matrix_sha and r['x_sha256'] == x_sha]
    if not rows:
        raise ValueError('baseline summary has no validated direct configuration for this exact A/X')
    algorithm = min(rows, key=lambda r: float(r['median_ms']))['direct_algorithm']
    if algorithm not in DIRECT_ALGORITHMS:
        raise ValueError('baseline summary contains an unknown direct algorithm')
    return algorithm


def benchmark(a):
    columns = sample_columns = None
    limit = torch.cuda.get_device_properties(0).shared_memory_per_block_optin
    with torch.inference_mode():
        for size in a.sizes:
            for profile in a.profiles:
                print(f'PANEL {profile} N={size}', flush=True)
                mat = load_matrix(size, profile, a.seed)
                csr = torch.sparse_csr_tensor(torch.from_numpy(mat.indptr.copy()),
                                               torch.from_numpy(mat.indices.copy()),
                                               torch.from_numpy(mat.data.copy()), size=mat.shape).cuda()
                variants = [(dict(algorithm='cusparse_csr', config='cusparse_csr'), csr)]
                structures = dict(csr=csr_structure(mat), boo={})
                for method in a.methods:
                    for k in a.groups:
                        config, boo = prepare_boo(a, mat, profile, size, method, k)
                        structures['boo'][config['config']] = boo_structure(boo)
                        if feature_tile(boo, 8, limit)[2]:
                            torch.cuda.synchronize()
                            t0 = time.perf_counter()
                            boo.to('cuda')
                            config['staging_ms'] = 1000 * (time.perf_counter() - t0)
                        variants.append((config, boo))
                (a.out / f'{profile}_n{size}_structure.json').write_text(json.dumps(structures, indent=2) + '\n')
                for f in a.features:
                    prepared, unsupported = [], []
                    for config, obj in variants:
                        config = dict(config)
                        if config['algorithm'] == 'boo':
                            p, arena, supported = feature_tile(obj, f, limit)
                            info = spmm.boo_info(obj, p=p, threads=256)
                            config.update(p=p, arena_bytes=arena, supported=supported,
                                          occupancy_pct=info.occupancy_pct, ctas_per_sm=info.ctas_per_sm,
                                          max_nr=obj.max_nr, max_nc=obj.max_nc, max_nnz_blk=obj.max_nnz_blk)
                            if not supported:
                                unsupported.append(dict(config, correctness='not_supported',
                                                        unsupported_reason='shared-memory limit exceeded at P=8',
                                                        timing_scope='full_call', median_ms=None))
                                continue
                        if config['algorithm'] == 'boo':
                            for fpt in a.boo_fpt:
                                suffix = '' if fpt == 1 else f'_fpt{fpt}'
                                prepared.append((dict(config, config=config['config'] + suffix,
                                                       features_per_thread=fpt), obj))
                        else:
                            prepared.append((config, obj))
                    x_cpu = torch.randn(mat.shape[1], f, generator=torch.Generator().manual_seed(a.seed))
                    rows, samples = measure_cell(prepared, mat, x_cpu, a.warmup, a.repeats, a.seed,
                                                  direct_algorithms=DIRECT_ALGORITHMS, symmetric=True,
                                                  diagnostics=False)
                    rows.extend(unsupported)
                    matched_ratios(rows)
                    common = dict(profile=profile, requested_size=size, features=f, seed=a.seed,
                                  nnz=mat.nnz, matrix_sha256=digest_arrays(np.asarray(mat.shape),
                                      mat.indptr, mat.indices, mat.data), x_sha256=digest_arrays(x_cpu.numpy()))
                    rows = [dict(common, **r) for r in rows]
                    samples = [dict(common, **r) for r in samples]
                    # Fixed schema handles later unsupported configs without dropping fields.
                    if columns is None:
                        columns = sorted(set().union(*(r.keys() for r in rows),
                            {'unsupported_reason', 'supported', 'staging_ms', 'max_abs_error'}))
                        sample_columns = sorted(set().union(*(r.keys() for r in samples)))
                    append_csv(a.out / 'summary.csv', rows, columns)
                    append_csv(a.out / 'samples.csv', samples, sample_columns)
                    print(f'  F={f}: {len(rows)} configurations, correctness checked', flush=True)
                del prepared, variants, csr, boo, obj, mat
                torch.cuda.empty_cache()


def profile(a):
    size, name, f = a.sizes[0], a.profiles[0], a.features[0]
    mat = load_matrix(size, name, a.seed)
    x_cpu = torch.randn(mat.shape[1], f, generator=torch.Generator().manual_seed(a.seed))
    x = x_cpu.cuda()
    details = dict(method=a.profile_method, profile=name, requested_size=size, features=f,
                   matrix_sha256=digest_arrays(np.asarray(mat.shape), mat.indptr, mat.indices, mat.data),
                   x_sha256=digest_arrays(x_cpu.numpy()), csr=csr_structure(mat))
    if a.profile_method == 'boo':
        config, obj = prepare_boo(a, mat, name, size, a.methods[0], a.groups[0])
        p, arena, supported = feature_tile(obj, f, torch.cuda.get_device_properties(0).shared_memory_per_block_optin)
        if not supported:
            raise RuntimeError('BOO profile exceeds shared-memory limit')
        details.update(config=config, p=p, features_per_thread=a.boo_fpt[0],
                       arena_bytes=arena, boo_info=asdict(spmm.boo_info(obj, p=p)))
        obj.to('cuda')
        call = lambda: spmm.boo_matmul(obj, x, threads=256, p=p, row_seg=True,
                                       csr_mode=0, launch_mode='auto', features_per_thread=a.boo_fpt[0])
    else:
        csr = torch.sparse_csr_tensor(torch.from_numpy(mat.indptr.copy()),
                                      torch.from_numpy(mat.indices.copy()), torch.from_numpy(mat.data.copy()),
                                      size=mat.shape).cuda()
        if a.profile_method == 'torch':
            call = lambda: torch.sparse.mm(csr, x)
        else:
            algorithm = (select_direct_algorithm(a.baseline_summary, details['matrix_sha256'],
                                                 details['x_sha256'])
                         if a.baseline_summary else a.direct_algorithm)
            obj = core.DirectSpMM(csr.crow_indices(), csr.col_indices(), csr.values(), x, algorithm)
            details.update(direct_algorithm=algorithm, direct_info=dict(obj.info()))
            call = obj.run
    with torch.inference_mode():
        reference = mat.astype(np.float64) @ x_cpu.numpy().astype(np.float64)
        actual = call().cpu().numpy().astype(np.float64)
        np.testing.assert_allclose(actual, reference, rtol=1e-4, atol=1e-4)
        details.update(correctness='passed', max_abs_error=float(np.max(np.abs(actual - reference))),
                       warmup=a.warmup, profiled_calls=1, timing_scope='full_call',
                       direct_output_reused=a.profile_method == 'direct')
        del actual, reference
        for _ in range(a.warmup):
            call()
        torch.cuda.synchronize()
        torch.cuda.profiler.start()
        y = call()
        torch.cuda.synchronize()
        torch.cuda.profiler.stop()
        del y
    (a.out / 'profile.json').write_text(json.dumps(details, indent=2) + '\n')


def main(argv=None):
    a = parse_args(argv)
    if a.phase == 'cluster':
        cluster_phase(a)
        return
    if not torch.cuda.is_available():
        raise RuntimeError('Run GPU phases on the remote GPU node')
    a.out.mkdir(parents=True, exist_ok=False)
    metadata = dict(source_metadata(Path(__file__).resolve().parents[3]),
                    arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()},
                    status='running', diagnostic_variants=False,
                    comparison='paired full-call and preallocated C++ scopes; no cross-scope ratios',
                    cpp_boo_launch='existing preallocated legacy fork/join; API uses auto',
                    tile_policy='largest fitting P in 64,32,16,8; record P per configuration')
    path = a.out / 'metadata.json'
    path.write_text(json.dumps(metadata, indent=2) + '\n')
    try:
        (benchmark if a.phase == 'benchmark' else profile)(a)
    except BaseException as exc:
        metadata.update(status='failed', error=str(exc))
        raise
    else:
        metadata['status'] = 'complete'
    finally:
        path.write_text(json.dumps(metadata, indent=2) + '\n')


if __name__ == '__main__':
    main()
