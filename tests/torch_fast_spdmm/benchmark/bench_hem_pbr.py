"""Controlled symmetric FEM ordering experiment; run on a CUDA node.

python -m benchmark.bench_hem_pbr --out results/hem-pbr
All cases run in one process with interleaved CUDA-event timings. No W&B upload.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import time

import numpy as np
import torch
import torch_fast_spdmm as spmm
from torch_fast_spdmm.clustering import cluster_axes

from benchmark.bench_spmm import (PBR_CONFIGS, append_csv, digest_arrays, pbr_config,
                                  round_orders, source_metadata)
from benchmark.profile_pbr import build_matrix


def canonical(mat):
    mat = mat.astype(np.float32).tocsr()
    mat.sum_duplicates()
    mat.eliminate_zeros()
    mat.sort_indices()
    mat.indptr = mat.indptr.astype(np.int32)
    mat.indices = mat.indices.astype(np.int32)
    return mat


def symmetric_reorder(mat, labels):
    """p[new] = old; P[i,p[i]]=1, so A' = A[p][:,p] = P A P.T.

    Preserve increasing original vertex IDs within each HEM group, matching BOO.
    No padding, regrouping, or second clustering pass is performed.
    """
    labels = np.asarray(labels)
    if mat.shape[0] != mat.shape[1] or labels.shape != (mat.shape[0],):
        raise ValueError('Expected a square matrix and one label per vertex')
    if not np.issubdtype(labels.dtype, np.integer) or np.any(labels < 0):
        raise ValueError('Labels must be nonnegative integers')
    if (mat != mat.T).nnz:
        raise ValueError('Expected numerically symmetric input')
    p = np.argsort(labels, kind='stable')
    return canonical(mat[p][:, p]), p


def prepare(size, seed, args, out):
    start = time.perf_counter()
    original = canonical(build_matrix(size, 'fem2d', seed=seed))
    # Match the existing fem2d_permuted generator: shuffle whole four-DOF nodes.
    q = (np.random.default_rng(seed).permutation(original.shape[0] // 4)[:, None]
         * 4 + np.arange(4)).ravel()
    permuted = canonical(original[q][:, q])
    assert (permuted != canonical(build_matrix(size, 'fem2d_permuted', seed=seed))).nnz == 0
    assert (original != original.T).nnz == 0
    np.savez_compressed(out / f'permutation_n{size}_s{seed}.npz', random_order=q)
    variants = []
    preparation = dict(matrix_build_ms=(time.perf_counter() - start) * 1000)

    def add(mat, order, case, k=0, labels=None, hem_ms=0., reorder_ms=0.):
        matrix_hash = digest_arrays(mat.indptr, mat.indices, mat.data)
        configs = args.pbr_configs if labels is None else [(0, 0)]
        for block, minimum in configs:
            t0 = time.perf_counter()
            if labels is None:
                obj = spmm.csr_to_pbr(mat, block_rows=block, block_cols=block,
                                      min_nnz_per_block=minimum)
                blocks = int(obj.accounted_blocks())
            else:
                obj = spmm.csr_to_boo(mat, row_labels=labels, col_labels=labels,
                                      target_nr=k, target_nc=k, nz_order='rowmajor',
                                      small_block_policy='remainder', small_block_threshold=4)
                blocks = len(obj.blk_row_grp)
            build_ms = (time.perf_counter() - t0) * 1000
            stats = dict(case=case, k=k, block=block, min_nnz=minimum,
                         matrix_sha256=matrix_hash, order_sha256=digest_arrays(order),
                         num_blocks=blocks, compressed_fraction=obj.compressed_nnz() / mat.nnz,
                         remainder_nnz=int(obj.remainder_nnz()), build_ms=build_ms,
                         hem_ms=hem_ms, sparse_reorder_ms=reorder_ms)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            obj = obj.to('cuda')
            torch.cuda.synchronize()
            stats['staging_ms'] = (time.perf_counter() - t0) * 1000
            kernels = ['block'] if labels is None else ['rowseg', 'shared_atomic']
            # The extra reordered BOO case holds groups and kernel fixed.
            if case == 'hem_reordered_boo':
                kernels = ['rowseg']
            for kernel in kernels:
                config = f'{case}_k{k}_b{block}_min{minimum}_{kernel}'
                variants.append((dict(stats, config=config, kernel=kernel), obj, order))

    add(original, np.arange(original.shape[0]), 'original_pbr')
    add(permuted, q, 'permuted_pbr')
    for k in args.hem_k:
        t0 = time.perf_counter()
        rows, cols = cluster_axes(permuted, method='hem', target_nr=k, target_nc=k,
                                  max_nr=k, max_nc=k, symmetric=True, seed=seed,
                                  feasible_first=True)
        np.testing.assert_array_equal(rows, cols)
        hem_ms = (time.perf_counter() - t0) * 1000
        t0 = time.perf_counter()
        reordered, p = symmetric_reorder(permuted, rows)
        reorder_ms = (time.perf_counter() - t0) * 1000
        np.savez_compressed(out / f'hem_n{size}_s{seed}_k{k}.npz', labels=rows,
                            hem_order=p, original_order=q[p])
        add(permuted, q, 'permuted_boo', k, rows, hem_ms)
        add(reordered, q[p], 'hem_reordered_pbr', k, hem_ms=hem_ms,
            reorder_ms=reorder_ms)
        add(reordered, q[p], 'hem_reordered_boo', k, rows[p], hem_ms, reorder_ms)
        counts = np.bincount(rows)
        preparation[f'k{k}'] = dict(hem_ms=hem_ms, sparse_reorder_ms=reorder_ms,
                                    groups=len(counts), group_min=int(counts.min()),
                                    group_median=float(np.median(counts)), group_max=int(counts.max()))
        print(f'Prepared N={size} k={k}: {len(counts)} groups, HEM {hem_ms:.1f} ms', flush=True)
    return original, variants, preparation


def invoke(stats, obj, x):
    if stats['kernel'] == 'block':
        return spmm.pbr_matmul(obj, x, kernel='block', smem_size=2048, csr_mode=0)
    return spmm.boo_matmul(obj, x, threads=256, p=64, csr_mode=0,
                           row_seg=stats['kernel'] == 'rowseg', launch_mode='legacy')


def measure(original, variants, features, seed, args):
    x_cpu = torch.randn(original.shape[1], features,
                        generator=torch.Generator().manual_seed(seed), dtype=torch.float32)
    reference = original.astype(np.float64) @ x_cpu.numpy().astype(np.float64)
    inputs, errors = {}, {}
    for stats, obj, order in variants:
        key = stats['order_sha256']
        if key not in inputs:
            inputs[key] = x_cpu[torch.from_numpy(order)].contiguous().to('cuda')
        actual = invoke(stats, obj, inputs[key]).cpu().numpy().astype(np.float64)
        # Inverse scatter returns every result to the same original FEM coordinates.
        restored = np.empty_like(actual)
        restored[order] = actual
        np.testing.assert_allclose(restored, reference, rtol=1e-4, atol=1e-4,
                                   err_msg=stats['config'])
        errors[stats['config']] = float(np.max(np.abs(restored - reference)))
    del reference, actual, restored
    for stats, obj, _ in variants:
        for _ in range(args.warmup):
            invoke(stats, obj, inputs[stats['order_sha256']])
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    end.record()
    end.synchronize()
    samples = []
    for repeat, order in enumerate(round_orders(len(variants), args.repeats, seed)):
        for position, index in enumerate(order):
            stats, obj, _ = variants[index]
            x = inputs[stats['order_sha256']]
            start.record()
            y = invoke(stats, obj, x)
            end.record()
            end.synchronize()
            ms = start.elapsed_time(end)
            if not np.isfinite(ms) or ms <= 0:
                raise RuntimeError(f'Invalid CUDA timing: {ms}')
            samples.append(dict(config=stats['config'], repeat=repeat, position=position, time_ms=ms))
            del y
    summaries = []
    for stats, _, _ in variants:
        values = [s['time_ms'] for s in samples if s['config'] == stats['config']]
        summaries.append(dict(stats, median_ms=statistics.median(values),
                              p25_ms=float(np.percentile(values, 25)),
                              p75_ms=float(np.percentile(values, 75)), min_ms=min(values),
                              max_ms=max(values), correctness='passed',
                              max_abs_error=errors[stats['config']], x_sha256=digest_arrays(x_cpu.numpy())))
    return summaries, samples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sizes', type=int, nargs='+', default=[16384, 65536, 262144])
    parser.add_argument('--features', type=int, nargs='+', default=[128, 512])
    parser.add_argument('--seeds', type=int, nargs='+', default=[0])
    parser.add_argument('--hem-k', type=int, nargs='+', default=[8, 16, 32, 64])
    parser.add_argument('--pbr-configs', type=pbr_config, nargs='+',
                        default=[pbr_config(s) for s in PBR_CONFIGS])
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--repeats', type=int, default=20)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if (min(args.sizes + args.features + args.hem_k) <= 0 or max(args.hem_k) > 256
            or args.repeats < 1 or args.warmup < 0 or min(args.seeds) < 0
            or max(args.seeds) >= 2**32):
        parser.error('Invalid size, features, HEM target, seed, warmup or repeats')
    for values in (args.sizes, args.features, args.seeds, args.hem_k, args.pbr_configs):
        if len(set(values)) != len(values):
            parser.error('Sweep lists must not contain duplicates')
    if not torch.cuda.is_available():
        parser.error('Run on the remote CUDA node')
    args.out.mkdir(parents=True, exist_ok=False)
    metadata = source_metadata(Path(__file__).resolve().parents[3])
    metadata['baseline'] = 'original_pbr; no cuSPARSE baseline in this experiment'
    metadata.update(arguments=dict(vars(args), out=str(args.out)), status='running',
                    preparation={}, completed_cells=0,
                    permutation_convention='p[new]=old; Xprime=X[p]; Y[order]=Yprime',
                    excluded_costs='HEM, sparse permutation, format build, staging, dense input/output permutation',
                    hem_feasible_first=True, pbr_smem=2048, boo_threads=256, boo_p=64,
                    boo_policy='remainder', boo_threshold=4, boo_launch='legacy')
    meta_path = args.out / 'metadata.json'
    meta_path.write_text(json.dumps(metadata, indent=2) + '\n')
    try:
        with torch.inference_mode():
            for size in args.sizes:
                for seed in args.seeds:
                    original, variants, preparation = prepare(size, seed, args, args.out)
                    metadata['preparation'][f'n{size}_s{seed}'] = preparation
                    for features in args.features:
                        summaries, samples = measure(original, variants, features, seed, args)
                        common = dict(requested_size=size, rows=original.shape[0], nnz=original.nnz,
                                      features=features, seed=seed)
                        summaries = [dict(common, **s) for s in summaries]
                        samples = [dict(common, **s) for s in samples]
                        append_csv(args.out / 'summary.csv', summaries, list(summaries[0]))
                        append_csv(args.out / 'samples.csv', samples, list(samples[0]))
                        metadata['completed_cells'] += 1
                        meta_path.write_text(json.dumps(metadata, indent=2) + '\n')
                        print(f'Completed N={size} f={features} seed={seed}: '
                              f'{len(summaries)} validated configurations', flush=True)
                    del original, variants
                    torch.cuda.empty_cache()
        metadata['status'] = 'complete'
    except BaseException as exc:
        metadata.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        meta_path.write_text(json.dumps(metadata, indent=2) + '\n')


if __name__ == '__main__':
    main()
