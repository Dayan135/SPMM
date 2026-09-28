"""Replicated A--F FEM ordering experiment, with auditable HEM hierarchy.

Run each replicate in a separate Slurm job. Timing matches bench_hem_pbr.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import socket
import statistics
import time

import numpy as np
import torch
import torch_fast_spdmm as spmm
from torch_fast_spdmm.clustering import cluster_axis

from benchmark.bench_hem_pbr import canonical, measure
from benchmark.bench_spmm import PBR_CONFIGS, append_csv, digest_arrays, pbr_config, source_metadata
from benchmark.profile_pbr import build_matrix


def hierarchy_order(labels, levels):
    """Final group first, then coarsest-to-finest matching labels, then vertex ID.

    np.lexsort's LAST key is primary. _hem_labels records levels finest first.
    """
    vertices = np.arange(len(labels), dtype=np.int64)
    return np.lexsort((vertices, *levels, labels))


def assert_identical_boo_blocks(c, e):
    # With stable label order, even local nz coordinates/values are identical.
    # Only row_ids/col_ids and remainder global coordinates can differ.
    for field in ('row_grp_ptr', 'col_grp_ptr', 'blk_row_grp', 'blk_col_grp',
                  'blk_nz_ptr', 'blk_shift', 'nz_idx', 'nz_val'):
        np.testing.assert_array_equal(getattr(c, field), getattr(e, field), err_msg=field)


def alignment(labels, order):
    position = np.argsort(order)
    counts = np.bincount(labels)
    lo = np.full(len(counts), len(labels), dtype=np.int64)
    hi = np.zeros(len(counts), dtype=np.int64)
    np.minimum.at(lo, labels, position)
    np.maximum.at(hi, labels, position)
    result = {}
    for size in (2, 4, 8):
        mask = counts == size
        result[str(size)] = dict(groups=int(mask.sum()),
                                contiguous=int(np.sum(mask & (hi - lo + 1 == size))),
                                aligned=int(np.sum(mask & (hi - lo + 1 == size) & (lo % size == 0))))
    return result


def prepare(size, args):
    original = canonical(build_matrix(size, 'fem2d', seed=args.seed))
    if (original != original.T).nnz:
        raise ValueError('FEM must be symmetric')
    n = original.shape[0]
    identity = np.arange(n, dtype=np.int64)
    node_order = (np.random.default_rng(args.seed).permutation(n // 4)[:, None]
                  * 4 + np.arange(4)).ravel()
    dof_order = np.random.default_rng(args.seed).permutation(n)
    node_mat = canonical(original[node_order][:, node_order])
    dof_mat = canonical(original[dof_order][:, dof_order])
    assert (node_mat != canonical(build_matrix(size, 'fem2d_permuted', seed=args.seed))).nnz == 0
    variants, permutations, diagnostics = [], {}, {}
    np.savez_compressed(args.out / f'input_orders_n{size}.npz', node=node_order, dof=dof_order)

    def add(mat, order, arm, k=0, labels=None, prep=None, p=None):
        prep = prep or dict(cluster_ms=0., order_ms=0., pap_ms=0.,
                            group_min=0, group_median=0., group_max=0, num_groups=0)
        key = digest_arrays(order)
        permutations[key] = (order, p)
        matrix_hash = digest_arrays(mat.indptr, mat.indices, mat.data)
        created = None
        for block, minimum in (args.pbr_configs if labels is None else [(0, 0)]):
            start = time.perf_counter()
            if labels is None:
                obj = spmm.csr_to_pbr(mat, block_rows=block, block_cols=block,
                                      min_nnz_per_block=minimum)
            else:
                obj = spmm.csr_to_boo(mat, row_labels=labels, col_labels=labels,
                                      target_nr=k, target_nc=k, nz_order='rowmajor',
                                      small_block_policy='remainder', small_block_threshold=4)
            build_ms = 1000 * (time.perf_counter() - start)
            boo = spmm.boo_info(obj, p=64, threads=256, row_seg=True) if labels is not None else None
            stats = dict(arm=arm, k=k, block=block, min_nnz=minimum,
                         kernel='rowseg' if boo else 'block', method='hem' if k else 'none',
                         smem_tier=2048 if not boo else 0, threads=256 if boo else 0,
                         p=64 if boo else 0, dtype='float32', index_dtype='int32',
                         target=k, max_size=k, row_seg=bool(boo), boo_launch='legacy' if boo else '',
                         policy='remainder' if boo else 'min_nnz', threshold=4 if boo else minimum,
                         feasible_first=bool(k), nz_order='rowmajor' if boo else '', csr_mode=0,
                         build_ms=build_ms, matrix_sha256=matrix_hash, order_sha256=key,
                         pbr_compressed_fraction=obj.compressed_nnz()/mat.nnz if not boo else '',
                         pbr_blocks=int(obj.accounted_blocks()) if not boo else '',
                         boo_blocks=boo.num_blocks if boo else '',
                         boo_x_loads=boo.x_loads if boo else '',
                         boo_occupancy_pct=boo.occupancy_pct if boo else '',
                         boo_arena_bytes=boo.arena_bytes if boo else '',
                         boo_compressed_fraction=boo.compressed_fraction if boo else '',
                         remainder_nnz=int(obj.remainder_nnz()), **prep)
            torch.cuda.synchronize()
            start = time.perf_counter()
            obj = obj.to('cuda')
            torch.cuda.synchronize()
            stats['staging_ms'] = 1000 * (time.perf_counter() - start)
            stats['config'] = f'{arm}_k{k}_b{block}_min{minimum}'
            variants.append((stats, obj, order))
            created = obj
        return created

    add(original, identity, 'A')
    add(node_mat, node_order, 'B')
    add(dof_mat, dof_order, 'F_PBR')
    for k in args.hem_k:
        for name, mat, order, arm in [('node', node_mat, node_order, 'C'),
                                      ('dof', dof_mat, dof_order, 'F_BOO'),
                                      ('natural', original, identity, 'NATURAL_BOO')]:
            levels = []
            start = time.perf_counter()
            labels = cluster_axis(mat, method='hem', target=k, max_size=k,
                                   feasible_first=True, hem_hierarchy=levels)
            cluster_ms = 1000 * (time.perf_counter() - start)
            counts = np.bincount(labels)
            prep = dict(cluster_ms=cluster_ms, order_ms=0., pap_ms=0., num_groups=len(counts),
                        group_min=int(counts.min()), group_median=float(np.median(counts)),
                        group_max=int(counts.max()))
            c = add(mat, order, arm, k, labels, prep)
            np.savez_compressed(args.out / f'hem_{name}_n{size}_k{k}.npz', labels=labels,
                                **{f'level_{i:02d}': level for i, level in enumerate(levels)})
            if name != 'node':
                continue
            for mode, new_arm in [('stable', 'D'), ('hierarchy', 'Dprime')]:
                start = time.perf_counter()
                p = np.argsort(labels, kind='stable') if mode == 'stable' else hierarchy_order(labels, levels)
                order_ms = 1000 * (time.perf_counter() - start)
                start = time.perf_counter()
                reordered = canonical(mat[p][:, p])
                pap_ms = 1000 * (time.perf_counter() - start)
                new_prep = dict(prep, order_ms=order_ms, pap_ms=pap_ms)
                add(reordered, order[p], new_arm, k, prep=new_prep, p=p)
                np.savez_compressed(args.out / f'order_{mode}_n{size}_k{k}.npz', p=p, original_order=order[p])
                diagnostics[f'n{size}_k{k}_{mode}'] = [alignment(level, p) for level in levels]
                if mode == 'stable':
                    e = add(reordered, order[p], 'E', k, labels[p], new_prep, p)
                    assert_identical_boo_blocks(c, e)
                    np.testing.assert_array_equal(e.row_ids, identity)
                    np.testing.assert_array_equal(e.col_ids, identity)
            print(f'Prepared N={size} k={k}: node HEM {cluster_ms:.1f} ms', flush=True)
    return original, variants, permutations, diagnostics


def time_permutations(permutations, n, f, args):
    """One allocating GPU index_select per sample; no host transfer/index build.

    External costs include the composed original->arm order. HEM-only costs
    additionally measure B->D/E (or Dprime), without charging the node shuffle.
    """
    x = torch.randn(n, f, device='cuda')
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    end.record()
    end.synchronize()
    records, raw, cache = {}, [], {}
    for key, (order, hem_order) in permutations.items():
        costs = {}
        for prefix, perm in [('external', order), ('hem_only', hem_order)]:
            if perm is None or np.array_equal(perm, np.arange(n)):
                costs.update({f'{prefix}_x_ms': 0., f'{prefix}_y_ms': 0.})
                continue
            perm_key = digest_arrays(perm)
            if perm_key in cache:
                costs.update({f'{prefix}_{axis}_ms': cache[perm_key][axis] for axis in ('x', 'y')})
                continue
            indices = {'x': torch.from_numpy(perm).to('cuda'),
                       'y': torch.from_numpy(np.argsort(perm)).to('cuda')}
            values = {'x': [], 'y': []}
            for index in indices.values():
                for _ in range(args.warmup):
                    x.index_select(0, index)
            torch.cuda.synchronize()
            for repeat in range(args.repeats):
                for axis in ('x', 'y') if repeat % 2 == 0 else ('y', 'x'):
                    start.record()
                    y = x.index_select(0, indices[axis])
                    end.record()
                    end.synchronize()
                    ms = start.elapsed_time(end)
                    values[axis].append(ms)
                    raw.append(dict(order_sha256=key, permutation_sha256=perm_key, scope=prefix,
                                    axis=axis, repeat=repeat, time_ms=ms))
                    del y
            cache[perm_key] = {axis: statistics.median(v) for axis, v in values.items()}
            costs.update({f'{prefix}_{axis}_ms': cache[perm_key][axis] for axis in ('x', 'y')})
        records[key] = costs
    return records, raw


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sizes', type=int, nargs='+', default=[65536, 262144])
    parser.add_argument('--features', type=int, nargs='+', default=[128, 512])
    parser.add_argument('--hem-k', type=int, nargs='+', default=[8, 16, 32, 64])
    parser.add_argument('--pbr-configs', type=pbr_config, nargs='+', default=[pbr_config(s) for s in PBR_CONFIGS])
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--repeats', type=int, default=20)
    parser.add_argument('--replicate', type=int, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if (min(args.sizes + args.features + args.hem_k) < 1 or max(args.hem_k) > 256
            or args.repeats < 1 or args.warmup < 0 or not 0 <= args.seed < 2**32):
        parser.error('Invalid dimensions, k, repeats, warmup or seed')
    for values in (args.sizes, args.features, args.hem_k, args.pbr_configs):
        if len(values) != len(set(values)):
            parser.error('Duplicate sweep entries')
    if not torch.cuda.is_available():
        parser.error('CUDA GPU required')
    args.out.mkdir(parents=True, exist_ok=False)
    metadata = source_metadata(Path(__file__).resolve().parents[3])
    metadata.update(baseline='A: natural FEM + PBR', node=socket.gethostname(),
                    slurm_nodelist=os.environ.get('SLURM_JOB_NODELIST'),
                    arguments=dict(vars(args), out=str(args.out)), status='running', completed_cells=0,
                    occupancy='boo_info SM86 shared-memory/thread-limit estimate; not measured achieved occupancy',
                    x_loads='sum of column-group sizes over BOO blocks, per feature; excludes CSR remainder',
                    dense_permutation='one GPU index_select per sample, allocation included; pre-staged indices',
                    hierarchy='identity and each contracted label vector, finest first; lexsort last key primary')
    path = args.out / 'metadata.json'
    path.write_text(json.dumps(metadata, indent=2) + '\n')
    try:
        with torch.inference_mode():
            for n in args.sizes:
                original, variants, permutations, diagnostics = prepare(n, args)
                (args.out / f'alignment_n{n}.json').write_text(json.dumps(diagnostics, indent=2) + '\n')
                for f in args.features:
                    permutation_costs, permutation_samples = time_permutations(permutations, original.shape[0], f, args)
                    summaries, samples = measure(original, variants, f, args.seed, args)
                    common = dict(replicate=args.replicate, job_id=metadata['slurm_job_id'], node=metadata['node'],
                                  commit=metadata['git_commit'], source_sha256=metadata['source_sha256'],
                                  requested_size=n, rows=original.shape[0], features=f, seed=args.seed,
                                  warmup=args.warmup, repeats=args.repeats)
                    summaries = [dict(common, **r, **permutation_costs[r['order_sha256']]) for r in summaries]
                    samples = [dict(common, **r) for r in samples]
                    permutation_samples = [dict(common, **r) for r in permutation_samples]
                    for name, records in [('summary', summaries), ('samples', samples),
                                          ('permutation_samples', permutation_samples)]:
                        if records:
                            append_csv(args.out / f'{name}.csv', records, list(records[0]))
                    metadata['completed_cells'] += 1
                    path.write_text(json.dumps(metadata, indent=2) + '\n')
                    print(f'Completed N={n}, F={f}: {len(summaries)} validated configurations', flush=True)
                del original, variants, permutations
                torch.cuda.empty_cache()
        metadata['status'] = 'complete'
    except BaseException as exc:
        metadata.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        path.write_text(json.dumps(metadata, indent=2) + '\n')


if __name__ == '__main__':
    main()
