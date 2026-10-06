"""Compare actual clustering backends through the uint16 square-group limit.

Run on a GPU node: python -m benchmark.bench_clustering --out results/clustering
No backend substitutions, external logging, or format changes are performed.
"""

import argparse
from dataclasses import asdict
from importlib.metadata import version
import json
from pathlib import Path
import time

import numpy as np
import torch
import torch_fast_spdmm as spmm
from torch_fast_spdmm.clustering import cluster_axes
from torch_fast_spdmm.boo import boo_arena_bytes

from benchmark.bench_spmm import append_csv, digest_arrays, measure_cell, source_metadata
from benchmark.profile_pbr import build_matrix, _MATRIX_PROFILES


def feature_tile(boo, features, limit):
    """Largest supported tile <=64 that fits, including CUDA alignment slack."""
    p = 64
    while p > 8 and p > features:
        p //= 2
    while True:
        arena = boo_arena_bytes(boo.max_nr, boo.max_nc, boo.max_nnz_blk, p, 4) + 16
        if not len(boo.blk_row_grp) or arena <= limit:
            return p, arena, True
        if p == 8:
            return p, arena, False
        p //= 2


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--methods', nargs='+', choices=['identity', 'hem', 'metis', 'graclus'],
                        default=['identity', 'hem', 'metis', 'graclus'])
    parser.add_argument('--groups', nargs='+', type=int, default=[8, 16, 32, 64, 128, 256])
    parser.add_argument('--sizes', nargs='+', type=int, default=[16384, 65536])
    parser.add_argument('--features', nargs='+', type=int, default=[128, 512])
    parser.add_argument('--profiles', nargs='+', choices=list(_MATRIX_PROFILES),
                        default=['fem2d', 'fem2d_permuted', 'diag_b0.3', 'diag_b1.0'])
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--repeats', type=int, default=20)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    if (min(args.groups + args.sizes + args.features) < 1 or max(args.groups) > 256
            or args.repeats < 1 or args.warmup < 0 or not 0 <= args.seed < 2**32):
        parser.error('Positive sizes required, groups <=256, seed in uint32, repeats >=1, warmup >=0')
    for values in (args.methods, args.groups, args.sizes, args.features, args.profiles):
        if len(set(values)) != len(values):
            parser.error('Sweep lists must not contain duplicates')
    if not torch.cuda.is_available():
        raise RuntimeError('This benchmark requires a GPU node')
    args.out.mkdir(parents=True, exist_ok=False)
    metadata = source_metadata(Path(__file__).resolve().parents[3])
    metadata.update(arguments={**vars(args), 'out': str(args.out)}, status='running',
                    backend_versions={name: version(name) for method, name in
                                      [('metis', 'pymetis'), ('graclus', 'torch-cluster')]
                                      if method in args.methods},
                    tile_policy='largest fitting p in 64,32,16,8, bounded by feature count')
    meta_path = args.out / 'metadata.json'
    meta_path.write_text(json.dumps(metadata, indent=2) + '\n')
    summary_columns = None
    try:
        limit = torch.cuda.get_device_properties(0).shared_memory_per_block_optin
        with torch.inference_mode():
            for size in args.sizes:
                for profile in args.profiles:
                    mat = build_matrix(size, profile, seed=args.seed).astype(np.float32).tocsr()
                    mat.sum_duplicates()
                    mat.sort_indices()
                    if max(*mat.shape, mat.nnz) > np.iinfo(np.int32).max:
                        raise ValueError('CSR exceeds int32 limits')
                    mat.indices = mat.indices.astype(np.int32)
                    mat.indptr = mat.indptr.astype(np.int32)
                    csr = torch.sparse_csr_tensor(torch.from_numpy(mat.indptr),
                                                  torch.from_numpy(mat.indices),
                                                  torch.from_numpy(mat.data), size=mat.shape).cuda()
                    variants = [(dict(algorithm='cusparse_csr', config='cusparse_csr'), csr)]
                    for method in args.methods:
                        for k in args.groups:
                            name = f'{method}_k{k}'
                            print(f'prepare {profile} N={size} {name}', flush=True)
                            start = time.perf_counter()
                            rows, cols = cluster_axes(mat, method=method, target_nr=k, target_nc=k,
                                                      max_nr=k, max_nc=k, seed=args.seed,
                                                      feasible_first=True)
                            cluster_ms = 1000 * (time.perf_counter() - start)
                            np.savez_compressed(args.out / f'{profile}_n{size}_{name}.npz',
                                                rows=rows, cols=cols)
                            start = time.perf_counter()
                            boo = spmm.csr_to_boo(mat, row_labels=rows, col_labels=cols)
                            build_ms = 1000 * (time.perf_counter() - start)
                            info = asdict(spmm.boo_info(boo))
                            # The CUDA launch computes this per actual group, with padded columns.
                            row_sizes, col_sizes = np.bincount(rows), np.bincount(cols)
                            config = dict(algorithm='boo', config=name, cluster=method, k=k,
                                          threads=256, launch_mode='auto', cluster_ms=cluster_ms,
                                          build_ms=build_ms, staging_ms=0.,
                                          num_blocks=info['num_blocks'], x_loads=info['x_loads'],
                                          y_writes=info['y_writes'],
                                          compressed_fraction=info['compressed_fraction'],
                                          remainder_nnz=int(boo.remainder_nnz()),
                                          max_nnz_blk=int(boo.max_nnz_blk),
                                          row_group_mean=float(row_sizes.mean()),
                                          row_group_max=int(row_sizes.max()),
                                          col_group_mean=float(col_sizes.mean()),
                                          col_group_max=int(col_sizes.max()),
                                          labels_sha256=digest_arrays(rows, cols))
                            # Oversized sparse payloads cannot launch even with P=8.
                            if feature_tile(boo, 8, limit)[2]:
                                start = time.perf_counter()
                                boo.to('cuda')
                                torch.cuda.synchronize()
                                config['staging_ms'] = 1000 * (time.perf_counter() - start)
                            variants.append((config, boo))
                    for features in args.features:
                        prepared = []
                        for config, obj in variants:
                            config = dict(config)
                            if config['algorithm'] == 'boo':
                                p, arena, supported = feature_tile(obj, features, limit)
                                config.update(p=p, arena_bytes=arena, supported=supported,
                                              shared_limit_bytes=limit,
                                              unsupported_reason='' if supported else
                                              'shared-memory arena exceeds device limit even at p=8')
                            prepared.append((config, obj))
                        x = torch.randn(mat.shape[1], features,
                                        generator=torch.Generator().manual_seed(args.seed))
                        summaries, samples = measure_cell(prepared, mat, x, args.warmup,
                                                          args.repeats, args.seed)
                        common = dict(profile=profile, requested_size=size, features=features,
                                      nnz=mat.nnz, seed=args.seed,
                                      matrix_sha256=digest_arrays(mat.indptr, mat.indices, mat.data),
                                      x_sha256=digest_arrays(x.numpy()))
                        summaries = [dict(common, **s) for s in summaries]
                        samples = [dict(common, **s) for s in samples]
                        if summary_columns is None:
                            summary_columns = sorted({key for row in summaries for key in row})
                        append_csv(args.out / 'summary.csv', summaries, summary_columns)
                        append_csv(args.out / 'samples.csv', samples, list(samples[0]))
                        for row in summaries:
                            print(f"{profile} N={size} F={features} {row['config']}: "
                                  f"{row['median_ms']} ms ({row['correctness']})", flush=True)
                    del prepared, variants, boo, obj, csr, mat
                    torch.cuda.empty_cache()
        metadata['status'] = 'complete'
    except BaseException as exc:
        metadata.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        meta_path.write_text(json.dumps(metadata, indent=2) + '\n')


if __name__ == '__main__':
    main()
