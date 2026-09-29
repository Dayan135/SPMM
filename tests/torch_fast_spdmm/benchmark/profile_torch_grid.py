"""Profile exactly one warmed call per workload with nsys, for torch.sparse.mm,
pbr_matmul, or boo_matmul (--method).

Run --grid from the GPU job after timing. Each child has one profiler region;
no kernel-name filter or launch limit is used. SQLite traces retain every kernel.
Profiler timings are diagnostic and are not substituted for benchmark timings.

Output files are named {profile}_{n}_f{features}_{method}.json/.sqlite, so all
three methods can be profiled into the same --out directory without collision.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

from benchmark.bench_4way import DEFAULT_PROFILES
from benchmark.bench_spmm import digest_arrays, source_metadata
from benchmark.profile_pbr import build_matrix

METHODS = ('torch', 'pbr', 'boo')


def _build_torch(mat, x):
    csr = torch.sparse_csr_tensor(torch.from_numpy(mat.indptr.copy()),
                                  torch.from_numpy(mat.indices.copy()),
                                  torch.from_numpy(mat.data.copy()), size=mat.shape).cuda()
    return csr, lambda: torch.sparse.mm(csr, x)


def _build_pbr(mat, x):
    import torch_fast_spdmm as spmm
    pbr = spmm.csr_to_pbr(mat, block_rows=4, block_cols=4, min_nnz_per_block=4).to('cuda')
    return pbr, lambda: spmm.pbr_matmul(pbr, x, kernel='block', smem_size=2048, csr_mode=0)


def _build_boo(mat, x):
    from torch_fast_spdmm.boo import csr_to_boo, boo_matmul
    boo = csr_to_boo(mat, cluster='hem', target_nr=32, target_nc=32,
                     small_block_policy='remainder', small_block_threshold=4,
                     nz_order='rowmajor').to('cuda')
    return boo, lambda: boo_matmul(boo, x, threads=256, p=64, row_seg=True, csr_mode=0)


_BUILDERS = {'torch': _build_torch, 'pbr': _build_pbr, 'boo': _build_boo}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--grid', action='store_true')
    p.add_argument('--method', choices=METHODS, default='torch')
    p.add_argument('--methods', nargs='+', choices=METHODS, default=None,
                   help='--grid only: profile multiple methods per workload cell. '
                       'Defaults to just --method.')
    p.add_argument('--sizes', nargs='+', type=int, default=[16384, 65536, 262144])
    p.add_argument('--features', nargs='+', type=int, default=[128, 512])
    p.add_argument('--profiles', nargs='+', default=DEFAULT_PROFILES)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    if a.grid:
        methods = a.methods or [a.method]
        for n in a.sizes:
            for profile in a.profiles:
                for f in a.features:
                    for method in methods:
                        name = f'{profile}_{n}_f{f}_{method}'
                        print(f'PROFILE {name}', flush=True)
                        nsys = os.environ.get('NSYS', 'nsys')
                        command = [nsys, 'profile', '--trace=cuda,nvtx', '--cuda-event-trace=false',
                                   '--sample=none', '--cpuctxsw=none',
                                   '--capture-range=cudaProfilerApi', '--capture-range-end=stop',
                                   '--export=sqlite', '-o', str(a.out / name), sys.executable,
                                   '-m', 'benchmark.profile_torch_grid', '--sizes', str(n),
                                   '--features', str(f), '--profiles', profile,
                                   '--method', method, '--out', str(a.out)]
                        with (a.out / f'{name}.log').open('w') as log:
                            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        return
    if len(a.sizes) != 1 or len(a.features) != 1 or len(a.profiles) != 1:
        p.error('Without --grid, specify exactly one workload')
    n, f, profile = a.sizes[0], a.features[0], a.profiles[0]
    mat = build_matrix(n, profile, seed=0).astype(np.float32).tocsr()
    mat.sum_duplicates()
    mat.sort_indices()
    mat.indptr = mat.indptr.astype(np.int32)
    mat.indices = mat.indices.astype(np.int32)
    x_cpu = torch.randn(mat.shape[1], f, generator=torch.Generator().manual_seed(0))
    x = x_cpu.cuda()
    with torch.inference_mode():
        _obj, call = _BUILDERS[a.method](mat, x)
        for _ in range(5):
            call()
        torch.cuda.synchronize()
        start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        start.record()
        end.record()
        end.synchronize()
        torch.cuda.profiler.start()
        start.record()
        torch.cuda.nvtx.range_push(f'{a.method}_spmm')
        t0 = time.perf_counter()
        y = call()
        host_call_ms = (time.perf_counter() - t0) * 1000
        torch.cuda.nvtx.range_pop()
        end.record()
        end.synchronize()
        profiled_event_ms = start.elapsed_time(end)
        torch.cuda.profiler.stop()
        del y
    metadata = dict(source_metadata(Path(__file__).resolve().parents[3]),
                    profile=profile, requested_size=n, features=f, method=a.method,
                    matrix_sha256=digest_arrays(np.asarray(mat.shape), mat.indptr, mat.indices, mat.data),
                    x_sha256=digest_arrays(x_cpu.numpy()),
                    profiled_event_ms=profiled_event_ms, host_call_ms=host_call_ms,
                    profiled_calls=1, warmup=5,
                    profiler=f'Nsight Systems CUDA trace; cudaProfilerApi capture; one full {a.method} call')
    (a.out / f'{profile}_{n}_f{f}_{a.method}.json').write_text(json.dumps(metadata, indent=2) + '\n')


if __name__ == '__main__':
    main()
