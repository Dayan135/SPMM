"""Profile exactly one warmed torch.sparse.mm call per workload with nsys.

Run --grid from the GPU job after timing. Each child has one profiler region;
no kernel-name filter or launch limit is used. SQLite traces retain every kernel.
Profiler timings are diagnostic and are not substituted for benchmark timings.
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


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--grid', action='store_true')
    p.add_argument('--sizes', nargs='+', type=int, default=[16384, 65536, 262144])
    p.add_argument('--features', nargs='+', type=int, default=[128, 512])
    p.add_argument('--profiles', nargs='+', default=DEFAULT_PROFILES)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    if a.grid:
        for n in a.sizes:
            for profile in a.profiles:
                for f in a.features:
                    name = f'{profile}_{n}_f{f}'
                    print(f'PROFILE {name}', flush=True)
                    nsys = os.environ.get('NSYS', 'nsys')
                    command = [nsys, 'profile', '--trace=cuda,nvtx', '--cuda-event-trace=false', '--sample=none',
                               '--cpuctxsw=none', '--capture-range=cudaProfilerApi',
                               '--capture-range-end=stop', '--export=sqlite', '-o',
                               str(a.out / name), sys.executable,
                               '-m', 'benchmark.profile_torch_grid', '--sizes', str(n),
                               '--features', str(f), '--profiles', profile, '--out', str(a.out)]
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
    csr = torch.sparse_csr_tensor(torch.from_numpy(mat.indptr.copy()),
                                 torch.from_numpy(mat.indices.copy()),
                                 torch.from_numpy(mat.data.copy()), size=mat.shape).cuda()
    x_cpu = torch.randn(mat.shape[1], f, generator=torch.Generator().manual_seed(0))
    x = x_cpu.cuda()
    with torch.inference_mode():
        for _ in range(5):
            torch.sparse.mm(csr, x)
        torch.cuda.synchronize()
        start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        start.record()
        end.record()
        end.synchronize()
        torch.cuda.profiler.start()
        start.record()
        torch.cuda.nvtx.range_push('torch_sparse_mm')
        t0 = time.perf_counter()
        y = torch.sparse.mm(csr, x)
        host_call_ms = (time.perf_counter() - t0) * 1000
        torch.cuda.nvtx.range_pop()
        end.record()
        end.synchronize()
        profiled_event_ms = start.elapsed_time(end)
        torch.cuda.profiler.stop()
    metadata = dict(source_metadata(Path(__file__).resolve().parents[3]),
                    profile=profile, requested_size=n, features=f,
                    matrix_sha256=digest_arrays(np.asarray(mat.shape), mat.indptr, mat.indices, mat.data),
                    x_sha256=digest_arrays(x_cpu.numpy()),
                    profiled_event_ms=profiled_event_ms, host_call_ms=host_call_ms,
                    profiled_calls=1, warmup=5,
                    profiler='Nsight Systems CUDA trace; cudaProfilerApi capture; one full call')
    (a.out / f'{profile}_{n}_f{f}.json').write_text(json.dumps(metadata, indent=2) + '\n')


if __name__ == '__main__':
    main()
