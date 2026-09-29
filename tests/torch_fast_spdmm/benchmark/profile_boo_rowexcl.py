"""One-call Nsight Compute occupancy/stall captures for row-exclusive and BOO."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch
import torch_fast_spdmm as spmm
from benchmark.bench_spmm import digest_arrays, source_metadata
from benchmark.profile_pbr import build_matrix

METRICS = ['gpu__time_duration.sum', 'sm__warps_active.avg.pct_of_peak_sustained_active',
           'smsp__warp_issue_stalled_no_instruction_per_warp_active.pct']


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--grid', action='store_true')
    p.add_argument('--selection', type=Path)
    p.add_argument('--profile', default='fem2d')
    p.add_argument('--size', type=int, default=262144)
    p.add_argument('--features', type=int, default=512)
    p.add_argument('--mode', choices=['rowexcl', 'boo'], default='rowexcl')
    p.add_argument('--k', type=int, default=32)
    p.add_argument('--slab', type=int, default=32)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    if a.grid:
        choices = json.loads(a.selection.read_text())
        for choice in choices:
            name = f"{choice['profile']}_{choice['mode']}"
            command = ['ncu', '--profile-from-start', 'off', '--metrics', ','.join(METRICS),
                       '--cache-control', 'none', '--clock-control', 'none', '--csv',
                       '--log-file', str(a.out/f'{name}.ncu.csv'), sys.executable,
                       '-m', 'benchmark.profile_boo_rowexcl', '--out', str(a.out),
                       '--profile', choice['profile'], '--mode', choice['mode'],
                       '--k', str(choice['k']), '--slab', str(choice.get('slab',32))]
            print('PROFILE', choice, flush=True)
            with (a.out/f'{name}.log').open('w') as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        return
    mat = build_matrix(a.size, a.profile, seed=0).astype(np.float32).tocsr()
    mat.sum_duplicates(); mat.sort_indices()
    mat.indptr = mat.indptr.astype(np.int32); mat.indices = mat.indices.astype(np.int32)
    boo = spmm.csr_to_boo(mat, cluster='hem', target_nr=a.k, target_nc=a.k,
                          seed=0, small_block_policy='remainder', small_block_threshold=4,
                          nz_order='rowmajor', hem_feasible_first=False).to('cuda')
    x_cpu = torch.randn(mat.shape[1], a.features, generator=torch.Generator().manual_seed(0))
    x = x_cpu.cuda()
    extra = {}
    if a.mode == 'rowexcl':
        plan = spmm.prepare_boo_rowexcl(boo, slab=a.slab)
        bound = plan.bind(x)
        call = bound.run
        extra = plan.stats
    else:
        call = lambda: spmm.boo_matmul(boo, x, threads=256, p=64, csr_mode=0,
                                       row_seg=True, launch_mode='legacy')
    with torch.inference_mode():
        for _ in range(5):
            call()
        torch.cuda.synchronize()
        torch.cuda.profiler.start()
        y = call()
        torch.cuda.synchronize()
        torch.cuda.profiler.stop()
    metadata = dict(source_metadata(Path(__file__).resolve().parents[3]),
                    arguments=dict(vars(a), out=str(a.out), selection=str(a.selection)),
                    matrix_sha256=digest_arrays(np.asarray(mat.shape), mat.indptr, mat.indices, mat.data),
                    x_sha256=digest_arrays(x_cpu.numpy()), metrics=METRICS, plan=extra,
                    compressed_fraction=boo.compressed_nnz()/max(mat.nnz,1))
    (a.out/f'{a.profile}_{a.mode}.json').write_text(json.dumps(metadata, indent=2)+'\n')


if __name__ == '__main__':
    main()
