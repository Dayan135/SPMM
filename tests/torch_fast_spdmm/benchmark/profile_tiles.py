"""Profile one explicit BOO (P, FPT, slab) variant, or a baseline, for one case.

Validates against the float64 reference, warms up, then brackets exactly one
call with cudaProfilerStart/Stop (run under `ncu --profile-from-start off` or
`nsys --capture-range=cudaProfilerApi`). FPT/slab are applied immediately
before every call; profile.json records the launcher's own report of the
selected specialization, which the trace's kernel name must match.

  python -m benchmark.profile_tiles --case pwtk --features 512 --method boo \
      --p 64 --fpt 2 --slab 512 --labels L --out DIR
  ... --method direct --direct-algorithm CSR_ALG2      # original-order cuSPARSE
  ... --structure                                       # CPU statistics only, no GPU call

Profiler durations diagnose; bench_tiles' unprofiled timings decide performance.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import time

import numpy as np

from benchmark import bench_real as br
from benchmark import real_matrices as rm
from benchmark import tiles_common as tc


def structure(a):
    from benchmark.panel_structure import boo_structure, csr_structure
    csr, boo, info = tc.load_case(tc.CASES[a.case], a.cache, a.labels, threshold=a.threshold, seed=a.seed)
    start = time.perf_counter()
    s = boo_structure(boo)
    nr = np.diff(np.asarray(boo.row_grp_ptr))[np.asarray(boo.blk_row_grp)].astype(np.int64)
    nc = np.diff(np.asarray(boo.col_grp_ptr))[np.asarray(boo.blk_col_grp)].astype(np.int64)
    nz = np.diff(np.asarray(boo.blk_nz_ptr)).astype(np.int64)
    warp = {f'p{p}_fpt{f}': tc.warp_row_efficiency(boo, p // f)
            for p in (16, 32, 64, 128) for f in (1, 2, 4)}
    # Per-call logical shared-memory traffic models (bytes, float32), F given:
    # X staging covers n_c x P per tile, padded to whole P tiles inside each slab;
    # sparse values/indices/ids/row table are re-staged once per slab (grid.z).
    models = {}
    for slab in (128, 256, 512, 1024):
        for p in (16, 32, 64, 128):
            if slab % p:
                continue
            for features in a.model_features:
                slabs = -(-features // slab)
                tiles = sum(-(-min(slab, features - z * slab) // p) for z in range(slabs))
                models[f'F{features}_s{slab}_p{p}'] = dict(
                    grid_z=slabs, tiles_per_block=tiles,
                    x_staged_bytes=int(4 * p * tiles * nc.sum()),
                    x_staged_active_bytes=int(4 * p * tiles * s['active_x_rows']),
                    sparse_restaged_bytes=int(slabs * (6 * nz.sum() + 4 * (2 * nr.sum() + nc.sum() + len(nr)))),
                    barriers_per_block=int(2 + 2 * tiles))
    info.update(csr=csr_structure(csr), boo=s, warp_row_efficiency=warp, traffic_models=models,
                block_rows=dict(mean=float(nr.mean()), max=int(nr.max())) if len(nr) else {},
                structure_s=time.perf_counter() - start)
    return info


def profile(a):
    import torch
    import torch_fast_spdmm as spmm
    import torch_fast_spdmm_core as core
    csr, boo, details = tc.load_case(tc.CASES[a.case], a.cache, a.labels, threshold=a.threshold,
                                     seed=a.seed)
    x_cpu = torch.randn(csr.shape[1], a.features, generator=torch.Generator().manual_seed(a.seed))
    details.update(features=a.features, method=a.method, x_sha256=rm.digest_arrays(x_cpu.numpy()),
                   warmup=a.warmup, profiled_calls=1)
    x = x_cpu.cuda()
    if a.method == 'boo':
        limit = torch.cuda.get_device_properties(0).shared_memory_per_block_optin
        boo.to('cuda')
        launch, reason = tc.launch_info(core, boo, a.features, a.threads, a.p, a.fpt, a.slab, limit)
        if reason:
            raise RuntimeError(f'unsupported variant: {reason}')
        details.update(launch=launch, threads=a.threads, p=a.p, features_per_thread=a.fpt,
                       feat_block=a.slab, launch_mode='auto',
                       expected_kernel=('boo_spmm_rowseg_kernel<float, {}, {}, false, {}, {}>'
                                        .format(*tc.expected_kernel(launch))))

        def call():
            tc.apply_settings(core, a.fpt, a.slab)
            return spmm.boo_matmul(boo, x, threads=a.threads, p=a.p, row_seg=True, csr_mode=0,
                                   launch_mode='auto', features_per_thread=a.fpt)
    else:
        t = br._to_torch_csr(torch, csr)
        if a.method == 'torch':
            call = lambda: torch.sparse.mm(t, x)
        else:
            plan = core.DirectSpMM(t.crow_indices(), t.col_indices(), t.values(), x, a.direct_algorithm)
            details.update(direct_algorithm=a.direct_algorithm, direct_info=dict(plan.info()))
            call = plan.run
    with torch.inference_mode():
        check = br.Reference(csr, x_cpu.numpy()).check(call())
        if check['bound_violations']:
            raise AssertionError(f'validation failed: {check}')
        details.update(check, correctness='passed')
        for _ in range(a.warmup):
            call()
        torch.cuda.synchronize()
        torch.cuda.profiler.start()
        torch.cuda.nvtx.range_push(f'{a.case}:{a.method}')
        y = call()
        torch.cuda.synchronize()
        torch.cuda.nvtx.range_pop()
        torch.cuda.profiler.stop()
        del y
    if a.method == 'boo':
        details['settings_after'] = dict(fpt=core.get_boo_rowseg_fpt(), feat_block=core.get_boo_feat_block())
        tc.apply_settings(core, 1, tc.DEFAULT_SLAB)
    return details


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--case', required=True, choices=sorted(tc.CASES))
    p.add_argument('--features', type=int, default=512)
    p.add_argument('--method', choices=['boo', 'direct', 'torch'], default='boo')
    p.add_argument('--p', type=int, default=64, choices=[8, 16, 32, 64, 128])
    p.add_argument('--fpt', type=int, default=2, choices=[1, 2, 4])
    p.add_argument('--slab', type=int, default=512, choices=[128, 256, 512, 1024])
    p.add_argument('--direct-algorithm', choices=br.DIRECT_PRIORITY, default='DEFAULT')
    p.add_argument('--structure', action='store_true', help='CPU statistics only, no GPU call')
    p.add_argument('--model-features', nargs='+', type=int, default=[128, 512, 1024])
    p.add_argument('--cache', type=Path, default=rm.default_cache())
    p.add_argument('--labels', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--threads', type=int, choices=[128, 256, 512], default=256)
    p.add_argument('--threshold', type=int, default=4)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--warmup', type=int, default=5)
    a = p.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    details = structure(a) if a.structure else profile(a)
    name = 'structure.json' if a.structure else 'profile.json'
    rm._write_json(a.out / name, details)
    print(f'PROFILE_TILES_DONE {a.out / name}', flush=True)


if __name__ == '__main__':
    main()
