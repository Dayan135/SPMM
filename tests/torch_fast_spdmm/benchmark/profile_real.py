"""Profile exactly one SpMM call on a real matrix with saved labels (ncu / nsys).

Uses the canonical CSR and labels produced by benchmark.real_matrices /
benchmark.bench_real (both hash-validated), so the profiled workload is the one
bench_real timed. The call is validated against the float64 reference, warmed
up, then bracketed by cudaProfilerStart/Stop -- run under
`ncu --profile-from-start off` or `nsys --capture-range=cudaProfilerApi`.

  python -m benchmark.profile_real --matrix pwtk --labeling identity_k32 \
      --features 512 --method boo --labels L --out DIR
  ... --method direct --direct-algorithm DEFAULT         # cuSPARSE, original order
  ... --method direct_perm --direct-algorithm DEFAULT    # cuSPARSE on A[pr][:, pc], steady state
  ... --structure                                        # no GPU call; write block statistics only

Profiled durations diagnose behaviour; bench_real's unprofiled timings decide performance.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

import numpy as np
import torch

from benchmark import bench_real as br
from benchmark import real_matrices as rm


def load_case(a):
    import torch_fast_spdmm as spmm
    spec, = rm.resolve([a.matrix])
    csr, record = rm.load(spec, a.cache, a.explicit_zeros)
    method, k = a.labeling.rsplit('_k', 1)
    rows, cols, provenance = br.load_labels(
        br.label_path(a.labels, spec, a.explicit_zeros, method, int(k)),
        matrix=spec.key, matrix_sha256=record['matrix_sha256'], shape=csr.shape,
        explicit_zeros=a.explicit_zeros, method=method, k=int(k), seed=a.seed)
    boo = spmm.csr_to_boo(csr, row_labels=rows, col_labels=cols, small_block_policy='remainder',
                          small_block_threshold=a.threshold, nz_order='rowmajor')
    pr, pc = br.permutation_from_labels(rows), br.permutation_from_labels(cols)
    np.testing.assert_array_equal(np.asarray(boo.row_ids), pr)
    np.testing.assert_array_equal(np.asarray(boo.col_ids), pc)
    details = dict(matrix=spec.key, labeling=a.labeling, features=a.features, method=a.method,
                   matrix_sha256=record['matrix_sha256'], label_sha256=provenance['label_sha256'],
                   threshold=a.threshold, seed=a.seed)
    return csr, boo, pr, pc, details


def structure(a):
    from benchmark.panel_structure import boo_structure, csr_structure
    csr, boo, _, _, details = load_case(a)
    start = time.perf_counter()
    details.update(csr=csr_structure(csr), boo=boo_structure(boo))
    details['structure_s'] = time.perf_counter() - start
    return details


def profile(a):
    import torch_fast_spdmm as spmm
    import torch_fast_spdmm_core as core
    from benchmark.bench_clustering import feature_tile
    csr, boo, pr, pc, details = load_case(a)
    x_cpu = torch.randn(csr.shape[1], a.features, generator=torch.Generator().manual_seed(a.seed))
    details['x_sha256'] = rm.digest_arrays(x_cpu.numpy())
    x = x_cpu.cuda()
    row_perm = None
    if a.method == 'boo':
        limit = torch.cuda.get_device_properties(0).shared_memory_per_block_optin
        p, arena, fits = feature_tile(boo, a.features, limit)
        if not fits:
            raise RuntimeError(f'BOO arena {arena} B exceeds shared-memory limit {limit} B')
        boo.to('cuda')
        details.update(p=p, arena_bytes=arena, threads=a.threads, launch_mode='auto',
                       boo_info=asdict(spmm.boo_info(boo, p=p, threads=a.threads)))
        call = lambda: spmm.boo_matmul(boo, x, threads=a.threads, p=p, row_seg=True,
                                       csr_mode=0, launch_mode='auto')
    else:
        target = csr
        if a.method == 'direct_perm':
            target, row_perm = br.reorder(csr, pr, pc), pr
            x = x.index_select(0, torch.from_numpy(pc).cuda())  # resident reordered X
        t = br._to_torch_csr(torch, target)
        if a.method == 'torch':
            call = lambda: torch.sparse.mm(t, x)
        else:
            plan = core.DirectSpMM(t.crow_indices(), t.col_indices(), t.values(), x, a.direct_algorithm)
            details.update(direct_algorithm=a.direct_algorithm, direct_info=dict(plan.info()))
            call = plan.run
    with torch.inference_mode():
        check = br.Reference(csr, x_cpu.numpy()).check(call(), row_perm)
        if check['bound_violations']:
            raise AssertionError(f'validation failed: {check}')
        details.update(check, correctness='passed', warmup=a.warmup, profiled_calls=1)
        for _ in range(a.warmup):
            call()
        torch.cuda.synchronize()
        torch.cuda.profiler.start()
        torch.cuda.nvtx.range_push(f'{details["matrix"]}:{a.method}')
        y = call()
        torch.cuda.synchronize()
        torch.cuda.nvtx.range_pop()
        torch.cuda.profiler.stop()
        del y
    return details


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--matrix', required=True)
    p.add_argument('--labeling', required=True, help='e.g. identity_k32, hem_k64')
    p.add_argument('--features', type=int, default=512)
    p.add_argument('--method', choices=['boo', 'direct', 'direct_perm', 'torch'], default='boo')
    p.add_argument('--direct-algorithm', choices=br.DIRECT_PRIORITY, default='DEFAULT')
    p.add_argument('--structure', action='store_true', help='CPU block statistics only, no GPU call')
    p.add_argument('--cache', type=Path, default=rm.default_cache())
    p.add_argument('--labels', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--explicit-zeros', choices=rm.EXPLICIT_ZERO_POLICIES, default='keep')
    p.add_argument('--threads', type=int, choices=[128, 256, 512], default=256)
    p.add_argument('--threshold', type=int, default=4)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--warmup', type=int, default=5)
    a = p.parse_args(argv)
    if not a.structure and not torch.cuda.is_available():
        raise SystemExit('CUDA required (or pass --structure)')
    a.out.mkdir(parents=True, exist_ok=True)
    details = structure(a) if a.structure else profile(a)
    name = 'structure.json' if a.structure else 'profile.json'
    rm._write_json(a.out / name, details)
    print(f'PROFILE_REAL_DONE {a.out / name}', flush=True)


if __name__ == '__main__':
    main()
