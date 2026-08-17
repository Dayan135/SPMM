"""Full 4-way benchmark: BOO vs PBR block / nnzflat_shared / nnzflat_onthefly.

All four kernels run in ONE build, in ONE process, against the SAME matrix
object, interleaved within a cell. That matters: SHOW_TO_ERAN.md sec. 4/7
measured zero_idle's own unchanged code swinging 29-41% between separate sweep
runs on this cluster -- larger than most of the effects being compared. Timing
them back to back removes build-to-build and job-to-job variance from the
comparison (absolute numbers still drift; ratios within a cell do not).

Each cell reports the median of `--repeats` interleaved rounds, so a transient
on the node perturbs every kernel in the cell rather than one of them.

Usage (from tests/torch_fast_spdmm)::

    python -m benchmark.bench_4way --sizes 65536 262144 --features 512 \
        --out bench_4way.csv
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

from benchmark.profile_pbr import build_matrix, _MATRIX_PROFILES
from torch_fast_spdmm import (
    boo_info, boo_matmul, csr_to_boo, csr_to_pbr, pbr_info, pbr_matmul,
)

PBR_KERNELS = ('block', 'nnzflat_shared', 'nnzflat_onthefly')
ALL_KERNELS = ('boo',) + PBR_KERNELS

# The two matrices SHOW_TO_ERAN sec. 7b found 100% confound-free, the fem2d pair
# that isolates the clustering claim, and a spread of the scattered/pattern
# families. Ordered cheapest-first so a truncated run still covers the headline.
DEFAULT_PROFILES = [
    'fem2d', 'fem2d_permuted',
    'diag_b1.0', 'pattern_corner_cluster',
    'diag_b0.3', 'diag_b0.1',
    'scattered_1x8', 'scattered_8x1', 'scattered_2x2', 'scattered_2x4',
    'pattern_row_skewed', 'pattern_checkerboard', 'pattern_diag_stripe',
    'diag_skewed',
]


def time_ms(fn, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--sizes', type=int, nargs='+', default=[65536, 262144])
    ap.add_argument('--features', type=int, nargs='+', default=[512])
    ap.add_argument('--profiles', nargs='+', default=DEFAULT_PROFILES)
    ap.add_argument('--repeats', type=int, default=3,
                    help='interleaved rounds; the median of these is reported')
    ap.add_argument('--warmup', type=int, default=5)
    ap.add_argument('--iters', type=int, default=20)
    # BOO config
    ap.add_argument('--cluster', default='hem')
    ap.add_argument('--target-nr', type=int, default=32)
    ap.add_argument('--target-nc', type=int, default=32)
    ap.add_argument('--threads', type=int, default=256)
    ap.add_argument('--p', type=int, default=64)
    ap.add_argument('--small-block-policy', default='keep',
                    choices=['keep', 'remainder', 'tail_batch'])
    ap.add_argument('--small-block-threshold', type=int, default=0)
    # PBR config
    ap.add_argument('--block', type=int, default=8)
    ap.add_argument('--min-nnz', type=int, default=2)
    ap.add_argument('--smem-size', type=int, default=2048)
    ap.add_argument('--out', default='bench_4way.csv')
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit('CUDA required')

    rows_out = []
    header = (f"{'matrix':24s} {'N':>7s} {'f':>4s} "
              + ' '.join(f'{k:>18s}' for k in ALL_KERNELS)
              + f" {'best':>18s}")

    for size in args.sizes:
        for profile in args.profiles:
            try:
                mat = build_matrix(size, profile).astype(np.float32)
            except Exception as exc:                       # noqa: BLE001
                print(f'  skip {profile}/{size}: {exc}', flush=True)
                continue

            t0 = time.perf_counter()
            boo = csr_to_boo(mat, cluster=args.cluster,
                             target_nr=args.target_nr, target_nc=args.target_nc,
                             small_block_policy=args.small_block_policy,
                             small_block_threshold=args.small_block_threshold)
            boo_build_s = time.perf_counter() - t0
            pbr = csr_to_pbr(mat, block_rows=args.block, block_cols=args.block,
                             min_nnz_per_block=args.min_nnz)
            binfo = boo_info(boo, p=args.p, threads=args.threads)
            pinfo = pbr_info(pbr)
            boo.to('cuda')
            pbr.to('cuda')

            print(header, flush=True)
            for features in args.features:
                x = torch.randn(mat.shape[1], features, device='cuda')

                runners = {'boo': (lambda: boo_matmul(boo, x, threads=args.threads, p=args.p))}
                for k in PBR_KERNELS:
                    runners[k] = (lambda k=k: pbr_matmul(pbr, x, smem_size=args.smem_size,
                                                         kernel=k))

                # Interleave rounds so a node transient hits every kernel equally.
                samples = {k: [] for k in ALL_KERNELS}
                for _ in range(args.repeats):
                    for k in ALL_KERNELS:
                        samples[k].append(time_ms(runners[k], args.warmup, args.iters))
                med = {k: statistics.median(v) for k, v in samples.items()}

                best = min(med, key=med.get)
                cells = ' '.join(f'{med[k]:>18.4f}' for k in ALL_KERNELS)
                print(f'{profile:24s} {size:>7d} {features:>4d} {cells} {best:>18s}',
                      flush=True)

                record = {
                    'matrix': profile, 'size': size, 'features': features,
                    'best_kernel': best,
                    'boo_cluster': args.cluster, 'boo_target_nr': args.target_nr,
                    'boo_target_nc': args.target_nc, 'boo_threads': args.threads,
                    'boo_p': args.p, 'boo_build_s': round(boo_build_s, 2),
                    'pbr_block': args.block, 'pbr_min_nnz': args.min_nnz,
                    'pbr_smem': args.smem_size,
                    # Reporting rules (BOO_FORMAT.md sec. 10.1): a silent CSR
                    # fallback or an occupancy tax must be visible in the row,
                    # not inferred later.
                    'boo_small_block_policy': args.small_block_policy,
                    'boo_small_block_threshold': args.small_block_threshold,
                    'boo_compressed_fraction': round(binfo.compressed_fraction, 4),
                    'boo_fraction': round(binfo.boo_fraction, 4),
                    'boo_tail_nnz': binfo.tail_nnz,
                    'pbr_compressed_fraction': round(pinfo.compressed_fraction, 4),
                    'boo_occupancy_pct': round(binfo.occupancy_pct, 1),
                    'boo_ctas_per_sm': binfo.ctas_per_sm,
                    'boo_blocks': binfo.num_blocks, 'pbr_blocks': pinfo.num_blocks,
                    'boo_x_loads': binfo.x_loads, 'boo_y_writes': binfo.y_writes,
                    'nnz': int(mat.nnz),
                }
                for k in ALL_KERNELS:
                    record[f'ms_{k}'] = round(med[k], 5)
                    record[f'spread_{k}'] = round(max(samples[k]) - min(samples[k]), 5)
                    record[f'speedup_boo_over_{k}'] = round(med[k] / med['boo'], 4)
                rows_out.append(record)

            if pinfo.compressed_fraction < 0.98:
                # SHOW_TO_ERAN sec. 2: a low PBR compressed_fraction means most of
                # the matrix ran through the shared CSR remainder, so that cell is
                # partly CSR-vs-CSR rather than a real kernel comparison.
                print(f'  !! pbr compressed_fraction={pinfo.compressed_fraction:.3f} '
                      f'-- PBR cells above are partly CSR-fallback, not a clean '
                      f'kernel comparison', flush=True)
            print(flush=True)

    if rows_out:
        out = Path(args.out)
        with out.open('w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows_out[0]))
            w.writeheader()
            w.writerows(rows_out)
        print(f'wrote {len(rows_out)} rows -> {out}', flush=True)

    print('BENCH4WAY_DONE', flush=True)


if __name__ == '__main__':
    main()
