"""Single-matmul driver for Nsight Compute profiling of the BOO SpMM kernel.

Mirrors ``profile_pbr.py`` and reuses its ``build_matrix`` / ``_MATRIX_PROFILES``
so both drivers profile the identical matrix.  With ``--kernel pbr`` it profiles
the PBR block kernel on the same matrix instead, which is how a BOO-vs-PBR cell
is made comparable.

Usage (under ncu)::

    ncu --profile-from-start off --metrics <...> --csv \
        python -m benchmark.profile_boo --size 262144 --features 512 \
            --matrix-profile fem2d_permuted --cluster graclus --target 32

Prints ``boo_info`` before the profiled region.  Per BOO_FORMAT.md sec. 10.1 the
reporting rules are non-negotiable: ``compressed_fraction`` (1.0 by construction
under the `keep` policy -- say so rather than letting it look suspicious) and
``ctas_per_sm`` / ``occupancy_pct`` next to any timing, so an occupancy tax is
visible immediately rather than inferred three experiments later.
"""

from __future__ import annotations

import argparse
import dataclasses
import time

import torch
import torch.cuda.profiler as profiler

from benchmark.profile_pbr import build_matrix, _MATRIX_PROFILES
from torch_fast_spdmm import (
    boo_info, boo_matmul, csr_to_boo, csr_to_pbr, pbr_info, pbr_matmul,
)


def _time_ms(fn, warmup: int = 5, iters: int = 20) -> float:
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--size', type=int, required=True)
    parser.add_argument('--features', type=int, required=True)
    parser.add_argument('--matrix-profile', required=True, choices=sorted(_MATRIX_PROFILES))
    parser.add_argument('--kernel', default='boo', choices=['boo', 'pbr'])
    # BOO knobs
    parser.add_argument('--cluster', default='identity')
    parser.add_argument('--target-nr', type=int, default=32)
    parser.add_argument('--target-nc', type=int, default=32)
    parser.add_argument('--threads', type=int, default=128)
    parser.add_argument('--p', type=int, default=8)
    parser.add_argument('--nz-order', default='rowmajor', choices=['rowmajor', 'colmajor'])
    parser.add_argument('--small-block-policy', default='keep',
                        choices=['keep', 'merge', 'remainder'])
    parser.add_argument('--small-block-threshold', type=int, default=0)
    # PBR knobs (for --kernel pbr)
    parser.add_argument('--block', type=int, default=8)
    parser.add_argument('--min-nnz', type=int, default=2)
    parser.add_argument('--smem-size', type=int, default=2048)
    parser.add_argument('--time', action='store_true',
                        help='also report a CUDA-event wall time (skip under ncu)')
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit('CUDA required')

    build_start = time.perf_counter()
    mat = build_matrix(args.size, args.matrix_profile)
    print(f'[matrix] {args.matrix_profile} n={args.size} nnz={mat.nnz} '
          f'({time.perf_counter() - build_start:.1f}s)')

    x = torch.randn(mat.shape[1], args.features, device='cuda')

    if args.kernel == 'boo':
        convert_start = time.perf_counter()
        boo = csr_to_boo(mat, cluster=args.cluster,
                         target_nr=args.target_nr, target_nc=args.target_nc,
                         small_block_policy=args.small_block_policy,
                         small_block_threshold=args.small_block_threshold,
                         nz_order=args.nz_order)
        convert_s = time.perf_counter() - convert_start
        info = boo_info(boo, p=args.p, threads=args.threads)
        for key, value in dataclasses.asdict(info).items():
            print(f'  {key:24s} {value}')
        print(f'  {"preprocess_s":24s} {convert_s:.1f}')
        boo.to('cuda')
        run = lambda: boo_matmul(boo, x, threads=args.threads, p=args.p)  # noqa: E731
    else:
        pbr = csr_to_pbr(mat, block_rows=args.block, block_cols=args.block,
                         min_nnz_per_block=args.min_nnz)
        for key, value in dataclasses.asdict(pbr_info(pbr)).items():
            print(f'  {key:24s} {value}')
        pbr.to('cuda')
        run = lambda: pbr_matmul(pbr, x, smem_size=args.smem_size)  # noqa: E731

    run()
    torch.cuda.synchronize()

    if args.time:
        print(f'\n[time_ms] {_time_ms(run):.4f}')

    profiler.start()
    run()
    torch.cuda.synchronize()
    profiler.stop()


if __name__ == '__main__':
    main()
