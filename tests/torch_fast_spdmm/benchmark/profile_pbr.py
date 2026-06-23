"""Single-matmul driver for Nsight Compute profiling of the PBR SpMM kernel.

Builds one scattered matrix, converts to PBR, warms up, then runs exactly
one ``pbr_matmul`` inside a cudaProfilerStart/Stop region so ncu (launched
with ``--profile-from-start off``) captures only that matmul's kernels.

Matrix profiles (--matrix-profile):
  scattered_8x1  — 8-row × 1-col scattered blocks  (default)
  scattered_1x8  — 1-row × 8-col scattered blocks
  scattered_2x4  — 2-row × 4-col scattered blocks

Usage (under ncu):
    ncu --profile-from-start off --metrics <...> --csv \
        python -m benchmark.profile_pbr --size 16384 --features 128 \
            --block 8 --min-nnz 2 --smem-size 2048 --matrix-profile scattered_1x8
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
import torch.cuda.profiler as profiler

from benchmark.matrices import generate_scattered_block_matrix
from torch_fast_spdmm import csr_to_pbr, pbr_matmul

_MATRIX_PROFILES = {
    "scattered_8x1": (8, 1),
    "scattered_1x8": (1, 8),
    "scattered_2x4": (2, 4),
}


def build(size: int, features: int, block: int, min_nnz: int,
          matrix_profile: str, smem_size: int):
    rows_per_block, cols_per_block = _MATRIX_PROFILES[matrix_profile]
    mat = generate_scattered_block_matrix(
        system_size=size, block_size=8,
        profiles=[(rows_per_block, cols_per_block, 0.00001)],
    ).astype(np.float32)
    pbr = csr_to_pbr(mat, block_rows=block, block_cols=block,
                     min_nnz_per_block=min_nnz).to("cuda")
    x = torch.randn((size, features), dtype=torch.float32, device="cuda")
    return pbr, x, smem_size


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--size",           type=int,  required=True)
    p.add_argument("--features",       type=int,  default=128)
    p.add_argument("--block",          type=int,  default=8)
    p.add_argument("--min-nnz",        type=int,  default=2)
    p.add_argument("--smem-size",      type=int,  default=2048,
                   help="shared-memory tile size in floats (512/1024/2048)")
    p.add_argument("--matrix-profile", type=str,  default="scattered_8x1",
                   choices=list(_MATRIX_PROFILES))
    p.add_argument("--warmup",         type=int,  default=10)
    args = p.parse_args()

    pbr, x, smem_size = build(
        args.size, args.features, args.block, args.min_nnz,
        args.matrix_profile, args.smem_size,
    )

    for _ in range(args.warmup):
        pbr_matmul(pbr, x, smem_size=smem_size)
    torch.cuda.synchronize()

    profiler.start()
    pbr_matmul(pbr, x, smem_size=smem_size)
    torch.cuda.synchronize()
    profiler.stop()


if __name__ == "__main__":
    main()
