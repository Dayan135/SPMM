"""Single-matmul driver for Nsight Compute profiling of the PBR SpMM kernel.

Builds one scattered_8x1 matrix, converts to PBR, warms up, then runs exactly
one ``pbr_matmul`` inside a cudaProfilerStart/Stop region so ncu (launched with
``--profile-from-start off``) captures only that matmul's kernels.

Usage (under ncu):
    ncu --profile-from-start off --metrics <...> --csv \
        python -m benchmark.profile_pbr --size 16384 --features 64 \
            --block 2 --min-nnz 2
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
import torch.cuda.profiler as profiler

from benchmark.matrices import generate_scattered_block_matrix
from torch_fast_spdmm import csr_to_pbr, pbr_matmul


def build(size: int, features: int, block: int, min_nnz: int):
    mat = generate_scattered_block_matrix(
        system_size=size, block_size=8, profiles=[(8, 1, 0.00001)]
    ).astype(np.float32)
    pbr = csr_to_pbr(mat, block_rows=block, block_cols=block,
                     min_nnz_per_block=min_nnz).to("cuda")
    x = torch.randn((size, features), dtype=torch.float32, device="cuda")
    return pbr, x


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--size", type=int, required=True)
    p.add_argument("--features", type=int, default=64)
    p.add_argument("--block", type=int, default=2)
    p.add_argument("--min-nnz", type=int, default=2)
    p.add_argument("--warmup", type=int, default=10)
    args = p.parse_args()

    pbr, x = build(args.size, args.features, args.block, args.min_nnz)

    for _ in range(args.warmup):
        pbr_matmul(pbr, x)
    torch.cuda.synchronize()

    profiler.start()
    pbr_matmul(pbr, x)
    torch.cuda.synchronize()
    profiler.stop()


if __name__ == "__main__":
    main()
