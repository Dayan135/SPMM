"""Single-matmul driver for Nsight Compute profiling of cusparse SpMM.

Mirrors profile_pbr.py but runs torch.sparse.mm (cusparse) so ncu captures
only that matmul inside the cudaProfilerStart/Stop region.

Usage:
    ncu --profile-from-start off --metrics <...> --csv \
        python -m benchmark.profile_cusparse --size 16384 --features 64
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
import torch.cuda.profiler as profiler

from benchmark.matrices import generate_scattered_block_matrix
from test_utils import scipy_csr_to_torch_csr


def build(size: int, features: int):
    mat = generate_scattered_block_matrix(
        system_size=size, block_size=8, profiles=[(8, 1, 0.00001)]
    ).astype(np.float32)
    sparse_torch = scipy_csr_to_torch_csr(mat).to(dtype=torch.float32, device="cuda")
    x = torch.randn((size, features), dtype=torch.float32, device="cuda")
    return sparse_torch, x


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--size", type=int, required=True)
    p.add_argument("--features", type=int, default=64)
    p.add_argument("--warmup", type=int, default=10)
    args = p.parse_args()

    sparse_torch, x = build(args.size, args.features)

    for _ in range(args.warmup):
        torch.sparse.mm(sparse_torch, x)
    torch.cuda.synchronize()

    profiler.start()
    torch.sparse.mm(sparse_torch, x)
    torch.cuda.synchronize()
    profiler.stop()


if __name__ == "__main__":
    main()
