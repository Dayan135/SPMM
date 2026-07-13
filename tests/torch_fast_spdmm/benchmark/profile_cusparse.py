"""Single-matmul driver for Nsight Compute profiling of cusparse SpMM.

Mirrors profile_pbr.py but runs torch.sparse.mm (cusparse) so ncu captures
only that matmul inside the cudaProfilerStart/Stop region. Shares
build_matrix() / _MATRIX_PROFILES with profile_pbr.py so the same
--matrix-profile name yields the identical matrix in both drivers --
required for an apples-to-apples PBR-vs-cusparse comparison at a given size.

Usage:
    ncu --profile-from-start off --metrics <...> --csv \
        python -m benchmark.profile_cusparse --size 16384 --features 64 \
            --matrix-profile diag_b0.3
"""

from __future__ import annotations

import argparse

import torch
import torch.cuda.profiler as profiler

from benchmark.profile_pbr import _MATRIX_PROFILES, build_matrix
from test_utils import scipy_csr_to_torch_csr


def build(size: int, features: int, matrix_profile: str):
    mat = build_matrix(size, matrix_profile)
    sparse_torch = scipy_csr_to_torch_csr(mat).to(dtype=torch.float32, device="cuda")
    x = torch.randn((size, features), dtype=torch.float32, device="cuda")
    return sparse_torch, x


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--size", type=int, required=True)
    p.add_argument("--features", type=int, default=64)
    p.add_argument("--matrix-profile", type=str, default="scattered_8x1",
                   choices=list(_MATRIX_PROFILES))
    p.add_argument("--warmup", type=int, default=10)
    args = p.parse_args()

    sparse_torch, x = build(args.size, args.features, args.matrix_profile)

    for _ in range(args.warmup):
        torch.sparse.mm(sparse_torch, x)
    torch.cuda.synchronize()

    profiler.start()
    torch.sparse.mm(sparse_torch, x)
    torch.cuda.synchronize()
    profiler.stop()


if __name__ == "__main__":
    main()
