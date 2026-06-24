"""Single-matmul driver for Nsight Compute profiling of the PBR SpMM kernels.

Builds one matrix, converts to PBR, prints pbr_info structural stats, warms
up, then runs exactly one ``pbr_matmul`` inside a cudaProfilerStart/Stop
region so ncu (launched with ``--profile-from-start off``) captures only that
matmul's kernels.

Matrix profiles (--matrix-profile):
  scattered_8x1    — 8-row × 1-col scattered blocks
  scattered_1x8    — 1-row × 8-col scattered blocks
  scattered_2x4    — 2-row × 4-col scattered blocks
  scattered_dense2 — dense 2×2 block scatter (block_size=2)

Kernels (--kernel):
  block    — original one-CUDA-block-per-PBR-block (default)
  rowgroup — row-group fusion kernel

Usage (under ncu):
    ncu --profile-from-start off --metrics <...> --csv \
        python -m benchmark.profile_pbr --size 262144 --features 512 \
            --block 2 --min-nnz 2 --smem-size 2048 \
            --matrix-profile scattered_dense2 --kernel rowgroup
"""

from __future__ import annotations

import argparse
import dataclasses

import numpy as np
import torch
import torch.cuda.profiler as profiler

import torch_fast_spdmm_core as _core
from benchmark.matrices import generate_scattered_block_matrix
from torch_fast_spdmm import csr_to_pbr, pbr_info, pbr_matmul

_MATRIX_PROFILES = {
    "scattered_8x1":    {"block_size": 8, "rows_per_block": 8, "cols_per_block": 1},
    "scattered_1x8":    {"block_size": 8, "rows_per_block": 1, "cols_per_block": 8},
    "scattered_2x4":    {"block_size": 8, "rows_per_block": 2, "cols_per_block": 4},
    "scattered_dense2": {"block_size": 2, "rows_per_block": 2, "cols_per_block": 2},
}


def build(size: int, features: int, block: int, min_nnz: int,
          matrix_profile: str, smem_size: int, kernel: str):
    cfg = _MATRIX_PROFILES[matrix_profile]
    mat = generate_scattered_block_matrix(
        system_size=size,
        block_size=cfg["block_size"],
        profiles=[(cfg["rows_per_block"], cfg["cols_per_block"], 0.00001)],
    ).astype(np.float32)

    pbr_mat = csr_to_pbr(mat, block_rows=block, block_cols=block,
                         min_nnz_per_block=min_nnz)

    # Print structural stats before transferring to GPU
    info = pbr_info(pbr_mat)
    print("[pbr_info]")
    for field in dataclasses.fields(info):
        print(f"  {field.name:26s} = {getattr(info, field.name)}")

    sm_count = torch.cuda.get_device_properties(0).multi_processor_count
    print(f"  {'sm_count':26s} = {sm_count}")
    print(f"  {'kernel':26s} = {kernel}")
    print(f"  {'smem_size':26s} = {smem_size}")
    print(f"  {'features':26s} = {features}")
    print()

    pbr_mat = pbr_mat.to("cuda")

    if kernel == "rowgroup":
        bps = _core.pbr_rowgroup_blocks_per_sm(block, smem_size)
        num_groups = int(pbr_mat._gpu_meta.group_row_starts.shape[0]) - 1
        print(f"  {'blocks_per_sm':26s} = {bps}")
        print(f"  {'target_groups':26s} = {sm_count * bps}")
        print(f"  {'num_groups':26s} = {num_groups}")
        print()

    x = torch.randn((size, features), dtype=torch.float32, device="cuda")
    return pbr_mat, x


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--size",           type=int, required=True)
    p.add_argument("--features",       type=int, default=128)
    p.add_argument("--block",          type=int, default=8)
    p.add_argument("--min-nnz",        type=int, default=2)
    p.add_argument("--smem-size",      type=int, default=2048,
                   help="shared-memory tile size in floats (512/1024/2048)")
    p.add_argument("--matrix-profile", type=str, default="scattered_8x1",
                   choices=list(_MATRIX_PROFILES))
    p.add_argument("--kernel",         type=str, default="block",
                   choices=["block", "rowgroup"],
                   help="which PBR kernel to profile")
    p.add_argument("--warmup",         type=int, default=10)
    args = p.parse_args()

    pbr_mat, x = build(
        args.size, args.features, args.block, args.min_nnz,
        args.matrix_profile, args.smem_size, args.kernel,
    )

    for _ in range(args.warmup):
        pbr_matmul(pbr_mat, x, smem_size=args.smem_size, kernel=args.kernel)
    torch.cuda.synchronize()

    profiler.start()
    pbr_matmul(pbr_mat, x, smem_size=args.smem_size, kernel=args.kernel)
    torch.cuda.synchronize()
    profiler.stop()


if __name__ == "__main__":
    main()
