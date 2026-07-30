"""Single-matmul driver for Nsight Compute profiling of the PBR SpMM kernel.

Builds one matrix, converts to PBR, prints pbr_info structural stats, warms
up, then runs exactly one ``pbr_matmul`` inside a cudaProfilerStart/Stop
region so ncu (launched with ``--profile-from-start off``) captures only that
matmul's kernels.

Matrix profiles (--matrix-profile):
  scattered_8x1    — 8-row × 1-col scattered blocks
  scattered_1x8    — 1-row × 8-col scattered blocks
  scattered_2x4    — 2-row × 4-col scattered blocks
  scattered_2x2    — 2-row × 2-col scattered blocks
  scattered_dense2 — dense 2×2 block scatter (block_size=2)
  diag_dd0.017     — diagonal blocks + noise (64×64 blocks, diag_density=0.017)
  diag_dd0.033     — diagonal blocks + noise (64×64 blocks, diag_density=0.033)
  diag_dd0.05      — diagonal blocks + noise (64×64 blocks, diag_density=0.05)
  diag_b0.1        — same as diag_b0.3 but diag_density=0.1 (sparser band)
  diag_b0.3        — diag_b0.3 from test_benchmark.py (64-wide band, no off-diag noise)
  diag_b1.0        — diag_b1.0 from test_benchmark.py (fully dense 64-wide band, no off-diag noise)
  diag_skewed      — fully-dense diagonal band + off-diagonal noise, deliberately
                     bimodal per-block nnz
  fem2d            — 2D FEM stiffness matrix (grid-aligned reuse)
  fem2d_permuted   — same, with mesh nodes relabeled (breaks grid-aligned reuse)
  pattern_diag_stripe    — fixed 8x8 diagonal-stripe pattern tiled across the grid
  pattern_corner_cluster — fixed 8x8 dense 4x4-corner pattern tiled across the grid
  pattern_checkerboard   — fixed 8x8 checkerboard pattern tiled across the grid
  pattern_row_skewed     — fixed 8x8 pattern with one fully-dense row + one nonzero
                           per remaining row -- maximal per-row density skew, the
                           discriminating case for intra-block work-balance experiments

Usage (under ncu):
    ncu --profile-from-start off --metrics <...> --csv \
        python -m benchmark.profile_pbr --size 262144 --features 512 \
            --block 8 --min-nnz 2 --smem-size 2048 \
            --matrix-profile diag_dd0.001
"""

from __future__ import annotations

import argparse
import dataclasses

import numpy as np
import torch
import torch.cuda.profiler as profiler

from benchmark.matrices import (generate_scattered_block_matrix, generate_diag_blocks_plus_noise,
                                generate_fem_2d, generate_permuted_fem_2d,
                                generate_patterned_block_matrix, PBR_PATTERN_PRESETS)
from torch_fast_spdmm import csr_to_pbr, pbr_info, pbr_matmul

# Each entry: (kind, cfg)
#   kind="scattered" → generate_scattered_block_matrix
#   kind="diag"      → generate_diag_blocks_plus_noise
#   kind="pattern"   → generate_patterned_block_matrix (fixed 8x8 mask, see PBR_PATTERN_PRESETS)
_MATRIX_PROFILES = {
    "scattered_8x1":    ("scattered", {"block_size": 8, "rows_per_block": 8, "cols_per_block": 1}),
    "scattered_1x8":    ("scattered", {"block_size": 8, "rows_per_block": 1, "cols_per_block": 8}),
    "scattered_2x4":    ("scattered", {"block_size": 8, "rows_per_block": 2, "cols_per_block": 4}),
    "scattered_2x2":    ("scattered", {"block_size": 8, "rows_per_block": 2, "cols_per_block": 2}),
    "scattered_dense2": ("scattered", {"block_size": 2, "rows_per_block": 2, "cols_per_block": 2}),
    "diag_dd0.017":     ("diag",      {"block_size": 64, "diag_density": 0.017, "off_diag_density": 0.0000002}),
    "diag_dd0.033":     ("diag",      {"block_size": 64, "diag_density": 0.033, "off_diag_density": 0.0000002}),
    "diag_dd0.05":      ("diag",      {"block_size": 64, "diag_density": 0.05,  "off_diag_density": 0.0000002}),
    "diag_b0.1":        ("diag",      {"block_size": 64, "diag_density": 0.1,   "off_diag_density": 0.0}),
    "diag_b0.3":        ("diag",      {"block_size": 64, "diag_density": 0.3,   "off_diag_density": 0.0}),
    "diag_b1.0":        ("diag",      {"block_size": 64, "diag_density": 1.0,   "off_diag_density": 0.0}),
    "diag_skewed":      ("diag",      {"block_size": 64, "diag_density": 1.0,   "off_diag_density": 0.00002}),
    "fem2d":            ("fem",       {"permuted": False}),
    "fem2d_permuted":   ("fem",       {"permuted": True}),
    "pattern_diag_stripe":    ("pattern", {"preset": "diag_stripe"}),
    "pattern_corner_cluster": ("pattern", {"preset": "corner_cluster"}),
    "pattern_checkerboard":   ("pattern", {"preset": "checkerboard"}),
    "pattern_row_skewed":     ("pattern", {"preset": "row_skewed"}),
}


# Scattered "density" anchored at N=1024 and scaled by 1024/size (same convention
# as test_benchmark.py's _sd) so a fixed hardcoded density doesn't produce zero
# blocks at small N while still staying sparse (not O(N^2) nnz) at large N.
_SCATTERED_DENSITY_AT_1024 = 0.005


def build_matrix(size: int, matrix_profile: str):
    """Build the raw scipy matrix for a --matrix-profile. Shared with
    profile_cusparse.py so both drivers profile the identical matrix."""
    kind, cfg = _MATRIX_PROFILES[matrix_profile]
    if kind == "scattered":
        density = _SCATTERED_DENSITY_AT_1024 * 1024 / size
        mat = generate_scattered_block_matrix(
            system_size=size,
            block_size=cfg["block_size"],
            profiles=[(cfg["rows_per_block"], cfg["cols_per_block"], density)],
        ).astype(np.float32)
    elif kind == "diag":
        mat = generate_diag_blocks_plus_noise(
            system_size=size,
            block_size=cfg["block_size"],
            diag_density=cfg["diag_density"],
            off_diag_density=cfg["off_diag_density"],
        ).astype(np.float32)
    elif kind == "pattern":
        density = _SCATTERED_DENSITY_AT_1024 * 1024 / size
        mat = generate_patterned_block_matrix(
            system_size=size, block_size=8,
            patterns=[(PBR_PATTERN_PRESETS[cfg["preset"]], density)],
        ).astype(np.float32)
    else:  # fem
        gen = generate_permuted_fem_2d if cfg["permuted"] else generate_fem_2d
        mat = gen(system_size=size).astype(np.float32)
    return mat


def build(size: int, features: int, block: int, min_nnz: int,
          matrix_profile: str, smem_size: int):
    mat = build_matrix(size, matrix_profile)

    pbr_mat = csr_to_pbr(mat, block_rows=block, block_cols=block,
                         min_nnz_per_block=min_nnz)

    # Print structural stats before transferring to GPU
    info = pbr_info(pbr_mat)
    print("[pbr_info]")
    for field in dataclasses.fields(info):
        print(f"  {field.name:26s} = {getattr(info, field.name)}")

    sm_count = torch.cuda.get_device_properties(0).multi_processor_count
    print(f"  {'sm_count':26s} = {sm_count}")
    print(f"  {'smem_size':26s} = {smem_size}")
    print(f"  {'features':26s} = {features}")
    print()

    pbr_mat = pbr_mat.to("cuda")

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
    p.add_argument("--warmup",         type=int, default=10)
    args = p.parse_args()

    pbr_mat, x = build(
        args.size, args.features, args.block, args.min_nnz,
        args.matrix_profile, args.smem_size,
    )

    for _ in range(args.warmup):
        pbr_matmul(pbr_mat, x, smem_size=args.smem_size)
    torch.cuda.synchronize()

    profiler.start()
    pbr_matmul(pbr_mat, x, smem_size=args.smem_size)
    torch.cuda.synchronize()
    profiler.stop()


if __name__ == "__main__":
    main()
