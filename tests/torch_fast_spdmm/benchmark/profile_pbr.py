"""Single-matmul driver for Nsight Compute profiling of the PBR SpMM kernels.

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
  diag_b1.0        — diag_b1.0 from test_benchmark.py (fully dense 64-wide band,
                     no off-diag noise); the case where 'tile' lost 2.4x to 'block'
                     at 262144 despite the 'grid' strategy producing perfectly
                     dense tiles — see LAST_EXPERIMENTS.md Experiment 3
  diag_skewed      — fully-dense diagonal band + off-diagonal noise, deliberately
                     bimodal per-block nnz (CV=2.08 vs diag_b0.3's 0.19) — tests
                     nnz-balanced grouping where there's real per-block work skew
  fem2d            — 2D FEM stiffness matrix (grid-aligned reuse)
  fem2d_permuted   — same, with mesh nodes relabeled (breaks grid-aligned reuse;
                     stress case for the 'grid' vs 'greedy' tile strategies)

Kernels (--kernel):
  block    — original one-CUDA-block-per-PBR-block (default)
  chunk    — diagnostic-only: identical per-block body to 'block', but
             --chunk-size PBR blocks per CUDA block via a #pragma-unrolled
             loop (no smem reuse across the loop) — isolates the CTA-count/
             occupancy effect from rowgroup/colgroup/tile's added reuse
             machinery. See WHY_BLOCK_KERNEL_FAILS_TO_SCALE.md.
  rowgroup — block-count-balanced fusion kernel, row-major order (smem_Y reuse)
  colgroup — block-count-balanced fusion kernel, col-sorted order (smem_X reuse)
  tile     — one CUDA block per tile_rows x tile_cols group of PBR blocks,
             reusing both smem_X and smem_Y for the whole tile (see
             --tile-rows/--tile-cols/--tile-strategy)

Usage (under ncu):
    ncu --profile-from-start off --metrics <...> --csv \
        python -m benchmark.profile_pbr --size 262144 --features 512 \
            --block 8 --min-nnz 2 --smem-size 2048 \
            --matrix-profile diag_dd0.001 --kernel block

    ncu ... python -m benchmark.profile_pbr --size 262144 --features 512 \
            --matrix-profile diag_dd0.001 --kernel tile \
            --tile-rows 2 --tile-cols 2 --tile-strategy greedy
"""

from __future__ import annotations

import argparse
import dataclasses

import numpy as np
import torch
import torch.cuda.profiler as profiler

import torch_fast_spdmm_core as _core
from benchmark.matrices import (generate_scattered_block_matrix, generate_diag_blocks_plus_noise,
                                generate_fem_2d, generate_permuted_fem_2d)
from torch_fast_spdmm import csr_to_pbr, pbr_info, pbr_matmul
# _build_tile_meta/_build_group_offsets are private (not in __all__) -- reached
# into here for diagnostics only, same spirit as this script's direct _core
# occupancy-query calls below.
from torch_fast_spdmm import _build_group_offsets
# _build_tile_meta is private (not in __all__) -- reached into here for diagnostics only,
# same spirit as this script's direct _core occupancy-query calls below.
from torch_fast_spdmm import _build_tile_meta

# Each entry: (kind, cfg)
#   kind="scattered" → generate_scattered_block_matrix
#   kind="diag"      → generate_diag_blocks_plus_noise
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
    # Deliberately bimodal per-8x8-block nnz: fully-dense in-band blocks (~64
    # nnz) + mostly-near-threshold off-band blocks from the off-diagonal noise
    # (structural check: min=1 max=64 mean=11.1 std=23.1, CV=2.08 at N=262144
    # -- vs diag_b0.3's CV=0.19). Tests nnz-balanced grouping (see
    # WHY_BLOCK_KERNEL_FAILS_TO_SCALE.md) on a matrix that actually HAS
    # per-block work skew for it to correct, unlike diag_b0.3.
    "diag_skewed":      ("diag",      {"block_size": 64, "diag_density": 1.0,   "off_diag_density": 0.00002}),
    "fem2d":            ("fem",       {"permuted": False}),
    "fem2d_permuted":   ("fem",       {"permuted": True}),
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
    else:  # fem
        gen = generate_permuted_fem_2d if cfg["permuted"] else generate_fem_2d
        mat = gen(system_size=size).astype(np.float32)
    return mat


def build(size: int, features: int, block: int, min_nnz: int,
          matrix_profile: str, smem_size: int, kernel: str,
          tile_rows: int = 2, tile_cols: int = 2, tile_strategy: str = "grid",
          chunk_size: int = 8, grouping: str = "count"):
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
    print(f"  {'kernel':26s} = {kernel}")
    print(f"  {'smem_size':26s} = {smem_size}")
    print(f"  {'features':26s} = {features}")
    print()

    pbr_mat = pbr_mat.to("cuda")

    if kernel == "chunk":
        num_pbr_blocks = int(pbr_mat.accounted_blocks())
        target_groups = max(1, (num_pbr_blocks + chunk_size - 1) // chunk_size)
        group_offsets = _build_group_offsets(pbr_mat._gpu_meta.block_offsets_np, target_groups, grouping)
        num_groups = len(group_offsets) - 1
        print(f"  {'grouping':26s} = {grouping}")
        print(f"  {'chunk_size (target)':26s} = {chunk_size}")
        print(f"  {'num_pbr_blocks':26s} = {num_pbr_blocks}")
        print(f"  {'target_groups':26s} = {target_groups}")
        print(f"  {'num_groups (grid.x)':26s} = {num_groups}")
        print()
    elif kernel in ("rowgroup", "colgroup"):
        num_pbr_blocks = int(pbr_mat.accounted_blocks())
        bps_fn = _core.pbr_rowgroup_blocks_per_sm if kernel == "rowgroup" else _core.pbr_colgroup_blocks_per_sm
        bps = bps_fn(block, smem_size)
        target_groups = sm_count * bps
        offsets_np = (pbr_mat._gpu_meta.block_offsets_np if kernel == "rowgroup"
                      else pbr_mat._gpu_meta.col_offsets_np)
        group_offsets = _build_group_offsets(offsets_np, target_groups, grouping)
        num_groups = len(group_offsets) - 1
        print(f"  {'grouping':26s} = {grouping}")
        print(f"  {'blocks_per_sm':26s} = {bps}")
        print(f"  {'target_groups':26s} = {target_groups}")
        print(f"  {'num_groups (grid.x)':26s} = {num_groups}")
        print()
    elif kernel == "tile":
        bps = _core.pbr_tile_blocks_per_sm(tile_rows, tile_cols, block, smem_size)
        tile_meta = _build_tile_meta(pbr_mat, tile_rows, tile_cols, tile_strategy,
                                     pbr_mat._gpu_meta.codes.device)
        num_tiles = int(tile_meta.tile_offsets.shape[0]) - 1
        print(f"  {'tile_shape':26s} = {tile_rows}x{tile_cols}")
        print(f"  {'tile_strategy':26s} = {tile_strategy}")
        print(f"  {'blocks_per_sm':26s} = {bps}")
        print(f"  {'num_tiles':26s} = {num_tiles}")
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
                   choices=["block", "chunk", "rowgroup", "colgroup", "tile", "scalar_diag"],
                   help="which PBR kernel to profile")
    p.add_argument("--tile-rows",      type=int, default=2, help="tile kernel only")
    p.add_argument("--tile-cols",      type=int, default=2, help="tile kernel only")
    p.add_argument("--tile-strategy",  type=str, default="grid", choices=["grid", "greedy"],
                   help="tile kernel only")
    p.add_argument("--chunk-size",     type=int, default=8, help="chunk kernel only")
    p.add_argument("--grouping",       type=str, default="count", choices=["count", "nnz"],
                   help="chunk/rowgroup/colgroup only")
    p.add_argument("--scalar-variant", type=str, default="baseline",
                   choices=["baseline", "constdata", "nosyncA", "nosyncB", "nosyncC", "mergeBC",
                            "mergeBC_alloc"],
                   help="scalar_diag kernel only")
    p.add_argument("--warmup",         type=int, default=10)
    args = p.parse_args()

    pbr_mat, x = build(
        args.size, args.features, args.block, args.min_nnz,
        args.matrix_profile, args.smem_size, args.kernel,
        args.tile_rows, args.tile_cols, args.tile_strategy, args.chunk_size, args.grouping,
    )

    matmul_kwargs = dict(smem_size=args.smem_size, kernel=args.kernel)
    if args.kernel in ("chunk", "rowgroup", "colgroup"):
        matmul_kwargs.update(grouping=args.grouping)
    if args.kernel == "tile":
        matmul_kwargs.update(tile_rows=args.tile_rows, tile_cols=args.tile_cols,
                             tile_strategy=args.tile_strategy)
    elif args.kernel == "chunk":
        matmul_kwargs.update(chunk_size=args.chunk_size)
    elif args.kernel == "scalar_diag":
        matmul_kwargs.update(scalar_variant=args.scalar_variant)

    for _ in range(args.warmup):
        pbr_matmul(pbr_mat, x, **matmul_kwargs)
    torch.cuda.synchronize()

    profiler.start()
    pbr_matmul(pbr_mat, x, **matmul_kwargs)
    torch.cuda.synchronize()
    profiler.stop()


if __name__ == "__main__":
    main()
