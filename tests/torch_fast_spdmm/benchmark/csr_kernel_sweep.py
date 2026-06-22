"""
CSR kernel head-to-head experiment: csr_spmm_kernel_fast vs. csr_spmm_kernel (fallback).

Part A — Isolation
  min_nnz_per_block=65 forces all nnz into the CSR remainder (8×8 max = 64 nnz/block).
  Random matrices sweep nnz/row from 1 to 256 so the crossover near threshold=24 is visible.

Part B — Full pipeline
  Real PBR+CSR split with 8×8/min4 on all standard matrix families.
  csr_mode ∈ {0=auto, 1=force_fast, 2=force_fallback} measured for the whole pipeline.

Output: two markdown tables to stdout (redirect to .md in Slurm job).

Run from repo root with PYTHONPATH set (Slurm job handles this):
  python tests/operations/torch_fast_spdmm/benchmark/csr_kernel_sweep.py
"""

import os
import sys
import time

import numpy as np
import torch

# Allow `from benchmark.matrices import ...` when run as a script from repo root.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from benchmark.matrices import (
    generate_diag_blocks_plus_noise,
    generate_random_sparse,
    generate_scattered_block_matrix,
)
from test_utils import scipy_csr_to_torch_csr
from torch_fast_spdmm import csr_to_pbr, pbr_matmul

# ── Timing ────────────────────────────────────────────────────────────────────

WARMUP  = 10
REPEATS = 100

def time_ms(fn, *args, warmup=WARMUP, repeats=REPEATS):
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()
    t0 = torch.cuda.Event(enable_timing=True)
    t1 = torch.cuda.Event(enable_timing=True)
    t0.record()
    for _ in range(repeats):
        fn(*args)
    t1.record()
    torch.cuda.synchronize()
    return t0.elapsed_time(t1) / repeats


# ── Part A config ─────────────────────────────────────────────────────────────

SIZE_A    = 65536
FEATURES  = [64, 128, 256, 512]
# Densities chosen so that SIZE_A × density ≈ target nnz/row.
NNZ_ROW_TARGETS = [1, 2, 4, 8, 16, 24, 32, 64, 128, 256]
DENSITIES_A = [t / SIZE_A for t in NNZ_ROW_TARGETS]

# min_nnz_per_block=65 exceeds the 8×8 block capacity (64), so csr_to_pbr
# produces zero PBR blocks — every nnz lands in the CSR remainder.
MIN_NNZ_ISOLATION = 65


# ── Part B config ─────────────────────────────────────────────────────────────

SIZE_B     = 262144
FEATURES_B = [64, 512]

MATRIX_FAMILIES_B = [
    (generate_scattered_block_matrix,
     lambda n: {"system_size": n, "block_size": 8, "profiles": [(1, 8, 0.00001)]},
     "scattered_1x8"),
    (generate_scattered_block_matrix,
     lambda n: {"system_size": n, "block_size": 8, "profiles": [(8, 1, 0.00001)]},
     "scattered_8x1"),
    (generate_scattered_block_matrix,
     lambda n: {"system_size": n, "block_size": 8, "profiles": [(2, 4, 0.00001)]},
     "scattered_2x4"),
    (generate_scattered_block_matrix,
     lambda n: {"system_size": n, "block_size": 2, "profiles": [(2, 2, 0.0001)]},
     "scattered_dense2"),
    (generate_diag_blocks_plus_noise,
     lambda n: {"system_size": n, "block_size": 8, "diag_density": 0.001, "off_diag_density": 1e-6},
     "diag_dd0.001"),
    (generate_diag_blocks_plus_noise,
     lambda n: {"system_size": n, "block_size": 8, "diag_density": 0.003, "off_diag_density": 1e-6},
     "diag_dd0.003"),
    (generate_random_sparse,
     lambda n: {"system_size": n, "density": 1e-6},
     "random_d1e-6"),
]

PBR_B = {"block_rows": 8, "block_cols": 8, "min_nnz": 4}


# ── Helpers ───────────────────────────────────────────────────────────────────

def build_x(size, features, dtype=torch.float32):
    return torch.randn(size, features, dtype=dtype, device="cuda")


def remainder_nnz_per_row(pbr_mat):
    return len(pbr_mat.remainder_col_ind) / pbr_mat.rows


def fmt(v):
    return f"{v:.3f}"

def fmt_speedup(v):
    s = f"{v:.3f}"
    return f"**{s}**" if v >= 1.0 else s


# ── Part A ────────────────────────────────────────────────────────────────────

def run_part_a():
    print("## Part A — CSR kernel isolation")
    print(f"All nnz in CSR remainder (min_nnz_per_block={MIN_NNZ_ISOLATION}), "
          f"size={SIZE_A}, dtype=float32, {WARMUP} warmup / {REPEATS} repeats\n")
    print("speedup = time_fallback / time_fast; **bold** = fast kernel wins (>1.0)\n")

    header = "| nnz/row | feat | time_fast(ms) | time_fallback(ms) | speedup | auto_decision |"
    sep    = "|---|---|---|---|---|---|"
    print(header)
    print(sep)

    for density, target in zip(DENSITIES_A, NNZ_ROW_TARGETS):
        scipy_mat = generate_random_sparse(system_size=SIZE_A, density=density)
        pbr_mat   = csr_to_pbr(scipy_mat, block_rows=8, block_cols=8,
                                min_nnz_per_block=MIN_NNZ_ISOLATION)
        pbr_mat   = pbr_mat.to("cuda")

        actual_nnz_per_row = remainder_nnz_per_row(pbr_mat)
        auto_decision = "fast" if actual_nnz_per_row >= 24 else "fallback"

        for feat in FEATURES:
            x = build_x(SIZE_A, feat)

            t_fast     = time_ms(pbr_matmul, pbr_mat, x, 1)
            t_fallback = time_ms(pbr_matmul, pbr_mat, x, 2)
            speedup    = t_fallback / t_fast if t_fast > 0 else float("nan")

            print(f"| {actual_nnz_per_row:.1f} | {feat} | {fmt(t_fast)} | "
                  f"{fmt(t_fallback)} | {fmt_speedup(speedup)} | {auto_decision} |")

        del pbr_mat
        torch.cuda.empty_cache()

    print()


# ── Part B ────────────────────────────────────────────────────────────────────

def run_part_b():
    br = PBR_B["block_rows"]
    bc = PBR_B["block_cols"]
    mn = PBR_B["min_nnz"]

    print(f"## Part B — Full pipeline (pbr_{br}x{bc}_min{mn}, size={SIZE_B}, dtype=float32)")
    print(f"{WARMUP} warmup / {REPEATS} repeats\n")
    print("auto_ok = auto time ≤ min(fast, fallback) × 1.02\n")

    header = ("| matrix | feat | rem_nnz/row | "
              "time_auto(ms) | time_fast(ms) | time_fallback(ms) | speedup | auto_ok |")
    sep    = "|---|---|---|---|---|---|---|---|"
    print(header)
    print(sep)

    for gen, make_kwargs, name in MATRIX_FAMILIES_B:
        scipy_mat = gen(**make_kwargs(SIZE_B))
        pbr_mat   = csr_to_pbr(scipy_mat, block_rows=br, block_cols=bc,
                                min_nnz_per_block=mn)
        pbr_mat   = pbr_mat.to("cuda")
        rem_nnz   = remainder_nnz_per_row(pbr_mat)

        for feat in FEATURES_B:
            x = build_x(SIZE_B, feat)

            t_auto     = time_ms(pbr_matmul, pbr_mat, x, 0)
            t_fast     = time_ms(pbr_matmul, pbr_mat, x, 1)
            t_fallback = time_ms(pbr_matmul, pbr_mat, x, 2)

            speedup  = t_fallback / t_fast if t_fast > 0 else float("nan")
            best     = min(t_fast, t_fallback)
            auto_ok  = "yes" if t_auto <= best * 1.02 else "NO"

            print(f"| {name} | {feat} | {rem_nnz:.1f} | "
                  f"{fmt(t_auto)} | {fmt(t_fast)} | {fmt(t_fallback)} | "
                  f"{fmt_speedup(speedup)} | {auto_ok} |")

        del pbr_mat
        torch.cuda.empty_cache()

    print()


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available", file=sys.stderr)
        sys.exit(1)

    device_name = torch.cuda.get_device_name(0)
    print(f"# CSR kernel sweep — {device_name}\n")
    print(f"**Threshold:** CSR_FAST_MIN_NNZ_PER_ROW = 24  "
          f"(above → fast kernel, below → fallback)\n")

    run_part_a()
    run_part_b()
