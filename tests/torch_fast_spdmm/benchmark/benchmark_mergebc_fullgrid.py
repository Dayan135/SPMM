"""Repeated-trial wall-clock benchmark: baseline vs. mergeBC, scalar_diag path,
across this project's STANDARD full benchmark grid (not the narrow 5-matrix/
2-feature diagnostic grid used in STALL_REASONS_ZEROIDLE_MICROBENCH.md /
STALL_REASONS_MERGEBC_ALLOC_ABLATION.md).

Grid is taken verbatim from tests/torch_fast_spdmm/test_benchmark.py -- the
project's real W&B-logged benchmark suite used for every other kernel-vs-block
comparison in this project's history:
  - MATRIX_FAMILIES: same 10 (generator, kwargs, name) entries as
    test_benchmark.py's MATRIX_FAMILIES (diag_b0.3, diag_b1.0, 4 scattered
    shapes x 2 densities each).
  - FEATURES: same [16, 32, 64, 128, 256, 512] as test_benchmark.py's FEATURES.
  - Block/min_nnz configs: same _SCALAR_DIAG_BLOCK_CONFIGS test_benchmark.py
    already defines specifically for its own scalar_diag/mergeBC sweep
    (SCALAR_DIAG_RUNS) -- (2,2) (2,3) (4,4) (4,8) (8,2) (8,12) (8,20), fixed
    smem_size=2048, col_major=True (reuses the colgroup kernel's pre-built
    col-sorted arrays, exactly as test_benchmark.py's SCALAR_DIAG_RUNS does).
  - SIZES: {16384, 65536, 262144} (this project's standard size list -- see
    CLAUDE.md's plot_benchmarks example and LAST_EXPERIMENTS.md), NOT
    test_benchmark.py's full _ALL_SIZES (which also has 1024/4096 -- excluded
    here since those are toy sizes nobody plots).

kernel="scalar_diag" forces the scalar pbr_spmm_zero_idle_kernel family
regardless of features%4 (no use_vec check, unlike kernel="block") -- so
unlike the vec-kernel colmajor sweeps, this grid does not need to avoid
features that are multiples of 4.

Usage (one size per invocation -- see jobs/bench_mergebc_fullgrid/ for the
per-size sbatch split):
    python -m benchmark.benchmark_mergebc_fullgrid --size 262144 \
        --out /path/to/mergebc_fullgrid_262144.csv
"""

from __future__ import annotations

import argparse
import csv
import time

import torch

from benchmark.matrices import generate_diag_blocks_plus_noise, generate_scattered_block_matrix
from torch_fast_spdmm import csr_to_pbr, pbr_matmul

# ── Grid definitions, copied verbatim from test_benchmark.py ──────────────────

FEATURES = [16, 32, 64, 128, 256, 512]

_SCAT_REF = 1024
def _sd(base_density, n):
    return base_density * _SCAT_REF / n

MATRIX_FAMILIES = [
    (generate_diag_blocks_plus_noise,
     lambda n: {"system_size": n, "block_size": 64, "diag_density": 0.3, "off_diag_density": 0.0},
     "diag_b0.3"),
    (generate_diag_blocks_plus_noise,
     lambda n: {"system_size": n, "block_size": 64, "diag_density": 1.0, "off_diag_density": 0.0},
     "diag_b1.0"),
    (generate_scattered_block_matrix,
     lambda n: {"system_size": n, "block_size": 8, "profiles": [(1, 8, _sd(0.005, n))]},
     "scattered_1x8_d0.5pct"),
    (generate_scattered_block_matrix,
     lambda n: {"system_size": n, "block_size": 8, "profiles": [(1, 8, _sd(0.02, n))]},
     "scattered_1x8_d2pct"),
    (generate_scattered_block_matrix,
     lambda n: {"system_size": n, "block_size": 8, "profiles": [(8, 1, _sd(0.005, n))]},
     "scattered_8x1_d0.5pct"),
    (generate_scattered_block_matrix,
     lambda n: {"system_size": n, "block_size": 8, "profiles": [(8, 1, _sd(0.02, n))]},
     "scattered_8x1_d2pct"),
    (generate_scattered_block_matrix,
     lambda n: {"system_size": n, "block_size": 2, "profiles": [(2, 2, _sd(0.005, n))]},
     "scattered_2x2_d0.5pct"),
    (generate_scattered_block_matrix,
     lambda n: {"system_size": n, "block_size": 2, "profiles": [(2, 2, _sd(0.02, n))]},
     "scattered_2x2_d2pct"),
    (generate_scattered_block_matrix,
     lambda n: {"system_size": n, "block_size": 8, "profiles": [(2, 4, _sd(0.005, n))]},
     "scattered_2x4_d0.5pct"),
    (generate_scattered_block_matrix,
     lambda n: {"system_size": n, "block_size": 8, "profiles": [(2, 4, _sd(0.02, n))]},
     "scattered_2x4_d2pct"),
]

# Same (block_size, min_nnz) grid test_benchmark.py's own SCALAR_DIAG_RUNS uses.
_SCALAR_DIAG_BLOCK_CONFIGS = [
    (2, 2), (2, 3),
    (4, 4), (4, 8),
    (8, 2), (8, 12), (8, 20),
]
_SCALAR_DIAG_SMEM = 2048

VARIANTS = ["baseline", "mergeBC"]
WARMUP = 3
REPEATS = 5


def time_one(pbr_mat, x, variant: str) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    pbr_matmul(pbr_mat, x, kernel="scalar_diag", scalar_variant=variant,
               col_major=True, smem_size=_SCALAR_DIAG_SMEM)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end)  # ms


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--size", type=int, required=True)
    p.add_argument("--out", type=str, required=True)
    args = p.parse_args()

    device = "cuda"
    size = args.size

    with open(args.out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["matrix", "size", "features", "block_size", "min_nnz",
                          "smem_size", "variant", "rep", "time_ms"])

        for gen, make_kwargs, name in MATRIX_FAMILIES:
            kwargs = make_kwargs(size)
            t0 = time.time()
            mat = gen(**kwargs).astype("float32")
            gen_s = time.time() - t0
            print(f"[gen] {name}/{size}: {gen_s:.2f}s, nnz={mat.nnz}", flush=True)

            for bs, mn in _SCALAR_DIAG_BLOCK_CONFIGS:
                try:
                    pbr_mat = csr_to_pbr(mat, block_rows=bs, block_cols=bs, min_nnz_per_block=mn)
                except Exception as e:
                    print(f"  [skip] {name}/{size} block={bs} min_nnz={mn}: csr_to_pbr failed: {e}", flush=True)
                    continue
                pbr_mat = pbr_mat.to(device)

                for features in FEATURES:
                    torch.manual_seed(0)
                    x = torch.randn((1, mat.shape[1], features), dtype=torch.float32, device=device)

                    for variant in VARIANTS:
                        try:
                            for _ in range(WARMUP):
                                pbr_matmul(pbr_mat, x, kernel="scalar_diag", scalar_variant=variant,
                                           col_major=True, smem_size=_SCALAR_DIAG_SMEM)
                            torch.cuda.synchronize()

                            for rep in range(REPEATS):
                                t = time_one(pbr_mat, x, variant)
                                writer.writerow([name, size, features, bs, mn,
                                                  _SCALAR_DIAG_SMEM, variant, rep, f"{t:.6f}"])
                        except torch.OutOfMemoryError:
                            print(f"  [oom] {name}/{size} block={bs} min_nnz={mn} feat={features} "
                                  f"variant={variant}", flush=True)
                            torch.cuda.empty_cache()
                            continue
                f.flush()
                print(f"  [done] {name}/{size} block={bs} min_nnz={mn}", flush=True)

    print(f"[done] wrote {args.out}")


if __name__ == "__main__":
    main()
