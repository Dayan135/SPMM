"""Repeated-trial wall-clock benchmark for the zero_idle-kernel microbench
variants, to cross-check the single-shot ncu timing numbers from
jobs/profile_zeroidle_microbench/ against averaged, repeated measurements.

Same 5 matrices x 2 features x 6 variants grid as the profiling job, each run
10 times (after warmup) with torch.cuda.Event timing. No ncu involved here --
pure wall-clock.

Usage:
    python -m benchmark.benchmark_scalar_diag
"""

from __future__ import annotations

import statistics

import torch

from benchmark.profile_pbr import build_matrix
from torch_fast_spdmm import csr_to_pbr, pbr_matmul

MATRICES = ["diag_b0.1", "diag_b0.3", "scattered_1x8", "scattered_8x1", "scattered_2x2"]
FEATURES = [128, 512]
SIZE = 262144
BLOCK = 8
MIN_NNZ = 2
VARIANTS = ["baseline", "constdata", "nosyncA", "nosyncB", "nosyncC", "mergeBC"]
WARMUP = 10
REPEATS = 10


def time_one(pbr_mat, x, variant) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    pbr_matmul(pbr_mat, x, kernel="scalar_diag", scalar_variant=variant)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end)  # ms


def main() -> None:
    device = "cuda"
    rows = []
    for matrix in MATRICES:
        mat = build_matrix(SIZE, matrix)
        pbr_mat = csr_to_pbr(mat, block_rows=BLOCK, block_cols=BLOCK, min_nnz_per_block=MIN_NNZ)
        pbr_mat = pbr_mat.to(device)

        for features in FEATURES:
            torch.manual_seed(0)
            x = torch.randn((1, mat.shape[1], features), dtype=torch.float32, device=device)

            for variant in VARIANTS:
                for _ in range(WARMUP):
                    pbr_matmul(pbr_mat, x, kernel="scalar_diag", scalar_variant=variant)
                torch.cuda.synchronize()

                times = [time_one(pbr_mat, x, variant) for _ in range(REPEATS)]
                mean = statistics.mean(times)
                std = statistics.stdev(times)
                rows.append((matrix, features, variant, mean, std, times))

    header = f"{'matrix':16s} {'feat':5s} {'variant':10s} {'mean_ms':10s} {'std_ms':10s} {'cv':6s}"
    print(header)
    print("-" * len(header))
    for matrix, features, variant, mean, std, _ in rows:
        cv = std / mean if mean > 0 else float("nan")
        print(f"{matrix:16s} {features:<5d} {variant:10s} {mean:<10.4f} {std:<10.4f} {cv:<6.3f}")


if __name__ == "__main__":
    main()
