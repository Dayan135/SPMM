"""Correctness check for the diagnostic-only zero_idle-kernel microbench
variants (2026-07-16, +mergeBC_alloc 2026-07-21, +nosyncAll +block/min_nnz
sweep 2026-07-22): baseline/constdata/nosyncA/nosyncB/nosyncC/mergeBC/
mergeBC_alloc/nosyncAll.

Unlike test_pbr.py's parametrized correctness suite, this is NOT a pytest file
-- 5 of the 8 variants are EXPECTED to fail (they read wrong data or race by
design), so a hard assert_close would always throw. This script instead
reports max abs/rel error and a pass/fail verdict per (matrix, features,
block_config, variant), using the SAME matrices/sizes as the profiling job
(jobs/profile_zeroidle_microbench/) rather than the tiny pytest fixtures --
races need enough concurrent blocks/threads to have a realistic chance of
manifesting.

BLOCK_CONFIGS sweeps all 7 (block_size, min_nnz) combos from test_benchmark.py's
_SCALAR_DIAG_BLOCK_CONFIGS (the same grid the racecheck full-grid sweep uses,
STALL_REASONS_NOSYNC_RACECHECK_FULLGRID.md) -- added 2026-07-22 because the
original version fixed block=8/min_nnz=2 and never checked output correctness
across the other 6 configs, even though race-freedom got checked at all 7 by
the racecheck sweep. n_r/n_c (and therefore the worker-pool shape every
variant's loop bounds depend on) changes with block_size, so a variant could
plausibly be correct at one block config and wrong at another -- this wasn't
exercised before. Kept to the original 5-matrix/2-feature set for this
correctness pass (the racecheck sweep is the one that needs the full
10-matrix/6-feature grid; this check's job is just to confirm nothing new
broke across the block/min_nnz dimension).

Usage:
    python -m benchmark.check_scalar_diag_correctness
"""

from __future__ import annotations

import torch

from benchmark.profile_pbr import build_matrix
from torch_fast_spdmm import csr_to_pbr, pbr_batched_matmul_cuda, pbr_batched_matmul_scalar_diag_cuda

MATRICES = ["diag_b0.1", "diag_b0.3", "scattered_1x8", "scattered_8x1", "scattered_2x2"]
FEATURES = [128, 512]
SIZE = 262144

# Same 7 (block_size, min_nnz) combos as test_benchmark.py's
# _SCALAR_DIAG_BLOCK_CONFIGS / benchmark_mergebc_fullgrid.py / the racecheck
# full-grid sweep -- (block=8, min_nnz=2) is the original single point this
# check used to be pinned to, now just one of the 7.
BLOCK_CONFIGS = [
    (2, 2), (2, 3),
    (4, 4), (4, 8),
    (8, 2), (8, 12), (8, 20),
]

VARIANTS = ["baseline", "constdata", "nosyncA", "nosyncB", "nosyncC", "mergeBC", "mergeBC_alloc",
            "nosyncAll"]
EXPECTED_CORRECT = {"baseline", "mergeBC", "mergeBC_alloc"}

TOL = 1e-4


def main() -> None:
    device = "cuda"
    rows = []
    for matrix in MATRICES:
        mat = build_matrix(SIZE, matrix)

        for block, min_nnz in BLOCK_CONFIGS:
            try:
                pbr_mat = csr_to_pbr(mat, block_rows=block, block_cols=block, min_nnz_per_block=min_nnz)
            except Exception as e:
                print(f"[skip] {matrix} block={block} min_nnz={min_nnz}: csr_to_pbr failed: {e}")
                continue
            pbr_mat = pbr_mat.to(device)

            for features in FEATURES:
                torch.manual_seed(0)
                x = torch.randn((1, mat.shape[1], features), dtype=torch.float32, device=device)

                y_ref = pbr_batched_matmul_cuda(pbr_mat, x)

                for variant in VARIANTS:
                    y = pbr_batched_matmul_scalar_diag_cuda(pbr_mat, x, variant=variant)
                    diff = (y - y_ref).abs()
                    max_abs = diff.max().item()
                    denom = y_ref.abs().max().item()
                    max_rel = max_abs / denom if denom > 0 else float("nan")
                    passed = max_abs < TOL or max_rel < TOL
                    expected = variant in EXPECTED_CORRECT
                    flag = "OK" if passed == expected else "**UNEXPECTED**"
                    rows.append((matrix, block, min_nnz, features, variant, passed, expected,
                                 max_abs, max_rel, flag))

    header = (f"{'matrix':16s} {'block':5s} {'min_nnz':7s} {'feat':5s} {'variant':10s} {'pass':6s} "
              f"{'expected':9s} {'max_abs':12s} {'max_rel':12s}")
    print(header)
    print("-" * len(header))
    for matrix, block, min_nnz, features, variant, passed, expected, max_abs, max_rel, flag in rows:
        print(f"{matrix:16s} {block:<5d} {min_nnz:<7d} {features:<5d} {variant:10s} {str(passed):6s} "
              f"{str(expected):9s} {max_abs:<12.6g} {max_rel:<12.6g} {flag}")

    n_unexpected = sum(1 for r in rows if r[-1] != "OK")
    print()
    print(f"{len(rows)} checks run, {n_unexpected} unexpected (pass != expected)")


if __name__ == "__main__":
    main()
