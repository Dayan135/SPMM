"""Correctness check for the diagnostic-only zero_idle-kernel microbench
variants (2026-07-16): baseline/constdata/nosyncA/nosyncB/nosyncC/mergeBC.

Unlike test_pbr.py's parametrized correctness suite, this is NOT a pytest file
-- 4 of the 6 variants are EXPECTED to fail (they read wrong data or race by
design), so a hard assert_close would always throw. This script instead
reports max abs/rel error and a pass/fail verdict per (matrix, features,
variant), using the SAME matrices/sizes as the profiling job
(jobs/profile_zeroidle_microbench/) rather than the tiny pytest fixtures --
races need enough concurrent blocks/threads to have a realistic chance of
manifesting.

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
BLOCK = 8
MIN_NNZ = 2

VARIANTS = ["baseline", "constdata", "nosyncA", "nosyncB", "nosyncC", "mergeBC"]
EXPECTED_CORRECT = {"baseline", "mergeBC"}

TOL = 1e-4


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
                rows.append((matrix, features, variant, passed, expected, max_abs, max_rel, flag))

    header = f"{'matrix':16s} {'feat':5s} {'variant':10s} {'pass':6s} {'expected':9s} {'max_abs':12s} {'max_rel':12s}"
    print(header)
    print("-" * len(header))
    for matrix, features, variant, passed, expected, max_abs, max_rel, flag in rows:
        print(f"{matrix:16s} {features:<5d} {variant:10s} {str(passed):6s} {str(expected):9s} "
              f"{max_abs:<12.6g} {max_rel:<12.6g} {flag}")

    n_unexpected = sum(1 for r in rows if r[-1] != "OK")
    print()
    print(f"{len(rows)} checks run, {n_unexpected} unexpected (pass != expected)")


if __name__ == "__main__":
    main()
