"""Batched compute-sanitizer racecheck driver for the nosyncA/B/C/All
scalar_diag kernel variants, covering the FULL standard benchmark grid
(STALL_REASONS_NOSYNC_RACECHECK_FULLGRID.md) instead of the earlier
one-process-per-config sweep (racecheck_probe.py, 30 configs, 3 variants).

At the full grid's scale (10 matrices x 7 block configs x 6 features x 4
variants = 1680 configs) spawning a fresh `compute-sanitizer python -m ...`
process per config would re-pay full Python/PyTorch import cost 1680 times --
far too slow. Instead this script takes ONE (variant, matrix) pair per
invocation and loops internally over all 7 (block_size, min_nnz) x 6 features
= 42 sub-configs in a single process, so `compute-sanitizer --tool racecheck`
attaches once per (variant, matrix) and monitors 42 sequential kernel launches
-- 40 processes total (4 variants x 10 matrices) instead of 1680.

Matrix/block-config/feature grid is the same one used for the mergeBC
full-grid benchmark (STALL_REASONS_MERGEBC_FULLGRID.md): test_benchmark.py's
MATRIX_FAMILIES and _SCALAR_DIAG_BLOCK_CONFIGS, size fixed at 262144 (the
size the original racecheck study used specifically because it's large
enough to give races a realistic chance to manifest -- race *existence* is
about thread/warp access patterns, not wall-clock scaling, so unlike the
benchmark sweep this does not need to repeat across all 3 standard sizes).

Prints a `[CONFIG]`/`[CONFIG-DONE]` marker line around every sub-config's
matmul call (flushed immediately) so a hazard report emitted by racecheck --
which reports synchronously per kernel launch, interleaved with the target
process's own stdout -- can be attributed to the specific (block_size,
min_nnz, features) sub-config it occurred in, not just the (variant, matrix)
pair the whole log covers. The final cumulative `RACECHECK SUMMARY` line
(hazards/errors/warnings across the WHOLE process) is still the authoritative
per-(variant, matrix) total; per-sub-config attribution is best-effort, only
needed if that total is ever nonzero.

Usage:
    compute-sanitizer --tool racecheck python -m benchmark.racecheck_probe_batch \
        --variant nosyncAll --matrix diag_b0.3 > out/nosyncAll_diag_b0.3.log 2>&1
"""

from __future__ import annotations

import argparse
import sys

import torch

from benchmark.matrices import generate_diag_blocks_plus_noise, generate_scattered_block_matrix
from torch_fast_spdmm import csr_to_pbr, pbr_batched_matmul_scalar_diag_cuda

# ── Grid definitions, same as benchmark_mergebc_fullgrid.py ───────────────────

FEATURES = [16, 32, 64, 128, 256, 512]
SIZE = 262144

_SCAT_REF = 1024
def _sd(base_density, n):
    return base_density * _SCAT_REF / n

MATRIX_FAMILIES = {
    "diag_b0.3": (generate_diag_blocks_plus_noise,
        {"system_size": SIZE, "block_size": 64, "diag_density": 0.3, "off_diag_density": 0.0}),
    "diag_b1.0": (generate_diag_blocks_plus_noise,
        {"system_size": SIZE, "block_size": 64, "diag_density": 1.0, "off_diag_density": 0.0}),
    "scattered_1x8_d0.5pct": (generate_scattered_block_matrix,
        {"system_size": SIZE, "block_size": 8, "profiles": [(1, 8, _sd(0.005, SIZE))]}),
    "scattered_1x8_d2pct": (generate_scattered_block_matrix,
        {"system_size": SIZE, "block_size": 8, "profiles": [(1, 8, _sd(0.02, SIZE))]}),
    "scattered_8x1_d0.5pct": (generate_scattered_block_matrix,
        {"system_size": SIZE, "block_size": 8, "profiles": [(8, 1, _sd(0.005, SIZE))]}),
    "scattered_8x1_d2pct": (generate_scattered_block_matrix,
        {"system_size": SIZE, "block_size": 8, "profiles": [(8, 1, _sd(0.02, SIZE))]}),
    "scattered_2x2_d0.5pct": (generate_scattered_block_matrix,
        {"system_size": SIZE, "block_size": 2, "profiles": [(2, 2, _sd(0.005, SIZE))]}),
    "scattered_2x2_d2pct": (generate_scattered_block_matrix,
        {"system_size": SIZE, "block_size": 2, "profiles": [(2, 2, _sd(0.02, SIZE))]}),
    "scattered_2x4_d0.5pct": (generate_scattered_block_matrix,
        {"system_size": SIZE, "block_size": 8, "profiles": [(2, 4, _sd(0.005, SIZE))]}),
    "scattered_2x4_d2pct": (generate_scattered_block_matrix,
        {"system_size": SIZE, "block_size": 8, "profiles": [(2, 4, _sd(0.02, SIZE))]}),
}

_SCALAR_DIAG_BLOCK_CONFIGS = [
    (2, 2), (2, 3),
    (4, 4), (4, 8),
    (8, 2), (8, 12), (8, 20),
]
_SCALAR_DIAG_SMEM = 2048


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--variant", type=str, required=True,
                    choices=["nosyncA", "nosyncB", "nosyncC", "nosyncAll"])
    p.add_argument("--matrix", type=str, required=True, choices=list(MATRIX_FAMILIES))
    args = p.parse_args()

    gen, kwargs = MATRIX_FAMILIES[args.matrix]
    mat = gen(**kwargs).astype("float32")
    print(f"[racecheck_probe_batch] variant={args.variant} matrix={args.matrix} size={SIZE} "
          f"nnz={mat.nnz}", flush=True)

    n_configs = 0
    n_failed = 0
    for bs, mn in _SCALAR_DIAG_BLOCK_CONFIGS:
        try:
            pbr_mat = csr_to_pbr(mat, block_rows=bs, block_cols=bs, min_nnz_per_block=mn)
        except Exception as e:
            print(f"[CONFIG-SKIP] variant={args.variant} matrix={args.matrix} block={bs} "
                  f"min_nnz={mn}: csr_to_pbr failed: {e}", flush=True)
            continue
        pbr_mat = pbr_mat.to("cuda")

        for features in FEATURES:
            torch.manual_seed(0)
            x = torch.randn((1, mat.shape[1], features), dtype=torch.float32, device="cuda")

            print(f"[CONFIG] variant={args.variant} matrix={args.matrix} block={bs} "
                  f"min_nnz={mn} feat={features} START", flush=True)
            n_configs += 1
            try:
                y = pbr_batched_matmul_scalar_diag_cuda(pbr_mat, x, variant=args.variant)
                torch.cuda.synchronize()
                print(f"[CONFIG] variant={args.variant} matrix={args.matrix} block={bs} "
                      f"min_nnz={mn} feat={features} END y.sum={y.sum().item():.4f}", flush=True)
            except Exception as e:
                n_failed += 1
                print(f"[CONFIG] variant={args.variant} matrix={args.matrix} block={bs} "
                      f"min_nnz={mn} feat={features} FAILED: {e}", flush=True)

    print(f"[racecheck_probe_batch] done variant={args.variant} matrix={args.matrix} "
          f"n_configs={n_configs} n_failed={n_failed}", flush=True)


if __name__ == "__main__":
    main()
