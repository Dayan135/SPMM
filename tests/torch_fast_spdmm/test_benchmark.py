"""
GPU benchmark suite for torch_fast_spdmm.

Sweep axes
----------
mat_config  (MATRIX_CONFIGS) — one entry per workload; adding a new workload = one line
features    (FEATURES)       — dense column count
dtype       (DTYPES)         — float32 / float64
algo        (BASELINE_ALGOS) — (name, callable) pairs; adding a new baseline = one line
pbr_cfg     (PBR_CONFIGS)    — block-size / min-nnz variants; adding a new variant = one line

All results are logged to the "torch-fast-spdmm" W&B project.
"""

import gc
import os

import numpy as np
import pytest
import torch
import wandb
from pytest import mark

from benchmark.matrices import (
    generate_diag_blocks_plus_noise,
    generate_scattered_block_matrix,
)
from test_utils import scipy_csr_to_torch_csr
from torch_fast_spdmm import csr_to_pbr, pbr_matmul, spdmm_csr_naive


# ── Configuration tables ───────────────────────────────────────────────────────

FEATURES = [16, 32, 64, 128, 256, 512]
DTYPES   = [torch.float32]

# Square-matrix dimensions to sweep. Powers of two stay divisible by every
# block size used below (~1k / 4k / 16k / 64k / 256k).
#
# One SLURM array task per size: set SPDMM_ONLY_SIZE=<n> to restrict this run
# (and thus pytest collection + the W&B run name) to a single size. Unset =
# all sizes (local/ad-hoc use).
_ALL_SIZES = [1024, 4096, 16384, 65536, 262144]
_ONLY_SIZE = os.environ.get("SPDMM_ONLY_SIZE")
SIZES = [int(_ONLY_SIZE)] if _ONLY_SIZE else _ALL_SIZES

# Scattered "density" = fraction of the N/bs x N/bs block grid, so a fixed
# fraction gives O(N^2) nnz -- fine at small N, but ~1.4B nnz (OOM) at 262144.
# Anchor the requested fractions at N=1024 and scale as 1024/N so absolute
# block count (hence nnz) grows ~O(N): matrices stay genuinely sparse and
# comparable across sizes, and N=1024 keeps the literal 0.5%/2% spec.
# (diag/FEM are already O(N) and are not scaled.)
_SCAT_REF = 1024
def _sd(base_density, n):
    return base_density * _SCAT_REF / n

# Matrix families, parametrised by system size. Each entry is
# (generator_func, size -> kwargs, family_name). Adding a family = one line.
MATRIX_FAMILIES = [
    # Block-diagonal (64-wide dense bands, no off-block noise) — dense super-blocks
    # of 8x8 = 64 PBR 8x8 blocks each; ideal territory for the 2D tile kernel.
    # Densities are the original {0.1,0.5,1.0} tripled and clamped at 1.0
    # (0.5*3 and 1.0*3 both saturate -> single 1.0 entry).
    (generate_diag_blocks_plus_noise,
     lambda n: {"system_size": n, "block_size": 64, "diag_density": 0.3, "off_diag_density": 0.0},
     "diag_b0.3"),
    (generate_diag_blocks_plus_noise,
     lambda n: {"system_size": n, "block_size": 64, "diag_density": 1.0, "off_diag_density": 0.0},
     "diag_b1.0"),
    # Scattered 1x8 / 8x1 blocks (row- vs col-oriented reuse), 0.5% and 2% of grid
    # at N=1024, scaled 1024/N for larger sizes (see _sd).
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
    # Scattered dense 2x2 blocks (block_size=2).
    (generate_scattered_block_matrix,
     lambda n: {"system_size": n, "block_size": 2, "profiles": [(2, 2, _sd(0.005, n))]},
     "scattered_2x2_d0.5pct"),
    (generate_scattered_block_matrix,
     lambda n: {"system_size": n, "block_size": 2, "profiles": [(2, 2, _sd(0.02, n))]},
     "scattered_2x2_d2pct"),
    # Scattered 2x4 blocks within 8x8 windows.
    (generate_scattered_block_matrix,
     lambda n: {"system_size": n, "block_size": 8, "profiles": [(2, 4, _sd(0.005, n))]},
     "scattered_2x4_d0.5pct"),
    (generate_scattered_block_matrix,
     lambda n: {"system_size": n, "block_size": 8, "profiles": [(2, 4, _sd(0.02, n))]},
     "scattered_2x4_d2pct"),
]

# Optional per-matrix gate: one SLURM task per (size, matrix) so each task is
# small (~1.4k benchmarks), never times out, and logs a small W&B table (the
# session-level table silently truncates at 10k rows). Unset = all matrices.
_ONLY_MATRIX = os.environ.get("SPDMM_ONLY_MATRIX")
if _ONLY_MATRIX:
    MATRIX_FAMILIES = [m for m in MATRIX_FAMILIES if m[2] == _ONLY_MATRIX]

# Cross size × family into the actual workload table.
MATRIX_CONFIGS = [
    pytest.param((gen, make_kwargs(n), name), id=f"{name}_{n}")
    for n in SIZES
    for (gen, make_kwargs, name) in MATRIX_FAMILIES
]

# PBR block-size / min-nnz variants, crossed with the shared-memory tier.
# smem_size controls the shared-memory tile (floats) per CTA.
_PBR_BLOCK_CONFIGS = [
    (2, 2), (2, 3),            # 2x2 PBR blocks
    (4, 2), (4, 4), (4, 8),    # 4x4 PBR blocks
    (8, 2), (8, 8), (8, 12),   # 8x8 PBR blocks
]  # (block_size, min_nnz)
_SMEM_TIERS = [512, 1024, 2048]

# Tile ("block of blocks") kernel: all 9 shapes x {grid, greedy}. v1 supports
# 8x8 PBR blocks only, so tile is paired only with the 8x8 configs below.
_TILE_SHAPES     = [(1, 1), (1, 2), (2, 1), (1, 4), (4, 1), (2, 2), (2, 4), (4, 2), (4, 4)]
_TILE_STRATEGIES = ["grid", "greedy"]

# Combined (kernel_spec, pbr_cfg) runs — VALID pairings only, so pytest collects
# exactly the runnable cases (no ~100k skipped tile/non-8x8 cross-product items):
#   block / rowgroup / colgroup — every (block_size, min_nnz, smem) config
#   tile (shape x strategy)      — 8x8 configs only
def _make_pbr_runs():
    runs = []
    for bs, mn in _PBR_BLOCK_CONFIGS:
        for sm in _SMEM_TIERS:
            cfg = {"block_rows": bs, "block_cols": bs, "min_nnz": mn, "smem_size": sm}
            cfg_id = f"pbr_{bs}x{bs}_min{mn}_sm{sm}"
            for k in ("block", "rowgroup", "colgroup"):
                runs.append(pytest.param(({"kernel": k}, cfg), id=f"{k}-{cfg_id}"))
            if bs == 8:
                for tr, tc in _TILE_SHAPES:
                    for ts in _TILE_STRATEGIES:
                        spec = {"kernel": "tile", "tile_rows": tr, "tile_cols": tc, "tile_strategy": ts}
                        runs.append(pytest.param((spec, cfg), id=f"tile_{tr}x{tc}_{ts}-{cfg_id}"))
    return runs

PBR_RUNS = _make_pbr_runs()

# zero_idle-kernel ('scalar_diag') sweep: user-requested (block_size, min_nnz)
# configs, col-sorted (col_major=True, reuses the colgroup kernel's pre-built
# col-sorted arrays -- no kernel change needed), fixed smem_size=2048 (not
# crossed with _SMEM_TIERS). 'baseline' (unmodified pbr_spmm_zero_idle_kernel,
# forced scalar dispatch) and 'mergeBC' (Step B/C fused, p held fixed) --
# see STALL_REASONS_ZEROIDLE_MICROBENCH.md.
_SCALAR_DIAG_BLOCK_CONFIGS = [
    (2, 2), (2, 3),           # 2x2 PBR blocks
    (4, 4), (4, 8),           # 4x4 PBR blocks
    (8, 2), (8, 12), (8, 20), # 8x8 PBR blocks
]  # (block_size, min_nnz)
_SCALAR_DIAG_SMEM = 2048


def _make_scalar_diag_runs():
    runs = []
    for bs, mn in _SCALAR_DIAG_BLOCK_CONFIGS:
        cfg = {"block_rows": bs, "block_cols": bs, "min_nnz": mn, "smem_size": _SCALAR_DIAG_SMEM}
        cfg_id = f"pbr_{bs}x{bs}_min{mn}_sm{_SCALAR_DIAG_SMEM}"
        for variant in ("baseline", "mergeBC"):
            spec = {"kernel": "scalar_diag", "scalar_variant": variant, "col_major": True}
            runs.append(pytest.param((spec, cfg), id=f"scalar_diag_{variant}_colmajor-{cfg_id}"))
    return runs


SCALAR_DIAG_RUNS = _make_scalar_diag_runs()
PBR_RUNS = PBR_RUNS + SCALAR_DIAG_RUNS

# Same sweep, but over the vectorized 'block' kernel (pbr_spmm_vec_kernel --
# auto-selected since every features value here is %4==0) instead of the
# forced-scalar zero_idle kernel. Same (block_size, min_nnz) configs, col-sorted
# (col_major=True), fixed smem_size=2048. No mergeBC this round -- baseline only.
def _make_vec_colmajor_runs():
    runs = []
    for bs, mn in _SCALAR_DIAG_BLOCK_CONFIGS:
        cfg = {"block_rows": bs, "block_cols": bs, "min_nnz": mn, "smem_size": _SCALAR_DIAG_SMEM}
        cfg_id = f"pbr_{bs}x{bs}_min{mn}_sm{_SCALAR_DIAG_SMEM}"
        spec = {"kernel": "block", "col_major": True}
        runs.append(pytest.param((spec, cfg), id=f"block_colmajor-{cfg_id}"))
    return runs


VEC_COLMAJOR_RUNS = _make_vec_colmajor_runs()
PBR_RUNS = PBR_RUNS + VEC_COLMAJOR_RUNS

# Adding a new baseline algorithm = one line here.
BASELINE_ALGOS = [
    pytest.param(("cusparse",  torch.sparse.mm),  id="cusparse"),
    pytest.param(("csr_naive", spdmm_csr_naive),  id="csr_naive"),
]


# ── Helpers ────────────────────────────────────────────────────────────────────

def _prepare(mat_config, features, dtype):
    """Build scipy mat, sparse torch tensor, and dense x from a mat_config entry."""
    generator, kwargs, profile_name = mat_config
    system_size = kwargs["system_size"]

    np_dtype     = np.float64 if dtype == torch.float64 else np.float32
    scipy_mat    = generator(**kwargs).astype(np_dtype)
    sparse_torch = scipy_csr_to_torch_csr(scipy_mat).to(dtype=dtype, device="cuda")
    x            = torch.randn((system_size, features), dtype=dtype, device="cuda")

    return scipy_mat, sparse_torch, x, system_size, profile_name


# Identifies which kernel version produced a run, so successive benchmark
# batches are orderable in W&B (group/filter by commit). The job script exports
# the short git SHA; falls back to "unknown" for ad-hoc local runs.
_COMMIT = os.environ.get("SPDMM_COMMIT", "unknown")


# One row per benchmarked case; column order for the W&B table.
_RESULT_COLUMNS = [
    "commit", "profile", "system_size", "features", "dtype",
    "algorithm", "kernel", "block_rows", "block_cols", "min_nnz", "smem_size", "time_ms",
]


@pytest.fixture(scope="session")
def bench_results():
    """Accumulate every case into one W&B run as a single table.

    Each test appends a row; at session teardown the rows become one
    wandb.Table logged to a single run. The Slurm array runs one task per
    (size, matrix), so each run's table is ~1.4k rows -- well under the
    wandb.Table 10k-row cap that silently truncated the earlier per-size runs.
    Runs share group=commit; plotting concatenates all of a commit's runs.
    """
    rows: list[dict] = []
    yield rows

    if not rows:
        return

    table = wandb.Table(columns=_RESULT_COLUMNS)
    for r in rows:
        table.add_data(*(r.get(c) for c in _RESULT_COLUMNS))

    sizes = sorted({r["system_size"] for r in rows})
    profiles = sorted({r["profile"] for r in rows})
    # Unique run name per (sizes, matrices) slice so per-matrix tasks don't collide.
    name = f"{_COMMIT}_" + "_".join(str(s) for s in sizes)
    if _ONLY_MATRIX:
        name += f"_{_ONLY_MATRIX}"
    run = wandb.init(
        project="torch-fast-spdmm",
        group=_COMMIT,
        name=name,
        config={"commit": _COMMIT, "sizes": sizes, "profiles": profiles},
        reinit=True,
    )
    run.log({"benchmarks": table})
    run.finish()


def _cleanup():
    gc.collect()
    torch.cuda.empty_cache()


# ── Baseline benchmarks ───────────────────────────────────────────────────────

@mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@mark.parametrize("algo",       BASELINE_ALGOS)
@mark.parametrize("dtype",      DTYPES)
@mark.parametrize("features",   FEATURES)
@mark.parametrize("mat_config", MATRIX_CONFIGS)
def test_baseline_spmm(benchmark, bench_results, mat_config, features, dtype, algo):
    algo_name, algo_fn = algo
    _, sparse_torch, x, size, profile = _prepare(mat_config, features, dtype)

    try:
        benchmark(algo_fn, sparse_torch, x)
    except torch.OutOfMemoryError:
        _cleanup()
        pytest.skip("OutOfMemory")
    finally:
        _cleanup()

    bench_results.append({
        "commit":      _COMMIT,
        "algorithm":   algo_name,
        "kernel":      "baseline",
        "system_size": size,
        "features":    features,
        "dtype":       str(dtype),
        "profile":     profile,
        "time_ms":     benchmark.stats['mean'] * 1000,
    })


# ── PBR benchmarks ─────────────────────────────────────────────────

@mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@mark.parametrize("pbr_run",    PBR_RUNS)
@mark.parametrize("dtype",      DTYPES)
@mark.parametrize("features",   FEATURES)
@mark.parametrize("mat_config", MATRIX_CONFIGS)
def test_pbr_spmm(benchmark, bench_results, mat_config, features, dtype, pbr_run):
    kernel, pbr_cfg = pbr_run
    scipy_mat, _, x, size, profile = _prepare(mat_config, features, dtype)

    pbr_mat = csr_to_pbr(
        scipy_mat,
        block_rows=pbr_cfg["block_rows"],
        block_cols=pbr_cfg["block_cols"],
        min_nnz_per_block=pbr_cfg["min_nnz"],
    )
    pbr_mat = pbr_mat.to("cuda")

    # Warm-up (excluded from timing): one-time CUDA launch overhead and, for the
    # tile kernel, the CPU-side tile-meta build (cached on pbr_mat after the first
    # call). Steady-state matmul is what we measure — the grouping is a one-time
    # preprocessing cost amortized across many matmuls on the same PBR matrix.
    try:
        pbr_matmul(pbr_mat, x, smem_size=pbr_cfg["smem_size"], **kernel)
        torch.cuda.synchronize()
        benchmark(pbr_matmul, pbr_mat, x, smem_size=pbr_cfg["smem_size"], **kernel)
    except torch.OutOfMemoryError:
        _cleanup()
        pytest.skip("OutOfMemory")
    finally:
        _cleanup()

    kernel_name = kernel["kernel"]
    variant = kernel_name
    if kernel_name == "tile":
        variant = f"tile_{kernel['tile_rows']}x{kernel['tile_cols']}_{kernel['tile_strategy']}"
    elif kernel_name == "scalar_diag":
        order = "colmajor" if kernel.get("col_major") else "rowmajor"
        variant = f"scalar_diag_{kernel['scalar_variant']}_{order}"
    elif kernel_name == "block" and kernel.get("col_major"):
        variant = "block_colmajor"
    algo_name = (
        f"pbr_{pbr_cfg['block_rows']}x{pbr_cfg['block_cols']}"
        f"_min{pbr_cfg['min_nnz']}_sm{pbr_cfg['smem_size']}_{variant}"
    )
    bench_results.append({
        "commit":      _COMMIT,
        "algorithm":   algo_name,
        "kernel":      kernel_name,
        "system_size": size,
        "features":    features,
        "dtype":       str(dtype),
        "profile":     profile,
        "time_ms":     benchmark.stats['mean'] * 1000,
        **pbr_cfg,
    })
