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
    generate_patterned_block_matrix,
    PBR_PATTERN_PRESETS,
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
    # of 8x8 = 64 PBR 8x8 blocks each.
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
    # Fixed 8x8 patterns tiled across the grid (same nonzero-cell layout in
    # every occupied block, unlike scattered_*'s per-block-random subset) --
    # for isolating how a specific intra-block shape affects kernel speedup.
    # pattern_row_skewed is the discriminating case for work-balance kernel
    # experiments (one fully-dense row + one nonzero per remaining row).
    (generate_patterned_block_matrix,
     lambda n: {"system_size": n, "block_size": 8,
                "patterns": [(PBR_PATTERN_PRESETS["diag_stripe"], _sd(0.02, n))]},
     "pattern_diag_stripe"),
    (generate_patterned_block_matrix,
     lambda n: {"system_size": n, "block_size": 8,
                "patterns": [(PBR_PATTERN_PRESETS["corner_cluster"], _sd(0.02, n))]},
     "pattern_corner_cluster"),
    (generate_patterned_block_matrix,
     lambda n: {"system_size": n, "block_size": 8,
                "patterns": [(PBR_PATTERN_PRESETS["checkerboard"], _sd(0.02, n))]},
     "pattern_checkerboard"),
    # row_skewed at 0.2%/1%/5% -- atomic-contention stress test (row 0's 8
    # nonzeros all target the same compacted smem_Y row-slot in nnzflat's
    # Step B) for the nnzflat vs nnzflat_shared vs zero_idle comparison.
    # SPDMM_ONLY_PBR_CONFIG below can still pin this to one PBR config for a
    # narrower/faster run; unset sweeps the full PBR_RUNS grid like every
    # other matrix family.
    (generate_patterned_block_matrix,
     lambda n: {"system_size": n, "block_size": 8,
                "patterns": [(PBR_PATTERN_PRESETS["row_skewed"], _sd(0.002, n))]},
     "pattern_row_skewed_d0.2pct"),
    (generate_patterned_block_matrix,
     lambda n: {"system_size": n, "block_size": 8,
                "patterns": [(PBR_PATTERN_PRESETS["row_skewed"], _sd(0.01, n))]},
     "pattern_row_skewed_d1pct"),
    (generate_patterned_block_matrix,
     lambda n: {"system_size": n, "block_size": 8,
                "patterns": [(PBR_PATTERN_PRESETS["row_skewed"], _sd(0.05, n))]},
     "pattern_row_skewed_d5pct"),
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
    (2, 2), (2, 3),    # 2x2 PBR blocks
    (4, 2), (4, 4), (4, 8),    # 4x4 PBR blocks
    (8, 2), (8, 8),    # 8x8 PBR blocks -- min_nnz capped at 8 (block max for
                       # 1x8/8x1-oriented scattered matrices) so no config
                       # here can silently 100%-CSR-shunt an otherwise-real
                       # PBR workload the way min_nnz=12 did (see
                       # SHOW_TO_ERAN.md sec. 2).
]  # (block_size, min_nnz)
_SMEM_TIERS = [512, 1024, 2048]

# One run per (block_size, min_nnz, smem) config — only the block kernel exists now.
def _make_pbr_runs():
    runs = []
    for bs, mn in _PBR_BLOCK_CONFIGS:
        for sm in _SMEM_TIERS:
            cfg = {"block_rows": bs, "block_cols": bs, "min_nnz": mn, "smem_size": sm}
            cfg_id = f"pbr_{bs}x{bs}_min{mn}_sm{sm}"
            runs.append(pytest.param(cfg, id=cfg_id))
    return runs

PBR_RUNS = _make_pbr_runs()

# Optional gate matching SPDMM_ONLY_MATRIX/SPDMM_ONLY_KERNEL's convention: restrict
# to PBR configs whose id starts with this prefix (e.g. "pbr_8x8_min8" selects all
# 3 smem tiers of the 8x8/min_nnz=8 config, still sweeping smem). Unset = all configs.
_ONLY_PBR_CONFIG = os.environ.get("SPDMM_ONLY_PBR_CONFIG")
if _ONLY_PBR_CONFIG:
    PBR_RUNS = [r for r in PBR_RUNS if r.id.startswith(_ONLY_PBR_CONFIG)]

# Which PBR kernel(s) to sweep -- (label, kwargs-for-pbr_matmul) pairs. "zero_idle"
# forces the scalar path (force_scalar=True) rather than the auto vec/scalar choice
# kernel='block' would otherwise make, for a fair scalar-vs-scalar comparison against
# nnzflat (which has no vectorized variant) -- see pbr_matmul's force_scalar docstring.
# One SLURM task per size already covers all matrices/features/pbr_runs/kernels (see
# SPDMM_ONLY_SIZE above); SPDMM_ONLY_KERNEL further restricts to one kernel if needed,
# matching the SPDMM_ONLY_SIZE/SPDMM_ONLY_MATRIX convention. Unset = all kernels.
_ALL_KERNEL_RUNS = [
    pytest.param(("zero_idle", dict(kernel='block', force_scalar=True)), id="zero_idle"),
    pytest.param(("nnzflat",   dict(kernel='nnzflat')),                  id="nnzflat"),
]
_ONLY_KERNEL = os.environ.get("SPDMM_ONLY_KERNEL")
KERNEL_RUNS = [k for k in _ALL_KERNEL_RUNS if k.id == _ONLY_KERNEL] if _ONLY_KERNEL else _ALL_KERNEL_RUNS

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
        benchmark.pedantic(algo_fn, args=(sparse_torch, x), rounds=50)
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
@mark.parametrize("kernel_run", KERNEL_RUNS)
@mark.parametrize("pbr_run",    PBR_RUNS)
@mark.parametrize("dtype",      DTYPES)
@mark.parametrize("features",   FEATURES)
@mark.parametrize("mat_config", MATRIX_CONFIGS)
def test_pbr_spmm(benchmark, bench_results, mat_config, features, dtype, pbr_run, kernel_run):
    pbr_cfg = pbr_run
    kernel_label, kernel_kwargs = kernel_run
    scipy_mat, _, x, size, profile = _prepare(mat_config, features, dtype)

    pbr_mat = csr_to_pbr(
        scipy_mat,
        block_rows=pbr_cfg["block_rows"],
        block_cols=pbr_cfg["block_cols"],
        min_nnz_per_block=pbr_cfg["min_nnz"],
    )
    pbr_mat = pbr_mat.to("cuda")

    # Warm-up (excluded from timing): one-time CUDA launch overhead. Steady-state
    # matmul is what we measure.
    try:
        pbr_matmul(pbr_mat, x, smem_size=pbr_cfg["smem_size"], **kernel_kwargs)
        torch.cuda.synchronize()
        benchmark.pedantic(
            pbr_matmul, args=(pbr_mat, x),
            kwargs={"smem_size": pbr_cfg["smem_size"], **kernel_kwargs}, rounds=50,
        )
    except torch.OutOfMemoryError:
        _cleanup()
        pytest.skip("OutOfMemory")
    finally:
        _cleanup()

    algo_name = (
        f"pbr_{pbr_cfg['block_rows']}x{pbr_cfg['block_cols']}"
        f"_min{pbr_cfg['min_nnz']}_sm{pbr_cfg['smem_size']}_{kernel_label}"
    )
    bench_results.append({
        "commit":      _COMMIT,
        "algorithm":   algo_name,
        "kernel":      kernel_label,
        "system_size": size,
        "features":    features,
        "dtype":       str(dtype),
        "profile":     profile,
        "time_ms":     benchmark.stats['mean'] * 1000,
        **pbr_cfg,
    })
