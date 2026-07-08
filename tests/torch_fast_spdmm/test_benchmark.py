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
    generate_random_sparse,
    generate_scattered_block_matrix,
)
from test_utils import scipy_csr_to_torch_csr
from torch_fast_spdmm import csr_to_pbr, pbr_matmul, spdmm_csr_naive


# ── Configuration tables ───────────────────────────────────────────────────────

FEATURES = [64, 128, 256, 512]
DTYPES   = [torch.float32]

# Square-matrix dimensions to sweep. Powers of two stay divisible by every
# block size used below.
SIZES = [16384, 65536, 262144]

# Matrix families, parametrised by system size. Each entry is
# (generator_func, size -> kwargs, family_name). Adding a family = one line.
MATRIX_FAMILIES = [
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
     lambda n: {"system_size": n, "block_size": 64, "diag_density": 0.017, "off_diag_density": 0.0000002},
     "diag_dd0.017"),
    (generate_diag_blocks_plus_noise,
     lambda n: {"system_size": n, "block_size": 64, "diag_density": 0.033, "off_diag_density": 0.0000002},
     "diag_dd0.033"),
    (generate_diag_blocks_plus_noise,
     lambda n: {"system_size": n, "block_size": 64, "diag_density": 0.05,  "off_diag_density": 0.0000002},
     "diag_dd0.05"),
    (generate_random_sparse,
     lambda n: {"system_size": n, "density": 0.000001},
     "random_d1e-6"),
]

# Cross size × family into the actual workload table.
MATRIX_CONFIGS = [
    pytest.param((gen, make_kwargs(n), name), id=f"{name}_{n}")
    for n in SIZES
    for (gen, make_kwargs, name) in MATRIX_FAMILIES
]

# PBR variants: (block_size, min_nnz, smem_size).
# smem_size controls the shared-memory tile (floats) per CTA; smaller values
# leave more of the 128 KB L1/smem pool as true L1 for X-column caching.
PBR_VARIANTS = [
    (bs, mn, sm)
    for bs, mn in [(2, 2), (2, 3), (4, 2), (4, 4), (8, 2), (8, 4)]
    for sm in [512, 1024, 2048]
]

PBR_CONFIGS = [
    pytest.param(
        {"block_rows": bs, "block_cols": bs, "min_nnz": mn, "smem_size": sm},
        id=f"pbr_{bs}x{bs}_min{mn}_sm{sm}",
    )
    for bs, mn, sm in PBR_VARIANTS
]

# Kernel variants swept per PBR config:
#   block    — one CUDA block per PBR block, atomicAdd commit (original)
#   rowgroup — one CUDA block per block-count-balanced group, row-major order
#   colgroup — same, but col-sorted order (smem_X reuse instead of smem_Y)
PBR_KERNELS = [
    pytest.param("block",    id="block"),
    pytest.param("rowgroup", id="rowgroup"),
    pytest.param("colgroup", id="colgroup"),
]

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
    wandb.Table logged to a single run. Run split by size (the Slurm array)
    this yields one run per size instead of 768 micro-runs, so the
    offline→server sync is near-instant and the UI is one sortable table.
    """
    rows: list[dict] = []
    yield rows

    if not rows:
        return

    table = wandb.Table(columns=_RESULT_COLUMNS)
    for r in rows:
        table.add_data(*(r.get(c) for c in _RESULT_COLUMNS))

    sizes = sorted({r["system_size"] for r in rows})
    run = wandb.init(
        project="torch-fast-spdmm",
        group=_COMMIT,
        name=f"{_COMMIT}_" + "_".join(str(s) for s in sizes),
        config={"commit": _COMMIT, "sizes": sizes},
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
@mark.parametrize("kernel",     PBR_KERNELS)
@mark.parametrize("pbr_cfg",    PBR_CONFIGS)
@mark.parametrize("dtype",      DTYPES)
@mark.parametrize("features",   FEATURES)
@mark.parametrize("mat_config", MATRIX_CONFIGS)
def test_pbr_spmm(benchmark, bench_results, mat_config, features, dtype, pbr_cfg, kernel):
    scipy_mat, _, x, size, profile = _prepare(mat_config, features, dtype)

    pbr_mat = csr_to_pbr(
        scipy_mat,
        block_rows=pbr_cfg["block_rows"],
        block_cols=pbr_cfg["block_cols"],
        min_nnz_per_block=pbr_cfg["min_nnz"],
    )
    pbr_mat = pbr_mat.to("cuda")

    try:
        benchmark(pbr_matmul, pbr_mat, x, smem_size=pbr_cfg["smem_size"], kernel=kernel)
    except torch.OutOfMemoryError:
        _cleanup()
        pytest.skip("OutOfMemory")
    finally:
        _cleanup()

    algo_name = (
        f"pbr_{pbr_cfg['block_rows']}x{pbr_cfg['block_cols']}"
        f"_min{pbr_cfg['min_nnz']}_sm{pbr_cfg['smem_size']}_{kernel}"
    )
    bench_results.append({
        "commit":      _COMMIT,
        "algorithm":   algo_name,
        "kernel":      kernel,
        "system_size": size,
        "features":    features,
        "dtype":       str(dtype),
        "profile":     profile,
        "time_ms":     benchmark.stats['mean'] * 1000,
        **pbr_cfg,
    })
