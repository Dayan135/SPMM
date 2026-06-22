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
from integration.conversions import scipy_csr_to_torch_csr
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
     lambda n: {"system_size": n, "block_size": 8, "diag_density": 0.001, "off_diag_density": 0.000001},
     "diag_dd0.001"),
    (generate_diag_blocks_plus_noise,
     lambda n: {"system_size": n, "block_size": 8, "diag_density": 0.002, "off_diag_density": 0.000001},
     "diag_dd0.002"),
    (generate_diag_blocks_plus_noise,
     lambda n: {"system_size": n, "block_size": 8, "diag_density": 0.003, "off_diag_density": 0.000001},
     "diag_dd0.003"),
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

# PBR variants. min_nnz_per_block sets the block-vs-COO split: min_nnz=1 keeps
# every populated block in the fast block kernel; raising it pushes more
# nonzeros into the COO-remainder kernel.
PBR_VARIANTS = [
    (2, 2), (2, 3),
    (4, 2), (4, 4),
    (8, 2), (8, 4),
]

PBR_CONFIGS = [
    pytest.param(
        {"block_rows": bs, "block_cols": bs, "min_nnz": mn},
        id=f"pbr_{bs}x{bs}_min{mn}",
    )
    for bs, mn in PBR_VARIANTS
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


def _log_wandb(request, config: dict, time_ms: float):
    run = wandb.init(
        project="torch-fast-spdmm",
        group=config.get("profile", ""),
        name=request.node.name,
        config=config,
        reinit=True,
    )
    run.log({"time_ms": time_ms})
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
def test_baseline_spmm(benchmark, request, mat_config, features, dtype, algo):
    algo_name, algo_fn = algo
    _, sparse_torch, x, size, profile = _prepare(mat_config, features, dtype)

    try:
        benchmark(algo_fn, sparse_torch, x)
    except torch.OutOfMemoryError:
        _cleanup()
        pytest.skip("OutOfMemory")
    finally:
        _cleanup()

    _log_wandb(request, {
        "algorithm":   algo_name,
        "system_size": size,
        "features":    features,
        "dtype":       str(dtype),
        "profile":     profile,
    }, benchmark.stats['mean'] * 1000)


# ── PBR benchmarks ─────────────────────────────────────────────────

@mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@mark.parametrize("pbr_cfg",    PBR_CONFIGS)
@mark.parametrize("dtype",      DTYPES)
@mark.parametrize("features",   FEATURES)
@mark.parametrize("mat_config", MATRIX_CONFIGS)
def test_pbr_spmm(benchmark, request, mat_config, features, dtype, pbr_cfg):
    scipy_mat, _, x, size, profile = _prepare(mat_config, features, dtype)

    pbr_mat = csr_to_pbr(
        scipy_mat,
        block_rows=pbr_cfg["block_rows"],
        block_cols=pbr_cfg["block_cols"],
        min_nnz_per_block=pbr_cfg["min_nnz"],
    )
    pbr_mat = pbr_mat.to("cuda")

    try:
        benchmark(pbr_matmul, pbr_mat, x)
    except torch.OutOfMemoryError:
        _cleanup()
        pytest.skip("OutOfMemory")
    finally:
        _cleanup()

    algo_name = f"pbr_{pbr_cfg['block_rows']}x{pbr_cfg['block_cols']}_min{pbr_cfg['min_nnz']}"
    _log_wandb(request, {
        "algorithm":   algo_name,
        "system_size": size,
        "features":    features,
        "dtype":       str(dtype),
        "profile":     profile,
        **pbr_cfg,
    }, benchmark.stats['mean'] * 1000)
