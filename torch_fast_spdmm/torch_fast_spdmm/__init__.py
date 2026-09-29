import types
from dataclasses import dataclass
import numpy as np
import scipy.sparse as sp
import torch
import torch_fast_spdmm_core as core

spdmm_csr_naive = core.spdmm_csr_naive

pbr_batched_matmul_cpu = core.pbr_batched_matmul_cpu


def _pin_to_device(arr, device, dtype=None):
    return torch.from_numpy(np.asarray(arr, dtype=dtype)).pin_memory().to(device, non_blocking=True)


def _pbr_to_method(self, device='cuda'):
    target_device = 'cpu' if device is None else str(torch.device(device))

    existing_meta = getattr(self, '_gpu_meta', None)
    current_device = str(existing_meta.codes.device) if existing_meta else 'cpu'

    if current_device == target_device:
        return self

    if target_device != 'cpu':
        self._gpu_meta = types.SimpleNamespace(
            codes       = _pin_to_device(self.block_codes,      device, dtype=np.int64),
            coords      = _pin_to_device(self.block_coords,     device),
            offsets     = _pin_to_device(self.block_offsets,    device),
            data        = _pin_to_device(self.block_data,       device),
            rem_indptr  = _pin_to_device(self.remainder_indptr,  device),
            rem_col_ind = _pin_to_device(self.remainder_col_ind, device),
            rem_vals    = _pin_to_device(self.remainder_data,    device),
        )
        torch.cuda.synchronize(device)
    else:
        self._gpu_meta = None

    return self


def _pbr_repr(self):
    return (
        f"PBRMatrix({self.rows}x{self.cols}, "
        f"blocks={self.accounted_blocks()}, "
        f"compressed_nnz={self.compressed_nnz()}, "
        f"remainder_nnz={self.remainder_nnz()}, "
        f"total_nnz={self.total_nnz})"
    )


# --- Inject the method into the C++ classes ---
for cls_name in ["PBRMatrixInt64Float", "PBRMatrixInt32Float",
                 "PBRMatrixInt64Double", "PBRMatrixInt32Double"]:
    if hasattr(core, cls_name):
        setattr(getattr(core, cls_name), "to", _pbr_to_method)
        setattr(getattr(core, cls_name), "__repr__", _pbr_repr)


def _build_cuda_dispatch():
    return {
        (torch.int32, torch.float32): core.pbr_spmm_cuda_i32_f32,
        (torch.int64, torch.float32): core.pbr_spmm_cuda_i64_f32,
        (torch.int32, torch.float64): core.pbr_spmm_cuda_i32_f64,
        (torch.int64, torch.float64): core.pbr_spmm_cuda_i64_f64,
    }

def _build_chunkcols_cuda_dispatch():
    return {
        (torch.int32, torch.float32): core.pbr_chunkcols_spmm_cuda_i32_f32,
        (torch.int64, torch.float32): core.pbr_chunkcols_spmm_cuda_i64_f32,
        (torch.int32, torch.float64): core.pbr_chunkcols_spmm_cuda_i32_f64,
        (torch.int64, torch.float64): core.pbr_chunkcols_spmm_cuda_i64_f64,
    }


def _build_nnzflat_cuda_dispatch():
    return {
        (torch.int32, torch.float32): core.pbr_nnzflat_spmm_cuda_i32_f32,
        (torch.int64, torch.float32): core.pbr_nnzflat_spmm_cuda_i64_f32,
        (torch.int32, torch.float64): core.pbr_nnzflat_spmm_cuda_i32_f64,
        (torch.int64, torch.float64): core.pbr_nnzflat_spmm_cuda_i64_f64,
    }


def _build_variant_cuda_dispatch(prefix):
    """(index_dtype, x_dtype) -> pybind symbol for a `pbr_<prefix>_spmm_cuda_*` family."""
    return {
        (torch.int32, torch.float32): getattr(core, f'pbr_{prefix}_spmm_cuda_i32_f32'),
        (torch.int64, torch.float32): getattr(core, f'pbr_{prefix}_spmm_cuda_i64_f32'),
        (torch.int32, torch.float64): getattr(core, f'pbr_{prefix}_spmm_cuda_i32_f64'),
        (torch.int64, torch.float64): getattr(core, f'pbr_{prefix}_spmm_cuda_i64_f64'),
    }


# nnzflat v5/v6, resurrected into this build so the 4-way comparison is not
# confounded by the 29-41% between-sweep variance SHOW_TO_ERAN sec. 4/7 measured.
_VARIANT_DISPATCH_CACHE = {}


def pbr_batched_matmul_variant_cuda(pbr_mat, x, prefix, csr_mode=0, smem_size=2048):
    meta = getattr(pbr_mat, '_gpu_meta', None)
    if meta is None:
        raise RuntimeError("PBRMatrix is not on GPU. Call .to('cuda') first.")
    if prefix not in _VARIANT_DISPATCH_CACHE:
        _VARIANT_DISPATCH_CACHE[prefix] = _build_variant_cuda_dispatch(prefix)
    kernel = _VARIANT_DISPATCH_CACHE[prefix][(meta.coords.dtype, x.dtype)]
    return kernel(
        pbr_mat.cols, pbr_mat.rows, pbr_mat.block_rows, pbr_mat.block_cols,
        meta.codes, meta.coords, meta.offsets, meta.data,
        meta.rem_indptr, meta.rem_col_ind, meta.rem_vals,
        x, csr_mode, smem_size,
    )


_CUDA_KERNEL_DISPATCH = None
_CHUNKCOLS_CUDA_KERNEL_DISPATCH = None
_NNZFLAT_CUDA_KERNEL_DISPATCH = None


def pbr_batched_matmul_cuda(pbr_mat, x: torch.Tensor, csr_mode: int = 0, smem_size: int = 2048) -> torch.Tensor:
    """
    Python wrapper that routes the GPU tensors to the correct PyBind11 CUDA function.
    csr_mode:  0=auto, 1=force_fast, 2=force_fallback
    smem_size: shared-memory tile size (floats) for the PBR block kernel; must be
               one of 512, 1024, 2048.
    """
    meta = getattr(pbr_mat, '_gpu_meta', None)
    if meta is None:
        raise RuntimeError("PBRMatrix is not on GPU. Call .to('cuda') first.")

    global _CUDA_KERNEL_DISPATCH
    if _CUDA_KERNEL_DISPATCH is None:
        _CUDA_KERNEL_DISPATCH = _build_cuda_dispatch()

    kernel = _CUDA_KERNEL_DISPATCH[(meta.coords.dtype, x.dtype)]
    return kernel(
        pbr_mat.cols,
        pbr_mat.rows,
        pbr_mat.block_rows,
        pbr_mat.block_cols,
        meta.codes,
        meta.coords,
        meta.offsets,
        meta.data,
        meta.rem_indptr,
        meta.rem_col_ind,
        meta.rem_vals,
        x,
        csr_mode,
        smem_size,
    )


def pbr_batched_matmul_chunkcols_cuda(pbr_mat, x: torch.Tensor, csr_mode: int = 0,
                                       smem_size: int = 2048) -> torch.Tensor:
    """Experimental: bounds per-thread row-work at CHUNK_K=4 active columns
    instead of the block kernel's up-to-BLOCK_COLS. See pbr_kernel.cu."""
    meta = getattr(pbr_mat, '_gpu_meta', None)
    if meta is None:
        raise RuntimeError("PBRMatrix is not on GPU. Call .to('cuda') first.")

    global _CHUNKCOLS_CUDA_KERNEL_DISPATCH
    if _CHUNKCOLS_CUDA_KERNEL_DISPATCH is None:
        _CHUNKCOLS_CUDA_KERNEL_DISPATCH = _build_chunkcols_cuda_dispatch()

    kernel = _CHUNKCOLS_CUDA_KERNEL_DISPATCH[(meta.coords.dtype, x.dtype)]
    return kernel(
        pbr_mat.cols,
        pbr_mat.rows,
        pbr_mat.block_rows,
        pbr_mat.block_cols,
        meta.codes,
        meta.coords,
        meta.offsets,
        meta.data,
        meta.rem_indptr,
        meta.rem_col_ind,
        meta.rem_vals,
        x,
        csr_mode,
        smem_size,
    )


def pbr_batched_matmul_nnzflat_cuda(pbr_mat, x: torch.Tensor, csr_mode: int = 0,
                                     smem_size: int = 2048) -> torch.Tensor:
    """Experimental: one thread per (nonzero, feature) work-item, nonzeros
    visited column-major. See pbr_kernel.cu."""
    meta = getattr(pbr_mat, '_gpu_meta', None)
    if meta is None:
        raise RuntimeError("PBRMatrix is not on GPU. Call .to('cuda') first.")

    global _NNZFLAT_CUDA_KERNEL_DISPATCH
    if _NNZFLAT_CUDA_KERNEL_DISPATCH is None:
        _NNZFLAT_CUDA_KERNEL_DISPATCH = _build_nnzflat_cuda_dispatch()

    kernel = _NNZFLAT_CUDA_KERNEL_DISPATCH[(meta.coords.dtype, x.dtype)]
    return kernel(
        pbr_mat.cols,
        pbr_mat.rows,
        pbr_mat.block_rows,
        pbr_mat.block_cols,
        meta.codes,
        meta.coords,
        meta.offsets,
        meta.data,
        meta.rem_indptr,
        meta.rem_col_ind,
        meta.rem_vals,
        x,
        csr_mode,
        smem_size,
    )


def pbr_matmul(pbr_mat, x: torch.Tensor, csr_mode: int = 0, smem_size: int = 2048,
               kernel: str = 'block') -> torch.Tensor:
    """
    Unified entry point for PBR MatMul.
    Dispatches to the correct backend based on x.device.
    csr_mode:  0=auto, 1=force_fast, 2=force_fallback (GPU only)
    smem_size: shared-memory tile size in floats; one of 512, 1024, 2048 (GPU only)
    kernel:    'block'     — one CUDA block per PBR block; atomicAdd to Y (default)
               'chunkcols' — experimental: bounds per-thread row-work at CHUNK_K=4
                             active columns instead of up-to-BLOCK_COLS
               'nnzflat'   — experimental: one thread per (nonzero, feature)
                             work-item, column-sorted
               'nnzflat_shared'   — nnzflat v6, shared position table
               'nnzflat_onthefly' — nnzflat v5, row-major on-the-fly derivation
    """
    if x.dim() not in (2, 3):
        raise ValueError(f"Expected 2D or 3D input, got {x.dim()}D")

    if x.is_cuda:
        meta = getattr(pbr_mat, '_gpu_meta', None)
        if meta is None:
            raise RuntimeError("[pbr_matmul] X is on GPU but PBRMatrix is not. Call .to('cuda') first.")
        if kernel in ('nnzflat_shared', 'nnzflat_onthefly'):
            y = pbr_batched_matmul_variant_cuda(pbr_mat, x, kernel,
                                                csr_mode=csr_mode, smem_size=smem_size)
        elif kernel == 'chunkcols':
            y = pbr_batched_matmul_chunkcols_cuda(pbr_mat, x, csr_mode=csr_mode, smem_size=smem_size)
        elif kernel == 'nnzflat':
            y = pbr_batched_matmul_nnzflat_cuda(pbr_mat, x, csr_mode=csr_mode, smem_size=smem_size)
        else:
            y = pbr_batched_matmul_cuda(pbr_mat, x, csr_mode=csr_mode, smem_size=smem_size)
    else:
        y = core.pbr_batched_matmul_cpu(pbr_mat, x)

    return y.squeeze(0) if x.dim() == 2 else y


def csr_to_pbr(mat: sp.csr_matrix | torch.Tensor,
               block_rows: int = 8, block_cols: int = 8,
               min_nnz_per_block: int = 4):
    if isinstance(mat, torch.Tensor):
        if mat.device.type != 'cpu':
            raise ValueError('Input tensor must be on CPU')

        return core.csr_to_pbr(mat.crow_indices().numpy(),
                               mat.col_indices().numpy(),
                               mat.values().cpu().numpy(),
                               mat.shape[0], mat.shape[1],
                               block_rows, block_cols,
                               min_nnz_per_block)
    elif isinstance(mat, sp.csr_matrix):
        return core.csr_to_pbr(mat.indptr, mat.indices, mat.data,
                               mat.shape[0], mat.shape[1],
                               block_rows, block_cols,
                               min_nnz_per_block)
    else:
        raise TypeError(f'Unsupported type {type(mat)}')


_PBR_TYPE_MAP = {
    'PBRMatrixInt64Float':  (np.int64,  np.float32),
    'PBRMatrixInt32Float':  (np.int32,  np.float32),
    'PBRMatrixInt64Double': (np.int64,  np.float64),
    'PBRMatrixInt32Double': (np.int32,  np.float64),
}

def pbr_to_csr(pbr_mat) -> sp.csr_matrix:
    index_dtype, scalar_dtype = _PBR_TYPE_MAP[type(pbr_mat).__name__]
    indptr  = np.zeros(pbr_mat.rows + 1, dtype=index_dtype)
    indices = np.zeros(pbr_mat.total_nnz, dtype=index_dtype)
    data    = np.zeros(pbr_mat.total_nnz, dtype=scalar_dtype)
    core.pbr_to_csr(pbr_mat, indptr, indices, data)
    return sp.csr_matrix((data, indices, indptr), shape=(pbr_mat.rows, pbr_mat.cols))


def pbr_analyze_csr(mat: sp.csr_matrix | torch.Tensor,
                    block_rows: int = 8, block_cols: int = 8,
                    min_nnz_coverage=1000):
    if isinstance(mat, torch.Tensor):
        if mat.device.type != 'cpu':
            raise ValueError('Input tensor must be on CPU')

        return core.pbr_analyze_csr(mat.crow_indices().numpy(),
                                    mat.col_indices().numpy(),
                                    mat.shape[0], mat.shape[1], mat._nnz(),
                                    block_rows, block_cols,
                                    min_nnz_coverage)
    elif isinstance(mat, sp.csr_matrix):
        return core.pbr_analyze_csr(mat.indptr, mat.indices,
                                    mat.shape[0], mat.shape[1], mat.nnz,
                                    block_rows, block_cols,
                                    min_nnz_coverage)
    else:
        raise TypeError(f'Unsupported type {type(mat)}')


@dataclass(frozen=True)
class BlockFillStats:  # nnz count per compressed block (popcount of its 64-bit code)
    min: int
    max: int
    mean: float
    max_possible: int


@dataclass(frozen=True)
class BlockStripeStats:  # number of compressed blocks per horizontal row-stripe
    min: int
    max: int
    mean: float


@dataclass(frozen=True)
class PBRInfo:  # full structural summary of a built PBRMatrix
    shape: tuple
    block_size: tuple
    total_nnz: int
    compressed_nnz: int
    remainder_nnz: int
    compressed_fraction: float
    num_blocks: int
    num_unique_patterns: int
    avg_reuse: float
    block_fill: BlockFillStats
    blocks_per_stripe: BlockStripeStats


def pbr_info(pbr_mat) -> PBRInfo:
    """
    Returns a PBRInfo dataclass summarising a built PBRMatrix.

    Derived entirely from the already-exposed fields on pbr_mat — no rebuild
    or C++ call needed.  Useful for quick debugging and comparing the effect
    of different block sizes / min_nnz_per_block thresholds.
    Use dataclasses.asdict(info) to convert to a plain dict.
    """
    codes = np.asarray(pbr_mat.block_codes, dtype=np.uint64)
    num_blocks      = len(codes)
    total_nnz       = int(pbr_mat.total_nnz)
    compressed_nnz  = int(pbr_mat.compressed_nnz())
    remainder_nnz   = int(pbr_mat.remainder_nnz())

    if num_blocks > 0:
        popcounts = np.array([bin(int(c)).count('1') for c in codes], dtype=np.int32)
        unique_codes, _ = np.unique(codes, return_counts=True)
        num_unique = int(len(unique_codes))

        coords_flat = np.asarray(pbr_mat.block_coords)
        block_rows_abs = coords_flat.reshape(-1, 2)[:, 0]
        _, stripe_counts = np.unique(block_rows_abs // pbr_mat.block_rows, return_counts=True)

        block_fill = BlockFillStats(
            min=int(popcounts.min()),
            max=int(popcounts.max()),
            mean=float(popcounts.mean()),
            max_possible=pbr_mat.block_rows * pbr_mat.block_cols,
        )
        blocks_per_stripe = BlockStripeStats(
            min=int(stripe_counts.min()),
            max=int(stripe_counts.max()),
            mean=float(stripe_counts.mean()),
        )
    else:
        num_unique        = 0
        block_fill        = BlockFillStats(min=0, max=0, mean=0.0,
                                           max_possible=pbr_mat.block_rows * pbr_mat.block_cols)
        blocks_per_stripe = BlockStripeStats(min=0, max=0, mean=0.0)

    return PBRInfo(
        shape=(pbr_mat.rows, pbr_mat.cols),
        block_size=(pbr_mat.block_rows, pbr_mat.block_cols),
        total_nnz=total_nnz,
        compressed_nnz=compressed_nnz,
        remainder_nnz=remainder_nnz,
        compressed_fraction=compressed_nnz / total_nnz if total_nnz > 0 else 0.0,
        num_blocks=num_blocks,
        num_unique_patterns=num_unique,
        avg_reuse=num_blocks / num_unique if num_unique > 0 else 0.0,
        block_fill=block_fill,
        blocks_per_stripe=blocks_per_stripe,
    )


from .boo import (  # noqa: E402  -- must follow the core import above
    csr_to_boo, boo_to_csr, boo_info, boo_matmul, boo_arena_bytes, BOOInfo,
    SMALL_BLOCK_POLICIES, NZ_ORDERS,
)
from .clustering import cluster_axes, cluster_axis, CLUSTER_METHODS  # noqa: E402

__all__ = [
    'spdmm_csr_naive',

    'csr_to_boo',
    'boo_to_csr',
    'boo_info',
    'boo_matmul',
    'boo_arena_bytes',
    'BOOInfo',
    'SMALL_BLOCK_POLICIES',
    'NZ_ORDERS',
    'cluster_axes',
    'cluster_axis',
    'CLUSTER_METHODS',


    'csr_to_pbr',
    'pbr_to_csr',
    'pbr_analyze_csr',
    'pbr_info',
    'PBRInfo',
    'BlockFillStats',
    'BlockStripeStats',
    'pbr_matmul',
    'pbr_batched_matmul_cpu',
    'pbr_batched_matmul_cuda',
    'pbr_batched_matmul_chunkcols_cuda',
    'pbr_batched_matmul_nnzflat_cuda',
    'pbr_batched_matmul_variant_cuda',
]

from .boo_plan import BOOExecutionPlan, prepare_boo_cuda
__all__ += ['BOOExecutionPlan', 'prepare_boo_cuda']

from .boo_rowexcl import BOORowExclusivePlan, prepare_boo_rowexcl, boo_rowexcl_matmul
__all__ += ['BOORowExclusivePlan', 'prepare_boo_rowexcl', 'boo_rowexcl_matmul']
