"""Python API for the BOO (Blocked COO) format.

Build:  ``boo = csr_to_boo(A, cluster='hem', target_nr=32, target_nc=32)``
Use:    ``boo.to('cuda'); y = boo_matmul(boo, x)``
Inspect: ``boo_info(boo)`` -- the offline metrics the M1 gate reads.

See ``BOO_FORMAT.md`` for the format itself.
"""

from __future__ import annotations

import types
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
import scipy.sparse as sp
import torch
import torch_fast_spdmm_core as core

from .clustering import cluster_axes

__all__ = [
    'csr_to_boo', 'boo_to_csr', 'boo_info', 'boo_matmul', 'BOOInfo',
    'SMALL_BLOCK_POLICIES', 'NZ_ORDERS',
]

SMALL_BLOCK_POLICIES = ('keep', 'merge', 'remainder', 'tail_batch')
NZ_ORDERS = ('rowmajor', 'colmajor')

_POLICY_CODE = {'keep': 0, 'merge': 1, 'remainder': 2, 'tail_batch': 3}
_ORDER_CODE = {'rowmajor': 0, 'colmajor': 1}

_BOO_TYPE_MAP = {
    'BOOMatrixInt64Float':  (np.int64, np.float32),
    'BOOMatrixInt32Float':  (np.int32, np.float32),
    'BOOMatrixInt64Double': (np.int64, np.float64),
    'BOOMatrixInt32Double': (np.int32, np.float64),
}

# SM86 (RTX 3090).  See BOO_FORMAT.md sec. 9.2 -- calibrated against the
# measured "11 blocks/SM at smem_size=2048" in SHOW_TO_ERAN.md sec. 3.
_SMEM_PER_SM = 102400          # 100 KB
_DRIVER_SMEM_PER_CTA = 1024
_MAX_THREADS_PER_SM = 1536     # 48 warps -- NOT 2048; cc 8.6 caps here
_MAX_CTAS_PER_SM = 16


def _pin_to_device(arr, device, dtype=None):
    return torch.from_numpy(np.ascontiguousarray(arr, dtype=dtype)).pin_memory().to(
        device, non_blocking=True)


def _boo_to_method(self, device='cuda'):
    """Stage the format onto the GPU, mirroring PBRMatrix.to()."""
    target = 'cpu' if device is None else str(torch.device(device))
    existing = getattr(self, '_gpu_meta', None)
    current = str(existing.nz_val.device) if existing else 'cpu'
    if current == target:
        return self

    if target != 'cpu':
        # row_ids/col_ids are int32 unconditionally on the GPU regardless of the
        # host index_t -- halves index traffic and the pybind symbol count.
        self._gpu_meta = types.SimpleNamespace(
            row_grp_ptr=_pin_to_device(self.row_grp_ptr, device, np.int32),
            row_ids=_pin_to_device(self.row_ids, device, np.int32),
            col_grp_ptr=_pin_to_device(self.col_grp_ptr, device, np.int32),
            col_ids=_pin_to_device(self.col_ids, device, np.int32),
            blk_row_grp=_pin_to_device(self.blk_row_grp, device, np.int32),
            blk_col_grp=_pin_to_device(self.blk_col_grp, device, np.int32),
            blk_nz_ptr=_pin_to_device(self.blk_nz_ptr, device, np.int32),
            blk_shift=_pin_to_device(self.blk_shift, device, np.uint8),
            # Bit-level int16 VIEW, not a cast: nz_idx is uint16 and values above
            # 32767 are legal, so converting would overflow.  The kernel reads
            # int16_t and reinterprets back to uint16_t.
            nz_idx=_pin_to_device(np.ascontiguousarray(self.nz_idx).view(np.int16), device),
            nz_val=_pin_to_device(self.nz_val, device),
            rem_indptr=_pin_to_device(self.rem_indptr, device, np.int32),
            rem_col_ind=_pin_to_device(self.rem_col_ind, device, np.int32),
            rem_vals=_pin_to_device(self.rem_vals, device),
            tail_row=_pin_to_device(self.tail_row, device, np.int32),
            tail_col=_pin_to_device(self.tail_col, device, np.int32),
            tail_val=_pin_to_device(self.tail_val, device),
        )
        torch.cuda.synchronize(device)
    else:
        self._gpu_meta = None
    return self


for _cls_name in _BOO_TYPE_MAP:
    if hasattr(core, _cls_name):
        setattr(getattr(core, _cls_name), 'to', _boo_to_method)


# ---------------------------------------------------------------------------
# build / convert
# ---------------------------------------------------------------------------

def csr_to_boo(mat, cluster: str = 'hem', target_nr: int = 32, target_nc: int = 32,
               max_nr: Optional[int] = None, max_nc: Optional[int] = None,
               max_block_nnz: Optional[int] = None,
               small_block_policy: str = 'keep', small_block_threshold: int = 0,
               nz_order: str = 'rowmajor', seed: int = 0,
               symmetric: Optional[bool] = None,
               row_labels: Optional[np.ndarray] = None,
               col_labels: Optional[np.ndarray] = None):
    """Cluster both axes, then assemble the BOO format.

    `row_labels`/`col_labels` bypass the clustering step entirely -- useful for
    reproducing a specific grid, and for the PBR-parity check.
    """
    if small_block_policy not in SMALL_BLOCK_POLICIES:
        raise ValueError(f'small_block_policy must be one of {SMALL_BLOCK_POLICIES}')
    if nz_order not in NZ_ORDERS:
        raise ValueError(f'nz_order must be one of {NZ_ORDERS}')

    if isinstance(mat, torch.Tensor):
        if mat.device.type != 'cpu':
            raise ValueError('Input tensor must be on CPU')
        csr = sp.csr_matrix(
            (mat.values().cpu().numpy(), mat.col_indices().numpy(), mat.crow_indices().numpy()),
            shape=tuple(mat.shape))
    elif isinstance(mat, sp.csr_matrix):
        csr = mat
    else:
        raise TypeError(f'Unsupported type {type(mat)}')

    # Caps default to the targets: the clustering aims for `target` and is not
    # allowed to overshoot, so the arena stays predictable.
    if max_nr is None:
        max_nr = target_nr
    if max_nc is None:
        max_nc = target_nc

    if row_labels is None or col_labels is None:
        row_labels, col_labels = cluster_axes(
            csr, method=cluster, target_nr=target_nr, target_nc=target_nc,
            max_nr=max_nr, max_nc=max_nc, max_block_nnz=max_block_nnz,
            seed=seed, symmetric=symmetric)

    return core.csr_to_boo(
        csr.indptr, csr.indices, csr.data,
        csr.shape[0], csr.shape[1],
        np.ascontiguousarray(row_labels, dtype=np.int32),
        np.ascontiguousarray(col_labels, dtype=np.int32),
        _POLICY_CODE[small_block_policy],
        int(small_block_threshold),
        _ORDER_CODE[nz_order],
    )


def boo_to_csr(boo_mat) -> sp.csr_matrix:
    index_dtype, scalar_dtype = _BOO_TYPE_MAP[type(boo_mat).__name__]
    indptr = np.zeros(boo_mat.rows + 1, dtype=index_dtype)
    indices = np.zeros(boo_mat.total_nnz, dtype=index_dtype)
    data = np.zeros(boo_mat.total_nnz, dtype=scalar_dtype)
    core.boo_to_csr(boo_mat, indptr, indices, data)
    return sp.csr_matrix((data, indices, indptr), shape=(boo_mat.rows, boo_mat.cols))


_BOO_CUDA_DISPATCH = None


def _build_boo_cuda_dispatch():
    return {torch.float32: core.boo_spmm_cuda_f32,
            torch.float64: core.boo_spmm_cuda_f64}


def boo_matmul(boo_mat, x: torch.Tensor, threads: int = 256, p: int = 64,
               csr_mode: int = 0, row_seg: Optional[bool] = None) -> torch.Tensor:
    """Y = A @ X.  Dispatches on x.device.

    `threads` in {128, 256, 512} and `p` in {8..128} are the two shared-memory
    levers (BOO_FORMAT.md sec. 9); both are swept rather than fixed. Measured
    optimum for the row-segmented kernel is threads=256, p=64.

    `row_seg` selects the register-accumulating Step B; it defaults to True
    whenever the matrix was built rowmajor, and is ~1.6x faster than the
    shared-atomic variant (BOO_FORMAT.md sec. 13).
    """
    if x.dim() not in (2, 3):
        raise ValueError(f'Expected 2D or 3D input, got {x.dim()}D')

    if not x.is_cuda:
        y = core.boo_batched_matmul_cpu(boo_mat, x)
        return y.squeeze(0) if x.dim() == 2 else y

    meta = getattr(boo_mat, '_gpu_meta', None)
    if meta is None:
        raise RuntimeError("[boo_matmul] X is on GPU but BOOMatrix is not. Call .to('cuda') first.")

    global _BOO_CUDA_DISPATCH
    if _BOO_CUDA_DISPATCH is None:
        _BOO_CUDA_DISPATCH = _build_boo_cuda_dispatch()
    if x.dtype not in _BOO_CUDA_DISPATCH:
        raise TypeError(f'BOO CUDA kernel supports float32/float64, got {x.dtype}')

    if row_seg is None:
        # The register-accumulating kernel needs nz_idx sorted by local_row,
        # which only rowmajor guarantees.
        row_seg = int(boo_mat.nz_order) == _ORDER_CODE['rowmajor']
    elif row_seg and int(boo_mat.nz_order) != _ORDER_CODE['rowmajor']:
        raise ValueError('row_seg=True requires a matrix built with nz_order="rowmajor"')

    y = _BOO_CUDA_DISPATCH[x.dtype](
        boo_mat.rows, boo_mat.cols,
        int(boo_mat.max_nr), int(boo_mat.max_nc), int(boo_mat.max_nnz_blk),
        meta.row_grp_ptr, meta.row_ids, meta.col_grp_ptr, meta.col_ids,
        meta.blk_row_grp, meta.blk_col_grp, meta.blk_nz_ptr, meta.blk_shift,
        meta.nz_idx, meta.nz_val,
        meta.rem_indptr, meta.rem_col_ind, meta.rem_vals,
        meta.tail_row, meta.tail_col, meta.tail_val,
        x, threads, p, csr_mode, row_seg,
    )
    return y.squeeze(0) if x.dim() == 2 else y


# ---------------------------------------------------------------------------
# offline metrics -- what the M1 gate reads
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BOOInfo:
    shape: Tuple[int, int]
    total_nnz: int
    compressed_nnz: int
    remainder_nnz: int
    tail_nnz: int
    # Strict: fraction of nnz encoded as a block (the uint16 nz_idx form).
    # Under BOO_REMAINDER this doubles as "did any nnz leave BOO for the
    # shared CSR kernel" (SHOW_TO_ERAN sec. 2's confound); under
    # BOO_TAIL_BATCH it does NOT -- nothing left BOO, so read boo_fraction
    # instead for that question.
    compressed_fraction: float
    # Fraction of nnz processed by a BOO-family kernel (block OR tail), i.e.
    # 1.0 unless BOO_REMAINDER actually spilled to the shared CSR kernel.
    boo_fraction: float

    num_blocks: int
    num_row_groups: int
    num_col_groups: int

    n_r_mean: float
    n_r_max: int
    n_c_mean: float
    n_c_max: int
    nnz_per_block_mean: float
    nnz_per_block_max: int
    nnz_per_block_hist: dict

    # The traffic algebra of BOO_FORMAT.md sec. 3.  Bigger row groups cut
    # x_loads; bigger column groups cut y_writes.
    x_loads: int
    y_writes: int
    arithmetic_intensity: float

    # Derived launch geometry, so an occupancy tax is visible before any GPU run.
    scalar_bytes: int
    p: int
    threads: int
    arena_bytes: int
    ctas_per_sm: int
    occupancy_pct: float


def _occupancy(arena_bytes: int, threads: int) -> Tuple[int, float]:
    by_smem = _SMEM_PER_SM // (arena_bytes + _DRIVER_SMEM_PER_CTA)
    by_warps = _MAX_THREADS_PER_SM // threads
    ctas = max(0, min(by_smem, by_warps, _MAX_CTAS_PER_SM))
    return ctas, 100.0 * ctas * threads / _MAX_THREADS_PER_SM


def boo_arena_bytes(max_nr: int, max_nc: int, max_nnz_blk: int, p: int,
                    scalar_bytes: int, row_seg: bool = True) -> int:
    """Bytes of dynamic shared memory the launcher must request.

    Sized from the GLOBAL maxima, because `extern __shared__` takes one size for
    the whole grid -- every CTA gets the same allocation (BOO_FORMAT.md 9.4).

    The row-segmented kernel drops smem_Y entirely (each thread accumulates a
    row in a register) at the cost of a small row-start table, so it is
    materially cheaper -- which is part of why it wins.
    """
    data = max_nnz_blk * (scalar_bytes + 2)          # nz_val + nz_idx
    if row_seg:
        ids = 4 * (max_nr + max_nc) + 4 * (max_nr + 1)   # + row_start
        tiles = scalar_bytes * p * max_nc                # smem_X only
    else:
        ids = 4 * (max_nr + max_nc)
        tiles = scalar_bytes * p * (max_nr + max_nc)     # smem_X + smem_Y
    return data + ids + tiles


def boo_info(boo_mat, p: int = 64, threads: int = 256,
             row_seg: bool = True) -> BOOInfo:
    """Structural summary, derived entirely from the built format -- no rebuild."""
    _, scalar_dtype = _BOO_TYPE_MAP[type(boo_mat).__name__]
    scalar_bytes = np.dtype(scalar_dtype).itemsize

    row_grp_ptr = np.asarray(boo_mat.row_grp_ptr, dtype=np.int64)
    col_grp_ptr = np.asarray(boo_mat.col_grp_ptr, dtype=np.int64)
    blk_row_grp = np.asarray(boo_mat.blk_row_grp, dtype=np.int64)
    blk_col_grp = np.asarray(boo_mat.blk_col_grp, dtype=np.int64)
    blk_nz_ptr = np.asarray(boo_mat.blk_nz_ptr, dtype=np.int64)

    rg_size = np.diff(row_grp_ptr)
    cg_size = np.diff(col_grp_ptr)
    nb = int(blk_row_grp.size)

    if nb:
        n_r = rg_size[blk_row_grp]
        n_c = cg_size[blk_col_grp]
        nnz_per_block = np.diff(blk_nz_ptr)
        x_loads = int(n_c.sum())
        y_writes = int(n_r.sum())
        # Log2 buckets: the long tail of 1-2 nonzero blocks is exactly what the
        # small-block policy has to absorb, so it must be visible, not averaged.
        buckets = np.where(nnz_per_block > 0,
                           np.floor(np.log2(np.maximum(nnz_per_block, 1))).astype(int), 0)
        hist = {f'{1 << b}-{(1 << (b + 1)) - 1}': int(count)
                for b, count in zip(*np.unique(buckets, return_counts=True))}
    else:
        n_r = n_c = nnz_per_block = np.zeros(0, dtype=np.int64)
        x_loads = y_writes = 0
        hist = {}

    compressed = int(boo_mat.compressed_nnz())
    tail = int(boo_mat.tail_nnz())
    total = int(boo_mat.total_nnz)
    arena = boo_arena_bytes(int(boo_mat.max_nr), int(boo_mat.max_nc),
                            int(boo_mat.max_nnz_blk), p, scalar_bytes, row_seg)
    ctas, occupancy = _occupancy(arena, threads)

    return BOOInfo(
        shape=(int(boo_mat.rows), int(boo_mat.cols)),
        total_nnz=total,
        compressed_nnz=compressed,
        remainder_nnz=int(boo_mat.remainder_nnz()),
        tail_nnz=tail,
        compressed_fraction=compressed / total if total else 0.0,
        boo_fraction=(compressed + tail) / total if total else 0.0,
        num_blocks=nb,
        num_row_groups=int(rg_size.size),
        num_col_groups=int(cg_size.size),
        n_r_mean=float(n_r.mean()) if nb else 0.0,
        n_r_max=int(n_r.max()) if nb else 0,
        n_c_mean=float(n_c.mean()) if nb else 0.0,
        n_c_max=int(n_c.max()) if nb else 0,
        nnz_per_block_mean=float(nnz_per_block.mean()) if nb else 0.0,
        nnz_per_block_max=int(nnz_per_block.max()) if nb else 0,
        nnz_per_block_hist=hist,
        x_loads=x_loads,
        y_writes=y_writes,
        arithmetic_intensity=compressed / x_loads if x_loads else 0.0,
        scalar_bytes=scalar_bytes,
        p=p,
        threads=threads,
        arena_bytes=arena,
        ctas_per_sm=ctas,
        occupancy_pct=occupancy,
    )
