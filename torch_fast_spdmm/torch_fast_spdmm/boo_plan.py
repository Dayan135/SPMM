"""Explicit, reusable CUDA execution plans for BOO optimization experiments."""

from dataclasses import dataclass
import time
import types

import numpy as np
import torch


@dataclass(frozen=True)
class BOOExecutionPlan:
    """Prepared metadata; tied to one matrix's current CUDA staging.

    Treat tensors as immutable. Rebuild after moving the matrix to another device
    or back to CPU. Preparation synchronizes its device; matmul never prepares
    tables or synchronizes. ``extra_device_bytes`` includes duplicated block
    storage retained alongside the original format for bucketed plans.
    """

    matrix: object
    source_meta: object
    meta: object
    buckets: tuple
    row_ptr: torch.Tensor
    row_starts: torch.Tensor
    p: int
    bucketed: bool
    precomputed_rows: bool
    build_ms: float
    staging_ms: float
    extra_device_bytes: int


def _plan_arrays(boo, bucketed, precomputed_rows, p):
    """CPU construction, separate from transfers and the measured multiply."""
    nr = np.diff(np.asarray(boo.row_grp_ptr, dtype=np.int64))[boo.blk_row_grp]
    nc = np.diff(np.asarray(boo.col_grp_ptr, dtype=np.int64))[boo.blk_col_grp]
    old_ptr = np.asarray(boo.blk_nz_ptr, dtype=np.int64)
    nnz = np.diff(old_ptr)
    count = len(nnz)
    scalar_bytes = np.asarray(boo.nz_val).dtype.itemsize
    labels = np.zeros(count, dtype=np.int64)
    if bucketed and count:
        required = (scalar_bytes + 2) * nnz + 4 * (2 * nr + nc + 1) + scalar_bytes * p * nc + 16
        cuts = np.unique(np.quantile(required, [0.5, 0.9], method='nearest'))
        labels = np.searchsorted(cuts, required, side='left')
    order = np.argsort(labels, kind='stable')
    arrays = {}
    if bucketed:
        new_ptr = np.r_[0, np.cumsum(nnz[order])]
        positions = np.arange(int(old_ptr[-1])) + np.repeat(old_ptr[order] - new_ptr[:-1], nnz[order])
        for name in ('blk_row_grp', 'blk_col_grp', 'blk_shift'):
            arrays[name] = np.asarray(getattr(boo, name))[order]
        arrays['blk_nz_ptr'] = new_ptr
        arrays['nz_idx'] = np.asarray(boo.nz_idx)[positions]
        arrays['nz_val'] = np.asarray(boo.nz_val)[positions]
    buckets = []
    start = 0
    for label in np.unique(labels):
        selected = order[labels[order] == label]
        buckets.extend((start, len(selected), int(nr[selected].max()),
                        int(nc[selected].max()), int(nnz[selected].max())))
        start += len(selected)
    if not bucketed and count:
        buckets = [0, count, int(boo.max_nr), int(boo.max_nc), int(boo.max_nnz_blk)]
    row_ptr = row_starts = np.empty(0, dtype=np.int32)
    if precomputed_rows and count:
        row_ptr = np.r_[0, np.cumsum(nr[order] + 1)]
        idx = arrays.get('nz_idx', np.asarray(boo.nz_idx))
        shifts = arrays.get('blk_shift', np.asarray(boo.blk_shift))
        ptr = arrays.get('blk_nz_ptr', old_ptr)
        local_rows = idx.astype(np.uint16) >> np.repeat(shifts, nnz[order])
        slots = np.repeat(row_ptr[:-1], nnz[order]) + local_rows
        counts = np.bincount(slots, minlength=int(row_ptr[-1]))
        # Exclusive prefix sum, reset to zero at each block's first row.
        row_starts = np.cumsum(counts) - counts - np.repeat(ptr[:-1], nr[order] + 1)
        if max(int(row_ptr[-1]), int(row_starts.max(initial=0))) > np.iinfo(np.int32).max:
            raise ValueError('BOO row tables must fit int32')
        row_ptr, row_starts = row_ptr.astype(np.int32), row_starts.astype(np.int32)
    return arrays, tuple(buckets), row_ptr, row_starts


def prepare_boo_cuda(boo, *, bucketed=False, precomputed_rows=False, p=64):
    """Prepare an opt-in row-segmented plan outside timed/repeated matmul calls.

    Buckets split estimated shared-memory requirements at the median and 90th
    percentile (at most three nonempty grids). Each grid uses its own maxima.
    ``p`` is the requested feature tile used to choose buckets; use the same
    value in boo_matmul. Smaller feature counts may still clamp the CUDA tile.
    """
    from .boo import _pin_to_device

    source = getattr(boo, '_gpu_meta', None)
    if source is None:
        raise ValueError("Call boo.to('cuda') before preparing an execution plan")
    if int(boo.nz_order) != 0:
        raise ValueError('BOO execution plans require rowmajor nonzeros')
    if p not in (8, 16, 32, 64, 128):
        raise ValueError('p must be one of 8, 16, 32, 64, 128')
    start = time.perf_counter()
    arrays, buckets, row_ptr, row_starts = _plan_arrays(boo, bucketed, precomputed_rows, p)
    build_ms = (time.perf_counter() - start) * 1000
    device = source.nz_val.device
    torch.cuda.synchronize(device)
    start = time.perf_counter()
    meta = types.SimpleNamespace(**vars(source))
    extra = []
    for name, array in arrays.items():
        if name == 'nz_idx':
            array = np.ascontiguousarray(array).view(np.int16)
        elif name != 'nz_val' and name != 'blk_shift':
            array = np.asarray(array, dtype=np.int32)
        tensor = _pin_to_device(array, device)
        setattr(meta, name, tensor)
        extra.append(tensor)
    row_ptr, row_starts = (_pin_to_device(a, device) for a in (row_ptr, row_starts))
    extra.extend((row_ptr, row_starts))
    torch.cuda.synchronize(device)
    staging_ms = (time.perf_counter() - start) * 1000
    return BOOExecutionPlan(boo, source, meta, buckets, row_ptr, row_starts, p,
                            bucketed, precomputed_rows, build_ms, staging_ms,
                            sum(t.numel() * t.element_size() for t in extra))
