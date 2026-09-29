"""Opt-in row-exclusive BOO execution; immutable auxiliary scheduling tables.

FP32, contiguous 2D CUDA X; rowmajor BOO with a CSR remainder (or no remainder).
Preparation is explicit and outside all repeated matmul/timing calls.
"""
from dataclasses import dataclass
import time

import numpy as np
import torch
import torch_fast_spdmm_core as core


@dataclass(frozen=True)
class BOORowExclusivePlan:
    matrix: object
    source_meta: object
    core_plan: object
    stats: dict
    unit_nnz: np.ndarray
    unit_chunks: np.ndarray
    schedule: tuple

    def _check(self):
        if getattr(self.matrix, '_gpu_meta', None) is not self.source_meta:
            raise ValueError('Row-exclusive plan no longer matches the matrix staging')

    def bind(self, x):
        """Bind resident X and allocate persistent Y/events, outside timing."""
        self._check()
        return self.core_plan.bind(x)


def prepare_boo_rowexcl(boo, *, slab=32, max_cta_nnz=2048, max_cta_rows=32,
                       smem_bytes=8192):
    """Keep BOO unchanged; split row groups only into disjoint row slices.

    The effective nnz cap is max(max_cta_nnz, maximum single-row degree): a
    single row cannot be divided without violating exclusive ownership.
    Whole column groups are packed into shared-memory chunks; the budget rises
    to fit one column group when needed (at most the default 48 KiB limit).
    Empty rows remain in the schedule and are explicitly written by the kernel.
    """
    source = getattr(boo, '_gpu_meta', None)
    if source is None:
        raise ValueError("Call boo.to('cuda') before preparing the row-exclusive plan")
    if source.nz_val.dtype != torch.float32:
        raise TypeError('Row-exclusive BOO currently supports FP32 values')
    if int(boo.nz_order) != 0:
        raise ValueError('Row-exclusive BOO requires rowmajor nonzeros')
    if int(boo.tail_nnz()):
        raise ValueError('Use the CSR-remainder or keep policy; COO tails are not supported')
    if slab not in (16, 32, 64):
        raise ValueError('slab must be 16, 32 or 64')
    t0 = time.perf_counter()
    names = ('row_grp_ptr', 'row_ids', 'col_grp_ptr', 'col_ids', 'blk_row_grp',
             'blk_col_grp', 'blk_nz_ptr', 'blk_shift', 'nz_idx', 'rem_indptr')
    host = []
    for name in names:
        if name == 'nz_idx':
            array = np.ascontiguousarray(getattr(boo, name), dtype=np.uint16).view(np.int16)
        else:
            array = np.ascontiguousarray(getattr(boo, name), dtype=np.uint8 if name == 'blk_shift' else np.int32)
        host.append(torch.from_numpy(array))
    arrays, unit_nnz, stats = core.build_boo_rowexcl_schedule(
        host, max_cta_nnz, max_cta_rows, slab, smem_bytes)
    stats = dict(stats)
    nnz = unit_nnz.numpy().copy()
    groups = arrays[0].numpy()
    chunks = np.diff(arrays[3].numpy())[groups].copy()
    stats.update(build_ms=1000*(time.perf_counter()-t0),
                 cta_nnz_min=int(nnz.min(initial=np.iinfo(np.int32).max)) if nnz.size else 0,
                 cta_nnz_median=float(np.median(nnz)) if nnz.size else 0.,
                 cta_nnz_p90=float(np.quantile(nnz, .9)) if nnz.size else 0.,
                 cta_nnz_p99=float(np.quantile(nnz, .99)) if nnz.size else 0.,
                 cta_nnz_max=int(nnz.max(initial=0)),
                 chunked_fraction=float(np.mean(chunks > 1)) if chunks.size else 0.,
                 chunks_per_cta_max=int(chunks.max(initial=0)))
    assert stats['cta_nnz_max'] <= stats['effective_nnz_cap']
    device = source.nz_val.device
    torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    gpu = tuple(a.pin_memory().to(device, non_blocking=True) for a in arrays)
    sources = [getattr(source, n) for n in ('row_grp_ptr', 'row_ids', 'rem_indptr',
               'rem_col_ind', 'blk_shift', 'nz_idx', 'nz_val', 'rem_vals')]
    torch.cuda.synchronize(device)
    plan = core.BOORowExclusivePlan(boo.rows, boo.cols, list(gpu), sources, slab, stats['smem_bytes'])
    stats.update(staging_ms=1000*(time.perf_counter()-t0),
                 extra_device_bytes=sum(a.numel()*a.element_size() for a in gpu),
                 **dict(plan.info()))
    return BOORowExclusivePlan(boo, source, plan, stats, nnz, chunks, gpu)


def boo_rowexcl_matmul(plan, x):
    """Full Python API call: allocate uninitialized Y, then one kernel launch."""
    if not isinstance(plan, BOORowExclusivePlan):
        raise TypeError('Expected a prepared BOORowExclusivePlan')
    plan._check()
    return plan.core_plan.matmul(x)
