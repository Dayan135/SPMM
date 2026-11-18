"""GPU checks for the additive C++ timing selectors and launch paths."""
import numpy as np
import pytest
import scipy.sparse as sp
import torch
import torch_fast_spdmm as fast
import torch_fast_spdmm_core as core

from benchmark import bench_real as br
from benchmark import nnzflat_common as nc


def _input():
    # Four dense 32x32 diagonal tiles plus sparse off-diagonal remainder.
    blocks = sp.block_diag([np.ones((32, 32), np.float32) for _ in range(4)], format='csr')
    blocks[0, 96] = 1
    blocks.sort_indices()
    return blocks


def _check(reference, actual, expected):
    assert reference.check(actual)['bound_violations'] == 0
    assert reference.check(expected)['bound_violations'] == 0
    assert torch.allclose(actual, expected, atol=1e-3, rtol=1e-4)


@pytest.mark.parametrize('variant', ['block', 'chunkcols', 'nnzflat',
                                     'nnzflat_shared', 'nnzflat_onthefly'])
def test_pbr_timed_selector(variant):
    assert torch.cuda.is_available()
    csr = _input()
    x = torch.from_numpy(np.random.default_rng(1).standard_normal((128, 128), dtype=np.float32)).cuda()
    reference = br.Reference(csr, x.cpu().numpy())
    pbr = fast.csr_to_pbr(csr, block_rows=8, block_cols=8, min_nnz_per_block=4).to('cuda')
    meta = pbr._gpu_meta
    timed = core.PBRTimedSpMM(pbr.cols, pbr.rows, pbr.block_rows, pbr.block_cols,
                             meta.codes, meta.coords, meta.offsets, meta.data,
                             meta.rem_indptr, meta.rem_col_ind, meta.rem_vals,
                             x, 0, 2048, False, variant)
    _check(reference, timed.run(), fast.pbr_matmul(pbr, x, kernel=variant))
    names = nc.launched_kernels(torch, timed.run)
    expected = ('pbr_spmm_zero_idle_kernel', 'pbr_spmm_vec_kernel') if variant == 'block' \
        else (f'pbr_spmm_{variant}_kernel',)
    assert any(any(want in name for want in expected) for name in names), names


@pytest.mark.parametrize('launch_mode,variant,impl,fpt', [
    ('legacy', 'rowseg', None, 1), ('auto', 'rowseg', None, 1),
    ('auto', 'nnzflat', 'reference', 1), ('auto', 'nnzflat', 'template', 2),
])
def test_boo_timed_selector(launch_mode, variant, impl, fpt):
    assert torch.cuda.is_available()
    csr = _input()
    x = torch.from_numpy(np.random.default_rng(2).standard_normal((128, 128), dtype=np.float32)).cuda()
    reference = br.Reference(csr, x.cpu().numpy())
    boo = fast.csr_to_boo(csr, cluster='hem', target_nr=32, target_nc=32,
                          small_block_policy='keep').to('cuda')
    meta = boo._gpu_meta
    rowseg = variant == 'rowseg'
    core.set_boo_rowseg_fpt(fpt if rowseg else 1)
    timed = core.BOOTimedSpMM(
        boo.rows, boo.cols, boo.max_nr, boo.max_nc, boo.max_nnz_blk,
        meta.row_grp_ptr, meta.row_ids, meta.col_grp_ptr, meta.col_ids,
        meta.blk_row_grp, meta.blk_col_grp, meta.blk_nz_ptr, meta.blk_shift,
        meta.nz_idx, meta.nz_val, meta.rem_indptr, meta.rem_col_ind, meta.rem_vals,
        meta.tail_row, meta.tail_col, meta.tail_val, x, 256, 64, 0,
        rowseg, False, fpt if not rowseg else 1, impl != 'template', launch_mode)
    actual = timed.run()
    expected = fast.boo_matmul(boo, x, threads=256, p=64, row_seg=rowseg,
                               launch_mode=launch_mode, features_per_thread=fpt,
                               nnzflat_impl=impl)
    _check(reference, actual, expected)
    arm = dict(kernel=variant, impl=impl, p=64, threads=256, fpt=fpt)
    status, found = nc.verify_call(torch, timed.run, arm)
    assert status == 'verified', (status, found)


def test_boo_timed_execution_plan():
    assert torch.cuda.is_available()
    csr = _input()
    x = torch.from_numpy(np.random.default_rng(3).standard_normal((128, 128), dtype=np.float32)).cuda()
    boo = fast.csr_to_boo(csr, cluster='hem', target_nr=32, target_nc=32,
                          small_block_policy='keep').to('cuda')
    plan = fast.prepare_boo_cuda(boo, bucketed=True, precomputed_rows=True, p=64)
    meta = plan.meta
    timed = core.BOOTimedSpMM(
        boo.rows, boo.cols, boo.max_nr, boo.max_nc, boo.max_nnz_blk,
        meta.row_grp_ptr, meta.row_ids, meta.col_grp_ptr, meta.col_ids,
        meta.blk_row_grp, meta.blk_col_grp, meta.blk_nz_ptr, meta.blk_shift,
        meta.nz_idx, meta.nz_val, meta.rem_indptr, meta.rem_col_ind, meta.rem_vals,
        meta.tail_row, meta.tail_col, meta.tail_val, x, 256, 64, 0,
        True, False, 1, True, 'auto', plan.buckets, plan.row_ptr, plan.row_starts)
    reference = br.Reference(csr, x.cpu().numpy())
    _check(reference, timed.run(), fast.boo_matmul(boo, x, threads=256, p=64,
                                                   row_seg=True, launch_mode='auto',
                                                   execution_plan=plan))
    status, found = nc.verify_call(torch, timed.run,
                                   dict(kernel='rowseg', impl=None, p=64, threads=256, fpt=1))
    assert status == 'verified', (status, found)
