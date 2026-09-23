"""Execution-plan gates: block partition, row tables, streams and mixed policies."""

import numpy as np
import pytest
import scipy.sparse as sp
import torch

from torch_fast_spdmm import boo_matmul, csr_to_boo, prepare_boo_cuda
from torch_fast_spdmm.boo_plan import _plan_arrays

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')


@pytest.mark.parametrize('bucketed', [False, True])
@pytest.mark.parametrize('high_bit', [False, True])
def test_plan_partition_and_row_boundaries(bucketed, high_bit):
    # Full 256x256 groups exercise uint16 packed coordinates above signed int16.
    group = 256 if high_bit else 8
    mat = sp.random(group * 2 + 1, group * 2 + 3, density=0.015 if high_bit else 0.2,
                    format='csr', dtype=np.float64, random_state=19)
    boo = csr_to_boo(mat, cluster='identity', target_nr=group, target_nc=group,
                      small_block_policy='keep')
    arrays, buckets, row_ptr, starts = _plan_arrays(boo, bucketed, True, 64)
    def data(name):
        return arrays.get(name, np.asarray(getattr(boo, name)))
    ptr, idx = data('blk_nz_ptr'), data('nz_idx')
    expected = {(int(r), int(c)): (np.asarray(boo.nz_idx)[s:e], np.asarray(boo.nz_val)[s:e])
                for r, c, s, e in zip(boo.blk_row_grp, boo.blk_col_grp,
                                      boo.blk_nz_ptr[:-1], boo.blk_nz_ptr[1:])}
    covered = 0
    for start, count, nrmax, ncmax, nnzmax in np.asarray(buckets).reshape(-1, 5):
        assert start == covered
        for b in range(start, start + count):
            r, c = int(data('blk_row_grp')[b]), int(data('blk_col_grp')[b])
            nr = int(boo.row_grp_ptr[r + 1] - boo.row_grp_ptr[r])
            nc = int(boo.col_grp_ptr[c + 1] - boo.col_grp_ptr[c])
            s, e = ptr[b:b + 2]
            assert nr <= nrmax and nc <= ncmax and e - s <= nnzmax
            np.testing.assert_array_equal(idx[s:e], expected[r, c][0])
            np.testing.assert_array_equal(data('nz_val')[s:e], expected[r, c][1])
            decoded = idx[s:e].astype(np.uint16) >> data('blk_shift')[b]
            np.testing.assert_array_equal(starts[row_ptr[b]:row_ptr[b + 1]],
                                          np.searchsorted(decoded, np.arange(nr + 1)))
        covered += count
    assert covered == len(boo.blk_row_grp)
    if high_bit:
        assert np.max(idx) > 32767


@CUDA
@pytest.mark.boo_sanitizer
@pytest.mark.parametrize('bucketed,precomputed', [(False, False), (True, False),
                                                (False, True), (True, True)])
@pytest.mark.parametrize('policy,threshold', [('keep', 0), ('remainder', 4),
                                             ('remainder', 10000), ('tail_batch', 4)])
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_plans_with_streams_batches_and_partial_features(bucketed, precomputed, policy, threshold, dtype):
    mat = sp.random(41, 53, density=0.15, format='csr', random_state=19,
                    dtype=np.float32 if dtype == torch.float32 else np.float64)
    boo = csr_to_boo(mat, cluster='identity', target_nr=8, target_nc=16,
                      small_block_policy=policy, small_block_threshold=threshold).to('cuda')
    plan = prepare_boo_cuda(boo, bucketed=bucketed, precomputed_rows=precomputed)
    base = torch.randn(2, 65, 53, dtype=dtype, generator=torch.Generator().manual_seed(19))
    stream = torch.cuda.Stream()
    outputs = []
    with torch.cuda.stream(stream):
        for i in range(3):
            x = (base.to('cuda') + i).transpose(1, 2)
            outputs.append(boo_matmul(boo, x, launch_mode='auto', execution_plan=plan) * 2)
    stream.synchronize()
    for i, y in enumerate(outputs):
        reference = torch.from_numpy(mat.toarray()).double() @ (base.double() + i).transpose(1, 2)
        torch.testing.assert_close(y.cpu().double(), reference * 2, rtol=3e-5, atol=3e-5)


@CUDA
@pytest.mark.boo_sanitizer
def test_high_bit_plan_on_gpu():
    mat = sp.eye(257, dtype=np.float64, format='csr')
    boo = csr_to_boo(mat, cluster='identity', target_nr=256, target_nc=256,
                      small_block_policy='keep').to('cuda')
    plan = prepare_boo_cuda(boo, bucketed=True, precomputed_rows=True, p=8)
    x = torch.randn(257, 7, device='cuda', dtype=torch.float64)
    torch.testing.assert_close(boo_matmul(boo, x, p=8, launch_mode='auto', execution_plan=plan), x)


@CUDA
@pytest.mark.parametrize('shape,features,batch', [((0, 7), 17, 2), ((7, 0), 17, 2),
                                                ((7, 9), 0, 2), ((7, 9), 17, 0), ((7, 9), 17, 2)])
def test_empty_plans(shape, features, batch):
    boo = csr_to_boo(sp.csr_matrix(shape, dtype=np.float32)).to('cuda')
    plan = prepare_boo_cuda(boo, bucketed=True, precomputed_rows=True)
    y = boo_matmul(boo, torch.empty(batch, shape[1], features, device='cuda'),
                   launch_mode='auto', execution_plan=plan)
    assert y.shape == (batch, shape[0], features)
    assert torch.count_nonzero(y).item() == 0


@CUDA
def test_plan_guards():
    mat = sp.eye(9, dtype=np.float32, format='csr')
    boo = csr_to_boo(mat)
    with pytest.raises(ValueError, match='before preparing'):
        prepare_boo_cuda(boo)
    boo.to('cuda')
    plan = prepare_boo_cuda(boo, precomputed_rows=True)
    x = torch.ones(9, 17, device='cuda')
    with pytest.raises(ValueError, match='launch_mode'):
        boo_matmul(boo, x, execution_plan=plan)
    with pytest.raises(ValueError, match='prepared p'):
        boo_matmul(boo, x, p=32, launch_mode='auto', execution_plan=plan)
    boo.to('cpu').to('cuda')
    with pytest.raises(ValueError, match='current CUDA staging'):
        boo_matmul(boo, x, launch_mode='auto', execution_plan=plan)
    colmajor = csr_to_boo(mat, nz_order='colmajor').to('cuda')
    with pytest.raises(ValueError, match='rowmajor'):
        prepare_boo_cuda(colmajor)
