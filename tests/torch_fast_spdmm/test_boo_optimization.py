"""Regression gates for selectable BOO optimization experiments."""

import numpy as np
import pytest
import scipy.sparse as sp
import torch

from torch_fast_spdmm import boo_matmul, boo_to_csr, csr_to_boo
from torch_fast_spdmm.clustering import _mutual_max_pairs

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')


@pytest.mark.parametrize('cap_kind', ['size', 'nnz'])
def test_feasible_partner_is_not_hidden_by_stronger_infeasible_edge(cap_kind):
    # A-B is strongest but forbidden; A-C and B-D are both feasible.
    graph = sp.csr_matrix(np.array([[0, 10, 9, 0], [10, 0, 0, 8],
                                    [9, 0, 0, 0], [0, 8, 0, 0]]))
    costs = np.array([30, 30, 2, 2])
    kwargs = dict(max_size=32 if cap_kind == 'size' else None,
                  max_nnz=32 if cap_kind == 'nnz' else None)
    active = np.ones(4, dtype=bool)
    old = _mutual_max_pairs(graph, active, costs, costs, **kwargs, feasible_first=False)
    assert len(old[0]) == 0
    left, right = _mutual_max_pairs(graph, active, costs, costs, **kwargs, feasible_first=True)
    assert set(zip(left.tolist(), right.tolist())) == {(0, 2), (1, 3)}


@pytest.mark.parametrize('shape', [(97, 97), (73, 89)])
@pytest.mark.parametrize('target', [(8, 8), (16, 8), (8, 16)])
def test_feasible_clustering_roundtrip_caps_and_determinism(shape, target):
    mat = sp.random(*shape, density=0.12, format='csr', dtype=np.float32, random_state=7)
    kw = dict(target_nr=target[0], target_nc=target[1], hem_feasible_first=True)
    first, second = csr_to_boo(mat, **kw), csr_to_boo(mat, **kw)
    assert (boo_to_csr(first) != mat).nnz == 0
    for name in ('row_grp_ptr', 'row_ids', 'col_grp_ptr', 'col_ids', 'nz_idx', 'nz_val'):
        np.testing.assert_array_equal(getattr(first, name), getattr(second, name))
    assert max(np.diff(first.row_grp_ptr), default=0) <= target[0]
    assert max(np.diff(first.col_grp_ptr), default=0) <= target[1]


@CUDA
@pytest.mark.boo_sanitizer
@pytest.mark.parametrize('mode', ['legacy', 'auto'])
@pytest.mark.parametrize('policy,threshold', [('keep', 0), ('remainder', 4),
                                             ('remainder', 10000), ('tail_batch', 4)])
@pytest.mark.parametrize('order', ['rowmajor', 'colmajor'])
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_launch_modes_on_nondefault_stream_with_strided_batched_inputs(mode, policy, threshold, order, dtype):
    mat = sp.random(17, 23, density=0.2, format='csr', random_state=3,
                    dtype=np.float32 if dtype == torch.float32 else np.float64)
    boo = csr_to_boo(mat, cluster='identity', target_nr=4, target_nc=4,
                      small_block_policy=policy, small_block_threshold=threshold,
                      nz_order=order).to('cuda')
    base = torch.randn(2, 65, 23, dtype=dtype, generator=torch.Generator().manual_seed(3))
    stream = torch.cuda.Stream()
    outputs = []
    with torch.cuda.stream(stream):
        for i in range(3):
            x = (base.to('cuda') + i).transpose(1, 2)
            outputs.append(boo_matmul(boo, x, launch_mode=mode) * 2)
    stream.synchronize()
    for i, y in enumerate(outputs):
        ref = torch.from_numpy(mat.toarray()).double() @ (base.double() + i).transpose(1, 2)
        torch.testing.assert_close(y.cpu().double(), ref * 2, rtol=3e-5, atol=3e-5)


@CUDA
@pytest.mark.parametrize('shape', [(0, 7), (7, 0), (7, 9)])
@pytest.mark.parametrize('features,batch', [(0, 2), (17, 0), (17, 2)])
def test_auto_launch_empty_outputs(shape, features, batch):
    boo = csr_to_boo(sp.csr_matrix(shape, dtype=np.float32)).to('cuda')
    y = boo_matmul(boo, torch.empty(batch, shape[1], features, device='cuda'), launch_mode='auto')
    assert y.shape == (batch, shape[0], features)
    assert torch.count_nonzero(y).item() == 0
