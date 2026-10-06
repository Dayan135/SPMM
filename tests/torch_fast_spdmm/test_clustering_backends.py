"""Real backend coverage: run on the remote GPU node with both extras installed."""

import sys

import numpy as np
import pytest
import scipy.sparse as sp
import torch

from torch_fast_spdmm import boo_matmul, boo_to_csr, csr_to_boo
from torch_fast_spdmm.clustering import cluster_axis, cluster_axes


@pytest.fixture(params=['metis', 'graclus'])
def method(request):
    pytest.importorskip('pymetis' if request.param == 'metis' else 'torch_cluster')
    return request.param


@pytest.mark.parametrize('target', [8, 16, 32, 64, 128, 256])
def test_backend_sizes_and_product(method, target):
    # Sparse enough to launch at 256, with shared neighbors across rows.
    a = sp.diags([np.ones(511), np.ones(512), np.ones(511)], [-1, 0, 1],
                 shape=(512, 512), format='csr', dtype=np.float32)
    rows, cols = cluster_axes(a, method=method, target_nr=target, target_nc=target,
                              max_nr=target, max_nc=target, seed=7)
    assert rows.dtype == np.int32
    assert np.array_equal(rows, cols)
    assert np.bincount(rows).max() <= target
    again = cluster_axis(a, method=method, target=target, max_size=target, seed=7)
    np.testing.assert_array_equal(rows, again)
    boo = csr_to_boo(a, row_labels=rows, col_labels=cols)
    assert (boo_to_csr(boo) != a).nnz == 0
    x = torch.randn(512, 13)
    actual = boo_matmul(boo.to('cuda'), x.cuda(), p=8).cpu().numpy()
    np.testing.assert_allclose(actual, a @ x.numpy(), rtol=1e-5, atol=1e-5)


def test_rectangular_caps_and_rng(method):
    a = sp.random(95, 131, density=.08, random_state=4, format='csr')
    before = torch.random.get_rng_state().clone()
    rows, cols = cluster_axes(a, method=method, target_nr=16, target_nc=8,
                              max_nr=16, max_nc=8, max_block_nnz=40, seed=11)
    assert torch.equal(before, torch.random.get_rng_state())
    for labels, mat, cap in [(rows, a, 16), (cols, a.T.tocsr(), 8)]:
        assert len(labels) == mat.shape[0]
        assert np.bincount(labels).max() <= cap
        assert np.bincount(labels, weights=np.diff(mat.indptr)).max() <= 40


@pytest.mark.parametrize('n', [0, 1, 19])
def test_empty_similarity_graph(method, n):
    labels = cluster_axis(sp.eye(n, format='csr'), method=method, target=8, max_size=8)
    assert len(labels) == n
    if n:
        assert np.bincount(labels).max() <= 8


@pytest.mark.parametrize('method,module', [('metis', 'pymetis'), ('graclus', 'torch_cluster')])
def test_missing_backend_never_runs_hem(monkeypatch, method, module):
    monkeypatch.setitem(sys.modules, module, None)
    with pytest.raises(ImportError, match='requires'):
        cluster_axis(sp.eye(16, format='csr'), method=method, target=8, max_size=8)


def test_graclus_preserves_noncontiguous_pairs_under_cap():
    pytest.importorskip('torch_cluster')
    # A permutation of two dense components. Splitting overgrown clusters in
    # input order would discard this relationship; cap-aware matching retains it.
    a = sp.csr_matrix(np.tile([[1, 0], [0, 1]], (32, 1)))
    labels = cluster_axis(a, method='graclus', target=8, max_size=8, seed=3)
    for group in np.unique(labels):
        members = np.flatnonzero(labels == group)
        assert len(members) <= 8
        assert len(set(members % 2)) == 1


def test_large_group_feature_tiles():
    from types import SimpleNamespace
    from benchmark.bench_clustering import feature_tile
    sparse = SimpleNamespace(max_nr=256, max_nc=256, max_nnz_blk=100,
                             blk_row_grp=[0])
    p, arena, supported = feature_tile(sparse, 128, 48 * 1024)
    assert supported and p == 32 and arena <= 48 * 1024
    dense = SimpleNamespace(max_nr=256, max_nc=256, max_nnz_blk=65536,
                            blk_row_grp=[0])
    p, arena, supported = feature_tile(dense, 128, 99 * 1024)
    assert not supported and p == 8 and arena > 99 * 1024
