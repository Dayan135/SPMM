import numpy as np
import pytest

from benchmark.bench_hem_pbr import canonical
from benchmark.bench_reorder_vs_boo import hierarchy_order, assert_identical_boo_blocks
from benchmark.profile_pbr import build_matrix
from torch_fast_spdmm import csr_to_boo, boo_to_csr
from torch_fast_spdmm.clustering import cluster_axis


def test_lexsort_prioritizes_coarse_then_fine_labels():
    identity = np.arange(8)
    fine = np.array([0, 1, 2, 3, 2, 3, 0, 1])
    coarse = np.array([0, 0, 1, 1, 1, 1, 0, 0])
    p = hierarchy_order(np.zeros(8, dtype=int), [identity, fine, coarse])
    np.testing.assert_array_equal(p, [0, 6, 1, 7, 2, 4, 3, 5])


@pytest.mark.parametrize('k', [1, 8, 32])
def test_hierarchy_records_nested_partitions_without_changing_hem(k):
    mat = canonical(build_matrix(1024, 'fem2d_permuted', seed=0))
    levels = []
    args = dict(method='hem', target=k, max_size=k, feasible_first=True)
    labels = cluster_axis(mat, **args, hem_hierarchy=levels)
    np.testing.assert_array_equal(labels, cluster_axis(mat, **args))
    np.testing.assert_array_equal(levels[0], np.arange(mat.shape[0]))
    np.testing.assert_array_equal(levels[-1], labels)
    p = hierarchy_order(labels, levels)
    for level in levels:
        sorted_labels = level[p]
        # Every matched subtree is one uninterrupted interval in the order.
        starts = sorted_labels[np.r_[True, sorted_labels[1:] != sorted_labels[:-1]]]
        assert len(starts) == len(np.unique(level))
    reordered = canonical(mat[p][:, p])
    inverse = np.argsort(p)
    assert (reordered[inverse][:, inverse] != mat).nnz == 0


def test_contiguous_boo_preserves_every_block_and_roundtrips():
    mat = canonical(build_matrix(1024, 'fem2d_permuted', seed=0))
    labels = cluster_axis(mat, method='hem', target=32, max_size=32, feasible_first=True)
    p = np.argsort(labels, kind='stable')
    reordered = canonical(mat[p][:, p])
    c = csr_to_boo(mat, row_labels=labels, col_labels=labels)
    e = csr_to_boo(reordered, row_labels=labels[p], col_labels=labels[p])
    assert_identical_boo_blocks(c, e)
    np.testing.assert_array_equal(e.row_ids, np.arange(len(p)))
    assert (boo_to_csr(e)[np.argsort(p)][:, np.argsort(p)] != mat).nnz == 0


def test_dof_permutation_changes_four_dof_node_layout():
    q = np.random.default_rng(0).permutation(1024)
    assert not np.all(q.reshape(-1, 4) // 4 == (q.reshape(-1, 4) // 4)[:, :1])
    mat = canonical(build_matrix(1024, 'fem2d', seed=0))
    inverse = np.argsort(q)
    assert (mat[q][:, q][inverse][:, inverse] != mat).nnz == 0


def test_hierarchy_rejects_reused_list_or_other_method():
    mat = canonical(build_matrix(16, 'fem2d', seed=0))
    with pytest.raises(ValueError, match='empty list'):
        cluster_axis(mat, hem_hierarchy=[np.arange(16)])
    with pytest.raises(ValueError, match='method=hem'):
        cluster_axis(mat, method='identity', hem_hierarchy=[])
