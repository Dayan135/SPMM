import numpy as np
import pytest
import scipy.sparse as sparse

from benchmark.bench_hem_pbr import symmetric_reorder, canonical
from benchmark.profile_pbr import build_matrix
from torch_fast_spdmm import csr_to_boo, boo_to_csr
from torch_fast_spdmm.clustering import cluster_axes


def test_permutation_direction_and_group_order():
    a = sparse.csr_matrix([[4., 1., 2.], [1., 5., 3.], [2., 3., 6.]])
    reordered, p = symmetric_reorder(a, np.array([1, 0, 0]))
    np.testing.assert_array_equal(p, [1, 2, 0])  # Non-self-inverse permutation.
    x = np.arange(6, dtype=np.float32).reshape(3, 2)
    restored = np.empty_like(x)
    restored[p] = reordered @ x[p]
    np.testing.assert_array_equal(restored, a @ x)
    with pytest.raises(ValueError, match='symmetric'):
        symmetric_reorder(sparse.csr_matrix([[1., 2.], [0., 1.]]), np.array([0, 1]))


@pytest.mark.parametrize('k', [8, 32])
def test_fem_hem_permutation_preserves_exact_boo_groups(k):
    a = canonical(build_matrix(1024, 'fem2d_permuted', seed=7))
    rows, cols = cluster_axes(a, method='hem', target_nr=k, target_nc=k,
                              max_nr=k, max_nc=k, symmetric=True, feasible_first=True)
    np.testing.assert_array_equal(rows, cols)
    reordered, p = symmetric_reorder(a, rows)
    boo = csr_to_boo(a, row_labels=rows, col_labels=cols, target_nr=k, target_nc=k)
    np.testing.assert_array_equal(boo.row_ids, p)
    np.testing.assert_array_equal(boo.col_ids, p)
    assert (boo_to_csr(boo) != a).nnz == 0
    assert (reordered != reordered.T).nnz == 0
    assert reordered.nnz == a.nnz
    inverse = np.argsort(p)
    assert (reordered[inverse][:, inverse] != a).nnz == 0
