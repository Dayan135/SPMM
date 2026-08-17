"""Correctness for the BOO (Blocked COO) format and kernel.

Ordered so a failure localizes: format round-trip and invariants first (pure
CPU), then a numpy oracle built straight from the BOO arrays, then the CUDA
kernel against the trusted PBR block kernel.
"""

import numpy as np
import pytest
import scipy.sparse as sp
import torch
from pytest import mark

from torch_fast_spdmm import (
    boo_info, boo_matmul, boo_to_csr, csr_to_boo, pbr_matmul, csr_to_pbr,
)

from benchmark.matrices import (
    PBR_PATTERN_PRESETS, generate_patterned_block_matrix, generate_random_sparse,
)

CUDA = mark.skipif(not torch.cuda.is_available(), reason='CUDA not available')

CLUSTERS = ['identity', 'hem', 'rcm', 'degree', 'random']


def _square(n=256, density=0.05, seed=0, dtype=np.float32):
    rng = np.random.default_rng(seed)
    mat = sp.random(n, n, density=density, format='csr', random_state=seed)
    mat.data = rng.standard_normal(mat.nnz)
    return mat.astype(dtype)


def _rectangular(rows=192, cols=320, density=0.05, seed=1, dtype=np.float32):
    rng = np.random.default_rng(seed)
    mat = sp.random(rows, cols, density=density, format='csr', random_state=seed)
    mat.data = rng.standard_normal(mat.nnz)
    return mat.astype(dtype)


def _same_csr(a, b):
    a = a.copy(); a.sort_indices(); a.eliminate_zeros()
    b = b.copy(); b.sort_indices(); b.eliminate_zeros()
    return (np.array_equal(a.indptr, b.indptr)
            and np.array_equal(a.indices, b.indices)
            and np.allclose(a.data, b.data))


def _numpy_reference(boo_mat, x):
    """Dense reference built from the BOO arrays alone.

    Independent of both scipy and the kernel, so a disagreement localizes to
    format-vs-kernel rather than leaving both suspect.
    """
    row_grp_ptr = np.asarray(boo_mat.row_grp_ptr, dtype=np.int64)
    col_grp_ptr = np.asarray(boo_mat.col_grp_ptr, dtype=np.int64)
    row_ids = np.asarray(boo_mat.row_ids, dtype=np.int64)
    col_ids = np.asarray(boo_mat.col_ids, dtype=np.int64)
    blk_row_grp = np.asarray(boo_mat.blk_row_grp, dtype=np.int64)
    blk_col_grp = np.asarray(boo_mat.blk_col_grp, dtype=np.int64)
    blk_nz_ptr = np.asarray(boo_mat.blk_nz_ptr, dtype=np.int64)
    blk_shift = np.asarray(boo_mat.blk_shift, dtype=np.int64)
    nz_idx = np.asarray(boo_mat.nz_idx, dtype=np.int64)
    nz_val = np.asarray(boo_mat.nz_val)

    y = np.zeros((boo_mat.rows, x.shape[1]), dtype=np.float64)
    for b in range(blk_row_grp.size):
        shift = blk_shift[b]
        mask = (1 << shift) - 1
        rbase = row_grp_ptr[blk_row_grp[b]]
        cbase = col_grp_ptr[blk_col_grp[b]]
        for k in range(blk_nz_ptr[b], blk_nz_ptr[b + 1]):
            idx = nz_idx[k]
            y[row_ids[rbase + (idx >> shift)]] += nz_val[k] * x[col_ids[cbase + (idx & mask)]]

    rem_indptr = np.asarray(boo_mat.rem_indptr, dtype=np.int64)
    rem_col_ind = np.asarray(boo_mat.rem_col_ind, dtype=np.int64)
    rem_vals = np.asarray(boo_mat.rem_vals)
    for r in range(boo_mat.rows):
        for j in range(rem_indptr[r], rem_indptr[r + 1]):
            y[r] += rem_vals[j] * x[rem_col_ind[j]]

    tail_row = np.asarray(boo_mat.tail_row, dtype=np.int64)
    tail_col = np.asarray(boo_mat.tail_col, dtype=np.int64)
    tail_val = np.asarray(boo_mat.tail_val)
    for k in range(tail_row.size):
        y[tail_row[k]] += tail_val[k] * x[tail_col[k]]

    return y


# ---------------------------------------------------------------------------
# 1. round-trip
# ---------------------------------------------------------------------------

@mark.parametrize('cluster', CLUSTERS)
@mark.parametrize('policy,threshold', [('keep', 0), ('remainder', 4), ('tail_batch', 4)])
@mark.parametrize('nz_order', ['rowmajor', 'colmajor'])
@mark.parametrize('dtype', [np.float32, np.float64])
@mark.parametrize('shape', ['square', 'rectangular'])
def test_boo_roundtrip(cluster, policy, threshold, nz_order, dtype, shape):
    if shape == 'rectangular' and cluster == 'rcm':
        pytest.skip('rcm requires a square matrix')
    mat = _square(dtype=dtype) if shape == 'square' else _rectangular(dtype=dtype)
    boo = csr_to_boo(mat, cluster=cluster, target_nr=16, target_nc=16,
                     small_block_policy=policy, small_block_threshold=threshold,
                     nz_order=nz_order)
    assert _same_csr(boo_to_csr(boo), mat)


@mark.parametrize('target', [8, 16, 64])
def test_boo_roundtrip_targets(target):
    mat = _square()
    boo = csr_to_boo(mat, cluster='hem', target_nr=target, target_nc=target)
    assert _same_csr(boo_to_csr(boo), mat)


# ---------------------------------------------------------------------------
# 2. format invariants
# ---------------------------------------------------------------------------

@mark.parametrize('nz_order', ['rowmajor', 'colmajor'])
@mark.parametrize('cluster', ['identity', 'hem'])
@mark.parametrize('policy,threshold', [('keep', 0), ('remainder', 4), ('tail_batch', 4)])
def test_boo_invariants(nz_order, cluster, policy, threshold):
    mat = _square()
    boo = csr_to_boo(mat, cluster=cluster, target_nr=16, target_nc=16,
                     nz_order=nz_order, small_block_policy=policy,
                     small_block_threshold=threshold)

    row_grp_ptr = np.asarray(boo.row_grp_ptr, dtype=np.int64)
    col_grp_ptr = np.asarray(boo.col_grp_ptr, dtype=np.int64)
    row_ids = np.asarray(boo.row_ids, dtype=np.int64)
    col_ids = np.asarray(boo.col_ids, dtype=np.int64)

    # row_ids / col_ids are exact partitions of [0, rows) / [0, cols)
    assert np.array_equal(np.sort(row_ids), np.arange(boo.rows))
    assert np.array_equal(np.sort(col_ids), np.arange(boo.cols))
    # ascending within each group
    for ptr, ids in ((row_grp_ptr, row_ids), (col_grp_ptr, col_ids)):
        for g in range(ptr.size - 1):
            member = ids[ptr[g]:ptr[g + 1]]
            assert np.all(np.diff(member) > 0)

    # the uint16 local index space fits
    assert boo.max_nr * (1 << int(np.ceil(np.log2(max(boo.max_nc, 1))))) <= 65536

    blk_nz_ptr = np.asarray(boo.blk_nz_ptr, dtype=np.int64)
    blk_shift = np.asarray(boo.blk_shift, dtype=np.int64)
    blk_row_grp = np.asarray(boo.blk_row_grp, dtype=np.int64)
    blk_col_grp = np.asarray(boo.blk_col_grp, dtype=np.int64)
    nz_idx = np.asarray(boo.nz_idx, dtype=np.int64)

    for b in range(blk_row_grp.size):
        shift = blk_shift[b]
        mask = (1 << shift) - 1
        n_r = row_grp_ptr[blk_row_grp[b] + 1] - row_grp_ptr[blk_row_grp[b]]
        n_c = col_grp_ptr[blk_col_grp[b] + 1] - col_grp_ptr[blk_col_grp[b]]
        block = nz_idx[blk_nz_ptr[b]:blk_nz_ptr[b + 1]]
        assert np.all((block >> shift) < n_r)
        assert np.all((block & mask) < n_c)
        if nz_order == 'rowmajor':
            # rowmajor makes nz_idx itself strictly increasing -- a free assertion
            assert np.all(np.diff(block) > 0)
        else:
            key = ((block & mask) << 32) | (block >> shift)
            assert np.all(np.diff(key) > 0)

    # every nonzero is accounted for exactly once
    assert (boo.compressed_nnz() + boo.remainder_nnz() + boo.tail_nnz()
            == boo.total_nnz == mat.nnz)

    if policy != 'tail_batch':
        assert boo.tail_nnz() == 0
    else:
        assert boo.remainder_nnz() == 0
        # tail rows/cols are GLOBAL indices, in range
        tail_row = np.asarray(boo.tail_row, dtype=np.int64)
        tail_col = np.asarray(boo.tail_col, dtype=np.int64)
        if tail_row.size:
            assert tail_row.min() >= 0 and tail_row.max() < boo.rows
            assert tail_col.min() >= 0 and tail_col.max() < boo.cols


@mark.parametrize('policy,threshold', [('keep', 0), ('remainder', 4), ('tail_batch', 4)])
def test_boo_determinism(policy, threshold):
    mat = _square()
    kw = dict(cluster='hem', target_nr=16, target_nc=16,
             small_block_policy=policy, small_block_threshold=threshold)
    first = csr_to_boo(mat, **kw)
    second = csr_to_boo(mat, **kw)
    for field in ('row_ids', 'col_ids', 'blk_row_grp', 'blk_col_grp',
                  'blk_nz_ptr', 'blk_shift', 'nz_idx', 'nz_val',
                  'tail_row', 'tail_col', 'tail_val'):
        assert np.array_equal(np.asarray(getattr(first, field)),
                              np.asarray(getattr(second, field))), field


def test_boo_keep_policy_has_no_remainder():
    """`keep` makes compressed_fraction == 1.0 structurally, which is what
    removes SHOW_TO_ERAN sec. 2's CSR-fallback artifact by construction."""
    boo = csr_to_boo(_square(), cluster='hem', small_block_policy='keep')
    assert boo.remainder_nnz() == 0
    assert boo.tail_nnz() == 0
    assert boo_info(boo).compressed_fraction == 1.0
    assert boo_info(boo).boo_fraction == 1.0


def test_boo_tail_batch_never_leaves_boo():
    """Unlike `remainder`, tail_batch keeps every nonzero inside a BOO-family
    kernel -- boo_fraction stays 1.0 even though compressed_fraction (the
    strict block-encoded count) drops below it."""
    boo = csr_to_boo(_square(n=512, density=0.02), cluster='hem',
                     target_nr=16, target_nc=16,
                     small_block_policy='tail_batch', small_block_threshold=4)
    assert boo.remainder_nnz() == 0
    assert boo.tail_nnz() > 0                       # this matrix actually has a tail
    info = boo_info(boo)
    assert info.compressed_fraction < 1.0            # some nnz left the block encoding
    assert info.boo_fraction == 1.0                  # but none left BOO for CSR


def test_boo_merge_policy_is_rejected_clearly():
    with pytest.raises(Exception, match='merge'):
        csr_to_boo(_square(), small_block_policy='merge')


# ---------------------------------------------------------------------------
# 3. numpy oracle (CPU)
# ---------------------------------------------------------------------------

@mark.parametrize('cluster', ['identity', 'hem'])
@mark.parametrize('policy,threshold', [('keep', 0), ('remainder', 4), ('tail_batch', 4)])
def test_boo_cpu_matmul_matches_numpy_oracle(cluster, policy, threshold):
    mat = _square(dtype=np.float64)
    boo = csr_to_boo(mat, cluster=cluster, target_nr=16, target_nc=16,
                     small_block_policy=policy, small_block_threshold=threshold)
    x = np.random.default_rng(3).standard_normal((mat.shape[1], 24))

    oracle = _numpy_reference(boo, x)
    dense = mat.toarray() @ x
    np.testing.assert_allclose(oracle, dense, rtol=1e-10, atol=1e-10)

    cpu = boo_matmul(boo, torch.from_numpy(x)).numpy()
    np.testing.assert_allclose(cpu, dense, rtol=1e-10, atol=1e-10)


# ---------------------------------------------------------------------------
# 4. CUDA kernel vs the trusted PBR block kernel
# ---------------------------------------------------------------------------

@CUDA
@mark.parametrize('cluster', ['identity', 'hem'])
@mark.parametrize('target', [8, 32])
@mark.parametrize('features', [32, 128])
@mark.parametrize('threads,p', [(128, 8), (256, 16), (512, 32)])
@mark.parametrize('dtype', [torch.float32, torch.float64])
def test_boo_cuda_matches_dense(cluster, target, features, threads, p, dtype):
    np_dtype = np.float32 if dtype == torch.float32 else np.float64
    mat = _square(n=256, density=0.05, dtype=np_dtype)
    boo = csr_to_boo(mat, cluster=cluster, target_nr=target, target_nc=target).to('cuda')

    x = torch.randn(mat.shape[1], features, dtype=dtype)
    expected = torch.from_numpy(mat.toarray()).to(dtype) @ x
    got = boo_matmul(boo, x.cuda(), threads=threads, p=p).cpu()

    tol = 1e-5 if dtype == torch.float32 else 1e-10
    torch.testing.assert_close(got, expected, rtol=tol, atol=tol)


@CUDA
@mark.parametrize('features', [100, 200, 520])
def test_boo_cuda_features_not_multiple_of_p(features):
    """Feature counts that are not power-of-2 multiples of p.

    This is the `current_p` bug class that all four PBR kernels get wrong
    (documented-not-fixed at pbr_kernel.cu:82-84). BOO indexes shared memory
    with the compile-time P and guards only the global accesses, so these must
    pass.
    """
    mat = _square(n=192, density=0.06)
    boo = csr_to_boo(mat, cluster='hem', target_nr=16, target_nc=16).to('cuda')
    x = torch.randn(mat.shape[1], features)
    expected = torch.from_numpy(mat.toarray()) @ x
    got = boo_matmul(boo, x.cuda()).cpu()
    torch.testing.assert_close(got, expected, rtol=1e-5, atol=1e-5)


@CUDA
@mark.parametrize('policy,threshold', [('keep', 0), ('remainder', 6), ('tail_batch', 6)])
def test_boo_cuda_matches_pbr_block_kernel(policy, threshold):
    """Against the trusted reference kernel, with the remainder leg exercised."""
    mat = _square(n=256, density=0.04)
    pbr = csr_to_pbr(mat, block_rows=8, block_cols=8, min_nnz_per_block=1).to('cuda')
    boo = csr_to_boo(mat, cluster='hem', target_nr=16, target_nc=16,
                     small_block_policy=policy,
                     small_block_threshold=threshold).to('cuda')

    x = torch.randn(mat.shape[1], 64).cuda()
    torch.testing.assert_close(boo_matmul(boo, x).cpu(), pbr_matmul(pbr, x).cpu(),
                               rtol=1e-5, atol=1e-5)


@CUDA
def test_boo_cuda_row_skewed_pattern():
    """row_skewed: row 0 dense, every other row exactly one nonzero.

    The maximum per-row imbalance available, and the pattern SHOW_TO_ERAN sec. 6
    used to isolate atomic pileup.
    """
    mat = generate_patterned_block_matrix(
        256, 8, [(PBR_PATTERN_PRESETS['row_skewed'], 0.5)], seed=0).astype(np.float32)
    boo = csr_to_boo(mat, cluster='hem', target_nr=16, target_nc=16).to('cuda')
    x = torch.randn(mat.shape[1], 64)
    expected = torch.from_numpy(mat.toarray()) @ x
    torch.testing.assert_close(boo_matmul(boo, x.cuda()).cpu(), expected,
                               rtol=1e-4, atol=1e-4)


@CUDA
def test_boo_cuda_highly_imbalanced_clusters():
    """A few very high-degree rows next to many near-empty ones, so block sizes
    and nnz-per-block spread as widely as the caps allow -- the case where the
    single grid-wide arena (sized from the global max) is most likely to bite."""
    rng = np.random.default_rng(11)
    n = 512
    dense_rows, dense_cols = [], []
    for r in range(4):
        cols = rng.choice(n, size=200, replace=False)
        dense_rows.extend([r] * len(cols))
        dense_cols.extend(cols.tolist())
    for r in range(4, n):
        dense_rows.append(r)
        dense_cols.append(int(rng.integers(0, n)))
    mat = sp.csr_matrix(
        (rng.standard_normal(len(dense_rows)), (dense_rows, dense_cols)),
        shape=(n, n)).astype(np.float32)

    boo = csr_to_boo(mat, cluster='hem', target_nr=32, target_nc=32).to('cuda')
    x = torch.randn(n, 64)
    expected = torch.from_numpy(mat.toarray()) @ x
    torch.testing.assert_close(boo_matmul(boo, x.cuda()).cpu(), expected,
                               rtol=1e-4, atol=1e-4)


@CUDA
def test_boo_cuda_tail_batch_imbalanced_clusters():
    """Same stress matrix as the KEEP test above, routed through tail_batch
    instead -- exercises boo_spmm_tail_kernel on the arena-imbalance case it
    was built for."""
    rng = np.random.default_rng(11)
    n = 512
    dense_rows, dense_cols = [], []
    for r in range(4):
        cols = rng.choice(n, size=200, replace=False)
        dense_rows.extend([r] * len(cols))
        dense_cols.extend(cols.tolist())
    for r in range(4, n):
        dense_rows.append(r)
        dense_cols.append(int(rng.integers(0, n)))
    mat = sp.csr_matrix(
        (rng.standard_normal(len(dense_rows)), (dense_rows, dense_cols)),
        shape=(n, n)).astype(np.float32)

    boo = csr_to_boo(mat, cluster='hem', target_nr=32, target_nc=32,
                     small_block_policy='tail_batch', small_block_threshold=8).to('cuda')
    assert boo.tail_nnz() > 0
    x = torch.randn(n, 64)
    expected = torch.from_numpy(mat.toarray()) @ x
    torch.testing.assert_close(boo_matmul(boo, x.cuda()).cpu(), expected,
                               rtol=1e-4, atol=1e-4)


@CUDA
def test_boo_cuda_tail_batch_bimodal_forces_grid_stride():
    """diag_skewed-shaped: a dense 64-wide diagonal band (captured as normal
    BOO blocks) plus heavy scattered off-diagonal noise (singleton blocks,
    spilled to the tail). tail_nnz is large enough here to force the tail
    kernel's grid-stride loop to iterate more than once per thread -- the
    code path a small tail_nnz in the other tests never reaches."""
    rng = np.random.default_rng(7)
    n, band = 2048, 64
    band_rows, band_cols = [], []
    for b in range(n // band):
        for i in range(band):
            for j in range(band):
                if rng.random() < 0.5:
                    band_rows.append(b * band + i)
                    band_cols.append(b * band + j)
    noise_rows = rng.integers(0, n, size=20000)
    noise_cols = rng.integers(0, n, size=20000)
    rows = np.concatenate([band_rows, noise_rows])
    cols = np.concatenate([band_cols, noise_cols])
    vals = rng.standard_normal(rows.size)
    mat = sp.csr_matrix((vals, (rows, cols)), shape=(n, n)).astype(np.float32)
    mat.sum_duplicates()

    boo = csr_to_boo(mat, cluster='hem', target_nr=32, target_nc=32,
                     small_block_policy='tail_batch', small_block_threshold=8).to('cuda')
    assert boo.tail_nnz() > 5000, 'test setup should produce a large tail'
    x = torch.randn(n, 64)
    expected = torch.from_numpy(mat.toarray()) @ x
    torch.testing.assert_close(boo_matmul(boo, x.cuda()).cpu(), expected,
                               rtol=1e-3, atol=1e-3)


@CUDA
@mark.parametrize('batch', [1, 3])
def test_boo_cuda_batched(batch):
    mat = _square(n=192, density=0.05)
    boo = csr_to_boo(mat, cluster='hem', target_nr=16, target_nc=16).to('cuda')
    x = torch.randn(batch, mat.shape[1], 32)
    expected = torch.stack([torch.from_numpy(mat.toarray()) @ x[b] for b in range(batch)])
    torch.testing.assert_close(boo_matmul(boo, x.cuda()).cpu(), expected,
                               rtol=1e-5, atol=1e-5)
