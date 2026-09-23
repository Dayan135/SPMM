"""BOO contract, adversarial layouts, and reproducible CUDA regressions.

Small explicit labelings isolate kernel behavior from clustering. GPU references
use the ORIGINAL matrix and CPU float64 multiplication, not a BOO/PBR decoder.
Run on a built GPU node with --boo-require-cuda; see BOO_TESTING.md.
"""

import numpy as np
import pytest
import scipy.sparse as sp
import torch

from torch_fast_spdmm import boo_matmul, boo_to_csr, csr_to_boo

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
PATHS = [('rowmajor', True), ('rowmajor', False), ('colmajor', None)]
PATH_IDS = ['rowseg', 'nnzflat-rowmajor', 'nnzflat-colmajor-auto']
POLICIES = ['keep', 'remainder', 'tail_batch']


def _example(dtype=np.float32, index_dtype=np.int32):
    # Noncontiguous groups, a non-power-of-two column group, empty local/global
    # rows, and small blocks sharing output rows with retained blocks.
    entries = [(0, 0, 2), (0, 2, -1), (2, 6, 3), (2, 0, 1),
               (0, 1, 4), (1, 1, 2), (1, 4, -3), (4, 7, 1),
               (4, 9, 2), (4, 3, -2), (3, 5, 1)]
    r, c, v = zip(*entries)
    mat = sp.csr_matrix((np.array(v, dtype=dtype), (r, c)), shape=(7, 11))
    mat.indptr = mat.indptr.astype(index_dtype)
    mat.indices = mat.indices.astype(index_dtype)
    return mat, np.array([0, 1, 0, 2, 1, 0, 2]), np.array([0, 1, 0, 2, 1, 2, 0, 1, 2, 1, 2])


def _build(mat, row_labels, col_labels, policy='keep', order='rowmajor', threshold=4):
    return csr_to_boo(mat, row_labels=row_labels, col_labels=col_labels,
                      small_block_policy=policy, small_block_threshold=threshold,
                      nz_order=order)


def _x(shape, dtype=torch.float32, seed=123):
    return torch.randn(shape, dtype=dtype, generator=torch.Generator().manual_seed(seed))


def _assert_product(mat, boo, x, **kwargs):
    expected = torch.matmul(torch.from_numpy(mat.toarray()).double(), x.cpu().double())
    got = boo_matmul(boo, x, **kwargs)
    assert got.shape == expected.shape
    assert got.dtype == x.dtype and got.device == x.device
    tol = 3e-5 if x.dtype == torch.float32 else 1e-10
    torch.testing.assert_close(got.cpu().double(), expected, rtol=tol, atol=tol)


@pytest.mark.parametrize('order', ['rowmajor', 'colmajor'])
@pytest.mark.parametrize('policy', POLICIES)
@pytest.mark.parametrize('index_dtype', [np.int32, np.int64])
def test_explicit_format_and_cpu_reference(order, policy, index_dtype):
    mat, rl, cl = _example(np.float64, index_dtype)
    boo = _build(mat, rl, cl, policy, order)
    assert f'Int{np.dtype(index_dtype).itemsize * 8}' in type(boo).__name__
    # Original matrix is the oracle, including entries sent to CSR/COO.
    np.testing.assert_array_equal(boo_to_csr(boo).toarray(), mat.toarray())
    assert boo.compressed_nnz() == (11 if policy == 'keep' else 8)
    assert boo.remainder_nnz() == (3 if policy == 'remainder' else 0)
    assert boo.tail_nnz() == (3 if policy == 'tail_batch' else 0)
    for ptr_name, length in [('row_grp_ptr', mat.shape[0]), ('col_grp_ptr', mat.shape[1]),
                             ('blk_nz_ptr', boo.compressed_nnz()), ('rem_indptr', boo.remainder_nnz())]:
        ptr = np.asarray(getattr(boo, ptr_name))
        assert ptr[0] == 0 and ptr[-1] == length
        assert np.all(np.diff(ptr) >= 0)
    _assert_product(mat, boo, _x((3, 11, 9), torch.float64))


@CUDA
@pytest.mark.boo_sanitizer
@pytest.mark.parametrize('order,row_seg', PATHS, ids=PATH_IDS)
@pytest.mark.parametrize('features', [1, 7, 8, 9, 31, 32, 33, 63, 64, 65, 127, 128, 129, 511, 512, 513, 1025])
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_cuda_feature_boundaries(order, row_seg, features, dtype):
    mat, rl, cl = _example(np.float32 if dtype == torch.float32 else np.float64)
    boo = _build(mat, rl, cl, order=order).to('cuda')
    assert boo.compressed_nnz() == mat.nnz and boo.remainder_nnz() == boo.tail_nnz() == 0
    _assert_product(mat, boo, _x((2, 11, features), dtype).cuda(), row_seg=row_seg)


@CUDA
@pytest.mark.parametrize('order,row_seg', PATHS, ids=PATH_IDS)
@pytest.mark.parametrize('threads', [128, 256, 512])
@pytest.mark.parametrize('p', [8, 16, 32, 64, 128])
def test_cuda_launch_specializations(order, row_seg, threads, p):
    mat, rl, cl = _example()
    boo = _build(mat, rl, cl, order=order).to('cuda')
    _assert_product(mat, boo, _x((11, 257)).cuda(), threads=threads, p=p, row_seg=row_seg)


@CUDA
@pytest.mark.boo_sanitizer
@pytest.mark.parametrize('order,row_seg', PATHS, ids=PATH_IDS)
@pytest.mark.parametrize('policy,threshold', [('remainder', 4), ('tail_batch', 4),
                                           ('remainder', 100), ('tail_batch', 100)])
def test_cuda_mixed_and_all_spilled(order, row_seg, policy, threshold):
    mat, rl, cl = _example()
    boo = _build(mat, rl, cl, policy, order, threshold).to('cuda')
    assert boo.compressed_nnz() == (8 if threshold == 4 else 0)
    spill = boo.remainder_nnz() if policy == 'remainder' else boo.tail_nnz()
    assert spill == (3 if threshold == 4 else mat.nnz)
    for csr_mode in (0, 2):
        _assert_product(mat, boo, _x((2, 11, 65)).cuda(), row_seg=row_seg, csr_mode=csr_mode)


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=CUDA)])
@pytest.mark.parametrize('shape', [(0, 0), (0, 5), (5, 0), (5, 7)])
@pytest.mark.parametrize('policy', POLICIES)
def test_empty_matrix(device, shape, policy):
    mat = sp.csr_matrix(shape, dtype=np.float32)
    boo = csr_to_boo(mat, cluster='identity', small_block_policy=policy).to(device)
    assert boo.num_blocks() == boo.total_nnz == 0
    np.testing.assert_array_equal(boo_to_csr(boo).toarray(), mat.toarray())
    for row_seg in (True, False):
        _assert_product(mat, boo, _x((2, shape[1], 9)).to(device), row_seg=row_seg)


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=CUDA)])
@pytest.mark.parametrize('x_shape', [(11, 0), (0, 11, 9), (2, 11, 0)])
@pytest.mark.parametrize('policy', POLICIES)
def test_empty_dense_dimensions(device, x_shape, policy):
    mat, rl, cl = _example()
    boo = _build(mat, rl, cl, policy).to(device)
    _assert_product(mat, boo, _x(x_shape).to(device))


@CUDA
@pytest.mark.parametrize('order,row_seg', PATHS, ids=PATH_IDS)
@pytest.mark.parametrize('shape', [(1, 1), (1, 17), (19, 1)])
def test_singleton_axes(order, row_seg, shape):
    mat = sp.csr_matrix(np.arange(1, np.prod(shape) + 1, dtype=np.float32).reshape(shape))
    boo = _build(mat, np.zeros(shape[0], int), np.zeros(shape[1], int), order=order).to('cuda')
    _assert_product(mat, boo, _x((shape[1], 33)).cuda(), row_seg=row_seg)


@CUDA
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_int64_host_indices_cuda(dtype):
    mat, rl, cl = _example(np.float32 if dtype == torch.float32 else np.float64, np.int64)
    boo = _build(mat, rl, cl, 'tail_batch').to('cuda')
    assert 'Int64' in type(boo).__name__
    for row_seg in (True, False):
        _assert_product(mat, boo, _x((2, 11, 33), dtype).cuda(), row_seg=row_seg)


@CUDA
@pytest.mark.parametrize('order,row_seg', PATHS, ids=PATH_IDS)
def test_many_blocks_accumulate_into_same_rows(order, row_seg):
    mat = sp.csr_matrix(np.ones((3, 257), dtype=np.float32))
    boo = _build(mat, np.zeros(3, int), np.arange(257), order=order).to('cuda')
    assert boo.num_blocks() == 257
    # Exact integer arithmetic makes a lost global update visible without
    # tolerance/rounding ambiguity, even under heavy cross-block contention.
    x = torch.ones(257, 17, device='cuda')
    for _ in range(3):
        got = boo_matmul(boo, x, row_seg=row_seg)
        torch.testing.assert_close(got.cpu(), torch.full((3, 17), 257.), rtol=0, atol=0)


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=CUDA)])
@pytest.mark.parametrize('layout', ['transpose', 'slice', 'expand', 'batch_transpose'])
def test_noncontiguous_x(device, layout):
    mat, rl, cl = _example()
    boo = _build(mat, rl, cl, 'remainder').to(device)
    if layout == 'transpose':
        x = _x((13, 11)).to(device).T
    elif layout == 'slice':
        x = _x((11, 26)).to(device)[:, 1::2]
    elif layout == 'expand':
        x = _x((11, 1)).to(device).expand(11, 13)
    else:
        x = _x((3, 13, 11)).to(device).transpose(1, 2)
    assert not x.is_contiguous()
    saved = x.clone()
    for row_seg in (True, False):
        _assert_product(mat, boo, x, row_seg=row_seg)
    torch.testing.assert_close(x, saved, rtol=0, atol=0)


@CUDA
@pytest.mark.boo_sanitizer
@pytest.mark.parametrize('order,row_seg', PATHS, ids=PATH_IDS)
def test_uint16_high_bit_and_last_coordinate(order, row_seg):
    # Sparse 256x256 group exercises codes >=32768 through the signed int16
    # staging view, row-table striding, empty rows, and the exact uint16 limit.
    mat = sp.csr_matrix((np.array([2, -3, 4], np.float64),
                         ([0, 128, 255], [0, 1, 255])), shape=(256, 256))
    labels = np.zeros(256, dtype=np.int32)
    boo = _build(mat, labels, labels, order=order).to('cuda')
    assert max(boo.nz_idx) == 65535
    _assert_product(mat, boo, _x((256, 9), torch.float64).cuda(),
                    p=8, threads=128, row_seg=row_seg)


@CUDA
@pytest.mark.boo_sanitizer
@pytest.mark.parametrize('order,row_seg', PATHS, ids=PATH_IDS)
def test_large_dynamic_shared_memory_and_strided_work(order, row_seg):
    # Both kernels need >48 KiB but <the RTX 3090 opt-in limit. This also
    # forces metadata, X, and compute loops to take multiple strides/thread.
    mat = sp.csr_matrix(np.ones((32, 64), dtype=np.float64))
    boo = _build(mat, np.zeros(32, int), np.zeros(64, int), order=order).to('cuda')
    assert boo.max_nnz_blk == 2048
    _assert_product(mat, boo, _x((2, 64, 129), torch.float64).cuda(),
                    p=64, threads=128, row_seg=row_seg)


@CUDA
@pytest.mark.parametrize('order,row_seg', PATHS, ids=PATH_IDS)
def test_shared_memory_overflow_is_reported(order, row_seg):
    # Legal uint16 encoding, impossible shared arena even at the requested
    # minimum P. It must raise instead of silently returning a zeroed Y.
    mat = sp.csr_matrix(np.ones((256, 256), dtype=np.float64))
    boo = _build(mat, np.zeros(256, int), np.zeros(256, int), order=order).to('cuda')
    with pytest.raises(RuntimeError, match='CUDA'):
        boo_matmul(boo, _x((256, 8), torch.float64).cuda(), p=8, row_seg=row_seg)


@pytest.mark.parametrize('order', ['rowmajor', 'colmajor'])
def test_coordinate_overflow_rejected(order):
    mat = sp.csr_matrix(([1.0], ([256], [255])), shape=(257, 256), dtype=np.float32)
    with pytest.raises(ValueError, match='uint16'):
        _build(mat, np.zeros(257, int), np.zeros(256, int), order=order)


@CUDA
@pytest.mark.parametrize('seed', range(12))
def test_seeded_randomized_product(seed):
    rng = np.random.default_rng(seed)
    rows, cols = int(rng.integers(2, 40)), int(rng.integers(2, 50))
    dtype = np.float32 if seed % 2 else np.float64
    torch_dtype = torch.float32 if seed % 2 else torch.float64
    dense = rng.normal(size=(rows, cols))
    dense[rng.random((rows, cols)) > (0.05 + 0.07 * seed)] = 0
    mat = sp.csr_matrix(dense.astype(dtype))
    rl = rng.integers(0, min(rows, 5), size=rows)
    cl = rng.integers(0, min(cols, 6), size=cols)
    features = [1, 17, 65, 129, 513, 1025][seed % 6]
    policy = POLICIES[seed % 3]
    # Both encodings and both kernels see the same generated problem.
    for order, row_seg in PATHS:
        boo = _build(mat, rl, cl, policy, order).to('cuda')
        _assert_product(mat, boo, _x((2, cols, features), torch_dtype, seed).cuda(),
                        threads=[128, 256, 512][seed % 3], row_seg=row_seg)


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=CUDA)])
@pytest.mark.parametrize('order', ['rowmajor', 'colmajor'])
def test_unsorted_duplicates_and_explicit_zeros(device, order):
    mat = sp.csr_matrix((np.array([2., -2., 0., 3., 4.]),
                         np.array([3, 3, 1, 0, 2]), np.array([0, 4, 5, 5])), shape=(3, 5))
    assert not mat.has_canonical_format
    boo = _build(mat, np.zeros(3, int), np.zeros(5, int), order=order).to(device)
    np.testing.assert_array_equal(boo_to_csr(boo).toarray(), mat.toarray())
    _assert_product(mat, boo, _x((5, 17), torch.float64).to(device))


@CUDA
@pytest.mark.boo_sanitizer
@pytest.mark.parametrize('policy', POLICIES)
def test_nondefault_stream_producer_consumer(policy):
    mat, rl, cl = _example()
    boo = _build(mat, rl, cl, policy).to('cuda')
    stream = torch.cuda.Stream()
    outputs = []
    base = _x((13, 11))
    # No synchronization inside the loop: exercise fork/join and temporary X3
    # storage lifetime with GPU producers and immediate consumers on this stream.
    with torch.cuda.stream(stream):
        for i in range(8):
            x = (base.to('cuda') + i).T
            outputs.append(boo_matmul(boo, x, row_seg=bool(i % 2)) * 2 + 1)
    stream.synchronize()
    for i, got in enumerate(outputs):
        expected = torch.from_numpy(mat.toarray()).double() @ (base.double() + i).T
        torch.testing.assert_close(got.cpu().double(), expected * 2 + 1, rtol=3e-5, atol=3e-5)


@CUDA
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason='requires two GPUs')
def test_noncurrent_device_and_device_mismatch():
    mat, rl, cl = _example()
    with torch.cuda.device(0):
        boo = _build(mat, rl, cl).to('cuda:1')
        _assert_product(mat, boo, _x((11, 17)).to('cuda:1'))
        assert torch.cuda.current_device() == 0
        with pytest.raises(ValueError, match='same CUDA device'):
            boo_matmul(boo, _x((11, 17)).to('cuda:0'))


@CUDA
def test_device_staging_is_idempotent_and_reversible():
    mat, rl, cl = _example()
    boo = _build(mat, rl, cl).to('cuda')
    meta = boo._gpu_meta
    assert boo.to('cuda') is boo and boo._gpu_meta is meta
    boo.to('cpu')
    _assert_product(mat, boo, _x((11, 17)))
    with pytest.raises(RuntimeError, match='not'):
        boo_matmul(boo, _x((11, 17)).cuda())
    _assert_product(mat, boo.to('cuda'), _x((11, 17)).cuda())


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=CUDA)])
def test_invalid_x_contract(device):
    mat, rl, cl = _example()
    boo = _build(mat, rl, cl).to(device)
    for shape in [(11,), (1, 1, 11, 3)]:
        with pytest.raises(ValueError, match='2D or 3D'):
            boo_matmul(boo, torch.zeros(shape, device=device))
    with pytest.raises(ValueError, match='rows'):
        boo_matmul(boo, torch.zeros(10, 3, device=device))
    for dtype in (torch.float64, torch.float16, torch.int32, torch.complex64):
        with pytest.raises(TypeError, match='dtype'):
            boo_matmul(boo, torch.zeros(11, 3, dtype=dtype, device=device))
    with pytest.raises(ValueError, match='dense'):
        boo_matmul(boo, torch.zeros(11, 3, device=device).to_sparse())


@CUDA
@pytest.mark.parametrize('kwargs', [{'p': 0}, {'p': 4}, {'p': 12}, {'p': 256},
                                   {'threads': 0}, {'threads': 192}, {'threads': 1024},
                                   {'csr_mode': 3}])
def test_invalid_launch_parameters(kwargs):
    mat, rl, cl = _example()
    boo = _build(mat, rl, cl).to('cuda')
    with pytest.raises(ValueError):
        boo_matmul(boo, _x((11, 17)).cuda(), **kwargs)


@CUDA
def test_rowseg_rejects_column_major():
    mat, rl, cl = _example()
    boo = _build(mat, rl, cl, order='colmajor').to('cuda')
    with pytest.raises(ValueError, match='rowmajor'):
        boo_matmul(boo, _x((11, 17)).cuda(), row_seg=True)


@pytest.mark.parametrize('kwargs', [{'target_nr': 0}, {'max_nc': -1}, {'max_block_nnz': 0},
                                   {'small_block_threshold': -1}, {'nz_order': 'unknown'},
                                   {'small_block_policy': 'unknown'}])
def test_invalid_builder_parameters(kwargs):
    with pytest.raises(ValueError):
        csr_to_boo(_example()[0], **kwargs)


@pytest.mark.parametrize('case', ['missing', 'length', 'rank', 'negative', 'too_large', 'float'])
def test_invalid_labels(case):
    mat, rl, cl = _example()
    if case == 'missing':
        cl = None
    elif case == 'length':
        rl = rl[:-1]
    elif case == 'rank':
        rl = rl.reshape(1, -1)
    elif case == 'negative':
        rl[0] = -1
    elif case == 'too_large':
        rl[0] = 2**40
    else:
        rl = rl.astype(float) + 0.5
    with pytest.raises((ValueError, TypeError), match='labels'):
        csr_to_boo(mat, row_labels=rl, col_labels=cl)


def test_builder_input_contract_and_torch_csr():
    mat, rl, cl = _example()
    with pytest.raises(TypeError, match='float32/float64'):
        csr_to_boo(mat.astype(np.int32))
    with pytest.raises(ValueError, match='sparse CSR'):
        csr_to_boo(torch.zeros(7, 11))
    broken = mat.copy()
    broken.indices[0] = 11
    with pytest.raises(ValueError):
        csr_to_boo(broken)
    tensor = torch.sparse_csr_tensor(torch.from_numpy(mat.indptr), torch.from_numpy(mat.indices),
                                     torch.from_numpy(mat.data), size=mat.shape)
    boo = csr_to_boo(tensor, row_labels=rl, col_labels=cl)
    np.testing.assert_array_equal(boo_to_csr(boo).toarray(), mat.toarray())
