"""Features-per-thread (FPT) variants of the row-segmented BOO kernel.

FPT=2/4 change only Step B+C's work mapping and read smem_X as float2/float4
(double2), which moves smem_X to a 16-byte boundary. References are the
original matrix multiplied on the CPU in float64, as in test_boo_edges.
Run on a GPU node with --boo-require-cuda; the boo_sanitizer subset also runs
under compute-sanitizer memcheck/racecheck/synccheck (BOO_TESTING.md).
"""

import numpy as np
import pytest
import scipy.sparse as sp
import torch
import torch_fast_spdmm_core as core

from test_boo_edges import CUDA, _assert_product, _build, _example, _x
from torch_fast_spdmm import boo_matmul

FPTS = [(torch.float32, 2), (torch.float32, 4), (torch.float64, 2)]
FPT_IDS = ['f32-fpt2', 'f32-fpt4', 'f64-fpt2']


def _np(dtype):
    return np.float32 if dtype == torch.float32 else np.float64


@CUDA
@pytest.mark.boo_sanitizer
@pytest.mark.parametrize('dtype,fpt', FPTS, ids=FPT_IDS)
@pytest.mark.parametrize('features', [1, 2, 3, 5, 8, 9, 63, 64, 65, 129, 511, 512, 513, 1025])
def test_feature_boundaries(dtype, fpt, features):
    mat, rl, cl = _example(_np(dtype))
    boo = _build(mat, rl, cl).to('cuda')
    _assert_product(mat, boo, _x((2, 11, features), dtype).cuda(), features_per_thread=fpt)


@CUDA
@pytest.mark.parametrize('fpt', [2, 4])
@pytest.mark.parametrize('threads', [128, 256, 512])
@pytest.mark.parametrize('p', [8, 16, 32, 64, 128])
def test_launch_specializations(fpt, threads, p):
    mat, rl, cl = _example()
    boo = _build(mat, rl, cl).to('cuda')
    _assert_product(mat, boo, _x((11, 257)).cuda(), threads=threads, p=p, features_per_thread=fpt)


@CUDA
@pytest.mark.boo_sanitizer
@pytest.mark.parametrize('dtype,fpt', FPTS, ids=FPT_IDS)
def test_large_dynamic_shared_memory_and_strided_work(dtype, fpt):
    # >48 KiB arena; each thread takes several strides through rows and features.
    mat = sp.csr_matrix(np.ones((32, 64), dtype=_np(dtype)))
    boo = _build(mat, np.zeros(32, int), np.zeros(64, int)).to('cuda')
    _assert_product(mat, boo, _x((2, 64, 129), dtype).cuda(), p=64, threads=128,
                    features_per_thread=fpt)


@CUDA
@pytest.mark.boo_sanitizer
@pytest.mark.parametrize('dtype,fpt', FPTS, ids=FPT_IDS)
@pytest.mark.parametrize('max_nnz', [1, 2, 3, 5, 6, 7])
def test_unaligned_smem_x_offsets(dtype, fpt, max_nnz):
    # One block whose nonzero count sets max_nnz_blk; every residue mod 4 shifts
    # the vector-aligned smem_X start differently.
    rows = np.arange(max_nnz) % 3
    cols = np.arange(max_nnz)
    mat = sp.csr_matrix((np.arange(1, max_nnz + 1, dtype=_np(dtype)), (rows, cols)),
                        shape=(3, 8))
    boo = _build(mat, np.zeros(3, int), np.zeros(8, int)).to('cuda')
    assert int(boo.max_nnz_blk) == max_nnz
    _assert_product(mat, boo, _x((8, 72), dtype).cuda(), features_per_thread=fpt)


@CUDA
@pytest.mark.parametrize('dtype,fpt', FPTS, ids=FPT_IDS)
@pytest.mark.parametrize('launch_mode', ['legacy', 'auto'])
def test_random_matrix_with_remainder_matches_fpt1(dtype, fpt, launch_mode):
    rng = np.random.default_rng(0)
    mat = sp.random(301, 257, density=0.05, format='csr', random_state=rng, dtype=_np(dtype))
    # ~5 nonzeros per block: some blocks are retained, others go to the CSR remainder.
    rl, cl = rng.integers(0, 30, 301), rng.integers(0, 26, 257)
    boo = _build(mat, rl, cl, policy='remainder', threshold=4).to('cuda')
    assert boo.compressed_nnz() > 0 and boo.remainder_nnz() > 0
    x = _x((3, 257, 200), dtype).cuda()
    _assert_product(mat, boo, x, features_per_thread=fpt, launch_mode=launch_mode)
    tol = 3e-5 if dtype == torch.float32 else 1e-10
    torch.testing.assert_close(boo_matmul(boo, x, features_per_thread=fpt),
                               boo_matmul(boo, x, features_per_thread=1), rtol=tol, atol=tol)


@CUDA
def test_invalid_and_process_wide_setting():
    mat, rl, cl = _example()
    boo = _build(mat, rl, cl).to('cuda')
    x = _x((11, 16)).cuda()
    with pytest.raises(ValueError):
        boo_matmul(boo, x, features_per_thread=3)
    with pytest.raises(ValueError):
        boo_matmul(boo, x, features_per_thread=2, row_seg=False)
    mat64, rl, cl = _example(np.float64)
    boo64 = _build(mat64, rl, cl).to('cuda')
    with pytest.raises(ValueError):
        boo_matmul(boo64, x.double(), features_per_thread=4)
    with pytest.raises(ValueError):
        core.set_boo_rowseg_fpt(8)
    boo_matmul(boo, x, features_per_thread=4)
    assert core.get_boo_rowseg_fpt() == 4
    boo_matmul(boo, x)  # the default re-applies FPT=1 on every call
    assert core.get_boo_rowseg_fpt() == 1
