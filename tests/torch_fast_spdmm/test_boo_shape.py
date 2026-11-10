"""Rectangular BOO shapes and the arena-padding path used by benchmark.bench_shape.

References are the original matrix multiplied on the CPU in float64. Run on a GPU node with
--boo-require-cuda; the boo_sanitizer subset also runs under compute-sanitizer.
"""

import numpy as np
import pytest
import scipy.sparse as sp
import torch
import torch_fast_spdmm_core as core

from benchmark import bench_shape as bs
from benchmark import nnzflat_common as nc
from test_boo_edges import CUDA, _assert_product, _build, _x

SHAPES = [(32, 32), (64, 32), (128, 32), (256, 32), (128, 16), (256, 16), (256, 64), (32, 128)]
TOL = 3e-5


def _matrix(n=2048, density=0.02, seed=7):
    rng = np.random.default_rng(seed)
    mat = sp.random(n, n, density=density, random_state=rng, format='csr', dtype=np.float32)
    mat.sort_indices()
    mat.indptr = mat.indptr.astype(np.int32)
    mat.indices = mat.indices.astype(np.int32)
    return mat


def _shaped(mat, kr, kc, order='rowmajor'):
    n, m = mat.shape
    return _build(mat, np.arange(n) // kr, np.arange(m) // kc, 'remainder', order)


@pytest.mark.parametrize('kr,kc', SHAPES)
def test_group_sizes_follow_the_shape(kr, kc):
    boo = _shaped(_matrix(), kr, kc)
    assert int(boo.max_nr) == kr
    assert int(boo.max_nc) == kc


@CUDA
@pytest.mark.parametrize('kr,kc', SHAPES)
@pytest.mark.parametrize('p,threads,fpt', [(64, 256, 1), (64, 256, 2), (64, 256, 4), (64, 512, 4),
                                           (32, 128, 2), (128, 256, 4)])
@pytest.mark.parametrize('features', [64, 300])
def test_rectangular_product(kr, kc, p, threads, fpt, features):
    mat = _matrix()
    boo = _shaped(mat, kr, kc).to('cuda')
    x = _x((mat.shape[1], features)).cuda()
    _assert_product(mat, boo, x, threads=threads, p=p, row_seg=True, launch_mode='auto',
                    features_per_thread=fpt)


@pytest.mark.parametrize('arena,target,expected', [(13000, 3, 102 * 32), (13000, 1, 457 * 32),
                                                   (60000, 2, None)])
def test_pad_for_cta(arena, target, expected):
    pad = bs.pad_for_cta(arena, target)
    assert pad == expected
    if pad is not None:
        assert pad % bs.PAD_UNIT == 0
        assert bs.SMEM_PER_SM // (arena + 6 * pad + bs.DRIVER_RESERVED) == target


@CUDA
@pytest.mark.boo_sanitizer
@pytest.mark.parametrize('kr,kc', [(32, 32), (128, 32)])
@pytest.mark.parametrize('p,threads,fpt,target', [(64, 256, 1, 3), (64, 256, 2, 2), (64, 256, 4, 1),
                                                  (64, 512, 4, 2), (32, 256, 2, 2)])
def test_padded_arena_keeps_the_product(kr, kc, p, threads, fpt, target):
    mat = _matrix()
    boo = _shaped(mat, kr, kc).to('cuda')
    features = 300
    x = _x((mat.shape[1], features)).cuda()
    pad = bs.pad_for_cta(nc.arena_rowseg(boo, p), target)
    assert pad and pad % bs.PAD_UNIT == 0
    core.set_boo_rowseg_fpt(fpt)
    try:
        info = core.boo_rowseg_launch_info(int(boo.max_nr), int(boo.max_nc), int(boo.max_nnz_blk) + pad,
                                           features, threads, p)
        assert info['max_active_blocks_per_sm'] == target
        got = bs.matmul_padded(core, boo, x, threads, p, pad)
    finally:
        core.set_boo_rowseg_fpt(1)
    expected = torch.matmul(torch.from_numpy(mat.toarray()).double(), x.cpu().double())
    torch.testing.assert_close(got.cpu().double(), expected, rtol=TOL, atol=TOL)
