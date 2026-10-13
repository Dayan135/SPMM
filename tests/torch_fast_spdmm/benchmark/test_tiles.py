"""Row-segmented BOO P x FPT x feature-slab variants (experiment parameterization).

References are float64 CPU products of the original matrix (test_boo_edges).
Every test restores the process-wide FPT/slab defaults. The boo_sanitizer subset
is the representative boundary/alignment set run under compute-sanitizer.
"""
import numpy as np
import pytest
import scipy.sparse as sp
import torch
import torch_fast_spdmm_core as core

from test_boo_edges import CUDA, _build, _example, _x
from torch_fast_spdmm import boo_matmul
from benchmark import tiles_common as tc

SLABS = [128, 256, 512, 1024]
PS = [16, 32, 64, 128]
FPTS = [1, 2, 4]


@pytest.fixture(autouse=True)
def _restore_defaults():
    yield
    if torch.cuda.is_available():
        tc.apply_settings(core, 1, tc.DEFAULT_SLAB)


def _product(mat, boo, x, p, fpt, slab, **kwargs):
    tc.apply_settings(core, fpt, slab)
    got = boo_matmul(boo, x, p=p, threads=256, row_seg=True, features_per_thread=fpt, **kwargs)
    assert core.get_boo_feat_block() == slab and core.get_boo_rowseg_fpt() == fpt
    expected = torch.matmul(torch.from_numpy(mat.toarray()).double(), x.cpu().double())
    torch.testing.assert_close(got.cpu().double(), expected, rtol=3e-5, atol=3e-5)
    return got


def _random(seed=0, shape=(301, 257), density=0.05):
    rng = np.random.default_rng(seed)
    mat = sp.random(*shape, density=density, format='csr', random_state=rng, dtype=np.float32)
    rl, cl = rng.integers(0, 30, shape[0]), rng.integers(0, 26, shape[1])
    boo = _build(mat, rl, cl, policy='remainder', threshold=4).to('cuda')
    assert boo.compressed_nnz() > 0 and boo.remainder_nnz() > 0
    return mat, boo


@CUDA
@pytest.mark.parametrize('slab', SLABS)
@pytest.mark.parametrize('fpt', FPTS)
@pytest.mark.parametrize('p', PS)
@pytest.mark.parametrize('features', [128, 130, 1000, 1024, 1025])
def test_grid_correctness(features, p, fpt, slab):
    if slab % p:
        pytest.skip('P must tile the slab')
    mat, boo = _random()
    _product(mat, boo, _x((257, features)).cuda(), p, fpt, slab, launch_mode='auto')


@CUDA
@pytest.mark.boo_sanitizer
@pytest.mark.parametrize('slab', SLABS)
@pytest.mark.parametrize('fpt', FPTS)
@pytest.mark.parametrize('features', [1, 7, 130, 1025])
def test_partial_slabs_and_narrow_inputs(features, fpt, slab):
    # Partial final slab, P reduced below the request, batch>1, legacy fork/join.
    mat, rl, cl = _example()
    boo = _build(mat, rl, cl, policy='remainder', threshold=2).to('cuda')
    _product(mat, boo, _x((2, 11, features)).cuda(), 64, fpt, slab, launch_mode='legacy')


@CUDA
@pytest.mark.boo_sanitizer
@pytest.mark.parametrize('slab', [128, 1024])
@pytest.mark.parametrize('fpt', [2, 4])
@pytest.mark.parametrize('max_nnz', [1, 3, 5, 7])
def test_unaligned_smem_x_offsets(max_nnz, fpt, slab):
    rows = np.arange(max_nnz) % 3
    mat = sp.csr_matrix((np.arange(1, max_nnz + 1, dtype=np.float32), (rows, np.arange(max_nnz))),
                        shape=(3, 8))
    boo = _build(mat, np.zeros(3, int), np.zeros(8, int)).to('cuda')
    _product(mat, boo, _x((8, 1030)).cuda(), 128, fpt, slab)


@CUDA
@pytest.mark.boo_sanitizer
@pytest.mark.parametrize('slab', [128, 1024])
def test_large_dynamic_shared_memory(slab):
    mat = sp.csr_matrix(np.ones((32, 64), dtype=np.float32))
    boo = _build(mat, np.zeros(32, int), np.zeros(64, int)).to('cuda')
    _product(mat, boo, _x((64, 1025)).cuda(), 128, 4, slab)


@CUDA
@pytest.mark.parametrize('features,p,slab,grid_z,effective_p', [
    (1024, 64, 128, 8, 64), (1024, 64, 1024, 1, 64), (1025, 128, 256, 5, 128),
    (130, 128, 512, 1, 128), (7, 64, 512, 1, 8), (40, 128, 128, 1, 32)])
def test_launch_info_matches_launcher(features, p, slab, grid_z, effective_p):
    mat, boo = _random()
    limit = torch.cuda.get_device_properties(0).shared_memory_per_block_optin
    info, reason = tc.launch_info(core, boo, features, 256, p, 2, slab, limit)
    assert reason is None
    assert info['grid_z'] == grid_z and info['effective_p'] == effective_p
    assert info['feat_block'] == slab and info['features_per_thread'] == 2
    assert info['arena_bytes'] == core.boo_arena_bytes_rowseg_cuda(
        int(boo.max_nr), int(boo.max_nc), int(boo.max_nnz_blk), effective_p, 4)
    assert info['registers_per_thread'] > 0 and info['max_active_blocks_per_sm'] >= 1
    # launch_info restores the defaults it temporarily applied.
    assert core.get_boo_feat_block() == 512 and core.get_boo_rowseg_fpt() == 1


@CUDA
def test_launched_kernel_is_the_requested_specialization():
    mat, boo = _random()
    x = _x((257, 1024)).cuda()
    limit = torch.cuda.get_device_properties(0).shared_memory_per_block_optin
    for p, fpt, slab in [(32, 4, 256), (128, 1, 1024), (64, 2, 512)]:
        info, reason = tc.launch_info(core, boo, 1024, 256, p, fpt, slab, limit)
        assert reason is None

        def call():
            tc.apply_settings(core, fpt, slab)
            return boo_matmul(boo, x, p=p, features_per_thread=fpt, launch_mode='auto')
        status = tc.verify_launched(tc.launched_kernels(torch, call), info)
        if status.startswith('UNVERIFIED'):
            pytest.skip(status)
        assert status == 'verified', status


@CUDA
def test_f128_slab_partition_equivalence():
    mat, boo = _random(1)
    x = _x((257, 128)).cuda()
    outs = [_product(mat, boo, x, 64, 2, slab) for slab in SLABS]
    for out in outs[1:]:
        torch.testing.assert_close(out, outs[0], rtol=1e-6, atol=1e-6)


@CUDA
def test_infeasible_and_unsupported_are_reported():
    mat, boo = _random()
    limit = torch.cuda.get_device_properties(0).shared_memory_per_block_optin
    with pytest.raises(ValueError):
        core.set_boo_feat_block(384)
    # Non-default slabs exist only for float32/threads=256: rejected, not substituted.
    _, reason = tc.launch_info(core, boo, 512, 128, 64, 1, 256, limit)
    assert reason and 'threads=256' in reason
    core.set_boo_feat_block(256)
    with pytest.raises(ValueError, match='threads=256'):
        boo_matmul(boo, _x((257, 64)).cuda(), threads=512)
    mat64, rl, cl = _example(np.float64)
    boo64 = _build(mat64, rl, cl).to('cuda')
    with pytest.raises(ValueError, match='float32'):
        boo_matmul(boo64, _x((11, 64), torch.float64).cuda())
    core.set_boo_feat_block(512)
    _, reason = tc.launch_info(core, boo, 512, 256, 64, 1, 512, 1024)
    assert reason and 'exceeds' in reason
    _, reason = tc.launch_info(core, boo, 512, 256, 128, 1, 64, limit)
    assert reason and 'multiple' in reason


@CUDA
def test_default_slab_unchanged_for_all_threads_and_dtypes():
    assert core.get_boo_feat_block() == 512
    mat, rl, cl = _example(np.float64)
    boo = _build(mat, rl, cl).to('cuda')
    for threads in (128, 256, 512):
        got = boo_matmul(boo, _x((11, 600), torch.float64).cuda(), threads=threads,
                         features_per_thread=2)
        expected = torch.matmul(torch.from_numpy(mat.toarray()), _x((11, 600), torch.float64))
        torch.testing.assert_close(got.cpu(), expected, rtol=1e-10, atol=1e-10)


def test_warp_row_efficiency_counts_idle_lanes():
    # One block, rows of length 4, 0, 1, 3 (local rows 0..3), 2 rows per warp at lanes=16.
    entries = [(0, c) for c in range(4)] + [(2, 0)] + [(3, c) for c in range(3)]
    r, c = zip(*entries)
    mat = sp.csr_matrix((np.ones(len(r), np.float32), (r, c)), shape=(4, 4))
    boo = _build(mat, np.zeros(4, int), np.zeros(4, int))
    out = tc.warp_row_efficiency(boo, 16)
    assert out['rows_per_warp'] == 2
    assert out['lane_efficiency'] == pytest.approx(8 / (2 * 4 + 2 * 3))
    assert tc.warp_row_efficiency(boo, 32)['lane_efficiency'] == 1.0
