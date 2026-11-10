"""nnzflat (row_seg=False) features-per-thread template: FPT=1 control and FPT=2.

The original boo_spmm_nnzflat_kernel stays the FPT=1 reference
(nnzflat_impl='reference'); boo_spmm_nnzflat_fpt_kernel<..., FPT> changes only
Step B's work mapping to (nonzero, FPT adjacent features) and issues FPT scalar
shared atomics. References are the original matrix multiplied on the CPU in
float64, as in test_boo_edges. Run on a GPU node with --boo-require-cuda; the
boo_sanitizer subset also runs under compute-sanitizer memcheck/racecheck/synccheck.
"""

import numpy as np
import pytest
import scipy.sparse as sp
import torch
import torch_fast_spdmm_core as core

from test_boo_edges import CUDA, _assert_product, _build, _example, _x
from torch_fast_spdmm import boo_matmul

# (dtype, fpt, impl): the template FPT=1 control, FPT=2 in both precisions.
VARIANTS = [(torch.float32, 1, 'template'), (torch.float32, 2, 'template'),
            (torch.float64, 1, 'template'), (torch.float64, 2, 'template')]
VARIANT_IDS = ['f32-tfpt1', 'f32-fpt2', 'f64-tfpt1', 'f64-fpt2']
ORDERS = ['rowmajor', 'colmajor']


def _np(dtype):
    return np.float32 if dtype == torch.float32 else np.float64


def _nnzflat(**kwargs):
    return dict(row_seg=False, **kwargs)


@CUDA
@pytest.mark.boo_sanitizer
@pytest.mark.parametrize('order', ORDERS)
@pytest.mark.parametrize('dtype,fpt,impl', VARIANTS, ids=VARIANT_IDS)
@pytest.mark.parametrize('features', [1, 2, 7, 8, 9, 33, 64, 65, 129, 511, 512, 513, 1025])
def test_feature_boundaries(order, dtype, fpt, impl, features):
    mat, rl, cl = _example(_np(dtype))
    boo = _build(mat, rl, cl, order=order).to('cuda')
    _assert_product(mat, boo, _x((2, 11, features), dtype).cuda(),
                    **_nnzflat(features_per_thread=fpt, nnzflat_impl=impl))


@CUDA
@pytest.mark.parametrize('order', ORDERS)
@pytest.mark.parametrize('fpt', [1, 2])
@pytest.mark.parametrize('threads', [128, 256, 512])
@pytest.mark.parametrize('p', [8, 16, 32, 64, 128])
def test_launch_specializations(order, fpt, threads, p):
    mat, rl, cl = _example()
    boo = _build(mat, rl, cl, order=order).to('cuda')
    _assert_product(mat, boo, _x((11, 257)).cuda(), threads=threads, p=p,
                    **_nnzflat(features_per_thread=fpt, nnzflat_impl='template'))


@CUDA
@pytest.mark.boo_sanitizer
@pytest.mark.parametrize('dtype,fpt,impl', VARIANTS, ids=VARIANT_IDS)
def test_large_dynamic_shared_memory_and_strided_work(dtype, fpt, impl):
    # >48 KiB arena (float64 smem_X+smem_Y at P64); 2048 nonzeros x P/FPT items
    # give every thread many Step-B strides.
    mat = sp.csr_matrix(np.ones((32, 64), dtype=_np(dtype)))
    boo = _build(mat, np.zeros(32, int), np.zeros(64, int)).to('cuda')
    _assert_product(mat, boo, _x((2, 64, 129), dtype).cuda(), p=64, threads=128,
                    **_nnzflat(features_per_thread=fpt, nnzflat_impl=impl))


@CUDA
@pytest.mark.boo_sanitizer
@pytest.mark.parametrize('dtype,fpt,impl', VARIANTS, ids=VARIANT_IDS)
@pytest.mark.parametrize('max_nnz', [1, 2, 3, 5, 6, 7])
def test_unaligned_smem_x_offsets(dtype, fpt, impl, max_nnz):
    # Every residue of max_nnz_blk mod 4 shifts the 16-byte-aligned smem_X start.
    rows = np.arange(max_nnz) % 3
    cols = np.arange(max_nnz)
    mat = sp.csr_matrix((np.arange(1, max_nnz + 1, dtype=_np(dtype)), (rows, cols)), shape=(3, 8))
    boo = _build(mat, np.zeros(3, int), np.zeros(8, int)).to('cuda')
    assert int(boo.max_nnz_blk) == max_nnz
    _assert_product(mat, boo, _x((8, 72), dtype).cuda(),
                    **_nnzflat(features_per_thread=fpt, nnzflat_impl=impl))


@CUDA
@pytest.mark.boo_sanitizer
@pytest.mark.parametrize('order', ORDERS)
@pytest.mark.parametrize('fpt', [1, 2])
@pytest.mark.parametrize('pattern', ['balanced', 'full_rows', 'skewed'])
def test_row_imbalance_and_shared_atomic_contention(order, fpt, pattern):
    # One 64x64 block: equal total nnz, very different local row lengths. 'full_rows'
    # packs every nonzero into 8 full rows (heavy shared-atomic row contention).
    rng = np.random.default_rng(7)
    n, total = 64, 512
    if pattern == 'balanced':
        lengths = np.full(n, total // n)
    elif pattern == 'full_rows':
        lengths = np.zeros(n, int)
        lengths[5:5 + total // n] = n  # 8 full rows, the rest empty
    else:
        lengths = np.zeros(n, int)
        lengths[:7] = n          # 448 nonzeros in 7 full rows
        lengths[7:] = 1          # 57 singletons
        lengths[7:14] += 1       # 7 doubletons

    assert lengths.sum() == total and lengths.max() <= n
    r = np.repeat(np.arange(n), lengths)
    c = np.concatenate([rng.choice(n, k, replace=False) for k in lengths])
    mat = sp.csr_matrix((rng.standard_normal(total).astype(np.float32), (r, c)), shape=(n, n))
    boo = _build(mat, np.zeros(n, int), np.zeros(n, int), order=order).to('cuda')
    for threads in (128, 512):
        for p in (8, 64):
            _assert_product(mat, boo, _x((n, 200)).cuda(), threads=threads, p=p,
                            **_nnzflat(features_per_thread=fpt, nnzflat_impl='template'))


@CUDA
@pytest.mark.boo_sanitizer
@pytest.mark.parametrize('order', ORDERS)
@pytest.mark.parametrize('fpt', [1, 2])
@pytest.mark.parametrize('policy,threshold', [('remainder', 4), ('tail_batch', 4),
                                              ('remainder', 1000), ('keep', 4)])
def test_mixed_and_all_spilled(order, fpt, policy, threshold):
    rng = np.random.default_rng(0)
    mat = sp.random(301, 257, density=0.05, format='csr', random_state=rng, dtype=np.float32)
    rl, cl = rng.integers(0, 30, 301), rng.integers(0, 26, 257)
    boo = _build(mat, rl, cl, policy=policy, threshold=threshold, order=order).to('cuda')
    if policy == 'remainder' and threshold == 4:
        assert boo.compressed_nnz() > 0 and boo.remainder_nnz() > 0
    if threshold == 1000:
        assert boo.compressed_nnz() == 0
    for csr_mode in (0, 1, 2):
        for launch_mode in ('legacy', 'auto'):
            _assert_product(mat, boo, _x((3, 257, 65)).cuda(), csr_mode=csr_mode,
                            launch_mode=launch_mode,
                            **_nnzflat(features_per_thread=fpt, nnzflat_impl='template'))


@CUDA
@pytest.mark.parametrize('order', ORDERS)
@pytest.mark.parametrize('fpt', [1, 2])
def test_duplicates_explicit_zeros_and_cancellation(order, fpt):
    r = np.array([0, 0, 0, 1, 1, 2, 2, 2, 3])
    c = np.array([1, 1, 4, 2, 2, 0, 3, 3, 4])
    v = np.array([2.0, -2.0, 0.0, 1.5, 0.5, 0.0, 3.0, -1.0, 7.0], dtype=np.float32)
    mat = sp.csr_matrix((v, (r, c)), shape=(4, 5))  # sums duplicates; keeps zeros
    unsorted = sp.csr_matrix((v, c, np.array([0, 3, 5, 8, 9])), shape=(4, 5))
    for m in (mat, unsorted):
        boo = _build(m, np.array([0, 1, 0, 1]), np.array([0, 0, 1, 1, 0]), order=order).to('cuda')
        _assert_product(m, boo, _x((5, 33)).cuda(),
                        **_nnzflat(features_per_thread=fpt, nnzflat_impl='template'))


@CUDA
@pytest.mark.parametrize('order', ORDERS)
@pytest.mark.parametrize('fpt', [1, 2])
def test_many_blocks_accumulate_into_same_rows(order, fpt):
    # 257 single-column groups all feed the same 3 output rows.
    rng = np.random.default_rng(3)
    mat = sp.csr_matrix(rng.standard_normal((3, 257 * 2)).astype(np.float32))
    boo = _build(mat, np.zeros(3, int), np.arange(257 * 2) // 2, order=order).to('cuda')
    assert int(boo.num_blocks()) == 257
    _assert_product(mat, boo, _x((257 * 2, 70)).cuda(),
                    **_nnzflat(features_per_thread=fpt, nnzflat_impl='template'))


@CUDA
@pytest.mark.parametrize('seed', range(6))
def test_seeded_randomized_product(seed):
    rng = np.random.default_rng(seed)
    n_rows, n_cols = rng.integers(20, 200, 2)
    mat = sp.random(n_rows, n_cols, density=float(rng.uniform(0.02, 0.3)), format='csr',
                    random_state=rng, dtype=np.float32)
    rl = rng.integers(0, max(1, n_rows // 8), n_rows)
    cl = rng.integers(0, max(1, n_cols // 8), n_cols)
    features = int(rng.integers(1, 700))
    x = _x((2, n_cols, features), seed=seed).cuda()
    for order in ORDERS:
        boo = _build(mat, rl, cl, policy='remainder', order=order).to('cuda')
        for fpt in (1, 2):
            _assert_product(mat, boo, x, p=[8, 16, 32, 64, 128][seed % 5],
                            threads=[128, 256, 512][seed % 3],
                            **_nnzflat(features_per_thread=fpt, nnzflat_impl='template'))


@CUDA
@pytest.mark.boo_sanitizer
@pytest.mark.parametrize('fpt', [1, 2])
def test_nondefault_stream_and_interleaved_settings(fpt):
    mat, rl, cl = _example()
    boo = _build(mat, rl, cl).to('cuda')
    x = _x((2, 11, 40)).cuda()
    expected = torch.matmul(torch.from_numpy(mat.toarray()).double(), x.cpu().double())
    stream = torch.cuda.Stream()
    outputs = []
    with torch.cuda.stream(stream):
        for i in range(6):
            # rowseg FPT4 / nnzflat reference / nnzflat template interleaved on one stream.
            kwargs = [dict(row_seg=True, features_per_thread=4),
                      dict(row_seg=False, features_per_thread=fpt, nnzflat_impl='template'),
                      dict(row_seg=False)][i % 3]
            outputs.append(boo_matmul(boo, x, **kwargs) * 2 + 1)
    stream.synchronize()
    for out in outputs:
        torch.testing.assert_close(((out - 1) / 2).cpu().double(), expected, rtol=3e-5, atol=3e-5)


@CUDA
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('order', ORDERS)
def test_template_fpt1_matches_reference(dtype, order):
    rng = np.random.default_rng(11)
    mat = sp.random(400, 300, density=0.05, format='csr', random_state=rng, dtype=_np(dtype))
    rl, cl = rng.integers(0, 20, 400), rng.integers(0, 15, 300)
    boo = _build(mat, rl, cl, policy='remainder', order=order).to('cuda')
    x = _x((300, 130), dtype).cuda()
    ref = boo_matmul(boo, x, row_seg=False, nnzflat_impl='reference')
    tmpl = boo_matmul(boo, x, row_seg=False, nnzflat_impl='template')
    fpt2 = boo_matmul(boo, x, row_seg=False, features_per_thread=2)
    tol = 3e-5 if dtype == torch.float32 else 1e-10
    torch.testing.assert_close(tmpl, ref, rtol=tol, atol=tol)
    torch.testing.assert_close(fpt2, ref, rtol=tol, atol=tol)


@CUDA
def test_settings_api_and_rejections():
    mat, rl, cl = _example()
    boo = _build(mat, rl, cl).to('cuda')
    x = _x((11, 16)).cuda()
    with pytest.raises(ValueError):
        boo_matmul(boo, x, row_seg=False, features_per_thread=4)
    with pytest.raises(ValueError):
        boo_matmul(boo, x, row_seg=False, features_per_thread=2, nnzflat_impl='reference')
    with pytest.raises(ValueError):
        boo_matmul(boo, x, row_seg=True, nnzflat_impl='template')
    with pytest.raises(ValueError):
        boo_matmul(boo, x, row_seg=False, nnzflat_impl='vector')
    with pytest.raises(ValueError):
        core.set_boo_nnzflat_fpt(4)
    # The launcher itself rejects a reference FPT=2 request (no silent fallback).
    core.set_boo_nnzflat_fpt(2)
    core.set_boo_nnzflat_reference(True)
    with pytest.raises((ValueError, RuntimeError)):
        core.boo_nnzflat_launch_info(1, 1, 4, 4, 8, 16, 256, 8)
    boo_matmul(boo, x, row_seg=False, features_per_thread=2)
    assert (core.get_boo_nnzflat_fpt(), core.get_boo_nnzflat_reference(),
            core.get_boo_rowseg_fpt()) == (2, False, 1)
    boo_matmul(boo, x, row_seg=True, features_per_thread=4)
    assert (core.get_boo_nnzflat_fpt(), core.get_boo_nnzflat_reference(),
            core.get_boo_rowseg_fpt()) == (1, True, 4)
    boo_matmul(boo, x, row_seg=False)  # default nnzflat: the original reference kernel
    assert (core.get_boo_nnzflat_fpt(), core.get_boo_nnzflat_reference(),
            core.get_boo_rowseg_fpt()) == (1, True, 1)
    mat64, rl, cl = _example(np.float64)
    boo64 = _build(mat64, rl, cl).to('cuda')
    boo_matmul(boo64, x.double(), row_seg=False, features_per_thread=2)  # float64 FPT2 supported


@CUDA
@pytest.mark.parametrize('order', ORDERS)
@pytest.mark.parametrize('fpt', [1, 2])
def test_shared_memory_overflow_is_reported(order, fpt):
    # A dense 256x256 block cannot fit any arena: an explicit CUDA error, never a fallback.
    mat = sp.csr_matrix(np.ones((256, 256), dtype=np.float64))
    boo = _build(mat, np.zeros(256, int), np.zeros(256, int), order=order).to('cuda')
    with pytest.raises(RuntimeError):
        boo_matmul(boo, _x((256, 8), torch.float64).cuda(), p=8,
                   **_nnzflat(features_per_thread=fpt, nnzflat_impl='template'))


@CUDA
@pytest.mark.parametrize('impl,fpt', [('reference', 1), ('template', 1), ('template', 2)])
@pytest.mark.parametrize('threads', [128, 256, 512])
@pytest.mark.parametrize('p', [8, 16, 32, 64, 128])
def test_launch_info_and_launched_kernel(impl, fpt, threads, p):
    from torch.profiler import ProfilerActivity, profile
    mat, rl, cl = _example()
    boo = _build(mat, rl, cl).to('cuda')
    features = 200
    x = _x((11, features)).cuda()
    core.set_boo_nnzflat_fpt(fpt)
    core.set_boo_nnzflat_reference(impl == 'reference')
    info = core.boo_nnzflat_launch_info(int(boo.num_blocks()), 1, int(boo.max_nr), int(boo.max_nc),
                                        int(boo.max_nnz_blk), features, threads, p)
    assert info['features_per_thread'] == fpt and info['reference'] == (impl == 'reference')
    assert info['effective_p'] == p and info['threads'] == threads and info['feat_block'] == 512
    assert info['grid_x'] == int(boo.num_blocks()) and info['grid_z'] == 1
    assert info['arena_bytes'] == core.boo_arena_bytes_cuda(int(boo.max_nr), int(boo.max_nc),
                                                            int(boo.max_nnz_blk), p, 4)
    assert info['max_active_blocks_per_sm'] >= 1
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        boo_matmul(boo, x, threads=threads, p=p, launch_mode='auto',
                   **_nnzflat(features_per_thread=fpt, nnzflat_impl=impl))
        torch.cuda.synchronize()
    names = [e.name for e in prof.events() if 'boo_spmm' in e.name]
    pl = p.bit_length() - 1
    want = (f'boo_spmm_nnzflat_kernel<float, {threads}, {pl}>' if impl == 'reference'
            else f'boo_spmm_nnzflat_fpt_kernel<float, {threads}, {pl}, {fpt}>')
    assert names and all(want in n for n in names), names


@CUDA
@pytest.mark.parametrize('fpt,reference', [(1, True), (1, False), (2, False)])
def test_cpp_timed_wrapper(fpt, reference):
    rng = np.random.default_rng(5)
    mat = sp.random(301, 257, density=0.05, format='csr', random_state=rng, dtype=np.float32)
    boo = _build(mat, rng.integers(0, 30, 301), rng.integers(0, 26, 257),
                 policy='remainder').to('cuda')
    x = _x((257, 96)).cuda()
    m = boo._gpu_meta
    args = (boo.rows, boo.cols, boo.max_nr, boo.max_nc, boo.max_nnz_blk,
            m.row_grp_ptr, m.row_ids, m.col_grp_ptr, m.col_ids, m.blk_row_grp, m.blk_col_grp,
            m.blk_nz_ptr, m.blk_shift, m.nz_idx, m.nz_val, m.rem_indptr, m.rem_col_ind, m.rem_vals,
            m.tail_row, m.tail_col, m.tail_val, x, 256, 16, 0)
    timed = core.BOOTimedSpMM(*args, False, False, fpt, reference)
    core.set_boo_nnzflat_fpt(1)          # the wrapper must re-apply its own setting
    core.set_boo_nnzflat_reference(True)
    assert timed.timed_run() > 0
    assert (core.get_boo_nnzflat_fpt(), core.get_boo_nnzflat_reference()) == (fpt, reference)
    expected = torch.from_numpy(mat.toarray()).double() @ x.cpu().double()
    torch.testing.assert_close(timed.run().cpu().double(), expected, rtol=3e-5, atol=3e-5)
    assert timed.info()['nnzflat_fpt'] == fpt
    with pytest.raises(RuntimeError):
        core.BOOTimedSpMM(*args, False, False, 2, True)
    with pytest.raises(RuntimeError):
        core.BOOTimedSpMM(*args, True, False, 2, False)
