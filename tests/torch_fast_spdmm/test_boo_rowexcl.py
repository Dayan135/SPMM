"""GPU correctness and ownership checks for the row-exclusive execution mode."""
import numpy as np
import pytest
import scipy.sparse as sparse
import torch
import torch_fast_spdmm as spmm


@pytest.mark.parametrize('slab', [16, 32, 64])
@pytest.mark.parametrize('features', [1, 16, 31, 32, 63, 64, 100, 128, 200, 512, 520])
@pytest.mark.parametrize('policy', ['keep', 'remainder'])
def test_row_exclusive_complete_output(slab, features, policy):
    rng = np.random.default_rng(12)
    a = sparse.random(23, 37, density=.22, random_state=rng,
                      data_rvs=lambda n: rng.normal(size=n), dtype=np.float32).tocsr()
    a.data[(a.indices % 7) == 0] *= -1
    a = a.tolil()
    a[0, :] = 0; a[7, :] = 0; a[22, :] = 0
    # An indivisible long row forces the effective nnz cap above the request.
    a[3, :] = np.linspace(-1, 1, 37, dtype=np.float32)
    a = a.tocsr(); a.eliminate_zeros(); a.sort_indices()
    # Noncontiguous row/column groups exercise gather mapping and empty rows.
    boo = spmm.csr_to_boo(a, row_labels=np.arange(23)%4, col_labels=np.arange(37)%7,
                          small_block_policy=policy, small_block_threshold=4).to('cuda')
    plan = spmm.prepare_boo_rowexcl(boo, slab=slab, max_cta_nnz=8,
                                   max_cta_rows=3, smem_bytes=slab*4*6)
    x_cpu = torch.randn(37, features, generator=torch.Generator().manual_seed(9))
    x = x_cpu.cuda()
    expected = a.astype(np.float64) @ x_cpu.numpy().astype(np.float64)
    runtime = plan.bind(x)
    ptr = runtime.output.data_ptr()
    for poison in (float('nan'), 1234.):
        runtime.output.fill_(poison)
        np.testing.assert_allclose(runtime.run().cpu().numpy(), expected, atol=1e-4, rtol=1e-4)
        assert runtime.output.data_ptr() == ptr
    assert runtime.timed_run() > 0
    np.testing.assert_allclose(runtime.output.cpu().numpy(), expected, atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(spmm.boo_rowexcl_matmul(plan, x).cpu().numpy(), expected, atol=1e-4, rtol=1e-4)
    assert plan.stats['cta_nnz_max'] <= plan.stats['effective_nnz_cap']
    assert plan.stats['effective_nnz_cap'] == plan.stats['max_row_nnz']
    assert plan.stats['split_row_groups'] > 0
    assert plan.stats['chunked_units'] > 0
    starts, ends = [t.cpu().numpy() for t in plan.schedule[1:3]]
    owned = np.concatenate([np.arange(lo, hi) for lo, hi in zip(starts, ends)])
    np.testing.assert_array_equal(owned, np.arange(23))
    assert np.max(ends-starts) <= 3


@pytest.mark.parametrize('slab', [16, 32, 64])
@pytest.mark.parametrize('density', [0., .03, 1.])
def test_empty_groups_and_pure_remainder(slab, density):
    rng = np.random.default_rng(5)
    a = sparse.random(19, 27, density=density, random_state=rng, dtype=np.float32).tocsr()
    boo = spmm.csr_to_boo(a, row_labels=np.arange(19)//4, col_labels=np.arange(27)//3,
                          small_block_policy='remainder', small_block_threshold=100000).to('cuda')
    plan = spmm.prepare_boo_rowexcl(boo, slab=slab)
    assert plan.stats['num_chunks'] == 0
    x = torch.randn(27, 100, device='cuda')
    runtime = plan.bind(x); runtime.output.fill_(float('nan'))
    actual = runtime.run().cpu().numpy()
    np.testing.assert_allclose(actual, a.astype(np.float64) @ x.cpu().numpy().astype(np.float64), rtol=1e-4, atol=1e-4)
    if density == 0:
        assert plan.stats['zero_nnz_units'] == plan.stats['num_units']
        assert np.all(actual == 0)


def test_input_validation_and_stale_plan():
    a = sparse.eye(9, format='csr', dtype=np.float32)
    boo = spmm.csr_to_boo(a).to('cuda')
    plan = spmm.prepare_boo_rowexcl(boo)
    with pytest.raises(RuntimeError, match='contiguous CUDA FP32'):
        plan.bind(torch.randn(9, 200, device='cuda')[:, ::2])
    with pytest.raises(RuntimeError, match='contiguous CUDA FP32'):
        plan.bind(torch.randn(9, 100, device='cuda', dtype=torch.float64))
    runtime = plan.bind(torch.randn(9, 100, device='cuda'))
    with torch.cuda.stream(torch.cuda.Stream()):
        with pytest.raises(RuntimeError, match='construction stream'):
            runtime.run()
    boo.to('cpu')
    with pytest.raises(ValueError, match='staging'):
        spmm.boo_rowexcl_matmul(plan, torch.randn(9, 100, device='cuda'))


def test_shared_preallocated_output():
    a = sparse.eye(9, format='csr', dtype=np.float32)
    boo = spmm.csr_to_boo(a).to('cuda')
    x = torch.randn(9, 100, device='cuda')
    out = torch.empty_like(x)
    plans = [spmm.prepare_boo_rowexcl(boo, slab=s) for s in (16, 32, 64)]
    for plan in plans:
        bound = plan.bind(x, out=out)
        out.fill_(float('nan'))
        assert bound.output.data_ptr() == out.data_ptr()
        bound.timed_run()
        torch.testing.assert_close(out, x)
    with pytest.raises(RuntimeError):
        plans[0].bind(x, out=x)
    with pytest.raises(RuntimeError, match='Y must be'):
        plans[0].bind(x, out=out[:, :50])
