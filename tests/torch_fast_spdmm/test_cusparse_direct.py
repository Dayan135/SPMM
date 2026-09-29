"""Run on the CUDA node after rebuilding the extension."""
import numpy as np
import pytest
import scipy.sparse as sparse
import torch
import torch_fast_spdmm_core as core

ALGORITHMS = ['DEFAULT', 'CSR_ALG1', 'CSR_ALG2', 'CSR_ALG3']


def inputs(empty=False):
    # Rectangular CSR, an empty row, irregular row lengths, and signed values.
    matrix = sparse.csr_matrix(np.array([[1., 0., -2., 3., 0.],
                                         [0., 0., 0., 0., 0.],
                                         [0., 4., 0., 0., 5.]], dtype=np.float32))
    if empty:
        matrix = sparse.csr_matrix((3, 5), dtype=np.float32)
    x = torch.randn(5, 128, device='cuda')
    arrays = [torch.from_numpy(a).to('cuda') for a in
              (matrix.indptr.astype(np.int32), matrix.indices.astype(np.int32), matrix.data)]
    reference = torch.from_numpy(matrix.toarray()).to('cuda') @ x
    return arrays, x, reference


@pytest.mark.parametrize('algorithm', ALGORITHMS)
@pytest.mark.parametrize('empty', [False, True])
def test_direct_beta_zero_persistent_buffers(algorithm, empty):
    arrays, x, reference = inputs(empty)
    plan = core.DirectSpMM(*arrays, x, algorithm)
    if not plan.info()['supported']:
        pytest.skip(plan.info()['unsupported_reason'])
    ptr = plan.output.data_ptr()
    for poison in (float('nan'), 321.):
        plan.output.fill_(poison)
        try:
            actual = plan.run()
        except RuntimeError as exc:
            if empty and 'CUSPARSE_STATUS_NOT_SUPPORTED' in str(exc):
                pytest.skip(str(exc))
            raise
        torch.testing.assert_close(actual, reference, rtol=1e-4, atol=1e-4)
        assert actual.data_ptr() == ptr
    assert np.isfinite(plan.timed_run())
    assert plan.timed_run() > 0
    assert plan.output.data_ptr() == ptr
    torch.testing.assert_close(plan.output, reference, rtol=1e-4, atol=1e-4)


def test_four_plans_have_independent_preprocess_state():
    arrays, x, reference = inputs()
    plans = [core.DirectSpMM(*arrays, x, algorithm) for algorithm in ALGORITHMS]
    for _ in range(3):
        for plan in reversed(plans):
            if plan.info()['supported']:
                torch.testing.assert_close(plan.run(), reference, rtol=1e-4, atol=1e-4)


def test_direct_rejects_invalid_inputs_and_stream_change():
    arrays, x, _ = inputs()
    with pytest.raises(RuntimeError, match='int32'):
        core.DirectSpMM(arrays[0].long(), arrays[1], arrays[2], x, 'DEFAULT')
    with pytest.raises(RuntimeError, match='contiguous'):
        core.DirectSpMM(*arrays, x[:, ::2], 'DEFAULT')
    with pytest.raises(RuntimeError, match='Unknown'):
        core.DirectSpMM(*arrays, x, 'OTHER')
    plan = core.DirectSpMM(*arrays, x, 'DEFAULT')
    with torch.cuda.stream(torch.cuda.Stream()):
        with pytest.raises(RuntimeError, match='construction stream'):
            plan.run()
