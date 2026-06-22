import torch
from pytest import fixture, mark
from torch.testing import assert_close
from torch_fast_spdmm import spdmm_csr_naive, spdmm_csr_shared_mem

from test_utils import TestTorchNoGrad, rel_abs_error


@fixture
def sparse_format():
    return torch.sparse_csr


@fixture
def batch_size():
    return None


@fixture
def device():
    return 'cuda'


@mark.parametrize('index_dtype', [torch.int32, torch.int64], ids=['int32', 'int64'])
@mark.parametrize('data_dtype', [torch.float32, torch.float64], ids=['float32', 'float64'])
@mark.parametrize('sparse_data_mat', ['sparse_identity_mat', 'sparse_laplacian_2d_mat'],
                  ids=['eye', 'laplacian'], indirect=True)
@mark.parametrize('system_size', [64, 256])
@mark.parametrize('features', [16, 32, 64, 128, 256, 512])
class TestCSR(TestTorchNoGrad):
    def test_csr_naive(self, sparse_mat, dense_mat, system_size):
        expected = torch.sparse.mm(sparse_mat, dense_mat)
        actual = spdmm_csr_naive(sparse_mat, dense_mat)

        assert_close(actual, expected)
        assert rel_abs_error(actual, expected)

    def test_csr_shared_mem(self, sparse_mat, dense_mat, system_size):
        expected = torch.sparse.mm(sparse_mat, dense_mat)
        actual = spdmm_csr_shared_mem(sparse_mat, dense_mat)

        assert_close(actual, expected)
        assert rel_abs_error(actual, expected)
