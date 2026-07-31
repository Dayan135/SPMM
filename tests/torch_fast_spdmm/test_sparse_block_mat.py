import torch
from pytest import fixture, mark
from torch.testing import assert_close

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


@fixture
def sparse_data_mat(sparse_random_block_mat):
    return sparse_random_block_mat


@mark.parametrize('index_dtype', [torch.int32, torch.int64], ids=['int32', 'int64'])
@mark.parametrize('data_dtype', [torch.float32, torch.float64], ids=['float32', 'float64'])
@mark.parametrize('features', [16, 32, 64, 128, 256, 512], ids=lambda x: f'f={x}')
class TestBlockMat(TestTorchNoGrad):
    @mark.skip(reason="TODO stub: block-format round-trip not implemented, "
                       "and sparse_data_mat depends on a sparse_random_block_mat "
                       "fixture that doesn't exist yet")
    def test_block_mat_sanity(self, sparse_mat, dense_mat):
        expected = torch.sparse.mm(sparse_mat, dense_mat)

        # TODO: Convert to block format, multiply, convert back, etc
        actual = ...

        assert_close(actual, expected)
        assert rel_abs_error(actual, expected)
