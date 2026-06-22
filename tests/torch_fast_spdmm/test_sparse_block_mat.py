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
@mark.parametrize('block_width', [8], ids=lambda x: f'bw={x}')
@mark.parametrize('block_height', [8], ids=lambda x: f'bh={x}')
@mark.parametrize('mat_width', [8 * 8], ids=lambda x: f'mw={x}')
@mark.parametrize('mat_height', [8 * 8], ids=lambda x: f'mh={x}')
@mark.parametrize('block_prob', [0.4], ids=lambda x: f'bp={x}')
@mark.parametrize('block_density', [0.4], ids=lambda x: f'bd={x}')
@mark.parametrize('features', [16, 32, 64, 128, 256, 512], ids=lambda x: f'f={x}')
class TestBlockMat(TestTorchNoGrad):
    def test_block_mat_sanity(self, sparse_mat, dense_mat):
        expected = torch.sparse.mm(sparse_mat, dense_mat)

        # TODO: Convert to block format, multiply, convert back, etc
        actual = ...

        assert_close(actual, expected)
        assert rel_abs_error(actual, expected)
