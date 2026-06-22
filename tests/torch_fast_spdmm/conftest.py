import numpy as np
import scipy.sparse as sp
import torch
from pytest import fixture
from pytest_benchmark.plugin import BenchmarkSession, NameWrapper
from scipy.sparse.linalg import LaplacianNd

from cuda_benchmark import CUDABenchmarkFixture
from test_utils import scipy_dtypes, torch_dtypes


# --------------------------------------------------------------------------- #
# dtype + sparse-format conversion helpers
# (self-contained extraction of integration.conversions from the fem repo)
# --------------------------------------------------------------------------- #
def numpy_dtype_to_torch_dtype(dtype):
    mapping = {
        np.float64: torch.float64, np.float32: torch.float32, np.float16: torch.float16,
        np.int64: torch.int64, np.int32: torch.int32, np.int16: torch.int16,
        np.int8: torch.int8, np.uint8: torch.uint8, bool: torch.bool,
    }
    return mapping[np.dtype(dtype).type]


def torch_dtype_to_numpy_dtype(dtype):
    mapping = {
        torch.float64: np.float64, torch.float32: np.float32, torch.float16: np.float16,
        torch.int64: np.int64, torch.int32: np.int32, torch.int16: np.int16,
        torch.int8: np.int8, torch.uint8: np.uint8, torch.bool: bool,
    }
    if isinstance(dtype, torch.dtype):
        return mapping[dtype]
    # numpy dtype passthrough (the fem tests parametrize with numpy dtypes too)
    return np.dtype(dtype).type


def scipy_csr_to_torch_csr(mat):
    mat = mat.tocsr()
    mat.sort_indices()
    return torch.sparse_csr_tensor(torch.tensor(mat.indptr), torch.tensor(mat.indices),
                                   torch.tensor(mat.data), mat.shape,
                                   dtype=numpy_dtype_to_torch_dtype(mat.dtype))


def scipy_csc_to_torch_csc(mat):
    mat = mat.tocsc()
    return torch.sparse_csc_tensor(torch.tensor(mat.indptr), torch.tensor(mat.indices),
                                   torch.tensor(mat.data), mat.shape,
                                   dtype=numpy_dtype_to_torch_dtype(mat.dtype))


def scipy_coo_to_torch_coo(mat):
    mat = mat.tocoo()
    return torch.sparse_coo_tensor(torch.tensor(np.array([mat.row, mat.col])),
                                   torch.tensor(mat.data), mat.shape,
                                   dtype=numpy_dtype_to_torch_dtype(mat.dtype))


# --------------------------------------------------------------------------- #
# CUDA-aware benchmark fixture (replaces pytest-benchmark's 'benchmark')
# --------------------------------------------------------------------------- #
@fixture
def benchmark(request):
    bs: BenchmarkSession = request.config._benchmarksession

    if bs.skip:
        from pytest import skip
        skip("Benchmarks are skipped (--benchmark-skip was used).")

    node = request.node
    marker = node.get_closest_marker("benchmark")
    options = dict(marker.kwargs) if marker else {}
    if "timer" in options:
        options["timer"] = NameWrapper(options["timer"])

    fixture = CUDABenchmarkFixture(
        node,
        add_stats=bs.benchmarks.append,
        logger=bs.logger,
        warner=node.warn,
        disabled=bs.disabled,
        **dict(bs.options, **options),
    )
    yield fixture
    fixture._cleanup()


# --------------------------------------------------------------------------- #
# Named sparse-matrix data fixtures (programmatic, no data files)
# --------------------------------------------------------------------------- #
@fixture
def sparse_identity_mat(system_size, data_dtype):
    return sp.eye(system_size, dtype=torch_dtype_to_numpy_dtype(data_dtype))


@fixture
def sparse_permutation_mat(system_size, data_dtype):
    np_dtype = torch_dtype_to_numpy_dtype(data_dtype)
    p = np.random.permutation(system_size)
    data = np.ones(system_size, dtype=np_dtype)
    return sp.csr_matrix((data, (np.arange(system_size), p)),
                         shape=(system_size, system_size))


@fixture
def sparse_laplacian_2d_mat(system_size, data_dtype):
    sqrt_size = int(np.sqrt(system_size))
    if sqrt_size * sqrt_size != system_size:
        raise ValueError('system_size must be a perfect square for 2D Laplacian matrix.')
    mat = -1 * LaplacianNd((sqrt_size, sqrt_size)).tosparse()
    return scipy_csr_to_torch_csr(mat).to(dtype=data_dtype)


@fixture
def sparse_random_low_sparsity_mat(system_size, data_dtype):
    return sp.random(system_size, system_size, density=0.9,
                     dtype=torch_dtype_to_numpy_dtype(data_dtype))


# --------------------------------------------------------------------------- #
# Core fixtures: sparse_data_mat -> sparse_mat -> dense_mat
# (defaults below are overridden per-file by the test modules)
# --------------------------------------------------------------------------- #
@fixture
def sparse_data_mat(request):
    return request.getfixturevalue(request.param)


@fixture
def batch_size():
    return None


@fixture
def device():
    return 'cpu'


@fixture
def sparse_format():
    return torch.sparse_csr


def sparse_mat_factory(sparse_data_mat, sparse_format, batch_size,
                       data_dtype, index_dtype, device=None):
    match sparse_format:
        case torch.sparse_coo:
            sparse_data_mat = (sparse_data_mat.to_sparse_coo().coalesce()
                               if isinstance(sparse_data_mat, torch.Tensor)
                               else scipy_coo_to_torch_coo(sparse_data_mat.tocoo()))
        case torch.sparse_csr:
            sparse_data_mat = (sparse_data_mat.to_sparse_csr()
                               if isinstance(sparse_data_mat, torch.Tensor)
                               else scipy_csr_to_torch_csr(sparse_data_mat.tocsr()))
        case torch.sparse_csc:
            sparse_data_mat = (sparse_data_mat.to_sparse_csc()
                               if isinstance(sparse_data_mat, torch.Tensor)
                               else scipy_csc_to_torch_csc(sparse_data_mat.tocsc()))
        case sp.csr_matrix:
            if isinstance(sparse_data_mat, sp.spmatrix):
                sparse_data_mat = sparse_data_mat.tocsr()
            else:
                raise NotImplementedError('torch -> scipy csr conversion is unused by these tests')
        case _:
            raise ValueError(f'Unsupported sparse format: {sparse_format}')

    if sparse_format in (torch.sparse_csr, torch.sparse_csc, torch.sparse_coo):
        sparse_data_mat = torch_dtypes(sparse_data_mat, data_dtype, index_dtype)
        sparse_data_mat = sparse_data_mat.to(device=device)
    else:
        sparse_data_mat = scipy_dtypes(sparse_data_mat, data_dtype, index_dtype)

    return sparse_data_mat


@fixture
def sparse_mat(sparse_data_mat, sparse_format, batch_size, index_dtype, data_dtype, device):
    return sparse_mat_factory(sparse_data_mat, sparse_format, batch_size,
                              data_dtype, index_dtype, device)


@fixture
def dense_mat(sparse_mat, features):
    data_dtype = sparse_mat.dtype
    device = sparse_mat.device

    if len(sparse_mat.shape) == 2:
        system_size = sparse_mat.shape[0]
        batch_size = None
    else:
        system_size = sparse_mat.shape[1]
        batch_size = sparse_mat.shape[0]

    shape = (system_size,)
    if features is not None:
        shape = shape + (features,)
    if batch_size is not None:
        shape = (batch_size,) + shape

    return torch.randn(shape, dtype=data_dtype, device=device)
