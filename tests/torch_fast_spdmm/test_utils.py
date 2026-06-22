import numpy as np
import torch
from scipy import sparse as sp


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
    # numpy dtype passthrough (some tests parametrize with numpy dtypes)
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


class TestTorchNoGrad:
    def setup_method(self):
        torch.set_grad_enabled(False)

    def teardown_method(self):
        torch.set_grad_enabled(True)


def rel_abs_error(t1: torch.Tensor, t2: torch.Tensor) -> torch.float64:
    total_error = ((t1 - t2).abs().sum() / t1.abs().sum()).to(torch.float64).item()

    return total_error < torch.finfo(t1.dtype).eps


def scipy_coo_dtypes(mat: sp.coo_matrix, data_dtype, index_dtype):
    mat = mat.copy()
    mat.indices = mat.indices.astype(index_dtype)
    mat.data = mat.data.astype(data_dtype)

    return mat


def scipy_csr_dtypes(mat: sp.csr_matrix, data_dtype, index_dtype):
    mat = mat.copy()
    mat.indptr = mat.indptr.astype(index_dtype)
    mat.indices = mat.indices.astype(index_dtype)
    mat.data = mat.data.astype(data_dtype)

    return mat


def scipy_csc_dtypes(mat: sp.csc_matrix, data_dtype, index_dtype):
    mat = sp.csc_matrix((mat.data.astype(data_dtype), mat.indices.astype(index_dtype),
                         mat.indptr.astype(index_dtype)),
                        shape=mat.shape, dtype=data_dtype)
    return mat


def scipy_dtypes(mat: sp.spmatrix, data_dtype, index_dtype):
    if isinstance(mat, sp.csr_matrix):
        return scipy_csr_dtypes(mat, data_dtype, index_dtype)
    elif isinstance(mat, sp.csc_matrix):
        return scipy_csc_dtypes(mat, data_dtype, index_dtype)
    elif isinstance(mat, sp.coo_matrix):
        return scipy_coo_dtypes(mat, data_dtype, index_dtype)
    else:
        raise ValueError(f'Unsupported sparse format: {mat.layout}')


def torch_coo_dtypes(mat: torch.Tensor, data_dtype, index_dtype):
    return torch.sparse_coo_tensor(mat.indices().to(index_dtype), mat.values(),
                                   size=mat.shape, dtype=data_dtype)


def torch_csr_dtypes(mat: torch.Tensor, data_dtype, index_dtype):
    return torch.sparse_csr_tensor(mat.crow_indices().to(index_dtype), mat.col_indices().to(index_dtype),
                                   mat.values(),
                                   size=mat.shape, dtype=data_dtype)


def torch_csc_dtypes(mat: torch.Tensor, data_dtype, index_dtype):
    return torch.sparse_csc_tensor(mat.ccol_indices().to(index_dtype), mat.row_indices().to(index_dtype),
                                   mat.values(),
                                   size=mat.shape, dtype=data_dtype)


def torch_dtypes(mat: torch.Tensor, data_dtype, index_dtype):
    if mat.layout == torch.sparse_csr:
        return torch_csr_dtypes(mat, data_dtype, index_dtype)
    elif mat.layout == torch.sparse_csc:
        return torch_csc_dtypes(mat, data_dtype, index_dtype)
    elif mat.layout == torch.sparse_coo:
        return torch_coo_dtypes(mat.coalesce(), data_dtype, index_dtype).coalesce()
    else:
        raise ValueError(f'Unsupported sparse format: {mat.layout}')
