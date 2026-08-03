import numpy as np
import scipy.sparse as sp
from pytest import fixture, mark
import torch
from torch.testing import assert_close
from torch_fast_spdmm import (pbr_to_csr, csr_to_pbr, pbr_batched_matmul_cuda,
                              pbr_batched_matmul_chunkcols_cuda, pbr_batched_matmul_nnzflat_cuda)


from test_utils import TestTorchNoGrad

@fixture
def sparse_format():
    return sp.csr_matrix


@fixture
def data_mat(crankseg_matrix):
    return crankseg_matrix


@fixture
def batch_size():
    return 1


@fixture
def device():
    return 'cpu'


@mark.parametrize('sparse_data_mat', ['sparse_identity_mat', 'sparse_permutation_mat', 'sparse_random_low_sparsity_mat'], indirect=True)
@mark.parametrize('index_dtype', [np.int32, np.int64], ids=['int32', 'int64'])
@mark.parametrize('data_dtype', [np.float32, np.float64], ids=['float32', 'float64'])
@mark.parametrize('system_size', [8, 16])
@mark.parametrize('block_size', [4, 8])
class TestTorchFastPBR(TestTorchNoGrad):

    @mark.parametrize('min_nnz_per_block', [1, 7])
    def test_pbr_roundtrip(self, sparse_mat, index_dtype, data_dtype, system_size, block_size, min_nnz_per_block):
        """
        Validates the encoder/decoder integrity.
        Original CSR -> PBR -> Reconstructed CSR.
        """
        mat = sparse_mat.astype(data_dtype)
        mat.indices = mat.indices.astype(index_dtype)
        mat.indptr = mat.indptr.astype(index_dtype)

        pbr_mat = csr_to_pbr(mat, min_nnz_per_block=min_nnz_per_block, block_rows=block_size, block_cols=block_size)

        result = pbr_to_csr(pbr_mat)

        np.testing.assert_array_equal(result.indptr, mat.indptr, err_msg="Indptr mismatch")
        np.testing.assert_array_equal(result.indices, mat.indices, err_msg="Indices mismatch")
        np.testing.assert_allclose(result.data, mat.data, err_msg="Data values mismatch")
    

    @mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_pbr_gpu_meta_dtypes(self, sparse_mat, index_dtype, data_dtype, system_size, block_size):
        """Verifies that .to('cuda') preserves the original index and scalar dtypes in GPU metadata."""
        mat = sparse_mat.astype(data_dtype)
        mat.indices = mat.indices.astype(index_dtype)
        mat.indptr = mat.indptr.astype(index_dtype)

        pbr_mat = csr_to_pbr(mat, block_rows=block_size, block_cols=block_size, min_nnz_per_block=1)
        pbr_mat = pbr_mat.to('cuda')

        meta = pbr_mat._gpu_meta
        expected_index_dtype  = torch.int32 if index_dtype == np.int32 else torch.int64
        expected_scalar_dtype = torch.float32 if data_dtype == np.float32 else torch.float64

        assert meta.coords.dtype      == expected_index_dtype,  f"coords dtype:      {meta.coords.dtype} != {expected_index_dtype}"
        assert meta.offsets.dtype     == expected_index_dtype,  f"offsets dtype:     {meta.offsets.dtype} != {expected_index_dtype}"
        assert meta.rem_indptr.dtype  == expected_index_dtype,  f"rem_indptr dtype:  {meta.rem_indptr.dtype} != {expected_index_dtype}"
        assert meta.rem_col_ind.dtype == expected_index_dtype,  f"rem_col_ind dtype: {meta.rem_col_ind.dtype} != {expected_index_dtype}"
        assert meta.data.dtype        == expected_scalar_dtype, f"data dtype:        {meta.data.dtype} != {expected_scalar_dtype}"
        assert meta.rem_vals.dtype    == expected_scalar_dtype, f"rem_vals dtype:    {meta.rem_vals.dtype} != {expected_scalar_dtype}"

    @mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @mark.parametrize('features', [8, 32], ids=['feat8', 'feat32'])
    @mark.parametrize('min_nnz', [1, 64], ids=['all_blocks', 'mostly_coo'])
    def test_pbr_batched_matmul_cuda_direct(self, sparse_mat, features, batch_size, data_dtype, block_size, system_size, min_nnz):
        device = 'cuda'
        torch_dtype = torch.from_numpy(np.array([], dtype=data_dtype)).dtype
        rows, cols = sparse_mat.shape
        
        x_gpu = torch.randn((batch_size, cols, features), dtype=torch_dtype, device=device)

        pbr_mat = csr_to_pbr(sparse_mat, block_rows=block_size, block_cols=block_size, min_nnz_per_block=min_nnz)
        pbr_mat = pbr_mat.to(device)

        y_gpu = pbr_batched_matmul_cuda(pbr_mat, x_gpu)

        # 6. Compute Ground Truth (on CPU using Torch Sparse)
        sparse_mat_torch = torch.sparse_csr_tensor(
            torch.from_numpy(sparse_mat.indptr.astype(np.int64)), 
            torch.from_numpy(sparse_mat.indices.astype(np.int64)), 
            torch.from_numpy(sparse_mat.data.astype(data_dtype)),
            size=(rows, cols)
        ).to(torch_dtype)

        x_cpu = x_gpu.cpu()
        expected = torch.stack([torch.sparse.mm(sparse_mat_torch, x_cpu[i]) for i in range(batch_size)])

        # 7. Final comparison (GPU vs CPU Reference)
        assert_close(y_gpu.cpu(), expected, rtol=1e-6, atol=1e-6)

    @mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @mark.parametrize('features', [8, 32], ids=['feat8', 'feat32'])
    @mark.parametrize('min_nnz', [1, 64], ids=['all_blocks', 'mostly_coo'])
    def test_pbr_force_scalar_matches_vec(self, sparse_mat, features, batch_size, data_dtype,
                                           block_size, system_size, min_nnz):
        """force_scalar=True forces pbr_spmm_zero_idle_kernel instead of the
        auto-selected pbr_spmm_vec_kernel (features=8/32 are both multiples of
        PbrVec<scalar_t>::lanes=4, so the default path here is vec) -- must
        agree with the default output, not just avoid crashing."""
        device = 'cuda'
        torch_dtype = torch.from_numpy(np.array([], dtype=data_dtype)).dtype

        x_gpu = torch.randn((batch_size, sparse_mat.shape[1], features), dtype=torch_dtype, device=device)

        pbr_mat = csr_to_pbr(sparse_mat, block_rows=block_size, block_cols=block_size, min_nnz_per_block=min_nnz)
        pbr_mat = pbr_mat.to(device)

        y_vec    = pbr_batched_matmul_cuda(pbr_mat, x_gpu, force_scalar=False)
        y_scalar = pbr_batched_matmul_cuda(pbr_mat, x_gpu, force_scalar=True)

        assert_close(y_scalar, y_vec, rtol=1e-6, atol=1e-6,
                     msg="force_scalar=True (zero_idle) output differs from the default vec output")


@mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@mark.parametrize('sparse_data_mat', ['sparse_identity_mat', 'sparse_permutation_mat', 'sparse_random_low_sparsity_mat'], indirect=True)
@mark.parametrize('index_dtype', [np.int32, np.int64], ids=['int32', 'int64'])
@mark.parametrize('data_dtype', [np.float32, np.float64], ids=['float32', 'float64'])
@mark.parametrize('system_size', [8, 16])
@mark.parametrize('block_size', [8])
@mark.parametrize('features', [128, 256], ids=['feat128', 'feat256'])
@mark.parametrize('min_nnz', [1, 64], ids=['all_blocks', 'mostly_coo'])
class TestPBRChunkcolsKernel(TestTorchNoGrad):
    """Correctness tests for the experimental chunkcols kernel (bounds
    per-thread row-work at CHUNK_K=4 columns) against the block kernel
    reference."""

    def test_chunkcols_matches_reference(self, sparse_mat, features, data_dtype, block_size,
                                          system_size, min_nnz, index_dtype):
        device = 'cuda'
        torch_dtype = torch.from_numpy(np.array([], dtype=data_dtype)).dtype
        mat = sparse_mat.astype(data_dtype)
        mat.indices = mat.indices.astype(index_dtype)
        mat.indptr  = mat.indptr.astype(index_dtype)
        cols = mat.shape[1]

        x_gpu = torch.randn((1, cols, features), dtype=torch_dtype, device=device)

        pbr_mat = csr_to_pbr(mat, block_rows=block_size, block_cols=block_size,
                              min_nnz_per_block=min_nnz)
        pbr_mat = pbr_mat.to(device)

        y_chunkcols = pbr_batched_matmul_chunkcols_cuda(pbr_mat, x_gpu)
        y_block     = pbr_batched_matmul_cuda(pbr_mat, x_gpu)

        assert_close(y_chunkcols, y_block, rtol=1e-5, atol=1e-5,
                     msg="chunkcols kernel output differs from block kernel")


@mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@mark.parametrize('sparse_data_mat', ['sparse_identity_mat', 'sparse_permutation_mat', 'sparse_random_low_sparsity_mat'], indirect=True)
@mark.parametrize('index_dtype', [np.int32, np.int64], ids=['int32', 'int64'])
@mark.parametrize('data_dtype', [np.float32, np.float64], ids=['float32', 'float64'])
@mark.parametrize('system_size', [8, 16])
@mark.parametrize('block_size', [8])
@mark.parametrize('features', [128, 256], ids=['feat128', 'feat256'])
@mark.parametrize('min_nnz', [1, 64], ids=['all_blocks', 'mostly_coo'])
class TestPBRNnzflatKernel(TestTorchNoGrad):
    """Correctness tests for the experimental nnzflat kernel (one thread per
    (nonzero, feature) work-item, column-sorted) against the block kernel
    reference."""

    def test_nnzflat_matches_reference(self, sparse_mat, features, data_dtype, block_size,
                                        system_size, min_nnz, index_dtype):
        device = 'cuda'
        torch_dtype = torch.from_numpy(np.array([], dtype=data_dtype)).dtype
        mat = sparse_mat.astype(data_dtype)
        mat.indices = mat.indices.astype(index_dtype)
        mat.indptr  = mat.indptr.astype(index_dtype)
        cols = mat.shape[1]

        x_gpu = torch.randn((1, cols, features), dtype=torch_dtype, device=device)

        pbr_mat = csr_to_pbr(mat, block_rows=block_size, block_cols=block_size,
                              min_nnz_per_block=min_nnz)
        pbr_mat = pbr_mat.to(device)

        y_nnzflat = pbr_batched_matmul_nnzflat_cuda(pbr_mat, x_gpu)
        y_block   = pbr_batched_matmul_cuda(pbr_mat, x_gpu)

        assert_close(y_nnzflat, y_block, rtol=1e-5, atol=1e-5,
                     msg="nnzflat kernel output differs from block kernel")


@mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@mark.parametrize('kernel_name,matmul_fn', [
    ('chunkcols', pbr_batched_matmul_chunkcols_cuda),
    ('nnzflat', pbr_batched_matmul_nnzflat_cuda),
])
def test_experimental_kernel_matches_reference_on_row_skewed_pattern(kernel_name, matmul_fn):
    """The row_skewed preset (one fully-dense row + one nonzero per remaining
    row, see benchmark/matrices.py) is the discriminating case both
    experimental kernels were built for -- maximal per-row density skew.
    Correctness here matters more than on any other single case."""
    from benchmark.matrices import generate_patterned_block_matrix, PBR_PATTERN_PRESETS

    device = 'cuda'
    system_size = 256
    mat = generate_patterned_block_matrix(
        system_size, 8, [(PBR_PATTERN_PRESETS['row_skewed'], 0.5)], seed=0,
    )
    x_gpu = torch.randn((1, system_size, 64), dtype=torch.float32, device=device)

    pbr_mat = csr_to_pbr(mat, block_rows=8, block_cols=8, min_nnz_per_block=1)
    pbr_mat = pbr_mat.to(device)

    y_experimental = matmul_fn(pbr_mat, x_gpu)
    y_block        = pbr_batched_matmul_cuda(pbr_mat, x_gpu)

    assert_close(y_experimental, y_block, rtol=1e-5, atol=1e-5,
                 msg=f"{kernel_name} kernel output differs from block kernel on row_skewed pattern")
