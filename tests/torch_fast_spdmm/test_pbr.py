import numpy as np
import scipy.sparse as sp
from pytest import fixture, mark
import torch
from torch.testing import assert_close
from torch_fast_spdmm import (pbr_to_csr, csr_to_pbr, pbr_batched_matmul_cuda,
                              pbr_batched_matmul_chunk_cuda,
                              pbr_batched_matmul_rowgroup_cuda, pbr_batched_matmul_colgroup_cuda,
                              pbr_batched_matmul_tile_cuda)


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
@mark.parametrize('sparse_data_mat', ['sparse_identity_mat', 'sparse_permutation_mat', 'sparse_random_low_sparsity_mat'], indirect=True)
@mark.parametrize('index_dtype', [np.int32, np.int64], ids=['int32', 'int64'])
@mark.parametrize('data_dtype', [np.float32, np.float64], ids=['float32', 'float64'])
@mark.parametrize('system_size', [8, 16])
@mark.parametrize('block_size', [8])
@mark.parametrize('features', [128, 256], ids=['feat128', 'feat256'])
@mark.parametrize('min_nnz', [1, 64], ids=['all_blocks', 'mostly_coo'])
class TestPBRRowgroupKernel(TestTorchNoGrad):
    """Correctness tests for the row-group fusion kernel against the block kernel reference."""

    def test_rowgroup_matches_reference(self, sparse_mat, features, data_dtype, block_size,
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

        y_rowgroup = pbr_batched_matmul_rowgroup_cuda(pbr_mat, x_gpu)
        y_block    = pbr_batched_matmul_cuda(pbr_mat, x_gpu)

        assert_close(y_rowgroup, y_block, rtol=1e-5, atol=1e-5,
                     msg="rowgroup kernel output differs from block kernel")


@mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@mark.parametrize('sparse_data_mat', ['sparse_identity_mat', 'sparse_permutation_mat', 'sparse_random_low_sparsity_mat'], indirect=True)
@mark.parametrize('index_dtype', [np.int32, np.int64], ids=['int32', 'int64'])
@mark.parametrize('data_dtype', [np.float32, np.float64], ids=['float32', 'float64'])
@mark.parametrize('system_size', [8, 16])
@mark.parametrize('block_size', [8])
@mark.parametrize('features', [128, 256], ids=['feat128', 'feat256'])
@mark.parametrize('min_nnz', [1, 64], ids=['all_blocks', 'mostly_coo'])
class TestPBRColgroupKernel(TestTorchNoGrad):
    """Correctness tests for the col-group fusion kernel against the block kernel reference."""

    def test_colgroup_matches_reference(self, sparse_mat, features, data_dtype, block_size,
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

        y_colgroup = pbr_batched_matmul_colgroup_cuda(pbr_mat, x_gpu)
        y_block    = pbr_batched_matmul_cuda(pbr_mat, x_gpu)

        assert_close(y_colgroup, y_block, rtol=1e-5, atol=1e-5,
                     msg="colgroup kernel output differs from block kernel")


@mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@mark.parametrize('sparse_data_mat', ['sparse_identity_mat', 'sparse_permutation_mat', 'sparse_random_low_sparsity_mat'], indirect=True)
@mark.parametrize('index_dtype', [np.int32, np.int64], ids=['int32', 'int64'])
@mark.parametrize('data_dtype', [np.float32, np.float64], ids=['float32', 'float64'])
@mark.parametrize('system_size', [8, 16])
@mark.parametrize('block_size', [8])
@mark.parametrize('features', [128, 256], ids=['feat128', 'feat256'])
@mark.parametrize('min_nnz', [1, 64], ids=['all_blocks', 'mostly_coo'])
@mark.parametrize('chunk_size', [2, 4, 8, 16, 32, 64])
class TestPBRChunkKernel(TestTorchNoGrad):
    """Correctness tests for the diagnostic-only chunk kernel against the block
    kernel reference. system_size=[8,16]/block_size=8 gives very few PBR blocks,
    so chunk_size=16/32 always exceeds num_pbr_blocks here -- exercises the
    tail-guard (b_idx >= num_pbr_blocks) path on every parametrization."""

    def test_chunk_matches_reference(self, sparse_mat, features, data_dtype, block_size,
                                     system_size, min_nnz, index_dtype, chunk_size):
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

        y_chunk = pbr_batched_matmul_chunk_cuda(pbr_mat, x_gpu, chunk_size=chunk_size)
        y_block = pbr_batched_matmul_cuda(pbr_mat, x_gpu)

        assert_close(y_chunk, y_block, rtol=1e-5, atol=1e-5,
                     msg="chunk kernel output differs from block kernel")


@mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@mark.parametrize('sparse_data_mat', ['sparse_identity_mat', 'sparse_permutation_mat', 'sparse_random_low_sparsity_mat'], indirect=True)
@mark.parametrize('index_dtype', [np.int32, np.int64], ids=['int32', 'int64'])
@mark.parametrize('data_dtype', [np.float32, np.float64], ids=['float32', 'float64'])
@mark.parametrize('system_size', [8, 16])
@mark.parametrize('block_size', [8])
@mark.parametrize('features', [128, 256], ids=['feat128', 'feat256'])
@mark.parametrize('min_nnz', [1, 64], ids=['all_blocks', 'mostly_coo'])
@mark.parametrize('tile_strategy', ['grid', 'greedy'])
@mark.parametrize('tile_rows,tile_cols', [
    (1, 1), (1, 2), (2, 1), (1, 4), (4, 1), (2, 2), (2, 4), (4, 2), (4, 4),
])
class TestPBRTileKernel(TestTorchNoGrad):
    """Correctness tests for the tile (block-of-blocks) kernel against the block kernel reference."""

    def test_tile_matches_reference(self, sparse_mat, features, data_dtype, block_size,
                                     system_size, min_nnz, index_dtype, tile_rows, tile_cols,
                                     tile_strategy):
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

        y_tile  = pbr_batched_matmul_tile_cuda(pbr_mat, x_gpu, tile_rows=tile_rows,
                                                tile_cols=tile_cols, tile_strategy=tile_strategy)
        y_block = pbr_batched_matmul_cuda(pbr_mat, x_gpu)

        assert_close(y_tile, y_block, rtol=1e-5, atol=1e-5,
                     msg=f"tile({tile_rows}x{tile_cols}, {tile_strategy}) output differs from block kernel")


@mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@mark.parametrize('tile_strategy', ['grid', 'greedy'])
def test_tile_sparse_fill_matches_reference(tile_strategy):
    """A (4,4)-capacity tile with only 2 of 16 slots populated (diagonal identity
    pattern) exercises the sentinel-guarded smem_Y flush -- most row/col slots
    in the tile have no contributing block and must stay untouched, not just zero."""
    device = 'cuda'
    system_size = 16
    mat = sp.eye(system_size, dtype=np.float32).tocsr()
    x_gpu = torch.randn((1, system_size, 64), dtype=torch.float32, device=device)

    pbr_mat = csr_to_pbr(mat, block_rows=8, block_cols=8, min_nnz_per_block=1)
    pbr_mat = pbr_mat.to(device)

    y_tile  = pbr_batched_matmul_tile_cuda(pbr_mat, x_gpu, tile_rows=4, tile_cols=4,
                                            tile_strategy=tile_strategy)
    y_block = pbr_batched_matmul_cuda(pbr_mat, x_gpu)

    assert_close(y_tile, y_block, rtol=1e-5, atol=1e-5,
                 msg=f"tile(4x4, {tile_strategy}) differs from block kernel on a sparse-fill tile")


@mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@mark.parametrize('tile_strategy', ['grid', 'greedy'])
@mark.parametrize('tile_rows,tile_cols', [(2, 2), (4, 4)])
def test_tile_non_aligned_size_matches_reference(tile_rows, tile_cols, tile_strategy):
    """system_size=20 with block_rows=block_cols=8 leaves a partial last block
    (rows/cols 16-19) -- exercises the tile flush's rows-bound clipping."""
    device = 'cuda'
    system_size = 20
    mat = sp.random(system_size, system_size, density=0.3, dtype=np.float32,
                    random_state=np.random.default_rng(0)).tocsr()
    x_gpu = torch.randn((1, system_size, 64), dtype=torch.float32, device=device)

    pbr_mat = csr_to_pbr(mat, block_rows=8, block_cols=8, min_nnz_per_block=1)
    pbr_mat = pbr_mat.to(device)

    y_tile  = pbr_batched_matmul_tile_cuda(pbr_mat, x_gpu, tile_rows=tile_rows,
                                            tile_cols=tile_cols, tile_strategy=tile_strategy)
    y_block = pbr_batched_matmul_cuda(pbr_mat, x_gpu)

    assert_close(y_tile, y_block, rtol=1e-5, atol=1e-5,
                 msg=f"tile({tile_rows}x{tile_cols}, {tile_strategy}) differs from block kernel "
                     f"at non-block-aligned system_size={system_size}")


@mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@mark.parametrize('tile_strategy', ['grid', 'greedy'])
def test_tile_scattered_corners_matches_reference(tile_strategy):
    """4 PBR blocks at the corners of a sparse block-grid (row/col bands {0, 2},
    nothing in between) -- the pattern a fixed grid partition structurally
    can't group into one (2,2) tile, but 'greedy' should (see CLAUDE.md). Both
    strategies must still match the block kernel numerically regardless of how
    many tiles the blocks land in."""
    device = 'cuda'
    system_size = 24  # 3 block-bands of 8
    rng = np.random.default_rng(1)
    row_idx, col_idx, data = [], [], []
    for r0, c0 in zip([0, 0, 2, 2], [0, 2, 0, 2]):
        for dr in range(8):
            for dc in range(8):
                row_idx.append(r0 * 8 + dr)
                col_idx.append(c0 * 8 + dc)
                data.append(rng.standard_normal())
    mat = sp.csr_matrix((data, (row_idx, col_idx)), shape=(system_size, system_size), dtype=np.float32)
    x_gpu = torch.randn((1, system_size, 64), dtype=torch.float32, device=device)

    pbr_mat = csr_to_pbr(mat, block_rows=8, block_cols=8, min_nnz_per_block=1)
    pbr_mat = pbr_mat.to(device)

    y_tile  = pbr_batched_matmul_tile_cuda(pbr_mat, x_gpu, tile_rows=2, tile_cols=2,
                                            tile_strategy=tile_strategy)
    y_block = pbr_batched_matmul_cuda(pbr_mat, x_gpu)

    assert_close(y_tile, y_block, rtol=1e-5, atol=1e-5,
                 msg=f"tile(2x2, {tile_strategy}) differs from block kernel on scattered corners")
