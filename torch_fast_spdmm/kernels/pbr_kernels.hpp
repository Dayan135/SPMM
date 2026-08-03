#pragma once
#include <cstdint>
#include <cuda_runtime.h>

template <typename index_t, typename scalar_t>
void launch_pbr_spmm(
    const int num_pbr_blocks,
    const int features,
    const int batch_size,
    const int cols,
    const int rows,
    const int BLOCK_ROWS,
    const int BLOCK_COLS,
    const int64_t* block_codes,
    const index_t* block_coords,
    const index_t* block_offsets,
    const scalar_t* block_data,
    const scalar_t* X,
    scalar_t* Y,
    cudaStream_t stream = 0,
    int smem_size = 2048,
    bool force_scalar = false
);

// Experimental: bounds per-thread row-work at CHUNK_K=4 active columns
// instead of zero_idle's up-to-BLOCK_COLS. See pbr_kernel.cu.
template <typename index_t, typename scalar_t>
void launch_pbr_chunkcols_spmm(
    const int num_pbr_blocks,
    const int features,
    const int batch_size,
    const int cols,
    const int rows,
    const int BLOCK_ROWS,
    const int BLOCK_COLS,
    const int64_t* block_codes,
    const index_t* block_coords,
    const index_t* block_offsets,
    const scalar_t* block_data,
    const scalar_t* X,
    scalar_t* Y,
    cudaStream_t stream = 0,
    int smem_size = 2048
);

// Experimental: one thread per (nonzero, feature) work-item, column-sorted.
// See pbr_kernel.cu.
template <typename index_t, typename scalar_t>
void launch_pbr_nnzflat_spmm(
    const int num_pbr_blocks,
    const int features,
    const int batch_size,
    const int cols,
    const int rows,
    const int BLOCK_ROWS,
    const int BLOCK_COLS,
    const int64_t* block_codes,
    const index_t* block_coords,
    const index_t* block_offsets,
    const scalar_t* block_data,
    const scalar_t* X,
    scalar_t* Y,
    cudaStream_t stream = 0,
    int smem_size = 2048
);

template <typename index_t, typename scalar_t>
void launch_csr_spmm(
    const int rows,
    const int features,
    const int batch_size,
    const int cols,
    const int nnz,
    const index_t* indptr,
    const index_t* col_ind,
    const scalar_t* csr_vals,
    const scalar_t* X,
    scalar_t* Y,
    cudaStream_t stream = 0,
    int csr_mode = 0
);
