#pragma once
#include <cstdint>
#include <cuda_runtime.h>
#include <ATen/core/Tensor.h>

// Zeroes Y then runs the block kernel + CSR remainder fork/join. Declared
// here (defined in bindings_kernels.cu, alongside pbr_spmm_cuda_dispatch
// which also calls it) so bindings_pbr_timed.cu's PBRTimedSpMM can reuse the
// exact same production code path for its C++-only timing. See
// bindings_kernels.cu for the full rationale.
template <typename index_t, typename scalar_t>
void pbr_run_into(
    at::Tensor Y,
    int cols, int rows,
    int BLOCK_ROWS, int BLOCK_COLS,
    const at::Tensor block_codes,
    const at::Tensor block_coords,
    const at::Tensor block_offsets,
    const at::Tensor block_data,
    const at::Tensor rem_indptr,
    const at::Tensor rem_col_ind,
    const at::Tensor rem_vals,
    const at::Tensor X3,
    int csr_mode,
    int smem_size,
    cudaStream_t cur_stream,
    cudaStream_t block_stream,
    cudaStream_t csr_stream
);

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
    int smem_size = 2048
);

// Diagnostic-only upper-bound timing variant (block kernel's Step C
// atomicAdd -> plain store; output incorrect by construction). fp32/int32
// only. See pbr_kernel.cu's launch_pbr_diag_block_spmm for the scoping
// rationale (only covers pbr_spmm_vec_kernel).
void launch_pbr_diag_block_spmm(
    const int num_pbr_blocks,
    const int features,
    const int batch_size,
    const int cols,
    const int rows,
    const int BLOCK_ROWS,
    const int BLOCK_COLS,
    const int64_t* block_codes,
    const int32_t* block_coords,
    const int32_t* block_offsets,
    const float* block_data,
    const float* X,
    float* Y,
    cudaStream_t stream = 0,
    int smem_size = 2048
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

// nnzflat shared -- v6: shared position table (feat/pbr-nnzflat-shared @ d2e4dce).
// Resurrected for the 4-way benchmark; see pbr_kernel.cu.
template <typename index_t, typename scalar_t>
void launch_pbr_nnzflat_shared_spmm(
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

// nnzflat onthefly -- v5: row-major on-the-fly derivation (feat/pbr-nnzflat-online-index @ 27c8759).
// Resurrected for the 4-way benchmark; see pbr_kernel.cu.
template <typename index_t, typename scalar_t>
void launch_pbr_nnzflat_onthefly_spmm(
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
