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
    int smem_size = 2048
);

// Diagnostic-only launcher for the zero_idle-kernel microbench variants
// (2026-07-16, +mergeBC_alloc 2026-07-21): block_data-read-as-constant, each
// __syncthreads() removed individually, a Step B/C merge (atomicAdd straight
// from register, no smem_Y), and an allocation-ablation control for that
// merge (same fused compute/commit, but an explicit dead smem_Y pointer kept
// at baseline's offset/size, isolating whether the merge's win comes from
// removing the smem_Y instructions vs. from incidentally freeing shared-
// memory allocation/occupancy). Always dispatches to a scalar
// (non-vectorized) kernel, selected by `variant` (0=baseline 1=constdata
// 2=nosyncA 3=nosyncB 4=nosyncC 5=mergeBC 6=mergeBC_alloc) -- never checks
// use_vec the way launch_pbr_spmm does. Several variants produce numerically
// wrong output by design; see pbr_kernel.cu. Never merge into launch_pbr_spmm.
template <typename index_t, typename scalar_t>
void launch_pbr_spmm_scalar_diag(
    int variant,
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

// group_offsets: CSR-style boundary array (size num_groups+1) built host-side
// by _build_group_offsets (__init__.py) -- either count-balanced (reproduces
// the original group_idx*group_size arithmetic) or nnz-balanced (searchsorted
// over the cumulative-nnz block_offsets array). See pbr_kernel.cu's rowgroup
// kernel comment and WHY_BLOCK_KERNEL_FAILS_TO_SCALE.md.
template <typename index_t, typename scalar_t>
void launch_pbr_rowgroup_spmm(
    const int  num_groups,
    const index_t* group_offsets,
    const int  num_pbr_blocks,
    const int  features,
    const int  batch_size,
    const int  cols,
    const int  rows,
    const int  BLOCK_ROWS,
    const int  BLOCK_COLS,
    const int64_t* block_codes,
    const index_t* block_coords,
    const index_t* block_offsets,
    const scalar_t* block_data,
    const scalar_t* X,
    scalar_t* Y,
    cudaStream_t stream = 0,
    int smem_size = 2048
);

// Returns cudaOccupancyMaxActiveBlocksPerMultiprocessor for the rowgroup kernel.
int get_pbr_rowgroup_blocks_per_sm(int block_rows, int smem_size);

// group_offsets: same contract as launch_pbr_rowgroup_spmm, but boundaries are
// indices into the col-sorted arrays (searchsorted over col_offsets for the
// nnz-balanced case).
template <typename index_t, typename scalar_t>
void launch_pbr_colgroup_spmm(
    const int  num_groups,
    const index_t* group_offsets,
    const int  num_pbr_blocks,
    const int  features,
    const int  batch_size,
    const int  cols,
    const int  rows,
    const int  BLOCK_ROWS,
    const int  BLOCK_COLS,
    const int64_t* col_codes,
    const index_t* col_coords,
    const index_t* col_offsets,
    const scalar_t* col_data,
    const scalar_t* X,
    scalar_t* Y,
    cudaStream_t stream = 0,
    int smem_size = 2048
);

// Returns cudaOccupancyMaxActiveBlocksPerMultiprocessor for the colgroup kernel.
int get_pbr_colgroup_blocks_per_sm(int block_rows, int smem_size);

template <typename index_t, typename scalar_t>
void launch_pbr_tile_spmm(
    const int  num_tiles,
    const int  features,
    const int  batch_size,
    const int  cols,
    const int  rows,
    const int  BLOCK_ROWS,
    const int  BLOCK_COLS,
    const int  tile_rows,
    const int  tile_cols,
    const int64_t*  block_codes,
    const index_t*  block_offsets,
    const scalar_t* block_data,
    const index_t*  tile_offsets,
    const index_t*  tile_row_origins,
    const index_t*  tile_col_origins,
    const uint8_t*  tile_row_local,
    const uint8_t*  tile_col_local,
    const scalar_t* X,
    scalar_t* Y,
    cudaStream_t stream = 0,
    int smem_size = 2048
);

// Returns cudaOccupancyMaxActiveBlocksPerMultiprocessor for the tile kernel.
int get_pbr_tile_blocks_per_sm(int tile_rows, int tile_cols, int block_rows, int smem_size);

// Diagnostic-only "chunk" kernel: identical per-block body to launch_pbr_spmm,
// but runs a runtime loop over [group_offsets[g], group_offsets[g+1]) PBR
// blocks per CTA instead of one, no smem reuse across iterations -- isolates
// the grouping-criterion variable from the smem-reuse machinery rowgroup/
// colgroup/tile also change. group_offsets: same contract as
// launch_pbr_rowgroup_spmm (count- or nnz-balanced). See
// WHY_BLOCK_KERNEL_FAILS_TO_SCALE.md.
template <typename index_t, typename scalar_t>
void launch_pbr_chunk_spmm(
    const int num_groups,
    const index_t* group_offsets,
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