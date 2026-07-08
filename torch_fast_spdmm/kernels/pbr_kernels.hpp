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

template <typename index_t, typename scalar_t>
void launch_pbr_rowgroup_spmm(
    const int  num_groups,
    const int  group_size,
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

template <typename index_t, typename scalar_t>
void launch_pbr_colgroup_spmm(
    const int  num_groups,
    const int  group_size,
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