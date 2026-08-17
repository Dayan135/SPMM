#pragma once

#include <cstdint>
#include <cuda_runtime.h>

// Launch the BOO SpMM kernel.  See BOO_FORMAT.md sec. 6.
//
// `threads` in {128, 256, 512} and `p` in {8, 16, 32} are swept, not fixed:
// `threads` is the 4x step lever on the shared-memory budget (128 -> 512
// quadruples it at identical 100% occupancy) and `p` is the linear one.
// Measured optimum for the row-segmented kernel: threads=256, p=64.
// (p=8, the original design point, is ~3x off it -- BOO_FORMAT.md sec. 12c/13.)
//
// The arena is sized from the format's GLOBAL maxima because `extern __shared__`
// takes one size for the whole grid -- CUDA has no per-CTA shared-memory size.
template <typename scalar_t>
void launch_boo_spmm(
    const int num_blocks,
    const int features,
    const int batch_size,
    const int rows,
    const int cols,
    const int max_nr,
    const int max_nc,
    const int max_nnz_blk,
    const int32_t* row_grp_ptr,
    const int32_t* row_ids,
    const int32_t* col_grp_ptr,
    const int32_t* col_ids,
    const int32_t* blk_row_grp,
    const int32_t* blk_col_grp,
    const int32_t* blk_nz_ptr,
    const uint8_t* blk_shift,
    const uint16_t* nz_idx,
    const scalar_t* nz_val,
    const scalar_t* X,
    scalar_t* Y,
    cudaStream_t stream = 0,
    int threads = 256,
    int p = 64,
    // row_seg: register-accumulating variant. Requires nz_order == rowmajor
    // (nz_idx sorted by local_row). Default; the caller falls back to false
    // for colmajor.
    bool row_seg = true);

// Dynamic shared memory the launcher requests, in bytes.  Mirrors
// torch_fast_spdmm.boo.boo_arena_bytes so the Python occupancy model and the
// real launch cannot drift apart.
int boo_arena_bytes(int max_nr, int max_nc, int max_nnz_blk, int p, int scalar_bytes);
int boo_arena_bytes_rowseg(int max_nr, int max_nc, int max_nnz_blk, int p, int scalar_bytes);
