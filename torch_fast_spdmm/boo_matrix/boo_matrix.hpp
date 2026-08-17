#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <ATen/ATen.h>

namespace py = pybind11;

#include <cstdint>
#include <string>
#include <vector>

// ===========================================================================
// BOO -- Blocked COO
// ===========================================================================
//
// A block is the non-empty intersection (row group i) x (col group j) of a
// clustered grid.  Row groups and column groups are each stored ONCE; a block
// is a pair of references plus its nonzeros.  See BOO_FORMAT.md.
//
// Occupancy is NOT a bitmap.  Each nonzero carries an explicit uint16 local
// coordinate
//
//     nz_idx = (local_row << blk_shift[b]) | local_col
//
// where local_row/local_col are positions within the block's own row/col group
// lists, and blk_shift[b] = log2(pad2(n_c(b))).  Decoding on the GPU is one
// shift and one AND -- no popcount, no bitmap scan, no select-kth-set-bit.
// This lifts all four of PBR's independent 8x8 caps at once.

// Policy for blocks that fall below `small_block_threshold` nonzeros.
// Deliberately open -- BOO_FORMAT.md sec. 7 -- and decided by measurement.
//
// Profiling diag_skewed (bimodal per-block nnz by construction) found KEEP
// launches one CUDA block per intersection regardless of how little work it
// holds: at N=262144, 97.4% of blocks carry <=3 nonzeros, each still paying
// the arena sized for the matrix's densest block (1.3% utilization), and the
// resulting 1.4M-CTA grid runs 6-7x slower than PBR's `block` kernel, which
// simply routes the same noise to a CSR remainder. See BOO_FORMAT.md sec. 15.
enum boo_small_block_policy_t {
    BOO_KEEP = 0,       // every non-empty intersection becomes a block
    BOO_MERGE = 1,      // fuse a row group's sparse col groups into one wider block
    BOO_REMAINDER = 2,  // spill below-threshold blocks to the CSR remainder
    BOO_TAIL_BATCH = 3  // spill to a flat COO tail, processed by boo_spmm_tail_kernel
};

// Nonzero ordering within a block.  A build parameter, not an assumption:
// column-major was inherited from the removed `colgroup` kernel and never
// independently validated (BOO_FORMAT.md sec. 4.3).
enum boo_nz_order_t {
    BOO_ROWMAJOR = 0,   // default; CSR's natural order, so no re-sort
    BOO_COLMAJOR = 1
};

template <typename index_t, typename scalar_t>
struct boo_matrix_t {
    const index_t rows;
    const index_t cols;
    const index_t total_nnz;

    // --- the clustered grid -------------------------------------------------
    // Row group g owns row_ids[row_grp_ptr[g] .. row_grp_ptr[g+1]).
    // row_ids is a permutation of [0, rows); col_ids of [0, cols).
    const std::vector<index_t> row_grp_ptr;
    const std::vector<index_t> row_ids;
    const std::vector<index_t> col_grp_ptr;
    const std::vector<index_t> col_ids;

    // --- blocks -------------------------------------------------------------
    const std::vector<index_t> blk_row_grp;   // size nb
    const std::vector<index_t> blk_col_grp;   // size nb
    const std::vector<index_t> blk_nz_ptr;    // size nb+1
    const std::vector<uint8_t> blk_shift;     // size nb
    const std::vector<uint16_t> nz_idx;       // size compressed_nnz
    const std::vector<scalar_t> nz_val;       // size compressed_nnz

    // --- remainder (may be empty; see boo_small_block_policy_t) -------------
    const std::vector<index_t> rem_indptr;    // size rows+1
    const std::vector<index_t> rem_col_ind;
    const std::vector<scalar_t> rem_vals;

    // --- tail (may be empty; BOO_TAIL_BATCH only) ----------------------------
    // Flat COO of every nonzero from below-threshold blocks, GLOBAL indices
    // (no block-local reencoding -- there is no shared arena to amortize for
    // entries this sparse). Consumed by boo_spmm_tail_kernel: a grid-stride
    // loop with no smem staging and no barriers, so its CTA count is bounded
    // by occupancy rather than by tail_nnz -- the fix for KEEP's CTA
    // explosion on bimodal matrices. Unlike BOO_REMAINDER this never leaves
    // BOO for the shared CSR kernel; boo_info folds tail_nnz into its
    // "handled inside BOO" fraction accordingly (see torch_fast_spdmm/boo.py).
    const std::vector<index_t> tail_row;
    const std::vector<index_t> tail_col;
    const std::vector<scalar_t> tail_val;

    // --- global maxima ------------------------------------------------------
    // CUDA has no per-CTA shared-memory size: `extern __shared__` takes its size
    // from the launch config and every CTA in the grid gets that same
    // allocation.  So the launcher must size the arena from the LARGEST block,
    // and these are what it reads.  (BOO_FORMAT.md sec. 9.4.)
    const index_t max_nr;
    const index_t max_nc;
    const index_t max_nnz_blk;

    const boo_nz_order_t nz_order;

    boo_matrix_t(const index_t rows, const index_t cols, const index_t total_nnz,
                 std::vector<index_t>&& row_grp_ptr, std::vector<index_t>&& row_ids,
                 std::vector<index_t>&& col_grp_ptr, std::vector<index_t>&& col_ids,
                 std::vector<index_t>&& blk_row_grp, std::vector<index_t>&& blk_col_grp,
                 std::vector<index_t>&& blk_nz_ptr, std::vector<uint8_t>&& blk_shift,
                 std::vector<uint16_t>&& nz_idx, std::vector<scalar_t>&& nz_val,
                 std::vector<index_t>&& rem_indptr, std::vector<index_t>&& rem_col_ind,
                 std::vector<scalar_t>&& rem_vals,
                 std::vector<index_t>&& tail_row, std::vector<index_t>&& tail_col,
                 std::vector<scalar_t>&& tail_val,
                 const index_t max_nr, const index_t max_nc, const index_t max_nnz_blk,
                 const boo_nz_order_t nz_order) :
        rows(rows), cols(cols), total_nnz(total_nnz),
        row_grp_ptr(std::move(row_grp_ptr)), row_ids(std::move(row_ids)),
        col_grp_ptr(std::move(col_grp_ptr)), col_ids(std::move(col_ids)),
        blk_row_grp(std::move(blk_row_grp)), blk_col_grp(std::move(blk_col_grp)),
        blk_nz_ptr(std::move(blk_nz_ptr)), blk_shift(std::move(blk_shift)),
        nz_idx(std::move(nz_idx)), nz_val(std::move(nz_val)),
        rem_indptr(std::move(rem_indptr)), rem_col_ind(std::move(rem_col_ind)),
        rem_vals(std::move(rem_vals)),
        tail_row(std::move(tail_row)), tail_col(std::move(tail_col)),
        tail_val(std::move(tail_val)),
        max_nr(max_nr), max_nc(max_nc), max_nnz_blk(max_nnz_blk),
        nz_order(nz_order) {}

    index_t num_blocks() const { return static_cast<index_t>(blk_row_grp.size()); }
    index_t num_row_groups() const { return static_cast<index_t>(row_grp_ptr.size()) - 1; }
    index_t num_col_groups() const { return static_cast<index_t>(col_grp_ptr.size()) - 1; }
    index_t compressed_nnz() const { return static_cast<index_t>(nz_val.size()); }
    index_t remainder_nnz() const { return static_cast<index_t>(rem_col_ind.size()); }
    index_t tail_nnz() const { return static_cast<index_t>(tail_val.size()); }
};

// Build from CSR plus a clustering label per row and per column.  The labels
// come from Python (torch_fast_spdmm.clustering); this function only assembles
// the format, so any clustering method plugs in unchanged.
template <typename index_t, typename scalar_t>
boo_matrix_t<index_t, scalar_t> csr_to_boo(
    const py::array_t<index_t, py::array::c_style> indptr,
    const py::array_t<index_t, py::array::c_style> indices,
    const py::array_t<scalar_t, py::array::c_style> data,
    const index_t rows, const index_t cols,
    const py::array_t<int32_t, py::array::c_style> row_labels,
    const py::array_t<int32_t, py::array::c_style> col_labels,
    const int small_block_policy,
    const index_t small_block_threshold,
    const int nz_order);

template <typename index_t, typename scalar_t>
void boo_to_csr(const boo_matrix_t<index_t, scalar_t>& boo_mat,
                py::array_t<index_t, py::array::c_style> indptr,
                py::array_t<index_t, py::array::c_style> indices,
                py::array_t<scalar_t, py::array::c_style> data);

template <typename index_t, typename scalar_t>
at::Tensor boo_batched_matmul_cpu(const boo_matrix_t<index_t, scalar_t>& boo_mat,
                                  at::Tensor x);
