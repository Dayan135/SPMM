#include "boo_matrix/boo_matrix.hpp"

#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <unordered_map>

namespace {

// Bits needed for the column half of nz_idx: ceil(log2(n_c)), i.e. n_c padded
// up to a power of two.  n_c == 1 -> 0 bits, which is correct.
inline uint8_t ceil_log2(uint32_t value) {
    uint8_t shift = 0;
    while ((1u << shift) < value) ++shift;
    return shift;
}

// Counting sort of `labels` into contiguous groups.
// Emits group_ptr (size n_groups+1) and ids (a permutation of [0, n)), with ids
// ASCENDING within each group -- the format's invariant, and what lets the
// kernel walk a group without sorting.  `position[i]` is i's local index inside
// its group, i.e. the local_row / local_col that goes into nz_idx.
template <typename index_t>
void build_groups(const int32_t* labels, index_t n, index_t n_groups,
                  std::vector<index_t>& group_ptr, std::vector<index_t>& ids,
                  std::vector<index_t>& position) {
    group_ptr.assign(static_cast<size_t>(n_groups) + 1, 0);
    for (index_t i = 0; i < n; ++i) {
        const int32_t label = labels[i];
        if (label < 0 || label >= n_groups) {
            throw std::invalid_argument("csr_to_boo: cluster label out of range");
        }
        group_ptr[static_cast<size_t>(label) + 1]++;
    }
    for (index_t g = 0; g < n_groups; ++g) {
        group_ptr[static_cast<size_t>(g) + 1] += group_ptr[static_cast<size_t>(g)];
    }

    ids.assign(static_cast<size_t>(n), 0);
    position.assign(static_cast<size_t>(n), 0);
    std::vector<index_t> cursor(group_ptr.begin(), group_ptr.end() - 1);
    // Ascending i means ascending ids within every group, for free.
    for (index_t i = 0; i < n; ++i) {
        const index_t slot = cursor[static_cast<size_t>(labels[i])]++;
        ids[static_cast<size_t>(slot)] = i;
        position[static_cast<size_t>(i)] = slot - group_ptr[static_cast<size_t>(labels[i])];
    }
}

}  // namespace


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
    const int nz_order) {

    if (small_block_policy == BOO_MERGE) {
        // Merging a row group's sparse column groups into one wider block would
        // give that row group a column set different from the global column
        // grouping -- but col_ids must stay a PARTITION of [0, cols), since a
        // column may belong to exactly one group.  So merging is a clustering
        // level relabel (fuse sparse column groups globally), not a builder
        // operation, and it belongs in clustering.py.  Left unimplemented until
        // the M1 histogram says it is needed at all (BOO_FORMAT.md sec. 7).
        throw std::invalid_argument(
            "csr_to_boo: 'merge' policy is a clustering-level column relabel, "
            "not a builder policy -- merge the column groups before building");
    }
    if (small_block_policy != BOO_KEEP && small_block_policy != BOO_REMAINDER) {
        throw std::invalid_argument("csr_to_boo: unknown small_block_policy");
    }
    if (nz_order != BOO_ROWMAJOR && nz_order != BOO_COLMAJOR) {
        throw std::invalid_argument("csr_to_boo: unknown nz_order");
    }

    const index_t* indptr_ptr = indptr.data();
    const index_t* indices_ptr = indices.data();
    const scalar_t* data_ptr = data.data();
    const int32_t* row_label_ptr = row_labels.data();
    const int32_t* col_label_ptr = col_labels.data();

    if (static_cast<index_t>(row_labels.size()) != rows ||
        static_cast<index_t>(col_labels.size()) != cols) {
        throw std::invalid_argument("csr_to_boo: label array length must match matrix dims");
    }

    const index_t total_nnz = static_cast<index_t>(data.size());

    index_t n_rg = 0, n_cg = 0;
    for (index_t r = 0; r < rows; ++r) n_rg = std::max(n_rg, static_cast<index_t>(row_label_ptr[r]) + 1);
    for (index_t c = 0; c < cols; ++c) n_cg = std::max(n_cg, static_cast<index_t>(col_label_ptr[c]) + 1);

    std::vector<index_t> row_grp_ptr, row_ids, row_position;
    std::vector<index_t> col_grp_ptr, col_ids, col_position;
    build_groups<index_t>(row_label_ptr, rows, n_rg, row_grp_ptr, row_ids, row_position);
    build_groups<index_t>(col_label_ptr, cols, n_cg, col_grp_ptr, col_ids, col_position);

    // --- pass 1: how many nonzeros land in each (row group, col group) -------
    std::unordered_map<int64_t, index_t> block_counts;
    block_counts.reserve(static_cast<size_t>(total_nnz) / 4 + 16);
    for (index_t r = 0; r < rows; ++r) {
        const int64_t rg = static_cast<int64_t>(row_label_ptr[r]);
        for (index_t j = indptr_ptr[r]; j < indptr_ptr[r + 1]; ++j) {
            const int64_t key = rg * static_cast<int64_t>(n_cg) + col_label_ptr[indices_ptr[j]];
            block_counts[key]++;
        }
    }

    // Sort the keys.  Hash-map iteration order is unspecified and varies with
    // libstdc++ version and rehash history -- csr_to_pbr emits blocks in that
    // order today, which makes its output non-reproducible.  Sorting by
    // (row group, col group) makes BOO deterministic and gives block-row-major
    // block ordering for free.
    std::vector<std::pair<int64_t, index_t>> sorted_blocks(block_counts.begin(), block_counts.end());
    std::sort(sorted_blocks.begin(), sorted_blocks.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });

    // --- decide which blocks are kept ---------------------------------------
    std::unordered_map<int64_t, index_t> key_to_block;
    key_to_block.reserve(sorted_blocks.size() * 2);
    std::vector<index_t> blk_row_grp, blk_col_grp, blk_nz_ptr;
    std::vector<uint8_t> blk_shift;
    blk_nz_ptr.push_back(0);

    index_t max_nr = 0, max_nc = 0, max_nnz_blk = 0;
    index_t running = 0;

    for (const auto& entry : sorted_blocks) {
        const index_t nnz_in_block = entry.second;
        if (small_block_policy == BOO_REMAINDER && nnz_in_block < small_block_threshold) {
            continue;  // handled by the remainder pass below
        }
        const index_t rg = static_cast<index_t>(entry.first / n_cg);
        const index_t cg = static_cast<index_t>(entry.first % n_cg);
        const index_t n_r = row_grp_ptr[rg + 1] - row_grp_ptr[rg];
        const index_t n_c = col_grp_ptr[cg + 1] - col_grp_ptr[cg];

        const uint8_t shift = ceil_log2(static_cast<uint32_t>(n_c));
        // nz_idx is uint16, so the block's local coordinate space must fit.
        // Shared memory binds long before this does (BOO_FORMAT.md sec. 4.2),
        // but assert rather than trust.
        const uint64_t span = (static_cast<uint64_t>(n_r - 1) << shift) + static_cast<uint64_t>(n_c - 1);
        if (span > 0xFFFFull) {
            throw std::invalid_argument(
                "csr_to_boo: block local index space exceeds uint16 -- "
                "reduce max_nr/max_nc in the clustering");
        }

        key_to_block[entry.first] = static_cast<index_t>(blk_row_grp.size());
        blk_row_grp.push_back(rg);
        blk_col_grp.push_back(cg);
        blk_shift.push_back(shift);
        running += nnz_in_block;
        blk_nz_ptr.push_back(running);

        max_nr = std::max(max_nr, n_r);
        max_nc = std::max(max_nc, n_c);
        max_nnz_blk = std::max(max_nnz_blk, nnz_in_block);
    }

    // --- pass 2: place every nonzero ----------------------------------------
    std::vector<uint16_t> nz_idx(static_cast<size_t>(running));
    std::vector<scalar_t> nz_val(static_cast<size_t>(running));
    std::vector<index_t> cursor(blk_nz_ptr.begin(), blk_nz_ptr.end() - 1);
    std::vector<std::vector<std::pair<index_t, scalar_t>>> rem_bins(static_cast<size_t>(rows));

    for (index_t r = 0; r < rows; ++r) {
        const int64_t rg = static_cast<int64_t>(row_label_ptr[r]);
        const index_t local_row = row_position[static_cast<size_t>(r)];
        for (index_t j = indptr_ptr[r]; j < indptr_ptr[r + 1]; ++j) {
            const index_t c = indices_ptr[j];
            const int64_t key = rg * static_cast<int64_t>(n_cg) + col_label_ptr[c];
            const auto found = key_to_block.find(key);
            if (found == key_to_block.end()) {
                rem_bins[static_cast<size_t>(r)].push_back({c, data_ptr[j]});
                continue;
            }
            const index_t b = found->second;
            const index_t slot = cursor[static_cast<size_t>(b)]++;
            nz_idx[static_cast<size_t>(slot)] = static_cast<uint16_t>(
                (static_cast<uint32_t>(local_row) << blk_shift[static_cast<size_t>(b)]) |
                static_cast<uint32_t>(col_position[static_cast<size_t>(c)]));
            nz_val[static_cast<size_t>(slot)] = data_ptr[j];
        }
    }

    // --- order the nonzeros inside each block -------------------------------
    // rowmajor: sort by nz_idx itself (== (local_row, local_col) lexicographic,
    //           since local_col occupies the low bits) -- so nz_idx comes out
    //           strictly increasing, a free invariant to assert.
    // colmajor: sort by (local_col, local_row).
    // CSR traversal already produces rowmajor order here, but sorting
    // unconditionally makes the invariant hold regardless of input ordering.
    {
        std::vector<std::pair<uint32_t, uint32_t>> keyed;
        for (size_t b = 0; b < blk_row_grp.size(); ++b) {
            const size_t begin = static_cast<size_t>(blk_nz_ptr[b]);
            const size_t end = static_cast<size_t>(blk_nz_ptr[b + 1]);
            if (end - begin < 2) continue;

            const uint8_t shift = blk_shift[b];
            const uint32_t col_mask = (1u << shift) - 1u;
            keyed.clear();
            keyed.reserve(end - begin);
            for (size_t k = begin; k < end; ++k) {
                const uint32_t idx = nz_idx[k];
                const uint32_t sort_key = (nz_order == BOO_ROWMAJOR)
                    ? idx
                    : (((idx & col_mask) << 16) | (idx >> shift));
                keyed.push_back({sort_key, static_cast<uint32_t>(k)});
            }
            std::sort(keyed.begin(), keyed.end());

            std::vector<uint16_t> idx_tmp(end - begin);
            std::vector<scalar_t> val_tmp(end - begin);
            for (size_t t = 0; t < keyed.size(); ++t) {
                idx_tmp[t] = nz_idx[keyed[t].second];
                val_tmp[t] = nz_val[keyed[t].second];
            }
            std::copy(idx_tmp.begin(), idx_tmp.end(), nz_idx.begin() + begin);
            std::copy(val_tmp.begin(), val_tmp.end(), nz_val.begin() + begin);
        }
    }

    // --- collapse the remainder bins to CSR ---------------------------------
    std::vector<index_t> rem_indptr(static_cast<size_t>(rows) + 1, 0);
    std::vector<index_t> rem_col_ind;
    std::vector<scalar_t> rem_vals;
    for (index_t r = 0; r < rows; ++r) {
        auto& bin = rem_bins[static_cast<size_t>(r)];
        std::sort(bin.begin(), bin.end());
        for (const auto& entry : bin) {
            rem_col_ind.push_back(entry.first);
            rem_vals.push_back(entry.second);
        }
        rem_indptr[static_cast<size_t>(r) + 1] = static_cast<index_t>(rem_col_ind.size());
    }

    return boo_matrix_t<index_t, scalar_t>(
        rows, cols, total_nnz,
        std::move(row_grp_ptr), std::move(row_ids),
        std::move(col_grp_ptr), std::move(col_ids),
        std::move(blk_row_grp), std::move(blk_col_grp),
        std::move(blk_nz_ptr), std::move(blk_shift),
        std::move(nz_idx), std::move(nz_val),
        std::move(rem_indptr), std::move(rem_col_ind), std::move(rem_vals),
        max_nr, max_nc, max_nnz_blk,
        static_cast<boo_nz_order_t>(nz_order));
}


template <typename index_t, typename scalar_t>
void boo_to_csr(const boo_matrix_t<index_t, scalar_t>& boo_mat,
                py::array_t<index_t, py::array::c_style> indptr,
                py::array_t<index_t, py::array::c_style> indices,
                py::array_t<scalar_t, py::array::c_style> data) {
    auto indptr_ptr = indptr.mutable_data();
    auto indices_ptr = indices.mutable_data();
    auto data_ptr = data.mutable_data();

    std::vector<std::vector<std::pair<index_t, scalar_t>>> row_bins(
        static_cast<size_t>(boo_mat.rows));

    for (index_t b = 0; b < boo_mat.num_blocks(); ++b) {
        const index_t rg = boo_mat.blk_row_grp[static_cast<size_t>(b)];
        const index_t cg = boo_mat.blk_col_grp[static_cast<size_t>(b)];
        const index_t row_base = boo_mat.row_grp_ptr[static_cast<size_t>(rg)];
        const index_t col_base = boo_mat.col_grp_ptr[static_cast<size_t>(cg)];
        const uint8_t shift = boo_mat.blk_shift[static_cast<size_t>(b)];
        const uint32_t col_mask = (1u << shift) - 1u;

        for (index_t k = boo_mat.blk_nz_ptr[static_cast<size_t>(b)];
             k < boo_mat.blk_nz_ptr[static_cast<size_t>(b) + 1]; ++k) {
            const uint32_t idx = boo_mat.nz_idx[static_cast<size_t>(k)];
            const index_t global_row = boo_mat.row_ids[static_cast<size_t>(row_base + (idx >> shift))];
            const index_t global_col = boo_mat.col_ids[static_cast<size_t>(col_base + (idx & col_mask))];
            row_bins[static_cast<size_t>(global_row)].push_back(
                {global_col, boo_mat.nz_val[static_cast<size_t>(k)]});
        }
    }

    for (index_t r = 0; r < boo_mat.rows; ++r) {
        for (index_t j = boo_mat.rem_indptr[static_cast<size_t>(r)];
             j < boo_mat.rem_indptr[static_cast<size_t>(r) + 1]; ++j) {
            row_bins[static_cast<size_t>(r)].push_back(
                {boo_mat.rem_col_ind[static_cast<size_t>(j)],
                 boo_mat.rem_vals[static_cast<size_t>(j)]});
        }
    }

    index_t current = 0;
    indptr_ptr[0] = 0;
    for (index_t r = 0; r < boo_mat.rows; ++r) {
        auto& bin = row_bins[static_cast<size_t>(r)];
        std::sort(bin.begin(), bin.end());
        for (const auto& entry : bin) {
            indices_ptr[current] = entry.first;
            data_ptr[current] = entry.second;
            ++current;
        }
        indptr_ptr[static_cast<size_t>(r) + 1] = current;
    }
}


template <typename index_t, typename scalar_t>
at::Tensor boo_batched_matmul_cpu(const boo_matrix_t<index_t, scalar_t>& boo_mat,
                                  at::Tensor x) {
    const bool batched = x.dim() == 3;
    at::Tensor x3 = batched ? x.contiguous() : x.unsqueeze(0).contiguous();
    const int64_t batch_size = x3.size(0);
    const int64_t features = x3.size(2);

    at::Tensor y = at::zeros({batch_size, static_cast<int64_t>(boo_mat.rows), features},
                             x3.options());

    const scalar_t* x_ptr = x3.template data_ptr<scalar_t>();
    scalar_t* y_ptr = y.template data_ptr<scalar_t>();

    for (int64_t batch = 0; batch < batch_size; ++batch) {
        const scalar_t* xb = x_ptr + batch * boo_mat.cols * features;
        scalar_t* yb = y_ptr + batch * boo_mat.rows * features;

        for (index_t b = 0; b < boo_mat.num_blocks(); ++b) {
            const index_t rg = boo_mat.blk_row_grp[static_cast<size_t>(b)];
            const index_t cg = boo_mat.blk_col_grp[static_cast<size_t>(b)];
            const index_t row_base = boo_mat.row_grp_ptr[static_cast<size_t>(rg)];
            const index_t col_base = boo_mat.col_grp_ptr[static_cast<size_t>(cg)];
            const uint8_t shift = boo_mat.blk_shift[static_cast<size_t>(b)];
            const uint32_t col_mask = (1u << shift) - 1u;

            for (index_t k = boo_mat.blk_nz_ptr[static_cast<size_t>(b)];
                 k < boo_mat.blk_nz_ptr[static_cast<size_t>(b) + 1]; ++k) {
                const uint32_t idx = boo_mat.nz_idx[static_cast<size_t>(k)];
                const index_t global_row = boo_mat.row_ids[static_cast<size_t>(row_base + (idx >> shift))];
                const index_t global_col = boo_mat.col_ids[static_cast<size_t>(col_base + (idx & col_mask))];
                const scalar_t value = boo_mat.nz_val[static_cast<size_t>(k)];
                const scalar_t* xrow = xb + static_cast<int64_t>(global_col) * features;
                scalar_t* yrow = yb + static_cast<int64_t>(global_row) * features;
                for (int64_t f = 0; f < features; ++f) yrow[f] += value * xrow[f];
            }
        }

        for (index_t r = 0; r < boo_mat.rows; ++r) {
            scalar_t* yrow = yb + static_cast<int64_t>(r) * features;
            for (index_t j = boo_mat.rem_indptr[static_cast<size_t>(r)];
                 j < boo_mat.rem_indptr[static_cast<size_t>(r) + 1]; ++j) {
                const scalar_t value = boo_mat.rem_vals[static_cast<size_t>(j)];
                const scalar_t* xrow =
                    xb + static_cast<int64_t>(boo_mat.rem_col_ind[static_cast<size_t>(j)]) * features;
                for (int64_t f = 0; f < features; ++f) yrow[f] += value * xrow[f];
            }
        }
    }

    return y;
}


// --- explicit instantiations ------------------------------------------------
#define BOO_INSTANTIATE(index_t, scalar_t)                                                  \
    template boo_matrix_t<index_t, scalar_t> csr_to_boo<index_t, scalar_t>(                 \
        const py::array_t<index_t, py::array::c_style>,                                     \
        const py::array_t<index_t, py::array::c_style>,                                     \
        const py::array_t<scalar_t, py::array::c_style>,                                    \
        const index_t, const index_t,                                                       \
        const py::array_t<int32_t, py::array::c_style>,                                     \
        const py::array_t<int32_t, py::array::c_style>,                                     \
        const int, const index_t, const int);                                               \
    template void boo_to_csr<index_t, scalar_t>(                                            \
        const boo_matrix_t<index_t, scalar_t>&,                                             \
        py::array_t<index_t, py::array::c_style>,                                           \
        py::array_t<index_t, py::array::c_style>,                                           \
        py::array_t<scalar_t, py::array::c_style>);                                         \
    template at::Tensor boo_batched_matmul_cpu<index_t, scalar_t>(                          \
        const boo_matrix_t<index_t, scalar_t>&, at::Tensor);

BOO_INSTANTIATE(int32_t, float)
BOO_INSTANTIATE(int64_t, float)
BOO_INSTANTIATE(int32_t, double)
BOO_INSTANTIATE(int64_t, double)
#undef BOO_INSTANTIATE
