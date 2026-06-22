#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <ATen/ATen.h>

namespace py = pybind11;

#include <bitset>
#include <unordered_map>
#include <vector>
#include <cstdint> 

#define BITS_PER_CODE 64
typedef std::bitset<BITS_PER_CODE> block_code_t;

template <typename index_t>
struct coord_t {
    index_t row;
    index_t col;

    coord_t() noexcept : row(0), col(0) {}

    coord_t(const index_t row, const index_t col) noexcept : row(row), col(col) {}

    coord_t(const coord_t& other) noexcept = default;
    coord_t& operator=(const coord_t& other) noexcept = default;

    coord_t(coord_t&& other) noexcept = default;
    coord_t& operator=(coord_t&& other) noexcept = default;
};

template <typename index_t, typename scalar_t>
struct coo_elem_t {
    index_t row;
    index_t col;
    scalar_t val;

    coo_elem_t(const index_t row, const index_t col, const scalar_t val) noexcept :
               row(row), col(col), val(val) {}

    coo_elem_t(const coo_elem_t& other) noexcept :
               row(other.row), col(other.col), val(other.val) {}
    coo_elem_t& operator=(const coo_elem_t& other) noexcept = default;

    coo_elem_t(coo_elem_t&& other) noexcept = default;
    coo_elem_t& operator=(coo_elem_t&& other) noexcept = default;
};

template <typename index_t, typename scalar_t, uint64_t hash_buckets = 12000>
struct pbr_stats_t {
    typedef struct coord_t<index_t> coord_t;

    typedef struct {
        index_t occurrence; // Number of times this block code has been seen in the matrix
        std::vector<coord_t> block_coords; // Coordinates of the instances of this block code in the matrix

        bool compress; // Whether this block should be compressed or moved to the remainder matrix
    } block_descriptor_t;

    typedef std::unordered_map<index_t, block_code_t> block_code_map;
    typedef std::unordered_map<index_t, block_descriptor_t> block_stats_map;

    const index_t rows;
    const index_t cols;
    const index_t nnz;
    const index_t block_rows;
    const index_t block_cols;
    const index_t min_nnz_coverage;

    pbr_stats_t::block_stats_map block_stats;

    // PBR statistics
    index_t accounted_nnz;
    index_t remainder_nnz;
    index_t accounted_patterns;
    index_t accounted_blocks;
    index_t remainder_coo_overhead;
    float csr_index_size;
    float pbr_index_size;

    pbr_stats_t(const index_t rows, const index_t cols, const index_t nnz,
                const index_t block_rows, const index_t block_cols, const index_t min_nnz_coverage) :
                    rows(rows), cols(cols), nnz(nnz), block_rows(block_rows), block_cols(block_cols),
                    min_nnz_coverage(min_nnz_coverage),
                    block_stats(hash_buckets),
                    accounted_nnz(0), remainder_nnz(0),
                    accounted_patterns(0), accounted_blocks(0), remainder_coo_overhead(0),
                    csr_index_size(0.0), pbr_index_size(0.0) {}
};

enum data_order_t {
    BLOCK_ROW_MAJOR,
    BLOCK_COLUMN_MAJOR,
    INTERLEAVED_ROW_MAJOR,
    INTERLEAVED_COLUMN_MAJOR
};

template <typename index_t, typename scalar_t>
struct pbr_matrix_t {
    typedef struct coord_t<index_t> coord_t;

    // Number of rows and columns in the matrix
    const index_t rows;
    const index_t cols;

    // Number of rows and columns in each block
    const index_t block_rows;
    const index_t block_cols;

    // Total number of nonzero elements in the matrix
    const index_t total_nnz;

    // Block descriptors
    // Each block is represented by a block code and the coordinates of its top left corner.
    const std::vector<block_code_t> block_codes;
    const std::vector<coord_t> block_coords;

    // The starting index in block_data for each block.
    // size = block_codes.size() + 1
    // block #3 data starts at block_offsets[2] and ends at block_offsets[3] (exclusive)
    const std::vector<index_t> block_offsets;

    // Nonzero values in each block, in the order specified by data_order.
    const data_order_t data_order;
    const std::vector<scalar_t> block_data;

    // Elements belonging to blocks that are not compressed, stored in CSR format.
    // remainder_indptr has size rows+1; remainder_col_ind and remainder_vals have size remainder_nnz().
    const std::vector<index_t> remainder_indptr;
    const std::vector<index_t> remainder_col_ind;
    const std::vector<scalar_t> remainder_vals;

    pbr_matrix_t(const index_t rows, const index_t cols,
                 const index_t block_rows, const index_t block_cols,
                 const index_t total_nnz,
                 const std::vector<block_code_t>&& block_codes,
                 const std::vector<coord_t>&& block_coords,
                 const std::vector<index_t>&& block_offsets,
                 const data_order_t data_order,
                 const std::vector<scalar_t>&& block_data,
                 const std::vector<index_t>&& remainder_indptr,
                 const std::vector<index_t>&& remainder_col_ind,
                 const std::vector<scalar_t>&& remainder_vals) :
                rows(rows), cols(cols),
                block_rows(block_rows), block_cols(block_cols),
                total_nnz(total_nnz),
                block_codes(std::move(block_codes)),
                block_coords(std::move(block_coords)),
                block_offsets(std::move(block_offsets)),
                data_order(data_order),
                block_data(std::move(block_data)),
                remainder_indptr(std::move(remainder_indptr)),
                remainder_col_ind(std::move(remainder_col_ind)),
                remainder_vals(std::move(remainder_vals)) {}

    const index_t accounted_blocks() const {
        return block_codes.size();
    }

    const index_t compressed_nnz() const {
        return block_data.size();
    }

    const index_t remainder_nnz() const {
        return remainder_col_ind.size();
    }
};

template <typename index_t, typename scalar_t>
at::Tensor pbr_batched_matmul_cpu(const pbr_matrix_t<index_t, scalar_t>& pbr_mat,
                                   at::Tensor x);

template <typename index_t, typename scalar_t>
pbr_matrix_t<index_t, scalar_t> csr_to_pbr(const py::array_t<index_t, py::array::c_style> indptr,
                                           const py::array_t<index_t, py::array::c_style> indices,
                                           const py::array_t<scalar_t, py::array::c_style> data,
                                           const index_t rows, const index_t cols,
                                           const index_t block_rows, const index_t block_cols,
                                           const uint8_t min_nnz_per_block);

template <typename index_t, typename scalar_t>
void pbr_to_csr(const pbr_matrix_t<index_t, scalar_t>& pbr_mat,
                py::array_t<index_t, py::array::c_style> indptr, 
                py::array_t<index_t, py::array::c_style> indices, 
                py::array_t<scalar_t, py::array::c_style> data);

template <typename index_t, typename scalar_t>
pbr_stats_t<index_t, scalar_t> pbr_analyze_csr(const py::array_t<index_t, py::array::c_style> indptr,
                                               const py::array_t<index_t, py::array::c_style> indices,
                                               const index_t rows, const index_t cols, const index_t nnz,
                                               const index_t block_rows, const index_t block_cols,
                                               const index_t min_nnz_coverage);