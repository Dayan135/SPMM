#include "csc_kernels.hpp"

#include <type_traits>

const dim3 thread_block_size(32, 32);

template <typename index_t, typename scalar_t>
__global__ void _naive_kernel(const scalar_t* A_values, const index_t* A_row_ind, const index_t* A_colptr,
                              const scalar_t* x, scalar_t* Ax,
                              const uint32_t x_rows, const uint32_t x_cols) {
    using uindex_t = typename std::make_unsigned<index_t>::type;

    // Each thread block/SM is responsible for a column of A.
    // Each thread within the block is responsible for computing one element of the column.
    // The x component of the thread block maps to a column of A and thus row of x.
    // x is assumed to be row-major.
    // For optimal performance, the x dimension of the thread block should be equal to the number of columns in x.
    const uint32_t col_id = blockIdx.x;

    const index_t col_start = A_colptr[col_id];
    const index_t col_end = A_colptr[col_id + 1];

    // Process each NNZ in the assigned column of A
    for (uindex_t nnz_id = col_start; nnz_id < col_end; ++nnz_id) {
        const scalar_t A_val = A_values[nnz_id];
        const index_t row_id = A_row_ind[nnz_id];

        // Process the columns of x in batches of blockDim.x
        for (uint32_t row_batch_start = 0; row_batch_start < x_cols; row_batch_start += blockDim.x) {
            const uint32_t row_batch_end = min(row_batch_start + blockDim.x, x_cols);
            if (threadIdx.x > row_batch_end)
                continue;

            // Each thread processes one element of the row of x
            const scalar_t x_val = x[row_id * x_cols + threadIdx.x];

            // Compute the product and accumulate
            atomicAdd(&Ax[row_id * x_cols + threadIdx.x], A_val * x_val);
        }
    }
}

template <typename index_t, typename scalar_t, uint32_t MAX_X_COLS = 2048>
__global__ void _shared_mem_kernel(const scalar_t* A_values, const index_t* A_row_ind, const index_t* A_colptr,
                                   const scalar_t* x, scalar_t* Ax,
                                   const uint32_t x_rows, const uint32_t x_cols) {
     using uindex_t = typename std::make_unsigned<index_t>::type;

     // Each thread block/SM is responsible for a column of A.
     // Each thread within the block is responsible for computing one element of the column.
     // The x component of the thread block maps to a column of A and thus row of x.
     // x is assumed to be row-major.
     // For optimal performance, the x dimension of the thread block should be equal to the number of columns in x.
     const uint32_t col_id = blockIdx.x;

    __shared__ scalar_t x_row_shared[MAX_X_COLS];

    // Load the entire corresponding row of x into shared memory
    if (threadIdx.x < x_cols)
        x_row_shared[threadIdx.x] = x[col_id * x_cols + threadIdx.x];

    __syncthreads();

    const index_t col_start = A_colptr[col_id];
    const index_t col_end = A_colptr[col_id + 1];

    // Process each NNZ in the assigned column of A
    for (uindex_t nnz_id = col_start; nnz_id < col_end; ++nnz_id) {
        const scalar_t A_val = A_values[nnz_id];
        const index_t row_id = A_row_ind[nnz_id];

        // Process the columns of x in batches of blockDim.x
        for (uint32_t row_batch_start = 0; row_batch_start < x_cols; row_batch_start += blockDim.x) {
            const uint32_t row_batch_end = min(row_batch_start + blockDim.x, x_cols);
            if (threadIdx.x > row_batch_end)
                continue;

            // Each thread processes one element of the row of x
            const scalar_t x_val = x_row_shared[threadIdx.x];

            // Compute the product and accumulate
            atomicAdd(&Ax[row_id * x_cols + threadIdx.x], A_val * x_val);
        }
    }
}

at::Tensor spdmm_csc_naive(const at::Tensor A, const at::Tensor x) {
    const uint32_t x_rows = x.size(0);
    const uint32_t x_cols = x.size(1);

    const at::Tensor& A_row_ind = A.row_indices();
    const at::Tensor& A_colptr = A.ccol_indices();
    const at::Tensor& A_values = A.values();

    at::Tensor Ax = at::zeros({A.size(0), x.size(1)}, x.options());

    const dim3 thread_blocks((x_cols + thread_block_size.x - 1) / thread_block_size.x,
                             (x_rows + thread_block_size.y - 1) / thread_block_size.y);

    AT_DISPATCH_FLOATING_TYPES(A_values.scalar_type(), "spdmm_csc_naive", ([&] {
        AT_DISPATCH_INDEX_TYPES(A_row_ind.scalar_type(), "spdmm_csc_naive", [&] {
            _naive_kernel<index_t, scalar_t><<<thread_blocks, thread_block_size>>>(
                A_values.data_ptr<scalar_t>(), A_row_ind.data_ptr<index_t>(), A_colptr.data_ptr<index_t>(),
                x.data_ptr<scalar_t>(), Ax.data_ptr<scalar_t>(), x_rows, x_cols);
        });
    }));

    return Ax;
}

at::Tensor spdmm_csc_shared_mem(const at::Tensor A, const at::Tensor x) {
    const uint32_t x_rows = x.size(0);
    const uint32_t x_cols = x.size(1);

    const at::Tensor& A_row_ind = A.row_indices();
    const at::Tensor& A_colptr = A.ccol_indices();
    const at::Tensor& A_values = A.values();

    at::Tensor Ax = at::zeros({A.size(0), x.size(1)}, x.options());

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    const dim3 thread_block_size(min(x_cols, prop.maxThreadsPerBlock), 1);
    const dim3 thread_blocks(x_cols, 1);

    AT_DISPATCH_FLOATING_TYPES(A_values.scalar_type(), "spdmm_csc_shared_mem", ([&] {
        AT_DISPATCH_INDEX_TYPES(A_row_ind.scalar_type(), "spdmm_csc_shared_mem", [&] {
            _shared_mem_kernel<index_t, scalar_t><<<thread_blocks, thread_block_size>>>(
                A_values.data_ptr<scalar_t>(), A_row_ind.data_ptr<index_t>(), A_colptr.data_ptr<index_t>(),
                x.data_ptr<scalar_t>(), Ax.data_ptr<scalar_t>(), x_rows, x_cols);
        });
    }));

    return Ax;
}