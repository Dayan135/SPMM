#include "csr_kernels.hpp"

const dim3 threads_per_block(32, 32);

template <typename index_t, typename scalar_t>
__global__ void _naive_kernel(const scalar_t* A_values, const index_t* A_col_ind, const index_t* A_rowptr,
                              const scalar_t* x, scalar_t* Ax,
                              const typename std::make_unsigned<index_t>::type A_rows,
                              const typename std::make_unsigned<index_t>::type x_cols) {
        using uindex_t = typename std::make_unsigned<index_t>::type;

		const uint32_t col_id = blockIdx.x * blockDim.x + threadIdx.x;
		const uint32_t row_id = blockIdx.y * blockDim.y + threadIdx.y;

		if ((row_id >= A_rows) || (col_id >= x_cols))
		    return;

        const index_t row_start = A_rowptr[row_id];
        const index_t row_end = A_rowptr[row_id + 1];

        // One thread handles all elements of the row assigned to it
        scalar_t element_val = 0.0;
        for (uindex_t nnz_id = row_start; nnz_id < row_end; ++nnz_id) {
            const auto x_row_id = A_col_ind[nnz_id];
            element_val += A_values[nnz_id] * x[x_row_id * x_cols + col_id];
        }

        Ax[row_id * x_cols + col_id] = element_val;
}

template <typename index_t, typename scalar_t>
__global__ void _shared_mem_kernel(const scalar_t* A_values, const index_t* A_col_ind, const index_t* A_rowptr,
                                   const scalar_t* x, scalar_t* Ax,
                                   const typename std::make_unsigned<index_t>::type A_rows,
                                   const typename std::make_unsigned<index_t>::type x_cols) {
    using uindex_t = typename std::make_unsigned<index_t>::type;

    const uint32_t thread_id = blockIdx.x * blockDim.x;
	const uint32_t col_id = thread_id + threadIdx.x;
	const uint32_t row_id = blockIdx.y * blockDim.y + threadIdx.y;

	if ((row_id >= A_rows) || (col_id >= x_cols))
		return;

	__shared__ scalar_t s_values[threads_per_block.x * threads_per_block.y];
	__shared__ index_t  s_col_ind[threads_per_block.x * threads_per_block.y];
	__shared__ scalar_t s_x[threads_per_block.x * threads_per_block.y];

	const index_t row_start = A_rowptr[row_id];
    const index_t row_end = A_rowptr[row_id + 1];

	if (threadIdx.y < row_end - row_start) {
		s_values[threadIdx.x + blockDim.x * threadIdx.y] = A_values[row_start + threadIdx.y];
		s_col_ind[threadIdx.x + blockDim.x * threadIdx.y] = A_col_ind[row_start + threadIdx.y];
	}
    s_x[threadIdx.x + blockDim.x * threadIdx.y] = x[row_id * x_cols + col_id];
	
	__syncthreads();

	// One thread handles all elements of the row assigned to it
	scalar_t element_val = 0.0;
	for (index_t col = row_start; col < row_end; ++col) {
		const uindex_t t = col - row_start;
		index_t x_row_id;
	    scalar_t A_val;
	    scalar_t x_val;

		if (t < blockDim.y) {
			x_row_id = s_col_ind[threadIdx.x + blockDim.x * t];
			A_val = s_values[threadIdx.x + blockDim.x * t];
		} else {
			x_row_id = A_col_ind[col];
			A_val = A_values[col];
		}

		if ((x_row_id >= thread_id) && (x_row_id < thread_id + blockDim.x)) {
			x_val = s_x[blockDim.x * threadIdx.y + x_row_id - thread_id];
		} else {
            x_val = x[x_row_id * x_cols + col_id];
		}
		element_val += A_val * x_val;
	}

    Ax[row_id * x_cols + col_id] = element_val;
}

at::Tensor spdmm_csr_naive(const at::Tensor A, const at::Tensor x) {
    const uint32_t A_rows = A.size(0);
    const uint32_t x_cols = x.size(1);

    const at::Tensor& A_col_ind = A.col_indices();
    const at::Tensor& A_rowptr = A.crow_indices();
    const at::Tensor& A_values = A.values();

    at::Tensor Ax = at::zeros({A_rows, x_cols}, x.options());

    const dim3 thread_blocks((x_cols + threads_per_block.x - 1) / threads_per_block.x,
                             (A_rows + threads_per_block.y - 1) / threads_per_block.y);

    AT_DISPATCH_FLOATING_TYPES(A_values.scalar_type(), "fast_spdmm_csr_naive", ([&] {
        AT_DISPATCH_INDEX_TYPES(A_col_ind.scalar_type(), "fast_spdmm_csr_naive", ([&] {
            _naive_kernel<index_t, scalar_t><<<thread_blocks, threads_per_block>>>(
                A_values.data_ptr<scalar_t>(), A_col_ind.data_ptr<index_t>(), A_rowptr.data_ptr<index_t>(),
                x.data_ptr<scalar_t>(), Ax.data_ptr<scalar_t>(), A_rows, x_cols);
        }));
    }));
//    cuda_check_err(cudaGetLastError());

    return Ax;
}

at::Tensor spdmm_csr_shared_mem(const at::Tensor A, const at::Tensor x) {
    const uint32_t A_rows = A.size(0);
    const uint32_t x_cols = x.size(1);

    const at::Tensor& A_col_ind = A.col_indices();
    const at::Tensor& A_rowptr = A.crow_indices();
    const at::Tensor& A_values = A.values();

    at::Tensor Ax = at::zeros({A_rows, x_cols}, x.options());

    const dim3 thread_blocks((x_cols + threads_per_block.x - 1) / threads_per_block.x,
                             (A_rows + threads_per_block.y - 1) / threads_per_block.y);

    AT_DISPATCH_FLOATING_TYPES(A_values.scalar_type(), "fast_spdmm_csr_shared_mem", ([&] {
        AT_DISPATCH_INDEX_TYPES(A_col_ind.scalar_type(), "fast_spdmm_csr_shared_mem", ([&] {
            _shared_mem_kernel<index_t, scalar_t><<<thread_blocks, threads_per_block>>>(
                A_values.data_ptr<scalar_t>(), A_col_ind.data_ptr<index_t>(), A_rowptr.data_ptr<index_t>(),
                x.data_ptr<scalar_t>(), Ax.data_ptr<scalar_t>(), A_rows, x_cols);
        }));
    }));
//    cuda_check_err(cudaGetLastError());

    return Ax;
}