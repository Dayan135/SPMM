#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/SparseCsrTensorUtils.h>
#include <ATen/cuda/CUDAContext.h>

#include "../kernels/csr_kernels.hpp"
#include "../kernels/pbr_kernels.hpp"

template <typename index_t, typename scalar_t>
at::Tensor pbr_spmm_cuda_dispatch(
    int cols, int rows,
    int BLOCK_ROWS, int BLOCK_COLS,
    const at::Tensor block_codes,
    const at::Tensor block_coords,
    const at::Tensor block_offsets,
    const at::Tensor block_data,
    const at::Tensor rem_indptr,
    const at::Tensor rem_col_ind,
    const at::Tensor rem_vals,
    const at::Tensor X,
    int csr_mode = 0,
    int smem_size = 2048
) {
    const int num_pbr_blocks = (int)block_codes.size(0);
    const int csr_nnz        = (int)rem_col_ind.size(0);
    const int batch_size     = (X.dim() == 3) ? (int)X.size(0) : 1;
    const int features       = (int)X.size(-1);

    at::Tensor X3 = (X.dim() == 2) ? X.unsqueeze(0) : X;
    at::Tensor Y  = at::zeros({batch_size, rows, features}, X.options());

    auto cur_stream   = at::cuda::getCurrentCUDAStream();
    auto block_stream = at::cuda::getStreamFromPool();
    auto csr_stream   = at::cuda::getStreamFromPool();

    // Both new streams wait for Y to be zeroed on cur_stream
    cudaEvent_t y_ready;
    cudaEventCreateWithFlags(&y_ready, cudaEventDisableTiming);
    cudaEventRecord(y_ready, cur_stream.stream());
    cudaStreamWaitEvent(block_stream.stream(), y_ready, 0);
    cudaStreamWaitEvent(csr_stream.stream(),   y_ready, 0);
    cudaEventDestroy(y_ready);

    if (num_pbr_blocks > 0) {
        launch_pbr_spmm<index_t, scalar_t>(
            num_pbr_blocks, features, batch_size, cols, rows, BLOCK_ROWS, BLOCK_COLS,
            block_codes.data_ptr<int64_t>(),
            block_coords.data_ptr<index_t>(),
            block_offsets.data_ptr<index_t>(),
            block_data.data_ptr<scalar_t>(),
            X3.data_ptr<scalar_t>(),
            Y.data_ptr<scalar_t>(),
            block_stream.stream(),
            smem_size
        );
    }

    if (csr_nnz > 0) {
        launch_csr_spmm<index_t, scalar_t>(
            rows, features, batch_size, cols, csr_nnz,
            rem_indptr.data_ptr<index_t>(),
            rem_col_ind.data_ptr<index_t>(),
            rem_vals.data_ptr<scalar_t>(),
            X3.data_ptr<scalar_t>(),
            Y.data_ptr<scalar_t>(),
            csr_stream.stream(),
            csr_mode
        );
    }

    // Rejoin both streams into cur_stream before returning Y
    cudaEvent_t block_done, csr_done;
    cudaEventCreateWithFlags(&block_done, cudaEventDisableTiming);
    cudaEventCreateWithFlags(&csr_done,   cudaEventDisableTiming);
    cudaEventRecord(block_done, block_stream.stream());
    cudaEventRecord(csr_done,   csr_stream.stream());
    cudaStreamWaitEvent(cur_stream.stream(), block_done, 0);
    cudaStreamWaitEvent(cur_stream.stream(), csr_done,   0);
    cudaEventDestroy(block_done);
    cudaEventDestroy(csr_done);

    return Y;
}

// Experimental chunkcols/nnzflat dispatch fns -- byte-for-byte the same shape
// as pbr_spmm_cuda_dispatch above, just forwarding to the corresponding
// launch_pbr_{chunkcols,nnzflat}_spmm. See pbr_kernel.cu.
template <typename index_t, typename scalar_t>
at::Tensor pbr_chunkcols_spmm_cuda_dispatch(
    int cols, int rows,
    int BLOCK_ROWS, int BLOCK_COLS,
    const at::Tensor block_codes,
    const at::Tensor block_coords,
    const at::Tensor block_offsets,
    const at::Tensor block_data,
    const at::Tensor rem_indptr,
    const at::Tensor rem_col_ind,
    const at::Tensor rem_vals,
    const at::Tensor X,
    int csr_mode = 0,
    int smem_size = 2048
) {
    const int num_pbr_blocks = (int)block_codes.size(0);
    const int csr_nnz        = (int)rem_col_ind.size(0);
    const int batch_size     = (X.dim() == 3) ? (int)X.size(0) : 1;
    const int features       = (int)X.size(-1);

    at::Tensor X3 = (X.dim() == 2) ? X.unsqueeze(0) : X;
    at::Tensor Y  = at::zeros({batch_size, rows, features}, X.options());

    auto cur_stream   = at::cuda::getCurrentCUDAStream();
    auto block_stream = at::cuda::getStreamFromPool();
    auto csr_stream   = at::cuda::getStreamFromPool();

    cudaEvent_t y_ready;
    cudaEventCreateWithFlags(&y_ready, cudaEventDisableTiming);
    cudaEventRecord(y_ready, cur_stream.stream());
    cudaStreamWaitEvent(block_stream.stream(), y_ready, 0);
    cudaStreamWaitEvent(csr_stream.stream(),   y_ready, 0);
    cudaEventDestroy(y_ready);

    if (num_pbr_blocks > 0) {
        launch_pbr_chunkcols_spmm<index_t, scalar_t>(
            num_pbr_blocks, features, batch_size, cols, rows, BLOCK_ROWS, BLOCK_COLS,
            block_codes.data_ptr<int64_t>(),
            block_coords.data_ptr<index_t>(),
            block_offsets.data_ptr<index_t>(),
            block_data.data_ptr<scalar_t>(),
            X3.data_ptr<scalar_t>(),
            Y.data_ptr<scalar_t>(),
            block_stream.stream(),
            smem_size
        );
    }

    if (csr_nnz > 0) {
        launch_csr_spmm<index_t, scalar_t>(
            rows, features, batch_size, cols, csr_nnz,
            rem_indptr.data_ptr<index_t>(),
            rem_col_ind.data_ptr<index_t>(),
            rem_vals.data_ptr<scalar_t>(),
            X3.data_ptr<scalar_t>(),
            Y.data_ptr<scalar_t>(),
            csr_stream.stream(),
            csr_mode
        );
    }

    cudaEvent_t block_done, csr_done;
    cudaEventCreateWithFlags(&block_done, cudaEventDisableTiming);
    cudaEventCreateWithFlags(&csr_done,   cudaEventDisableTiming);
    cudaEventRecord(block_done, block_stream.stream());
    cudaEventRecord(csr_done,   csr_stream.stream());
    cudaStreamWaitEvent(cur_stream.stream(), block_done, 0);
    cudaStreamWaitEvent(cur_stream.stream(), csr_done,   0);
    cudaEventDestroy(block_done);
    cudaEventDestroy(csr_done);

    return Y;
}

template <typename index_t, typename scalar_t>
at::Tensor pbr_nnzflat_spmm_cuda_dispatch(
    int cols, int rows,
    int BLOCK_ROWS, int BLOCK_COLS,
    const at::Tensor block_codes,
    const at::Tensor block_coords,
    const at::Tensor block_offsets,
    const at::Tensor block_data,
    const at::Tensor rem_indptr,
    const at::Tensor rem_col_ind,
    const at::Tensor rem_vals,
    const at::Tensor X,
    int csr_mode = 0,
    int smem_size = 2048
) {
    const int num_pbr_blocks = (int)block_codes.size(0);
    const int csr_nnz        = (int)rem_col_ind.size(0);
    const int batch_size     = (X.dim() == 3) ? (int)X.size(0) : 1;
    const int features       = (int)X.size(-1);

    at::Tensor X3 = (X.dim() == 2) ? X.unsqueeze(0) : X;
    at::Tensor Y  = at::zeros({batch_size, rows, features}, X.options());

    auto cur_stream   = at::cuda::getCurrentCUDAStream();
    auto block_stream = at::cuda::getStreamFromPool();
    auto csr_stream   = at::cuda::getStreamFromPool();

    cudaEvent_t y_ready;
    cudaEventCreateWithFlags(&y_ready, cudaEventDisableTiming);
    cudaEventRecord(y_ready, cur_stream.stream());
    cudaStreamWaitEvent(block_stream.stream(), y_ready, 0);
    cudaStreamWaitEvent(csr_stream.stream(),   y_ready, 0);
    cudaEventDestroy(y_ready);

    if (num_pbr_blocks > 0) {
        launch_pbr_nnzflat_spmm<index_t, scalar_t>(
            num_pbr_blocks, features, batch_size, cols, rows, BLOCK_ROWS, BLOCK_COLS,
            block_codes.data_ptr<int64_t>(),
            block_coords.data_ptr<index_t>(),
            block_offsets.data_ptr<index_t>(),
            block_data.data_ptr<scalar_t>(),
            X3.data_ptr<scalar_t>(),
            Y.data_ptr<scalar_t>(),
            block_stream.stream(),
            smem_size
        );
    }

    if (csr_nnz > 0) {
        launch_csr_spmm<index_t, scalar_t>(
            rows, features, batch_size, cols, csr_nnz,
            rem_indptr.data_ptr<index_t>(),
            rem_col_ind.data_ptr<index_t>(),
            rem_vals.data_ptr<scalar_t>(),
            X3.data_ptr<scalar_t>(),
            Y.data_ptr<scalar_t>(),
            csr_stream.stream(),
            csr_mode
        );
    }

    cudaEvent_t block_done, csr_done;
    cudaEventCreateWithFlags(&block_done, cudaEventDisableTiming);
    cudaEventCreateWithFlags(&csr_done,   cudaEventDisableTiming);
    cudaEventRecord(block_done, block_stream.stream());
    cudaEventRecord(csr_done,   csr_stream.stream());
    cudaStreamWaitEvent(cur_stream.stream(), block_done, 0);
    cudaStreamWaitEvent(cur_stream.stream(), csr_done,   0);
    cudaEventDestroy(block_done);
    cudaEventDestroy(csr_done);

    return Y;
}

// 4-way benchmark variant -- same stream orchestration, different kernel.
template <typename index_t, typename scalar_t>
at::Tensor pbr_nnzflat_shared_spmm_cuda_dispatch(
    int cols, int rows,
    int BLOCK_ROWS, int BLOCK_COLS,
    const at::Tensor block_codes,
    const at::Tensor block_coords,
    const at::Tensor block_offsets,
    const at::Tensor block_data,
    const at::Tensor rem_indptr,
    const at::Tensor rem_col_ind,
    const at::Tensor rem_vals,
    const at::Tensor X,
    int csr_mode = 0,
    int smem_size = 2048
) {
    const int num_pbr_blocks = (int)block_codes.size(0);
    const int csr_nnz        = (int)rem_col_ind.size(0);
    const int batch_size     = (X.dim() == 3) ? (int)X.size(0) : 1;
    const int features       = (int)X.size(-1);

    at::Tensor X3 = (X.dim() == 2) ? X.unsqueeze(0) : X;
    at::Tensor Y  = at::zeros({batch_size, rows, features}, X.options());

    auto cur_stream   = at::cuda::getCurrentCUDAStream();
    auto block_stream = at::cuda::getStreamFromPool();
    auto csr_stream   = at::cuda::getStreamFromPool();

    cudaEvent_t y_ready;
    cudaEventCreateWithFlags(&y_ready, cudaEventDisableTiming);
    cudaEventRecord(y_ready, cur_stream.stream());
    cudaStreamWaitEvent(block_stream.stream(), y_ready, 0);
    cudaStreamWaitEvent(csr_stream.stream(),   y_ready, 0);
    cudaEventDestroy(y_ready);

    if (num_pbr_blocks > 0) {
        launch_pbr_nnzflat_shared_spmm<index_t, scalar_t>(
            num_pbr_blocks, features, batch_size, cols, rows, BLOCK_ROWS, BLOCK_COLS,
            block_codes.data_ptr<int64_t>(),
            block_coords.data_ptr<index_t>(),
            block_offsets.data_ptr<index_t>(),
            block_data.data_ptr<scalar_t>(),
            X3.data_ptr<scalar_t>(),
            Y.data_ptr<scalar_t>(),
            block_stream.stream(),
            smem_size
        );
    }

    if (csr_nnz > 0) {
        launch_csr_spmm<index_t, scalar_t>(
            rows, features, batch_size, cols, csr_nnz,
            rem_indptr.data_ptr<index_t>(),
            rem_col_ind.data_ptr<index_t>(),
            rem_vals.data_ptr<scalar_t>(),
            X3.data_ptr<scalar_t>(),
            Y.data_ptr<scalar_t>(),
            csr_stream.stream(),
            csr_mode
        );
    }

    cudaEvent_t block_done, csr_done;
    cudaEventCreateWithFlags(&block_done, cudaEventDisableTiming);
    cudaEventCreateWithFlags(&csr_done,   cudaEventDisableTiming);
    cudaEventRecord(block_done, block_stream.stream());
    cudaEventRecord(csr_done,   csr_stream.stream());
    cudaStreamWaitEvent(cur_stream.stream(), block_done, 0);
    cudaStreamWaitEvent(cur_stream.stream(), csr_done,   0);
    cudaEventDestroy(block_done);
    cudaEventDestroy(csr_done);

    return Y;
}

// 4-way benchmark variant -- same stream orchestration, different kernel.
template <typename index_t, typename scalar_t>
at::Tensor pbr_nnzflat_onthefly_spmm_cuda_dispatch(
    int cols, int rows,
    int BLOCK_ROWS, int BLOCK_COLS,
    const at::Tensor block_codes,
    const at::Tensor block_coords,
    const at::Tensor block_offsets,
    const at::Tensor block_data,
    const at::Tensor rem_indptr,
    const at::Tensor rem_col_ind,
    const at::Tensor rem_vals,
    const at::Tensor X,
    int csr_mode = 0,
    int smem_size = 2048
) {
    const int num_pbr_blocks = (int)block_codes.size(0);
    const int csr_nnz        = (int)rem_col_ind.size(0);
    const int batch_size     = (X.dim() == 3) ? (int)X.size(0) : 1;
    const int features       = (int)X.size(-1);

    at::Tensor X3 = (X.dim() == 2) ? X.unsqueeze(0) : X;
    at::Tensor Y  = at::zeros({batch_size, rows, features}, X.options());

    auto cur_stream   = at::cuda::getCurrentCUDAStream();
    auto block_stream = at::cuda::getStreamFromPool();
    auto csr_stream   = at::cuda::getStreamFromPool();

    cudaEvent_t y_ready;
    cudaEventCreateWithFlags(&y_ready, cudaEventDisableTiming);
    cudaEventRecord(y_ready, cur_stream.stream());
    cudaStreamWaitEvent(block_stream.stream(), y_ready, 0);
    cudaStreamWaitEvent(csr_stream.stream(),   y_ready, 0);
    cudaEventDestroy(y_ready);

    if (num_pbr_blocks > 0) {
        launch_pbr_nnzflat_onthefly_spmm<index_t, scalar_t>(
            num_pbr_blocks, features, batch_size, cols, rows, BLOCK_ROWS, BLOCK_COLS,
            block_codes.data_ptr<int64_t>(),
            block_coords.data_ptr<index_t>(),
            block_offsets.data_ptr<index_t>(),
            block_data.data_ptr<scalar_t>(),
            X3.data_ptr<scalar_t>(),
            Y.data_ptr<scalar_t>(),
            block_stream.stream(),
            smem_size
        );
    }

    if (csr_nnz > 0) {
        launch_csr_spmm<index_t, scalar_t>(
            rows, features, batch_size, cols, csr_nnz,
            rem_indptr.data_ptr<index_t>(),
            rem_col_ind.data_ptr<index_t>(),
            rem_vals.data_ptr<scalar_t>(),
            X3.data_ptr<scalar_t>(),
            Y.data_ptr<scalar_t>(),
            csr_stream.stream(),
            csr_mode
        );
    }

    cudaEvent_t block_done, csr_done;
    cudaEventCreateWithFlags(&block_done, cudaEventDisableTiming);
    cudaEventCreateWithFlags(&csr_done,   cudaEventDisableTiming);
    cudaEventRecord(block_done, block_stream.stream());
    cudaEventRecord(csr_done,   csr_stream.stream());
    cudaStreamWaitEvent(cur_stream.stream(), block_done, 0);
    cudaStreamWaitEvent(cur_stream.stream(), csr_done,   0);
    cudaEventDestroy(block_done);
    cudaEventDestroy(csr_done);

    return Y;
}

void bind_cuda_functions(py::module_& m) {
    m.def("spdmm_csr_naive", &spdmm_csr_naive);

    m.def("pbr_spmm_cuda_i32_f32", &pbr_spmm_cuda_dispatch<int32_t, float>);
    m.def("pbr_spmm_cuda_i64_f32", &pbr_spmm_cuda_dispatch<int64_t, float>);
    m.def("pbr_spmm_cuda_i32_f64", &pbr_spmm_cuda_dispatch<int32_t, double>);
    m.def("pbr_spmm_cuda_i64_f64", &pbr_spmm_cuda_dispatch<int64_t, double>);

    m.def("pbr_chunkcols_spmm_cuda_i32_f32", &pbr_chunkcols_spmm_cuda_dispatch<int32_t, float>);
    m.def("pbr_chunkcols_spmm_cuda_i64_f32", &pbr_chunkcols_spmm_cuda_dispatch<int64_t, float>);
    m.def("pbr_chunkcols_spmm_cuda_i32_f64", &pbr_chunkcols_spmm_cuda_dispatch<int32_t, double>);
    m.def("pbr_chunkcols_spmm_cuda_i64_f64", &pbr_chunkcols_spmm_cuda_dispatch<int64_t, double>);

    m.def("pbr_nnzflat_spmm_cuda_i32_f32", &pbr_nnzflat_spmm_cuda_dispatch<int32_t, float>);
    m.def("pbr_nnzflat_spmm_cuda_i64_f32", &pbr_nnzflat_spmm_cuda_dispatch<int64_t, float>);
    m.def("pbr_nnzflat_spmm_cuda_i32_f64", &pbr_nnzflat_spmm_cuda_dispatch<int32_t, double>);
    m.def("pbr_nnzflat_spmm_cuda_i64_f64", &pbr_nnzflat_spmm_cuda_dispatch<int64_t, double>);

    m.def("pbr_nnzflat_shared_spmm_cuda_i32_f32", &pbr_nnzflat_shared_spmm_cuda_dispatch<int32_t, float>);
    m.def("pbr_nnzflat_shared_spmm_cuda_i64_f32", &pbr_nnzflat_shared_spmm_cuda_dispatch<int64_t, float>);
    m.def("pbr_nnzflat_shared_spmm_cuda_i32_f64", &pbr_nnzflat_shared_spmm_cuda_dispatch<int32_t, double>);
    m.def("pbr_nnzflat_shared_spmm_cuda_i64_f64", &pbr_nnzflat_shared_spmm_cuda_dispatch<int64_t, double>);

    m.def("pbr_nnzflat_onthefly_spmm_cuda_i32_f32", &pbr_nnzflat_onthefly_spmm_cuda_dispatch<int32_t, float>);
    m.def("pbr_nnzflat_onthefly_spmm_cuda_i64_f32", &pbr_nnzflat_onthefly_spmm_cuda_dispatch<int64_t, float>);
    m.def("pbr_nnzflat_onthefly_spmm_cuda_i32_f64", &pbr_nnzflat_onthefly_spmm_cuda_dispatch<int32_t, double>);
    m.def("pbr_nnzflat_onthefly_spmm_cuda_i64_f64", &pbr_nnzflat_onthefly_spmm_cuda_dispatch<int64_t, double>);
}
