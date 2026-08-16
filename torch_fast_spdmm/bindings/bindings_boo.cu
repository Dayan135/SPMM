#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>

#include "../kernels/boo_kernels.hpp"
#include "../kernels/csr_kernels.hpp"
#include "../kernels/pbr_kernels.hpp"

// BOO GPU entry point.
//
// Index arrays are int32 unconditionally -- BOO does not template on index_t
// the way PBR does, which halves both the index traffic and the pybind symbol
// count (2 symbols instead of 4).
//
// The two-stream fork/join is kept from the PBR path even though the default
// `keep` small-block policy leaves the remainder empty: the policy is
// deliberately open (BOO_FORMAT.md sec. 7), and `remainder` must work without a
// later rewrite.  With an empty remainder the CSR leg is simply skipped.
template <typename scalar_t>
at::Tensor boo_spmm_cuda_dispatch(
    int rows, int cols,
    int max_nr, int max_nc, int max_nnz_blk,
    const at::Tensor row_grp_ptr,
    const at::Tensor row_ids,
    const at::Tensor col_grp_ptr,
    const at::Tensor col_ids,
    const at::Tensor blk_row_grp,
    const at::Tensor blk_col_grp,
    const at::Tensor blk_nz_ptr,
    const at::Tensor blk_shift,
    const at::Tensor nz_idx,
    const at::Tensor nz_val,
    const at::Tensor rem_indptr,
    const at::Tensor rem_col_ind,
    const at::Tensor rem_vals,
    const at::Tensor X,
    int threads = 256,
    int p = 32,
    int csr_mode = 0,
    bool row_seg = true
) {
    const int num_blocks = (int)blk_row_grp.size(0);
    const int csr_nnz    = (int)rem_col_ind.size(0);
    const int batch_size = (X.dim() == 3) ? (int)X.size(0) : 1;
    const int features   = (int)X.size(-1);

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

    if (num_blocks > 0) {
        launch_boo_spmm<scalar_t>(
            num_blocks, features, batch_size, rows, cols,
            max_nr, max_nc, max_nnz_blk,
            row_grp_ptr.data_ptr<int32_t>(),
            row_ids.data_ptr<int32_t>(),
            col_grp_ptr.data_ptr<int32_t>(),
            col_ids.data_ptr<int32_t>(),
            blk_row_grp.data_ptr<int32_t>(),
            blk_col_grp.data_ptr<int32_t>(),
            blk_nz_ptr.data_ptr<int32_t>(),
            blk_shift.data_ptr<uint8_t>(),
            reinterpret_cast<const uint16_t*>(nz_idx.data_ptr<int16_t>()),
            nz_val.data_ptr<scalar_t>(),
            X3.data_ptr<scalar_t>(),
            Y.data_ptr<scalar_t>(),
            block_stream.stream(),
            threads, p, row_seg
        );
    }

    if (csr_nnz > 0) {
        launch_csr_spmm<int32_t, scalar_t>(
            rows, features, batch_size, cols, csr_nnz,
            rem_indptr.data_ptr<int32_t>(),
            rem_col_ind.data_ptr<int32_t>(),
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

void bind_boo_functions(pybind11::module_& m) {
    m.def("boo_spmm_cuda_f32", &boo_spmm_cuda_dispatch<float>);
    m.def("boo_spmm_cuda_f64", &boo_spmm_cuda_dispatch<double>);
    m.def("boo_arena_bytes_cuda", &boo_arena_bytes);
}
