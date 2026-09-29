// C++-only timing harness for BOO, mirroring bindings_cusparse.cu's
// DirectSpMM and bindings_pbr_timed.cu's PBRTimedSpMM: allocation, stream
// setup, and event creation happen once at construction (outside
// measurement); timed_run() brackets only the work needed to produce Y with
// CUDA events on the construction stream. Plain (non-bucketed, non-plan)
// three-way fork/join only -- see boo_run_into in bindings_boo.cu.
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>

#include "../kernels/boo_kernels.hpp"

namespace {

void cuda_check(cudaError_t s) {
    TORCH_CHECK(s == cudaSuccess, "CUDA: ", cudaGetErrorString(s));
}

class BOOTimedSpMM {
    at::Tensor row_grp_ptr_, row_ids_, col_grp_ptr_, col_ids_,
               blk_row_grp_, blk_col_grp_, blk_nz_ptr_, blk_shift_,
               nz_idx_, nz_val_, rem_indptr_, rem_col_ind_, rem_vals_,
               tail_row_, tail_col_, tail_val_, x3_, y_;
    int rows_, cols_, max_nr_, max_nc_, max_nnz_blk_, threads_, p_, csr_mode_, device_;
    bool row_seg_, diagnostic_;
    cudaStream_t cur_stream_ = nullptr, block_stream_ = nullptr,
                 csr_stream_ = nullptr, tail_stream_ = nullptr;
    cudaEvent_t start_ = nullptr, end_ = nullptr;

    void cleanup() noexcept {
        if (cur_stream_) cudaStreamSynchronize(cur_stream_);
        if (start_) cudaEventDestroy(start_);
        if (end_) cudaEventDestroy(end_);
    }
    void check_stream() const {
        TORCH_CHECK(c10::cuda::getCurrentCUDAStream(device_).stream() == cur_stream_,
                    "BOOTimedSpMM must run on its construction stream");
    }

public:
    BOOTimedSpMM(int rows, int cols, int max_nr, int max_nc, int max_nnz_blk,
                 at::Tensor row_grp_ptr, at::Tensor row_ids,
                 at::Tensor col_grp_ptr, at::Tensor col_ids,
                 at::Tensor blk_row_grp, at::Tensor blk_col_grp,
                 at::Tensor blk_nz_ptr, at::Tensor blk_shift,
                 at::Tensor nz_idx, at::Tensor nz_val,
                 at::Tensor rem_indptr, at::Tensor rem_col_ind, at::Tensor rem_vals,
                 at::Tensor tail_row, at::Tensor tail_col, at::Tensor tail_val,
                 at::Tensor x, int threads, int p, int csr_mode, bool row_seg,
                 bool diagnostic_raw_store)
        : row_grp_ptr_(row_grp_ptr), row_ids_(row_ids), col_grp_ptr_(col_grp_ptr), col_ids_(col_ids),
          blk_row_grp_(blk_row_grp), blk_col_grp_(blk_col_grp), blk_nz_ptr_(blk_nz_ptr),
          blk_shift_(blk_shift), nz_idx_(nz_idx), nz_val_(nz_val),
          rem_indptr_(rem_indptr), rem_col_ind_(rem_col_ind), rem_vals_(rem_vals),
          tail_row_(tail_row), tail_col_(tail_col), tail_val_(tail_val),
          x3_(((x.dim() == 2) ? x.unsqueeze(0) : x).contiguous()),
          rows_(rows), cols_(cols), max_nr_(max_nr), max_nc_(max_nc), max_nnz_blk_(max_nnz_blk),
          threads_(threads), p_(p), csr_mode_(csr_mode), device_(x.get_device()),
          row_seg_(row_seg), diagnostic_(diagnostic_raw_store) {
        TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kFloat,
                    "BOOTimedSpMM: X must be a CUDA FP32 tensor");
        TORCH_CHECK(x.dim() == 2 || x.dim() == 3, "BOOTimedSpMM: X must be 2D or 3D");
        TORCH_CHECK(threads == 128 || threads == 256 || threads == 512,
                    "BOOTimedSpMM: threads must be 128, 256 or 512");
        TORCH_CHECK(p == 8 || p == 16 || p == 32 || p == 64 || p == 128,
                    "BOOTimedSpMM: p must be 8, 16, 32, 64 or 128");
        for (const auto& t : {row_grp_ptr, row_ids, col_grp_ptr, col_ids,
                              blk_row_grp, blk_col_grp, blk_nz_ptr, blk_shift,
                              nz_idx, nz_val, rem_indptr, rem_col_ind, rem_vals,
                              tail_row, tail_col, tail_val}) {
            TORCH_CHECK(t.device() == x.device() && t.is_contiguous(),
                        "BOOTimedSpMM: BOO metadata must be contiguous tensors on X's device");
        }
        TORCH_CHECK(nz_val.scalar_type() == at::kFloat && rem_vals.scalar_type() == at::kFloat,
                    "BOOTimedSpMM: nz_val/rem_vals must be fp32 (this harness is fp32 only)");
        if (diagnostic_) {
            TORCH_CHECK(row_seg, "BOOTimedSpMM: diagnostic_raw_store requires row_seg=True");
        }

        c10::cuda::CUDAGuard guard(x.device());
        cur_stream_   = c10::cuda::getCurrentCUDAStream(device_).stream();
        block_stream_ = at::cuda::getStreamFromPool().stream();
        csr_stream_   = at::cuda::getStreamFromPool().stream();
        tail_stream_  = at::cuda::getStreamFromPool().stream();
        const int batch_size = (int)x3_.size(0);
        const int features   = (int)x3_.size(-1);
        y_ = at::empty({batch_size, rows, features}, x.options());
        try {
            cuda_check(cudaEventCreate(&start_));
            cuda_check(cudaEventCreate(&end_));
            cuda_check(cudaEventRecord(start_, cur_stream_));
            cuda_check(cudaEventRecord(end_, cur_stream_));
            cuda_check(cudaEventSynchronize(end_));
        } catch (...) {
            cleanup();
            throw;
        }
    }
    ~BOOTimedSpMM() {
        c10::cuda::CUDAGuard guard(device_);
        cleanup();
    }

    at::Tensor run() {
        TORCH_CHECK(!diagnostic_,
                    "BOOTimedSpMM(diagnostic_raw_store=True) is not correctness-checkable "
                    "-- its output is incorrect by construction");
        c10::cuda::CUDAGuard guard(device_);
        check_stream();
        boo_run_into<float>(
            y_, rows_, cols_, max_nr_, max_nc_, max_nnz_blk_,
            row_grp_ptr_, row_ids_, col_grp_ptr_, col_ids_,
            blk_row_grp_, blk_col_grp_, blk_nz_ptr_, blk_shift_, nz_idx_, nz_val_,
            rem_indptr_, rem_col_ind_, rem_vals_, tail_row_, tail_col_, tail_val_,
            x3_, threads_, p_, csr_mode_, row_seg_,
            cur_stream_, block_stream_, csr_stream_, tail_stream_);
        return y_;
    }

    // Brackets exactly: the Y memset, the block kernel, the CSR-remainder
    // kernel and tail kernel on their own streams, and the join
    // (non-diagnostic); or, in diagnostic mode, just the block kernel with
    // no memset and no remainder/tail legs. No Python, no allocation, in
    // the timed region.
    float timed_run() {
        c10::cuda::CUDAGuard guard(device_);
        check_stream();
        cuda_check(cudaEventRecord(start_, cur_stream_));
        if (diagnostic_) {
            const int num_blocks = (int)blk_row_grp_.size(0);
            launch_boo_diag_spmm(
                num_blocks, (int)x3_.size(-1), (int)x3_.size(0), rows_, cols_,
                max_nr_, max_nc_, max_nnz_blk_,
                row_grp_ptr_.data_ptr<int32_t>(), row_ids_.data_ptr<int32_t>(),
                col_grp_ptr_.data_ptr<int32_t>(), col_ids_.data_ptr<int32_t>(),
                blk_row_grp_.data_ptr<int32_t>(), blk_col_grp_.data_ptr<int32_t>(),
                blk_nz_ptr_.data_ptr<int32_t>(), blk_shift_.data_ptr<uint8_t>(),
                reinterpret_cast<const uint16_t*>(nz_idx_.data_ptr<int16_t>()),
                nz_val_.data_ptr<float>(), x3_.data_ptr<float>(), y_.data_ptr<float>(),
                cur_stream_, threads_, p_);
        } else {
            boo_run_into<float>(
                y_, rows_, cols_, max_nr_, max_nc_, max_nnz_blk_,
                row_grp_ptr_, row_ids_, col_grp_ptr_, col_ids_,
                blk_row_grp_, blk_col_grp_, blk_nz_ptr_, blk_shift_, nz_idx_, nz_val_,
                rem_indptr_, rem_col_ind_, rem_vals_, tail_row_, tail_col_, tail_val_,
                x3_, threads_, p_, csr_mode_, row_seg_,
                cur_stream_, block_stream_, csr_stream_, tail_stream_);
        }
        cuda_check(cudaEventRecord(end_, cur_stream_));
        cuda_check(cudaEventSynchronize(end_));
        float ms;
        cuda_check(cudaEventElapsedTime(&ms, start_, end_));
        return ms;
    }

    at::Tensor output() const {
        TORCH_CHECK(!diagnostic_,
                    "BOOTimedSpMM(diagnostic_raw_store=True)'s output is not meaningful");
        return y_;
    }
    py::dict info() const {
        py::dict d;
        d["diagnostic"] = diagnostic_;
        return d;
    }
};

} // namespace

void bind_boo_timed_functions(py::module_& m) {
    py::class_<BOOTimedSpMM, std::shared_ptr<BOOTimedSpMM>>(m, "BOOTimedSpMM")
        .def(py::init<int, int, int, int, int, at::Tensor, at::Tensor, at::Tensor, at::Tensor,
                      at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor,
                      at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor,
                      at::Tensor, int, int, int, bool, bool>(),
             py::arg("rows"), py::arg("cols"), py::arg("max_nr"), py::arg("max_nc"),
             py::arg("max_nnz_blk"), py::arg("row_grp_ptr"), py::arg("row_ids"),
             py::arg("col_grp_ptr"), py::arg("col_ids"), py::arg("blk_row_grp"),
             py::arg("blk_col_grp"), py::arg("blk_nz_ptr"), py::arg("blk_shift"),
             py::arg("nz_idx"), py::arg("nz_val"), py::arg("rem_indptr"), py::arg("rem_col_ind"),
             py::arg("rem_vals"), py::arg("tail_row"), py::arg("tail_col"), py::arg("tail_val"),
             py::arg("x"), py::arg("threads") = 256, py::arg("p") = 64, py::arg("csr_mode") = 0,
             py::arg("row_seg") = true, py::arg("diagnostic_raw_store") = false)
        .def("run", &BOOTimedSpMM::run)
        .def("timed_run", &BOOTimedSpMM::timed_run)
        .def_property_readonly("output", &BOOTimedSpMM::output)
        .def("info", &BOOTimedSpMM::info);
}
