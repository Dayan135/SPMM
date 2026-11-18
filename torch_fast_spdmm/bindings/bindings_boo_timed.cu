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
#include "../kernels/csr_kernels.hpp"
#include "../kernels/pbr_kernels.hpp"  // launch_csr_spmm declaration

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
    // nnzflat (row_seg=false) variant, applied process-wide before every entry
    // and outside the timed interval: interleaved harnesses cannot inherit
    // another variant's setting. Ignored by the rowseg path.
    int nnzflat_fpt_;
    bool nnzflat_reference_;
    bool squeeze_output_;
    std::string launch_mode_;
    std::vector<int64_t> buckets_;
    at::Tensor plan_row_ptr_, plan_row_starts_;
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
    void apply_nnzflat() const {
        if (!row_seg_) {
            set_boo_nnzflat_fpt(nnzflat_fpt_);
            set_boo_nnzflat_reference(nnzflat_reference_);
        }
    }
    void run_selected() {
        // The default takes precisely the pre-existing production helper.
        if (launch_mode_ == "legacy" && buckets_.empty()) {
            boo_run_into<float>(
                y_, rows_, cols_, max_nr_, max_nc_, max_nnz_blk_,
                row_grp_ptr_, row_ids_, col_grp_ptr_, col_ids_,
                blk_row_grp_, blk_col_grp_, blk_nz_ptr_, blk_shift_, nz_idx_, nz_val_,
                rem_indptr_, rem_col_ind_, rem_vals_, tail_row_, tail_col_, tail_val_,
                x3_, threads_, p_, csr_mode_, row_seg_,
                cur_stream_, block_stream_, csr_stream_, tail_stream_);
            return;
        }
        const int blocks = (int)blk_row_grp_.size(0);
        const int csr_nnz = (int)rem_col_ind_.size(0);
        const int tail_nnz = (int)tail_col_.size(0);
        const int f = (int)x3_.size(-1), batch = (int)x3_.size(0);
        cuda_check(cudaMemsetAsync(y_.data_ptr<float>(), 0, y_.numel() * sizeof(float), cur_stream_));
        auto run_blocks = [&](cudaStream_t stream) {
            auto launch_range = [&](int start, int count, int nr, int nc, int nnz) {
                launch_boo_spmm<float>(
                    count, f, batch, rows_, cols_, nr, nc, nnz,
                    row_grp_ptr_.data_ptr<int32_t>(), row_ids_.data_ptr<int32_t>(),
                    col_grp_ptr_.data_ptr<int32_t>(), col_ids_.data_ptr<int32_t>(),
                    blk_row_grp_.data_ptr<int32_t>() + start,
                    blk_col_grp_.data_ptr<int32_t>() + start,
                    blk_nz_ptr_.data_ptr<int32_t>() + start,
                    blk_shift_.data_ptr<uint8_t>() + start,
                    reinterpret_cast<const uint16_t*>(nz_idx_.data_ptr<int16_t>()),
                    nz_val_.data_ptr<float>(), x3_.data_ptr<float>(), y_.data_ptr<float>(),
                    stream, threads_, p_, row_seg_,
                    plan_row_ptr_.defined() && plan_row_ptr_.numel()
                        ? plan_row_ptr_.data_ptr<int32_t>() + start : nullptr,
                    plan_row_starts_.defined() && plan_row_starts_.numel()
                        ? plan_row_starts_.data_ptr<int32_t>() : nullptr);
            };
            if (!blocks) return;
            if (buckets_.empty()) launch_range(0, blocks, max_nr_, max_nc_, max_nnz_blk_);
            else for (size_t i = 0; i < buckets_.size(); i += 5)
                launch_range((int)buckets_[i], (int)buckets_[i+1], (int)buckets_[i+2],
                             (int)buckets_[i+3], (int)buckets_[i+4]);
        };
        // boo_matmul(auto) uses this shortcut only when there is no remainder/tail.
        if (launch_mode_ == "auto" && !csr_nnz && !tail_nnz) {
            run_blocks(cur_stream_);
            return;
        }
        cudaEvent_t ready, block_done, csr_done, tail_done;
        cuda_check(cudaEventCreateWithFlags(&ready, cudaEventDisableTiming));
        cuda_check(cudaEventRecord(ready, cur_stream_));
        for (auto stream : {block_stream_, csr_stream_, tail_stream_})
            cuda_check(cudaStreamWaitEvent(stream, ready, 0));
        cuda_check(cudaEventDestroy(ready));
        run_blocks(block_stream_);
        if (csr_nnz) {
            launch_csr_spmm<int32_t, float>(
                rows_, f, batch, cols_, csr_nnz, rem_indptr_.data_ptr<int32_t>(),
                rem_col_ind_.data_ptr<int32_t>(), rem_vals_.data_ptr<float>(),
                x3_.data_ptr<float>(), y_.data_ptr<float>(), csr_stream_, csr_mode_);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
        if (tail_nnz) {
            launch_boo_tail_spmm<float>(
                tail_nnz, f, batch, rows_, cols_, tail_row_.data_ptr<int32_t>(),
                tail_col_.data_ptr<int32_t>(), tail_val_.data_ptr<float>(),
                x3_.data_ptr<float>(), y_.data_ptr<float>(), tail_stream_, threads_);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
        cuda_check(cudaEventCreateWithFlags(&block_done, cudaEventDisableTiming));
        cuda_check(cudaEventCreateWithFlags(&csr_done, cudaEventDisableTiming));
        cuda_check(cudaEventCreateWithFlags(&tail_done, cudaEventDisableTiming));
        cuda_check(cudaEventRecord(block_done, block_stream_));
        cuda_check(cudaEventRecord(csr_done, csr_stream_));
        cuda_check(cudaEventRecord(tail_done, tail_stream_));
        for (auto event : {block_done, csr_done, tail_done}) {
            cuda_check(cudaStreamWaitEvent(cur_stream_, event, 0));
            cuda_check(cudaEventDestroy(event));
        }
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
                 bool diagnostic_raw_store, int nnzflat_fpt, bool nnzflat_reference,
                 std::string launch_mode, std::vector<int64_t> buckets,
                 py::object plan_row_ptr, py::object plan_row_starts)
        : row_grp_ptr_(row_grp_ptr), row_ids_(row_ids), col_grp_ptr_(col_grp_ptr), col_ids_(col_ids),
          blk_row_grp_(blk_row_grp), blk_col_grp_(blk_col_grp), blk_nz_ptr_(blk_nz_ptr),
          blk_shift_(blk_shift), nz_idx_(nz_idx), nz_val_(nz_val),
          rem_indptr_(rem_indptr), rem_col_ind_(rem_col_ind), rem_vals_(rem_vals),
          tail_row_(tail_row), tail_col_(tail_col), tail_val_(tail_val),
          x3_(((x.dim() == 2) ? x.unsqueeze(0) : x).contiguous()),
          rows_(rows), cols_(cols), max_nr_(max_nr), max_nc_(max_nc), max_nnz_blk_(max_nnz_blk),
          threads_(threads), p_(p), csr_mode_(csr_mode), device_(x.get_device()),
          row_seg_(row_seg), diagnostic_(diagnostic_raw_store),
          nnzflat_fpt_(nnzflat_fpt), nnzflat_reference_(nnzflat_reference),
          squeeze_output_(x.dim() == 2), launch_mode_(std::move(launch_mode)),
          buckets_(std::move(buckets)) {
        TORCH_CHECK(launch_mode_ == "auto" || launch_mode_ == "legacy",
                    "launch_mode must be 'auto' or 'legacy'");
        TORCH_CHECK(buckets_.empty() || (launch_mode_ == "auto" && row_seg_ &&
                    buckets_.size() % 5 == 0),
                    "BOO execution plans require auto rowseg and 5-element buckets");
        if (!plan_row_ptr.is_none()) plan_row_ptr_ = plan_row_ptr.cast<at::Tensor>();
        if (!plan_row_starts.is_none()) plan_row_starts_ = plan_row_starts.cast<at::Tensor>();
        TORCH_CHECK(plan_row_ptr_.defined() == plan_row_starts_.defined(),
                    "BOO plan row tables must be provided together");
        if (plan_row_ptr_.defined()) {
            for (const auto& table : {plan_row_ptr_, plan_row_starts_})
                TORCH_CHECK(table.device() == x.device() && table.is_contiguous() &&
                            table.scalar_type() == at::kInt && table.dim() == 1,
                            "BOO plan row tables must be contiguous CUDA int32 vectors");
            TORCH_CHECK(plan_row_ptr_.numel() == 0 ||
                        (plan_row_ptr_.numel() == blk_row_grp_.numel() + 1 &&
                         plan_row_starts_.numel() >= blk_row_grp_.numel()),
                        "invalid BOO plan row table sizes");
        }
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
        TORCH_CHECK(nnzflat_fpt == 1 || nnzflat_fpt == 2,
                    "BOOTimedSpMM: nnzflat_fpt must be 1 or 2");
        TORCH_CHECK(!row_seg || (nnzflat_fpt == 1 && nnzflat_reference),
                    "BOOTimedSpMM: nnzflat settings require row_seg=False");
        TORCH_CHECK(nnzflat_fpt == 1 || !nnzflat_reference,
                    "BOOTimedSpMM: nnzflat FPT=2 requires nnzflat_reference=False");

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
        apply_nnzflat();
        run_selected();
        // Matches boo_matmul's y.squeeze(0) for 2D input.
        return squeeze_output_ ? y_.squeeze(0) : y_;
    }

    // Brackets exactly: the Y memset, the block kernel, the CSR-remainder
    // kernel and tail kernel on their own streams, and the join
    // (non-diagnostic); or, in diagnostic mode, just the block kernel with
    // no memset and no remainder/tail legs. No Python, no allocation, in
    // the timed region.
    float timed_run() {
        c10::cuda::CUDAGuard guard(device_);
        check_stream();
        apply_nnzflat();
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
            run_selected();
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
        return squeeze_output_ ? y_.squeeze(0) : y_;
    }
    py::dict info() const {
        py::dict d;
        d["diagnostic"] = diagnostic_;
        d["row_seg"] = row_seg_;
        d["nnzflat_fpt"] = nnzflat_fpt_;
        d["nnzflat_reference"] = nnzflat_reference_;
        return d;
    }
};

} // namespace

void bind_boo_timed_functions(py::module_& m) {
    py::class_<BOOTimedSpMM, std::shared_ptr<BOOTimedSpMM>>(m, "BOOTimedSpMM")
        .def(py::init<int, int, int, int, int, at::Tensor, at::Tensor, at::Tensor, at::Tensor,
                      at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor,
                      at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor,
                      at::Tensor, int, int, int, bool, bool, int, bool,
                      std::string, std::vector<int64_t>, py::object, py::object>(),
             py::arg("rows"), py::arg("cols"), py::arg("max_nr"), py::arg("max_nc"),
             py::arg("max_nnz_blk"), py::arg("row_grp_ptr"), py::arg("row_ids"),
             py::arg("col_grp_ptr"), py::arg("col_ids"), py::arg("blk_row_grp"),
             py::arg("blk_col_grp"), py::arg("blk_nz_ptr"), py::arg("blk_shift"),
             py::arg("nz_idx"), py::arg("nz_val"), py::arg("rem_indptr"), py::arg("rem_col_ind"),
             py::arg("rem_vals"), py::arg("tail_row"), py::arg("tail_col"), py::arg("tail_val"),
             py::arg("x"), py::arg("threads") = 256, py::arg("p") = 64, py::arg("csr_mode") = 0,
             py::arg("row_seg") = true, py::arg("diagnostic_raw_store") = false,
             py::arg("nnzflat_fpt") = 1, py::arg("nnzflat_reference") = true,
             py::arg("launch_mode") = "legacy", py::arg("buckets") = std::vector<int64_t>{},
             py::arg("plan_row_ptr") = py::none(), py::arg("plan_row_starts") = py::none())
        .def("run", &BOOTimedSpMM::run)
        .def("timed_run", &BOOTimedSpMM::timed_run)
        .def_property_readonly("output", &BOOTimedSpMM::output)
        .def("info", &BOOTimedSpMM::info);
}
