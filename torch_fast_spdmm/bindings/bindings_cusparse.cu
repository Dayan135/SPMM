// Benchmark-only persistent FP32/int32 CSR SpMM plans. No output zeroing here:
// beta=0 delegates all required output initialization to cusparseSpMM itself.
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cusparse.h>
#include <cuda_runtime.h>
#include <memory>
#include <algorithm>
#include <string>

namespace {
void cuda_check(cudaError_t s) {
    TORCH_CHECK(s == cudaSuccess, "CUDA: ", cudaGetErrorString(s));
}
void sparse_check(cusparseStatus_t s) {
    TORCH_CHECK(s == CUSPARSE_STATUS_SUCCESS, "cuSPARSE: ", cusparseGetErrorName(s),
                " (", cusparseGetErrorString(s), ")");
}

class DirectSpMM {
    at::Tensor crow_, col_, val_, x_, y_, workspace_;
    cusparseHandle_t handle_ = nullptr;
    cusparseSpMatDescr_t a_ = nullptr;
    cusparseDnMatDescr_t b_ = nullptr, c_ = nullptr;
    cudaEvent_t start_ = nullptr, end_ = nullptr;
    cudaStream_t stream_ = nullptr;
    cusparseSpMMAlg_t alg_;
    float alpha_ = 1.f, beta_ = 0.f;
    int device_;
    bool supported_ = true;
    size_t bytes_ = 0;
    std::string preprocess_ = "not_attempted", reason_;

    void cleanup() noexcept {
        if (handle_) cudaStreamSynchronize(stream_);
        if (start_) cudaEventDestroy(start_);
        if (end_) cudaEventDestroy(end_);
        if (c_) cusparseDestroyDnMat(c_);
        if (b_) cusparseDestroyDnMat(b_);
        if (a_) cusparseDestroySpMat(a_);
        if (handle_) cusparseDestroy(handle_);
    }
    cusparseStatus_t call() {
        return cusparseSpMM(handle_, CUSPARSE_OPERATION_NON_TRANSPOSE,
                            CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha_, a_, b_, &beta_, c_,
                            CUDA_R_32F, alg_, workspace_.data_ptr());
    }
    void check_stream() const {
        TORCH_CHECK(supported_, reason_);
        TORCH_CHECK(c10::cuda::getCurrentCUDAStream(device_).stream() == stream_,
                    "DirectSpMM must run on its construction stream");
    }
public:
    DirectSpMM(at::Tensor crow, at::Tensor col, at::Tensor val, at::Tensor x,
               const std::string& algorithm)
        : crow_(crow), col_(col), val_(val), x_(x), device_(x.get_device()) {
        TORCH_CHECK(x.is_cuda() && x.dim() == 2 && x.is_contiguous() &&
                    x.scalar_type() == at::kFloat, "X must be contiguous CUDA FP32, two-dimensional");
        for (const auto& t : {crow, col, val})
            TORCH_CHECK(t.device() == x.device() && t.dim() == 1 && t.is_contiguous(),
                        "CSR arrays must be contiguous 1D tensors on X's device");
        TORCH_CHECK(crow.scalar_type() == at::kInt && col.scalar_type() == at::kInt &&
                    val.scalar_type() == at::kFloat, "CSR must use int32 indices and FP32 values");
        TORCH_CHECK(crow.numel() >= 1 && col.numel() == val.numel(), "Invalid CSR dimensions");
        if (algorithm == "DEFAULT") alg_ = CUSPARSE_SPMM_ALG_DEFAULT;
        else if (algorithm == "CSR_ALG1") alg_ = CUSPARSE_SPMM_CSR_ALG1;
        else if (algorithm == "CSR_ALG2") alg_ = CUSPARSE_SPMM_CSR_ALG2;
        else if (algorithm == "CSR_ALG3") alg_ = CUSPARSE_SPMM_CSR_ALG3;
        else TORCH_CHECK(false, "Unknown cuSPARSE algorithm: ", algorithm);
        c10::cuda::CUDAGuard guard(x.device());
        stream_ = c10::cuda::getCurrentCUDAStream(device_).stream();
        try {
            // Separate sparse descriptor AND workspace for every algorithm:
            // a descriptor can have only one active preprocess buffer.
            sparse_check(cusparseCreate(&handle_));
            sparse_check(cusparseSetStream(handle_, stream_));
            sparse_check(cusparseSetPointerMode(handle_, CUSPARSE_POINTER_MODE_HOST));
            const auto rows = crow.numel() - 1;
            y_ = at::empty({rows, x.size(1)}, x.options());
            sparse_check(cusparseCreateCsr(&a_, rows, x.size(0), val.numel(),
                         crow.data_ptr<int>(), col.data_ptr<int>(), val.data_ptr<float>(),
                         CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
            sparse_check(cusparseCreateDnMat(&b_, x.size(0), x.size(1), x.size(1),
                                           x.data_ptr<float>(), CUDA_R_32F, CUSPARSE_ORDER_ROW));
            sparse_check(cusparseCreateDnMat(&c_, rows, x.size(1), x.size(1),
                                           y_.data_ptr<float>(), CUDA_R_32F, CUSPARSE_ORDER_ROW));
            auto status = cusparseSpMM_bufferSize(handle_, CUSPARSE_OPERATION_NON_TRANSPOSE,
                CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha_, a_, b_, &beta_, c_, CUDA_R_32F, alg_, &bytes_);
            if (status == CUSPARSE_STATUS_NOT_SUPPORTED) {
                supported_ = false;
                reason_ = "CUSPARSE_STATUS_NOT_SUPPORTED at bufferSize";
                return;
            }
            sparse_check(status);
            workspace_ = at::empty({static_cast<int64_t>(std::max(bytes_, size_t(1)))},
                                   x.options().dtype(at::kByte));
            status = cusparseSpMM_preprocess(handle_, CUSPARSE_OPERATION_NON_TRANSPOSE,
                CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha_, a_, b_, &beta_, c_, CUDA_R_32F, alg_,
                workspace_.data_ptr());
            preprocess_ = cusparseGetErrorName(status);
            if (status != CUSPARSE_STATUS_NOT_SUPPORTED) sparse_check(status);
            cuda_check(cudaStreamSynchronize(stream_));
            cuda_check(cudaEventCreate(&start_));
            cuda_check(cudaEventCreate(&end_));
            // Initialize event resources outside measurement.
            cuda_check(cudaEventRecord(start_, stream_));
            cuda_check(cudaEventRecord(end_, stream_));
            cuda_check(cudaEventSynchronize(end_));
        } catch (...) {
            cleanup();
            throw;
        }
    }
    ~DirectSpMM() {
        c10::cuda::CUDAGuard guard(x_.device());
        cleanup();
    }
    at::Tensor run() {
        c10::cuda::CUDAGuard guard(x_.device());
        check_stream();
        sparse_check(call());
        return y_;  // Aliases persistent output; subsequent runs overwrite it.
    }
    float timed_run() {
        c10::cuda::CUDAGuard guard(x_.device());
        check_stream();
        cuda_check(cudaEventRecord(start_, stream_));
        auto status = call();  // The only operation between the CUDA events.
        cuda_check(cudaEventRecord(end_, stream_));
        sparse_check(status);
        cuda_check(cudaEventSynchronize(end_));
        float ms;
        cuda_check(cudaEventElapsedTime(&ms, start_, end_));
        return ms;
    }
    at::Tensor output() const { return y_; }
    py::dict info() const {
        py::dict d;
        int version = 0;
        sparse_check(cusparseGetVersion(handle_, &version));
        d["cusparse_version"] = version;
        d["cusparse_header_version"] = CUSPARSE_VERSION;
        d["workspace_bytes"] = bytes_;
        d["preprocess_status"] = preprocess_;
        d["supported"] = supported_;
        d["unsupported_reason"] = reason_;
        return d;
    }
};
} // namespace

void bind_cusparse_functions(py::module_& m) {
    py::class_<DirectSpMM, std::shared_ptr<DirectSpMM>>(m, "DirectSpMM")
        .def(py::init<at::Tensor, at::Tensor, at::Tensor, at::Tensor, const std::string&>())
        .def("run", &DirectSpMM::run)
        .def("timed_run", &DirectSpMM::timed_run)
        .def_property_readonly("output", &DirectSpMM::output)
        .def("info", &DirectSpMM::info);
}
