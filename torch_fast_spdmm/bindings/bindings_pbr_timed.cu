// C++-only timing harness for PBR, mirroring bindings_cusparse.cu's
// DirectSpMM exactly: allocation, stream setup, and event creation happen
// once at construction (outside measurement); timed_run() brackets only the
// work needed to produce Y with CUDA events on the construction stream. This
// makes PBR's timing symmetric with DirectSpMM's -- neither includes Python/
// pybind11 call overhead -- unlike bench_spmm.py's default Python-side event
// wrapping around the whole pbr_matmul() call, which does.
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>

#include "../kernels/pbr_kernels.hpp"

namespace {

void cuda_check(cudaError_t s) {
    TORCH_CHECK(s == cudaSuccess, "CUDA: ", cudaGetErrorString(s));
}

class PBRTimedSpMM {
    at::Tensor block_codes_, block_coords_, block_offsets_, block_data_,
               rem_indptr_, rem_col_ind_, rem_vals_, x3_, y_;
    int cols_, rows_, block_rows_, block_cols_, csr_mode_, smem_size_, device_;
    bool diagnostic_;
    bool squeeze_output_;
    cudaStream_t cur_stream_ = nullptr, block_stream_ = nullptr, csr_stream_ = nullptr;
    cudaEvent_t start_ = nullptr, end_ = nullptr;

    void cleanup() noexcept {
        if (cur_stream_) cudaStreamSynchronize(cur_stream_);
        if (start_) cudaEventDestroy(start_);
        if (end_) cudaEventDestroy(end_);
    }
    void check_stream() const {
        TORCH_CHECK(c10::cuda::getCurrentCUDAStream(device_).stream() == cur_stream_,
                    "PBRTimedSpMM must run on its construction stream");
    }

public:
    PBRTimedSpMM(int cols, int rows, int block_rows, int block_cols,
                 at::Tensor block_codes, at::Tensor block_coords, at::Tensor block_offsets,
                 at::Tensor block_data, at::Tensor rem_indptr, at::Tensor rem_col_ind,
                 at::Tensor rem_vals, at::Tensor x, int csr_mode, int smem_size,
                 bool diagnostic_raw_store)
        : block_codes_(block_codes), block_coords_(block_coords), block_offsets_(block_offsets),
          block_data_(block_data), rem_indptr_(rem_indptr), rem_col_ind_(rem_col_ind),
          rem_vals_(rem_vals), x3_((x.dim() == 2) ? x.unsqueeze(0) : x),
          cols_(cols), rows_(rows), block_rows_(block_rows), block_cols_(block_cols),
          csr_mode_(csr_mode), smem_size_(smem_size), device_(x.get_device()),
          diagnostic_(diagnostic_raw_store), squeeze_output_(x.dim() == 2) {
        TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kFloat,
                    "PBRTimedSpMM: X must be a CUDA FP32 tensor");
        TORCH_CHECK(x.dim() == 2 || x.dim() == 3, "PBRTimedSpMM: X must be 2D or 3D");
        TORCH_CHECK(x3_.is_contiguous(), "PBRTimedSpMM: X must be contiguous");
        for (const auto& t : {block_codes, block_coords, block_offsets, block_data,
                              rem_indptr, rem_col_ind, rem_vals}) {
            TORCH_CHECK(t.device() == x.device() && t.is_contiguous(),
                        "PBRTimedSpMM: PBR metadata must be contiguous tensors on X's device");
        }
        TORCH_CHECK(block_coords.scalar_type() == at::kInt && block_offsets.scalar_type() == at::kInt,
                    "PBRTimedSpMM: block_coords/block_offsets must be int32 (this harness is int32/fp32 only)");
        TORCH_CHECK(block_data.scalar_type() == at::kFloat && rem_vals.scalar_type() == at::kFloat,
                    "PBRTimedSpMM: block_data/rem_vals must be fp32");

        c10::cuda::CUDAGuard guard(x.device());
        cur_stream_   = c10::cuda::getCurrentCUDAStream(device_).stream();
        block_stream_ = at::cuda::getStreamFromPool().stream();
        csr_stream_   = at::cuda::getStreamFromPool().stream();
        const int batch_size = (int)x3_.size(0);
        const int features   = (int)x3_.size(-1);
        y_ = at::empty({batch_size, rows, features}, x.options());
        try {
            cuda_check(cudaEventCreate(&start_));
            cuda_check(cudaEventCreate(&end_));
            // Initialize event resources outside measurement (matches
            // DirectSpMM's ctor -- warms up event overhead once, up front).
            cuda_check(cudaEventRecord(start_, cur_stream_));
            cuda_check(cudaEventRecord(end_, cur_stream_));
            cuda_check(cudaEventSynchronize(end_));
        } catch (...) {
            cleanup();
            throw;
        }
    }
    ~PBRTimedSpMM() {
        c10::cuda::CUDAGuard guard(device_);
        cleanup();
    }

    at::Tensor run() {
        TORCH_CHECK(!diagnostic_,
                    "PBRTimedSpMM(diagnostic_raw_store=True) is not correctness-checkable "
                    "-- its output is incorrect by construction");
        c10::cuda::CUDAGuard guard(device_);
        check_stream();
        pbr_run_into<int32_t, float>(
            y_, cols_, rows_, block_rows_, block_cols_,
            block_codes_, block_coords_, block_offsets_, block_data_,
            rem_indptr_, rem_col_ind_, rem_vals_, x3_, csr_mode_, smem_size_,
            cur_stream_, block_stream_, csr_stream_);
        // Matches pbr_matmul's y.squeeze(0) for 2D input.
        return squeeze_output_ ? y_.squeeze(0) : y_;
    }

    // Brackets exactly: the Y memset, the block kernel, the CSR-remainder
    // kernel on the second stream, and the join (non-diagnostic); or, in
    // diagnostic mode, just the block kernel with no memset and no
    // remainder leg. No Python, no allocation, in the timed region.
    float timed_run() {
        c10::cuda::CUDAGuard guard(device_);
        check_stream();
        cuda_check(cudaEventRecord(start_, cur_stream_));
        if (diagnostic_) {
            const int num_pbr_blocks = (int)block_codes_.size(0);
            launch_pbr_diag_block_spmm(
                num_pbr_blocks, (int)x3_.size(-1), (int)x3_.size(0), cols_, rows_,
                block_rows_, block_cols_,
                block_codes_.data_ptr<int64_t>(), block_coords_.data_ptr<int32_t>(),
                block_offsets_.data_ptr<int32_t>(), block_data_.data_ptr<float>(),
                x3_.data_ptr<float>(), y_.data_ptr<float>(), cur_stream_, smem_size_);
        } else {
            pbr_run_into<int32_t, float>(
                y_, cols_, rows_, block_rows_, block_cols_,
                block_codes_, block_coords_, block_offsets_, block_data_,
                rem_indptr_, rem_col_ind_, rem_vals_, x3_, csr_mode_, smem_size_,
                cur_stream_, block_stream_, csr_stream_);
        }
        cuda_check(cudaEventRecord(end_, cur_stream_));
        cuda_check(cudaEventSynchronize(end_));
        float ms;
        cuda_check(cudaEventElapsedTime(&ms, start_, end_));
        return ms;
    }

    at::Tensor output() const {
        TORCH_CHECK(!diagnostic_,
                    "PBRTimedSpMM(diagnostic_raw_store=True)'s output is not meaningful");
        return squeeze_output_ ? y_.squeeze(0) : y_;
    }
    py::dict info() const {
        py::dict d;
        d["diagnostic"] = diagnostic_;
        return d;
    }
};

} // namespace

void bind_pbr_timed_functions(py::module_& m) {
    py::class_<PBRTimedSpMM, std::shared_ptr<PBRTimedSpMM>>(m, "PBRTimedSpMM")
        .def(py::init<int, int, int, int, at::Tensor, at::Tensor, at::Tensor, at::Tensor,
                      at::Tensor, at::Tensor, at::Tensor, at::Tensor, int, int, bool>(),
             py::arg("cols"), py::arg("rows"), py::arg("block_rows"), py::arg("block_cols"),
             py::arg("block_codes"), py::arg("block_coords"), py::arg("block_offsets"),
             py::arg("block_data"), py::arg("rem_indptr"), py::arg("rem_col_ind"),
             py::arg("rem_vals"), py::arg("x"), py::arg("csr_mode") = 0,
             py::arg("smem_size") = 2048, py::arg("diagnostic_raw_store") = false)
        .def("run", &PBRTimedSpMM::run)
        .def("timed_run", &PBRTimedSpMM::timed_run)
        .def_property_readonly("output", &PBRTimedSpMM::output)
        .def("info", &PBRTimedSpMM::info);
}
