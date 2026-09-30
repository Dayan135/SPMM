#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

#include "../kernels/boo_kernels.hpp"
#include "../kernels/csr_kernels.hpp"
#include "../kernels/pbr_kernels.hpp"

// An execution plan uses contiguous ranges in physically reordered block arrays.
// Empty buckets selects the original single grid; optional row tables are local
// nnz offsets, indexed through one pointer per block.
struct BOOLaunchPlan {
    std::vector<int64_t> buckets;
    at::Tensor row_ptr, row_starts;
};

// Zeroes Y (via memset, on cur_stream) then runs the three-way fork/join:
// block_stream runs the main BOO-block kernel; csr_stream the shared CSR
// remainder kernel (BOO_REMAINDER policy); tail_stream the in-format tail
// kernel (BOO_TAIL_BATCH policy). Exactly one of csr_nnz/tail_nnz is normally
// nonzero (the two small-block policies are mutually exclusive at build
// time), but nothing here assumes that -- an empty leg is simply skipped.
// All three commit into the same zeroed Y via atomicAdd. Y must already be
// allocated with the right shape/dtype/device; this function is the sole
// writer of the zero. Split out of boo_spmm_cuda_impl's plain (non-fast-path,
// non-plan) body so a C++-only timing harness (BOOTimedSpMM) can bracket
// exactly this work with CUDA events, with allocation and stream setup done
// once outside the timed region -- see bindings_boo_timed.cu. Bucketed plans
// and the PURE_FAST_PATH single-stream shortcut stay in boo_spmm_cuda_impl,
// out of scope for the timing harness.
template <typename scalar_t>
void boo_run_into(
    at::Tensor Y,
    int rows, int cols, int max_nr, int max_nc, int max_nnz_blk,
    const at::Tensor row_grp_ptr, const at::Tensor row_ids,
    const at::Tensor col_grp_ptr, const at::Tensor col_ids,
    const at::Tensor blk_row_grp, const at::Tensor blk_col_grp,
    const at::Tensor blk_nz_ptr, const at::Tensor blk_shift,
    const at::Tensor nz_idx, const at::Tensor nz_val,
    const at::Tensor rem_indptr, const at::Tensor rem_col_ind, const at::Tensor rem_vals,
    const at::Tensor tail_row, const at::Tensor tail_col, const at::Tensor tail_val,
    const at::Tensor X3, int threads, int p, int csr_mode, bool row_seg,
    cudaStream_t cur_stream, cudaStream_t block_stream,
    cudaStream_t csr_stream, cudaStream_t tail_stream
) {
    const int num_blocks = (int)blk_row_grp.size(0);
    const int csr_nnz    = (int)rem_col_ind.size(0);
    const int tail_nnz   = (int)tail_col.size(0);
    const int batch_size = (int)X3.size(0);
    const int features   = (int)X3.size(-1);

    cudaMemsetAsync(Y.data_ptr<scalar_t>(), 0, Y.numel() * sizeof(scalar_t), cur_stream);

    cudaEvent_t y_ready;
    cudaEventCreateWithFlags(&y_ready, cudaEventDisableTiming);
    cudaEventRecord(y_ready, cur_stream);
    cudaStreamWaitEvent(block_stream, y_ready, 0);
    cudaStreamWaitEvent(csr_stream,   y_ready, 0);
    cudaStreamWaitEvent(tail_stream,  y_ready, 0);
    cudaEventDestroy(y_ready);

    if (num_blocks > 0) {
        launch_boo_spmm<scalar_t>(
            num_blocks, features, batch_size, rows, cols, max_nr, max_nc, max_nnz_blk,
            row_grp_ptr.data_ptr<int32_t>(), row_ids.data_ptr<int32_t>(),
            col_grp_ptr.data_ptr<int32_t>(), col_ids.data_ptr<int32_t>(),
            blk_row_grp.data_ptr<int32_t>(), blk_col_grp.data_ptr<int32_t>(),
            blk_nz_ptr.data_ptr<int32_t>(), blk_shift.data_ptr<uint8_t>(),
            reinterpret_cast<const uint16_t*>(nz_idx.data_ptr<int16_t>()),
            nz_val.data_ptr<scalar_t>(), X3.data_ptr<scalar_t>(), Y.data_ptr<scalar_t>(),
            block_stream, threads, p, row_seg, nullptr, nullptr);
    }

    if (csr_nnz > 0) {
        launch_csr_spmm<int32_t, scalar_t>(
            rows, features, batch_size, cols, csr_nnz,
            rem_indptr.data_ptr<int32_t>(),
            rem_col_ind.data_ptr<int32_t>(),
            rem_vals.data_ptr<scalar_t>(),
            X3.data_ptr<scalar_t>(),
            Y.data_ptr<scalar_t>(),
            csr_stream,
            csr_mode
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    if (tail_nnz > 0) {
        launch_boo_tail_spmm<scalar_t>(
            tail_nnz, features, batch_size, rows, cols,
            tail_row.data_ptr<int32_t>(),
            tail_col.data_ptr<int32_t>(),
            tail_val.data_ptr<scalar_t>(),
            X3.data_ptr<scalar_t>(),
            Y.data_ptr<scalar_t>(),
            tail_stream,
            threads
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    cudaEvent_t block_done, csr_done, tail_done;
    cudaEventCreateWithFlags(&block_done, cudaEventDisableTiming);
    cudaEventCreateWithFlags(&csr_done,   cudaEventDisableTiming);
    cudaEventCreateWithFlags(&tail_done,  cudaEventDisableTiming);
    cudaEventRecord(block_done, block_stream);
    cudaEventRecord(csr_done,   csr_stream);
    cudaEventRecord(tail_done,  tail_stream);
    cudaStreamWaitEvent(cur_stream, block_done, 0);
    cudaStreamWaitEvent(cur_stream, csr_done,   0);
    cudaStreamWaitEvent(cur_stream, tail_done,  0);
    cudaEventDestroy(block_done);
    cudaEventDestroy(csr_done);
    cudaEventDestroy(tail_done);
}

// BOO GPU entry point.
//
// Index arrays are int32 unconditionally -- BOO does not template on index_t
// the way PBR does, which halves both the index traffic and the pybind symbol
// count (2 symbols instead of 4).
template <typename scalar_t, bool PURE_FAST_PATH = false>
at::Tensor boo_spmm_cuda_impl(
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
    const at::Tensor tail_row,
    const at::Tensor tail_col,
    const at::Tensor tail_val,
    const at::Tensor X,
    int threads = 128,
    int p = 64,
    int csr_mode = 0,
    bool row_seg = true,
    const BOOLaunchPlan* plan = nullptr
) {
    TORCH_CHECK(X.is_cuda(), "X must be a CUDA tensor");
    TORCH_CHECK(X.dim() == 2 || X.dim() == 3, "X must be 2D or 3D");
    TORCH_CHECK(X.size(-2) == cols, "X rows must match matrix columns");
    TORCH_CHECK(threads == 128 || threads == 256 || threads == 512,
                "threads must be 128, 256 or 512");
    TORCH_CHECK(p == 8 || p == 16 || p == 32 || p == 64 || p == 128,
                "p must be 8, 16, 32, 64 or 128");
    TORCH_CHECK(csr_mode >= 0 && csr_mode <= 2, "invalid csr_mode");
    for (const auto& tensor : {row_grp_ptr, row_ids, col_grp_ptr, col_ids,
                              blk_row_grp, blk_col_grp, blk_nz_ptr, blk_shift,
                              nz_idx, nz_val, rem_indptr, rem_col_ind, rem_vals,
                              tail_row, tail_col, tail_val}) {
        TORCH_CHECK(tensor.device() == X.device(), "BOO metadata and X must share a device");
        TORCH_CHECK(tensor.is_contiguous(), "BOO metadata must be contiguous");
    }
    const c10::cuda::CUDAGuard device_guard(X.device());
    const int num_blocks = (int)blk_row_grp.size(0);
    const int csr_nnz    = (int)rem_col_ind.size(0);
    const int tail_nnz   = (int)tail_col.size(0);
    const int batch_size = (X.dim() == 3) ? (int)X.size(0) : 1;
    const int features   = (int)X.size(-1);

    at::Tensor X3 = ((X.dim() == 2) ? X.unsqueeze(0) : X).contiguous();
    // Allocated here, zeroed by whichever branch below actually runs (each
    // needs Y zeroed on cur_stream before any worker stream's atomicAdd) --
    // keeps the zero a single memset instead of double-zeroing when the
    // plain (non-plan, non-fast-path) branch delegates to boo_run_into,
    // which does its own zero for the C++-only timing harness's benefit.
    at::Tensor Y  = at::empty({batch_size, rows, features}, X.options());

    auto run_blocks = [&](cudaStream_t stream) {
        auto launch_range = [&](int start, int count, int nr, int nc, int nnz) {
            launch_boo_spmm<scalar_t>(
                count, features, batch_size, rows, cols, nr, nc, nnz,
                row_grp_ptr.data_ptr<int32_t>(), row_ids.data_ptr<int32_t>(),
                col_grp_ptr.data_ptr<int32_t>(), col_ids.data_ptr<int32_t>(),
                blk_row_grp.data_ptr<int32_t>() + start,
                blk_col_grp.data_ptr<int32_t>() + start,
                blk_nz_ptr.data_ptr<int32_t>() + start,
                blk_shift.data_ptr<uint8_t>() + start,
                reinterpret_cast<const uint16_t*>(nz_idx.data_ptr<int16_t>()),
                nz_val.data_ptr<scalar_t>(), X3.data_ptr<scalar_t>(), Y.data_ptr<scalar_t>(),
                stream, threads, p, row_seg,
                plan && plan->row_ptr.numel() ? plan->row_ptr.data_ptr<int32_t>() + start : nullptr,
                plan && plan->row_starts.numel() ? plan->row_starts.data_ptr<int32_t>() : nullptr);
        };
        if (num_blocks == 0) return;
        if (!plan || plan->buckets.empty()) {
            launch_range(0, num_blocks, max_nr, max_nc, max_nnz_blk);
        } else {
            for (size_t i = 0; i < plan->buckets.size(); i += 5) {
                const auto* b = &plan->buckets[i];
                launch_range(b[0], b[1], b[2], b[3], b[4]);
            }
        }
    };
    if (plan) {
        TORCH_CHECK(row_seg, "BOO execution plans require row_seg=True");
        TORCH_CHECK(plan->buckets.size() % 5 == 0, "invalid BOO bucket specifications");
        int64_t covered = 0;
        for (size_t i = 0; i < plan->buckets.size(); i += 5) {
            const auto* b = &plan->buckets[i];
            TORCH_CHECK(b[0] == covered && b[1] > 0 && b[1] <= num_blocks - covered,
                        "BOO buckets must partition the blocks");
            TORCH_CHECK(b[2] > 0 && b[2] <= max_nr && b[3] > 0 && b[3] <= max_nc &&
                        b[4] > 0 && b[4] <= max_nnz_blk, "invalid BOO bucket maxima");
            covered += b[1];
        }
        TORCH_CHECK(plan->buckets.empty() || covered == num_blocks, "incomplete BOO buckets");
        for (const auto& table : {plan->row_ptr, plan->row_starts}) {
            TORCH_CHECK(table.device() == X.device() && table.is_contiguous() &&
                        table.scalar_type() == at::kInt && table.dim() == 1,
                        "BOO row tables must be contiguous int32 vectors on X's device");
        }
        TORCH_CHECK((plan->row_ptr.numel() == 0 && plan->row_starts.numel() == 0) ||
                    (plan->row_ptr.numel() == num_blocks + 1 &&
                     plan->row_starts.numel() >= num_blocks), "invalid BOO row table sizes");
    }
    auto cur_stream = at::cuda::getCurrentCUDAStream();
    if constexpr (PURE_FAST_PATH) {
        if (csr_nnz == 0 && tail_nnz == 0) {
            cudaMemsetAsync(Y.data_ptr<scalar_t>(), 0, Y.numel() * sizeof(scalar_t), cur_stream.stream());
            run_blocks(cur_stream.stream());
            return Y;
        }
    }

    auto block_stream = at::cuda::getStreamFromPool();
    auto csr_stream   = at::cuda::getStreamFromPool();
    auto tail_stream  = at::cuda::getStreamFromPool();

    if (!plan) {
        // Plain (non-bucketed) path: delegate to the shared helper also used
        // by BOOTimedSpMM's C++-only timing harness (bindings_boo_timed.cu).
        boo_run_into<scalar_t>(
            Y, rows, cols, max_nr, max_nc, max_nnz_blk,
            row_grp_ptr, row_ids, col_grp_ptr, col_ids,
            blk_row_grp, blk_col_grp, blk_nz_ptr, blk_shift, nz_idx, nz_val,
            rem_indptr, rem_col_ind, rem_vals, tail_row, tail_col, tail_val,
            X3, threads, p, csr_mode, row_seg,
            cur_stream.stream(), block_stream.stream(), csr_stream.stream(), tail_stream.stream());
        return Y;
    }

    // Bucketed/plan path: boo_run_into doesn't support buckets, so this keeps
    // the original inline zero + run_blocks(with bucket iteration) + join.
    cudaMemsetAsync(Y.data_ptr<scalar_t>(), 0, Y.numel() * sizeof(scalar_t), cur_stream.stream());

    cudaEvent_t y_ready;
    cudaEventCreateWithFlags(&y_ready, cudaEventDisableTiming);
    cudaEventRecord(y_ready, cur_stream.stream());
    cudaStreamWaitEvent(block_stream.stream(), y_ready, 0);
    cudaStreamWaitEvent(csr_stream.stream(),   y_ready, 0);
    cudaStreamWaitEvent(tail_stream.stream(),  y_ready, 0);
    cudaEventDestroy(y_ready);

    run_blocks(block_stream.stream());

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
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    if (tail_nnz > 0) {
        launch_boo_tail_spmm<scalar_t>(
            tail_nnz, features, batch_size, rows, cols,
            tail_row.data_ptr<int32_t>(),
            tail_col.data_ptr<int32_t>(),
            tail_val.data_ptr<scalar_t>(),
            X3.data_ptr<scalar_t>(),
            Y.data_ptr<scalar_t>(),
            tail_stream.stream(),
            threads
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    cudaEvent_t block_done, csr_done, tail_done;
    cudaEventCreateWithFlags(&block_done, cudaEventDisableTiming);
    cudaEventCreateWithFlags(&csr_done,   cudaEventDisableTiming);
    cudaEventCreateWithFlags(&tail_done,  cudaEventDisableTiming);
    cudaEventRecord(block_done, block_stream.stream());
    cudaEventRecord(csr_done,   csr_stream.stream());
    cudaEventRecord(tail_done,  tail_stream.stream());
    cudaStreamWaitEvent(cur_stream.stream(), block_done, 0);
    cudaStreamWaitEvent(cur_stream.stream(), csr_done,   0);
    cudaStreamWaitEvent(cur_stream.stream(), tail_done,  0);
    cudaEventDestroy(block_done);
    cudaEventDestroy(csr_done);
    cudaEventDestroy(tail_done);

    return Y;
}

template <typename scalar_t, bool PURE_FAST_PATH = false>
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
    const at::Tensor tail_row,
    const at::Tensor tail_col,
    const at::Tensor tail_val,
    const at::Tensor X,
    int threads = 128,
    int p = 64,
    int csr_mode = 0,
    bool row_seg = true
) {
    return boo_spmm_cuda_impl<scalar_t, PURE_FAST_PATH>(
        rows, cols, max_nr, max_nc, max_nnz_blk,
        row_grp_ptr, row_ids, col_grp_ptr, col_ids,
        blk_row_grp, blk_col_grp, blk_nz_ptr, blk_shift, nz_idx, nz_val,
        rem_indptr, rem_col_ind, rem_vals, tail_row, tail_col, tail_val,
        X, threads, p, csr_mode, row_seg);
}

template <typename scalar_t, bool PURE_FAST_PATH = false>
at::Tensor boo_spmm_cuda_planned(
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
    const at::Tensor tail_row,
    const at::Tensor tail_col,
    const at::Tensor tail_val,
    const at::Tensor X,
    int threads,
    int p,
    int csr_mode,
    bool row_seg,
    const std::vector<int64_t>& buckets,
    const at::Tensor row_ptr, const at::Tensor row_starts
) {
    const BOOLaunchPlan plan{buckets, row_ptr, row_starts};
    return boo_spmm_cuda_impl<scalar_t, true>(
        rows, cols, max_nr, max_nc, max_nnz_blk,
        row_grp_ptr, row_ids, col_grp_ptr, col_ids,
        blk_row_grp, blk_col_grp, blk_nz_ptr, blk_shift, nz_idx, nz_val,
        rem_indptr, rem_col_ind, rem_vals, tail_row, tail_col, tail_val,
        X, threads, p, csr_mode, row_seg, &plan);
}

// Explicit instantiation so bindings_boo_timed.cu (a separate translation
// unit) can call this template function -- only fp32 needed there. Must
// match boo_run_into's declared parameter list exactly: Y; rows, cols,
// max_nr, max_nc, max_nnz_blk; row_grp_ptr, row_ids, col_grp_ptr, col_ids;
// blk_row_grp, blk_col_grp, blk_nz_ptr, blk_shift; nz_idx, nz_val;
// rem_indptr, rem_col_ind, rem_vals; tail_row, tail_col, tail_val; X3;
// threads, p, csr_mode; row_seg; cur/block/csr/tail_stream (17 Tensors
// total after Y, not counting X3 separately from the rest -- X3 is the
// 17th).
template void boo_run_into<float>(
    /* Y */            at::Tensor,
    /* rows..nnz_blk */ int, int, int, int, int,
    /* row/col ptrs */  const at::Tensor, const at::Tensor, const at::Tensor, const at::Tensor,
    /* blk_* */         const at::Tensor, const at::Tensor, const at::Tensor, const at::Tensor,
    /* nz_idx/val */    const at::Tensor, const at::Tensor,
    /* rem_* */         const at::Tensor, const at::Tensor, const at::Tensor,
    /* tail_* */        const at::Tensor, const at::Tensor, const at::Tensor,
    /* X3 */            const at::Tensor,
    /* threads,p,csr_mode */ int, int, int,
    /* row_seg */       bool,
    /* streams */       cudaStream_t, cudaStream_t, cudaStream_t, cudaStream_t);

void bind_boo_functions(pybind11::module_& m) {
    m.def("boo_spmm_cuda_f32", &boo_spmm_cuda_dispatch<float>);
    m.def("boo_spmm_cuda_f64", &boo_spmm_cuda_dispatch<double>);
    m.def("boo_spmm_cuda_auto_f32", &boo_spmm_cuda_dispatch<float, true>);
    m.def("boo_spmm_cuda_auto_f64", &boo_spmm_cuda_dispatch<double, true>);
    m.def("boo_spmm_cuda_plan_f32", &boo_spmm_cuda_planned<float>);
    m.def("boo_spmm_cuda_plan_f64", &boo_spmm_cuda_planned<double>);
    m.def("boo_arena_bytes_cuda", &boo_arena_bytes);
}
