#include <cuda_runtime.h>
#include <cstdint>
#include <stdexcept>
#include <c10/cuda/CUDAException.h>

#include "kernels/boo_kernels.hpp"

// ===========================================================================
// BOO SpMM -- load / work / flush over a clustered, variable-size block grid
// ===========================================================================
//
// One CUDA block per BOO block.  A BOO block is the intersection
// (row group) x (col group), so its dimensions are runtime data, not template
// parameters -- hence a dynamic `extern __shared__` arena rather than PBR's
// static smem[TOTAL_SHARED].
//
// Occupancy inside a block is an explicit uint16 per nonzero:
//     r = idx >> shift;  c = idx & mask
// which replaces PBR's bitmap scan, its 3-bit/7-bit packed registers, and
// nnzflat's per-thread COO rebuild all at once.  See BOO_FORMAT.md sec. 4.2.
//
// Feature slabs: blockIdx.z slices `features` into FEAT_BLOCK-wide slabs, and
// inside a slab the `base_f` loop steps by the compile-time constant P.
//
// NOTE the deliberate absence of PBR's `current_p`.  All four PBR kernels
// decode with stride `p` but index shared memory with stride
// `current_p = min(p, feat_end - base_f)`, which is wrong whenever they differ
// (documented-not-fixed at pbr_kernel.cu:82-84).  Here shared memory is ALWAYS
// indexed with the template constant P and only the *global* accesses are
// guarded by `global_f < features`, so the bug class cannot occur.

constexpr int BOO_FEAT_BLOCK = 512;

int boo_arena_bytes(int max_nr, int max_nc, int max_nnz_blk, int p, int scalar_bytes) {
    const int data = max_nnz_blk * (scalar_bytes + 2);       // sh_nz_val + sh_nz_idx
    const int ids = 4 * (max_nr + max_nc);                   // sh_row_ids + sh_col_ids
    const int tiles = scalar_bytes * p * (max_nr + max_nc);  // smem_X + smem_Y
    return data + ids + tiles + 16;                          // + alignment slack
}

// The row-segmented kernel needs no smem_Y at all -- each thread accumulates a
// whole row in a register -- but does need a per-block row-start table.
int boo_arena_bytes_rowseg(int max_nr, int max_nc, int max_nnz_blk, int p, int scalar_bytes) {
    const int data = max_nnz_blk * (scalar_bytes + 2);
    const int ids = 4 * (max_nr + max_nc) + 4 * (max_nr + 1);  // + sh_row_start
    const int tiles = scalar_bytes * p * max_nc;               // smem_X only
    return data + ids + tiles + 16;
}


template <typename scalar_t, int THREADS, int P_LOG2>
__global__ void __launch_bounds__(THREADS, 1536 / THREADS) boo_spmm_nnzflat_kernel(
    const uint32_t num_blocks,
    const uint32_t features,
    const uint32_t batch_size,
    const uint32_t rows,
    const uint32_t cols,
    const uint32_t max_nr,
    const uint32_t max_nc,
    const uint32_t max_nnz_blk,
    const int32_t* __restrict__ row_grp_ptr,
    const int32_t* __restrict__ row_ids,
    const int32_t* __restrict__ col_grp_ptr,
    const int32_t* __restrict__ col_ids,
    const int32_t* __restrict__ blk_row_grp,
    const int32_t* __restrict__ blk_col_grp,
    const int32_t* __restrict__ blk_nz_ptr,
    const uint8_t* __restrict__ blk_shift,
    const uint16_t* __restrict__ nz_idx,
    const scalar_t* __restrict__ nz_val,
    const scalar_t* __restrict__ X,
    scalar_t* __restrict__ Y,
    const int32_t* __restrict__ block_row_ptr,
    const int32_t* __restrict__ row_starts
) {
    constexpr uint32_t P = 1u << P_LOG2;
    constexpr uint32_t P_MASK = P - 1u;

    const uint32_t b_idx = blockIdx.x;
    const uint32_t batch_idx = blockIdx.y;
    const uint32_t feat_slab = blockIdx.z;
    const uint32_t tx = threadIdx.x;

    if (b_idx >= num_blocks || batch_idx >= batch_size) return;

    // Widest alignment first (scalar_t, then int32, then uint16) so every
    // region is naturally aligned without explicit padding.
    extern __shared__ __align__(16) char boo_smem_raw[];
    scalar_t* sh_nz_val = reinterpret_cast<scalar_t*>(boo_smem_raw);
    scalar_t* smem_X = sh_nz_val + max_nnz_blk;
    scalar_t* smem_Y = smem_X + max_nc * P;
    int32_t* sh_col_ids = reinterpret_cast<int32_t*>(smem_Y + max_nr * P);
    int32_t* sh_row_ids = sh_col_ids + max_nc;
    uint16_t* sh_nz_idx = reinterpret_cast<uint16_t*>(sh_row_ids + max_nr);

    const uint32_t rg = blk_row_grp[b_idx];
    const uint32_t cg = blk_col_grp[b_idx];
    const uint32_t row_base = row_grp_ptr[rg];
    const uint32_t n_r = row_grp_ptr[rg + 1] - row_base;
    const uint32_t col_base = col_grp_ptr[cg];
    const uint32_t n_c = col_grp_ptr[cg + 1] - col_base;
    const uint32_t nz_base = blk_nz_ptr[b_idx];
    const uint32_t n_nz = blk_nz_ptr[b_idx + 1] - nz_base;

    if (n_r == 0 || n_c == 0 || n_nz == 0) return;

    const uint32_t shift = blk_shift[b_idx];
    const uint32_t col_mask = (1u << shift) - 1u;

    // --- LOAD: hoisted ABOVE the feature-tile loop, so it runs once per CTA --
    // PBR re-reads block_data from global on every feature-tile pass, making
    // A-traffic 4*nnz*(FEAT_BLOCK/p) -- which is why a small p used to be
    // unaffordable. Hoisting makes it 6*nnz, independent of p, and that is what
    // funds p=8. (BOO_FORMAT.md sec. 6.1.)
    for (uint32_t i = tx; i < n_nz; i += THREADS) {
        sh_nz_val[i] = nz_val[nz_base + i];
        sh_nz_idx[i] = nz_idx[nz_base + i];
    }
    for (uint32_t i = tx; i < n_c; i += THREADS) sh_col_ids[i] = col_ids[col_base + i];
    for (uint32_t i = tx; i < n_r; i += THREADS) sh_row_ids[i] = row_ids[row_base + i];
    __syncthreads();

    const uint32_t feat_start = feat_slab * BOO_FEAT_BLOCK;
    if (feat_start >= features) return;
    const uint32_t feat_end = min(feat_start + BOO_FEAT_BLOCK, features);

    const size_t x_batch_base = static_cast<size_t>(batch_idx) * cols * features;
    const size_t y_batch_base = static_cast<size_t>(batch_idx) * rows * features;

    for (uint32_t base_f = feat_start; base_f < feat_end; base_f += P) {

        // --- STEP A: stage X, zero the accumulator -------------------------
        for (uint32_t i = tx; i < n_r * P; i += THREADS) smem_Y[i] = scalar_t(0);

        for (uint32_t i = tx; i < n_c * P; i += THREADS) {
            const uint32_t c = i >> P_LOG2;
            const uint32_t global_f = base_f + (i & P_MASK);
            smem_X[i] = (global_f < features)
                ? X[x_batch_base + static_cast<size_t>(sh_col_ids[c]) * features + global_f]
                : scalar_t(0);
        }
        __syncthreads();

        // --- STEP B: one thread per (nonzero, feature) ----------------------
        for (uint32_t i = tx; i < n_nz * P; i += THREADS) {
            const uint32_t k = i >> P_LOG2;
            const uint32_t f = i & P_MASK;
            const uint32_t idx = sh_nz_idx[k];
            atomicAdd(&smem_Y[(idx >> shift) * P + f],
                      sh_nz_val[k] * smem_X[(idx & col_mask) * P + f]);
        }
        __syncthreads();

        // --- STEP C: flush ---------------------------------------------------
        // No row-activity branch: a row group only lists rows the clustering
        // assigned to it, so every slot is live by construction. That is
        // SHOW_TO_ERAN.md sec. 4's row-compaction, free from the format.
        // atomicAdd because a row's result is split across its block-row.
        for (uint32_t i = tx; i < n_r * P; i += THREADS) {
            const uint32_t global_f = base_f + (i & P_MASK);
            if (global_f < features) {
                atomicAdd(&Y[y_batch_base
                             + static_cast<size_t>(sh_row_ids[i >> P_LOG2]) * features
                             + global_f],
                          smem_Y[i]);
            }
        }
        __syncthreads();
    }
}


// ===========================================================================
// Row-segmented variant -- the default when nz_order == rowmajor
// ===========================================================================
//
// Profiling the nnzflat kernel above on fem2d/65536/f512 found it *compute*
// bound, not memory bound: sm_throughput 92%, alu_pipe 40%, long_scoreboard
// only 10.8%, and the single largest stall short_scoreboard at 29.3% -- i.e.
// warps waiting on shared-memory dependencies. That is the per-(nonzero,
// feature) atomicAdd into smem_Y.
//
// Here each thread instead owns one (row, feature) work-item and accumulates
// that row's whole contribution in a REGISTER, then commits once. Removes:
//   * every shared atomic,
//   * smem_Y entirely (and its zeroing pass) -- a smaller arena, so more CTAs,
//   * one __syncthreads() per feature tile.
// Global atomic count is unchanged. This is zero_idle's structure on BOO's
// format, and zero_idle is the PBR kernel that actually wins.
//
// REQUIRES nz_order == rowmajor, which makes nz_idx sorted by local_row, so a
// per-block row-start table can be recovered by binary search. The launcher
// falls back to the nnzflat kernel for colmajor.

template <typename scalar_t, int THREADS, int P_LOG2, bool DIAG_RAW_STORE = false>
__global__ void __launch_bounds__(THREADS, 1536 / THREADS) boo_spmm_rowseg_kernel(
    const uint32_t num_blocks,
    const uint32_t features,
    const uint32_t batch_size,
    const uint32_t rows,
    const uint32_t cols,
    const uint32_t max_nr,
    const uint32_t max_nc,
    const uint32_t max_nnz_blk,
    const int32_t* __restrict__ row_grp_ptr,
    const int32_t* __restrict__ row_ids,
    const int32_t* __restrict__ col_grp_ptr,
    const int32_t* __restrict__ col_ids,
    const int32_t* __restrict__ blk_row_grp,
    const int32_t* __restrict__ blk_col_grp,
    const int32_t* __restrict__ blk_nz_ptr,
    const uint8_t* __restrict__ blk_shift,
    const uint16_t* __restrict__ nz_idx,
    const scalar_t* __restrict__ nz_val,
    const scalar_t* __restrict__ X,
    scalar_t* __restrict__ Y,
    const int32_t* __restrict__ block_row_ptr,
    const int32_t* __restrict__ row_starts
) {
    constexpr uint32_t P = 1u << P_LOG2;
    constexpr uint32_t P_MASK = P - 1u;

    const uint32_t b_idx = blockIdx.x;
    const uint32_t batch_idx = blockIdx.y;
    const uint32_t feat_slab = blockIdx.z;
    const uint32_t tx = threadIdx.x;

    if (b_idx >= num_blocks || batch_idx >= batch_size) return;

    extern __shared__ __align__(16) char boo_smem_raw[];
    scalar_t* sh_nz_val = reinterpret_cast<scalar_t*>(boo_smem_raw);                // The Sparse Values
    scalar_t* smem_X = sh_nz_val + max_nnz_blk;                                     // The Staged X Tile
    int32_t* sh_col_ids = reinterpret_cast<int32_t*>(smem_X + max_nc * P);          // Global Column IDs
    int32_t* sh_row_ids = sh_col_ids + max_nc;                                      // Global Row IDs
    int32_t* sh_row_start = sh_row_ids + max_nr;                                    // Row-Start Table
    uint16_t* sh_nz_idx = reinterpret_cast<uint16_t*>(sh_row_start + max_nr + 1);   // The Sparse Indices (packed local_row, local_col)

    const uint32_t rg = blk_row_grp[b_idx];                     // Row group index for this block
    const uint32_t cg = blk_col_grp[b_idx];                     // Column group index for this block
    const uint32_t row_base = row_grp_ptr[rg];                  // Base index into row_ids for this row group
    const uint32_t n_r = row_grp_ptr[rg + 1] - row_base;        // Number of (active) rows in this row group
    const uint32_t col_base = col_grp_ptr[cg];                  // Base index into col_ids for this column group
    const uint32_t n_c = col_grp_ptr[cg + 1] - col_base;        // Number of (active) columns in this column group
    const uint32_t nz_base = blk_nz_ptr[b_idx];                 // Base index into nz_idx/nz_val for this block
    const uint32_t n_nz = blk_nz_ptr[b_idx + 1] - nz_base;      // Number of nonzeros in this block

    if (n_r == 0 || n_c == 0 || n_nz == 0) return;

    const uint32_t shift = blk_shift[b_idx];
    const uint32_t col_mask = (1u << shift) - 1u;

    // --- LOAD, once per CTA -------------------------------------------------
    for (uint32_t i = tx; i < n_nz; i += THREADS) {
        sh_nz_val[i] = nz_val[nz_base + i];
        sh_nz_idx[i] = nz_idx[nz_base + i];
    }
    for (uint32_t i = tx; i < n_c; i += THREADS) sh_col_ids[i] = col_ids[col_base + i];
    for (uint32_t i = tx; i < n_r; i += THREADS) sh_row_ids[i] = row_ids[row_base + i];
    __syncthreads();

    // Row-start table by binary search. nz_idx is sorted by local_row (rowmajor)
    // and local_col occupies the low bits, so `idx >> shift` is non-decreasing.
    // Binary search rather than a scan because it is branch-free per thread,
    // needs no atomics, and handles rows with NO nonzero in this block (a row
    // group member the column group simply does not touch) for free.
    for (uint32_t r = tx; r <= n_r; r += THREADS) {
        if (block_row_ptr != nullptr) {
            sh_row_start[r] = row_starts[block_row_ptr[b_idx] + r];
            continue;
        }
        uint32_t lo = 0, hi = n_nz;
        while (lo < hi) {
            const uint32_t mid = (lo + hi) >> 1;
            if ((uint32_t)(sh_nz_idx[mid] >> shift) < r) lo = mid + 1;
            else hi = mid;
        }
        sh_row_start[r] = (int32_t)lo;
    }
    __syncthreads();

    const uint32_t feat_start = feat_slab * BOO_FEAT_BLOCK;
    if (feat_start >= features) return;
    const uint32_t feat_end = min(feat_start + BOO_FEAT_BLOCK, features);

    const size_t x_batch_base = static_cast<size_t>(batch_idx) * cols * features;
    const size_t y_batch_base = static_cast<size_t>(batch_idx) * rows * features;

    for (uint32_t base_f = feat_start; base_f < feat_end; base_f += P) {

        // --- STEP A: stage X (no smem_Y to zero) ---------------------------
        for (uint32_t i = tx; i < n_c * P; i += THREADS) {
            const uint32_t c = i >> P_LOG2;
            const uint32_t global_f = base_f + (i & P_MASK);
            smem_X[i] = (global_f < features)
                ? X[x_batch_base + static_cast<size_t>(sh_col_ids[c]) * features + global_f]
                : scalar_t(0);
        }
        __syncthreads();

        // --- STEP B+C: one thread per (row, feature), register accumulate ---
        for (uint32_t i = tx; i < n_r * P; i += THREADS) {
            const uint32_t r = i >> P_LOG2;
            const uint32_t f = i & P_MASK;
            const uint32_t global_f = base_f + f;

            const uint32_t begin = sh_row_start[r];
            const uint32_t end = sh_row_start[r + 1];

            scalar_t acc = scalar_t(0);
            for (uint32_t k = begin; k < end; ++k) {
                acc += sh_nz_val[k] * smem_X[(sh_nz_idx[k] & col_mask) * P + f];
            }

            if (acc != scalar_t(0) && global_f < features) {
                const size_t y_idx = y_batch_base
                    + static_cast<size_t>(sh_row_ids[r]) * features + global_f;
                // DIAG_RAW_STORE: diagnostic-only upper-bound timing variant.
                // Unlike PBR's, this atomicAdd is not purely a synchronization
                // tax -- a row's contribution is genuinely split across
                // multiple blocks in its row-group, so a plain store here
                // drops real cross-block accumulation, not just atomic
                // overhead. Output is incorrect by construction. Never used
                // outside BOOTimedSpMM(diagnostic_raw_store=True); see
                // bindings_boo_timed.cu.
                if constexpr (DIAG_RAW_STORE) {
                    Y[y_idx] = acc;
                } else {
                    atomicAdd(&Y[y_idx], acc);
                }
            }
        }
        __syncthreads();
    }
}


template <typename scalar_t>
void launch_boo_spmm(
    const int num_blocks, const int features, const int batch_size,
    const int rows, const int cols,
    const int max_nr, const int max_nc, const int max_nnz_blk,
    const int32_t* row_grp_ptr, const int32_t* row_ids,
    const int32_t* col_grp_ptr, const int32_t* col_ids,
    const int32_t* blk_row_grp, const int32_t* blk_col_grp,
    const int32_t* blk_nz_ptr, const uint8_t* blk_shift,
    const uint16_t* nz_idx, const scalar_t* nz_val,
    const scalar_t* X, scalar_t* Y,
    cudaStream_t stream, int threads, int p, bool row_seg,
    const int32_t* block_row_ptr, const int32_t* row_starts
) {
    if (threads != 128 && threads != 256 && threads != 512)
        throw std::invalid_argument("BOO threads must be 128, 256 or 512");
    if (p != 8 && p != 16 && p != 32 && p != 64 && p != 128)
        throw std::invalid_argument("BOO p must be 8, 16, 32, 64 or 128");
    if (num_blocks == 0 || features == 0 || batch_size == 0) return;

    // Match an actual P specialization when sizing the arena. P=8 is the
    // smallest kernel; narrower inputs use padding, not a smaller allocation.
    while (p > 8 && p > features) p >>= 1;

    const int scalar_bytes = static_cast<int>(sizeof(scalar_t));
    const int arena = row_seg
        ? boo_arena_bytes_rowseg(max_nr, max_nc, max_nnz_blk, p, scalar_bytes)
        : boo_arena_bytes(max_nr, max_nc, max_nnz_blk, p, scalar_bytes);

    const int feat_slabs = (features + BOO_FEAT_BLOCK - 1) / BOO_FEAT_BLOCK;
    const dim3 grid(num_blocks, batch_size, feat_slabs);

    #define BOO_LAUNCH(T, PL)                                                                  \
        do {                                                                                   \
            auto kernel = row_seg ? boo_spmm_rowseg_kernel<scalar_t, T, PL>                    \
                                  : boo_spmm_nnzflat_kernel<scalar_t, T, PL>;                  \
            /* Only required above 48 KB/CTA, but set it guarded anyway: without */            \
            /* it an oversized arena fails at launch with a swallowed             */            \
            /* cudaErrorInvalidValue rather than anything diagnosable.            */            \
            if (arena > 48 * 1024) {                                                           \
                C10_CUDA_CHECK(cudaFuncSetAttribute(kernel,                                    \
                                     cudaFuncAttributeMaxDynamicSharedMemorySize, arena));     \
            }                                                                                  \
            kernel<<<grid, dim3(T), arena, stream>>>(                                          \
                (uint32_t)num_blocks, (uint32_t)features, (uint32_t)batch_size,                \
                (uint32_t)rows, (uint32_t)cols,                                                \
                (uint32_t)max_nr, (uint32_t)max_nc, (uint32_t)max_nnz_blk,                     \
                row_grp_ptr, row_ids, col_grp_ptr, col_ids,                                    \
                blk_row_grp, blk_col_grp, blk_nz_ptr, blk_shift,                               \
                nz_idx, nz_val, X, Y, block_row_ptr, row_starts);                                                         \
            C10_CUDA_KERNEL_LAUNCH_CHECK();                                                    \
        } while (0)

    #define BOO_DISPATCH_P(T)                                                    \
        do {                                                                     \
            if      (p <= 8)   BOO_LAUNCH(T, 3);                                 \
            else if (p <= 16)  BOO_LAUNCH(T, 4);                                 \
            else if (p <= 32)  BOO_LAUNCH(T, 5);                                 \
            else if (p <= 64)  BOO_LAUNCH(T, 6);                                 \
            else               BOO_LAUNCH(T, 7);                                 \
        } while (0)

    if      (threads <= 128) BOO_DISPATCH_P(128);
    else if (threads <= 256) BOO_DISPATCH_P(256);
    else                     BOO_DISPATCH_P(512);

    #undef BOO_DISPATCH_P
    #undef BOO_LAUNCH
}

template void launch_boo_spmm<float>(
    const int, const int, const int, const int, const int, const int, const int, const int,
    const int32_t*, const int32_t*, const int32_t*, const int32_t*,
    const int32_t*, const int32_t*, const int32_t*, const uint8_t*,
    const uint16_t*, const float*, const float*, float*, cudaStream_t, int, int, bool, const int32_t*, const int32_t*);

template void launch_boo_spmm<double>(
    const int, const int, const int, const int, const int, const int, const int, const int,
    const int32_t*, const int32_t*, const int32_t*, const int32_t*,
    const int32_t*, const int32_t*, const int32_t*, const uint8_t*,
    const uint16_t*, const double*, const double*, double*, cudaStream_t, int, int, bool, const int32_t*, const int32_t*);

// Diagnostic-only upper-bound timing variant: the rowseg kernel's Step
// B+C atomicAdd is replaced with a plain store (DIAG_RAW_STORE=true).
// Output is incorrect by construction (drops real cross-block row
// accumulation -- see the comment at the atomicAdd call site). fp32 only,
// matching BOOTimedSpMM's scope; row_seg-only (no nnzflat/colmajor
// diagnostic); no precomputed row-boundary table support (block_row_ptr/
// row_starts always null -- binary search only).
void launch_boo_diag_spmm(
    const int num_blocks, const int features, const int batch_size,
    const int rows, const int cols,
    const int max_nr, const int max_nc, const int max_nnz_blk,
    const int32_t* row_grp_ptr, const int32_t* row_ids,
    const int32_t* col_grp_ptr, const int32_t* col_ids,
    const int32_t* blk_row_grp, const int32_t* blk_col_grp,
    const int32_t* blk_nz_ptr, const uint8_t* blk_shift,
    const uint16_t* nz_idx, const float* nz_val,
    const float* X, float* Y,
    cudaStream_t stream, int threads, int p
) {
    if (threads != 128 && threads != 256 && threads != 512)
        throw std::invalid_argument("BOO threads must be 128, 256 or 512");
    if (p != 8 && p != 16 && p != 32 && p != 64 && p != 128)
        throw std::invalid_argument("BOO p must be 8, 16, 32, 64 or 128");
    if (num_blocks == 0 || features == 0 || batch_size == 0) return;

    while (p > 8 && p > features) p >>= 1;

    const int arena = boo_arena_bytes_rowseg(max_nr, max_nc, max_nnz_blk, p, (int)sizeof(float));
    const int feat_slabs = (features + BOO_FEAT_BLOCK - 1) / BOO_FEAT_BLOCK;
    const dim3 grid(num_blocks, batch_size, feat_slabs);

    #define BOO_DIAG_LAUNCH(T, PL)                                                               \
        do {                                                                                     \
            auto kernel = boo_spmm_rowseg_kernel<float, T, PL, true>;                            \
            if (arena > 48 * 1024) {                                                             \
                C10_CUDA_CHECK(cudaFuncSetAttribute(kernel,                                      \
                                     cudaFuncAttributeMaxDynamicSharedMemorySize, arena));       \
            }                                                                                    \
            kernel<<<grid, dim3(T), arena, stream>>>(                                            \
                (uint32_t)num_blocks, (uint32_t)features, (uint32_t)batch_size,                  \
                (uint32_t)rows, (uint32_t)cols,                                                  \
                (uint32_t)max_nr, (uint32_t)max_nc, (uint32_t)max_nnz_blk,                       \
                row_grp_ptr, row_ids, col_grp_ptr, col_ids,                                      \
                blk_row_grp, blk_col_grp, blk_nz_ptr, blk_shift,                                 \
                nz_idx, nz_val, X, Y, nullptr, nullptr);                                         \
            C10_CUDA_KERNEL_LAUNCH_CHECK();                                                      \
        } while (0)

    #define BOO_DIAG_DISPATCH_P(T)                                          \
        do {                                                                \
            if      (p <= 8)   BOO_DIAG_LAUNCH(T, 3);                       \
            else if (p <= 16)  BOO_DIAG_LAUNCH(T, 4);                       \
            else if (p <= 32)  BOO_DIAG_LAUNCH(T, 5);                       \
            else if (p <= 64)  BOO_DIAG_LAUNCH(T, 6);                       \
            else               BOO_DIAG_LAUNCH(T, 7);                       \
        } while (0)

    if      (threads <= 128) BOO_DIAG_DISPATCH_P(128);
    else if (threads <= 256) BOO_DIAG_DISPATCH_P(256);
    else                     BOO_DIAG_DISPATCH_P(512);

    #undef BOO_DIAG_DISPATCH_P
    #undef BOO_DIAG_LAUNCH
}


// ===========================================================================
// Tail kernel -- flat COO for below-threshold blocks (BOO_TAIL_BATCH policy)
// ===========================================================================
//
// KEEP launches one CUDA block per (row group x col group) intersection no
// matter how little work it holds. Profiling diag_skewed (bimodal per-block
// nnz by construction) found 97.4% of blocks holding <=3 nonzeros at
// N=262144, each still paying the arena sized for the matrix's densest block
// (1024 nnz there -- 1.3% utilization), and the resulting 1.4M-CTA grid ran
// 6-7x slower than PBR's `block` kernel, which simply routes the same noise
// to a CSR remainder. See BOO_FORMAT.md sec. 15.
//
// This kernel is the in-format alternative to that CSR spill (BOO_REMAINDER):
// below-threshold blocks are flattened to a GLOBAL-index COO list at build
// time (boo_matrix.cpp), with no block-local reencoding -- there is no
// shared arena to amortize for entries this sparse. Each thread owns one
// nonzero and grid-strides over the tail, looping over `features` for that
// nonzero. No smem, no barriers, no per-block arena, and critically the CTA
// count is bounded by GPU occupancy rather than by tail_nnz -- this is what
// fixes KEEP's CTA explosion, in contrast to BOO_REMAINDER which fixes it by
// leaving BOO entirely for the shared CSR kernel (and dropping
// compressed_fraction below 1.0, reintroducing SHOW_TO_ERAN sec. 2's
// CSR-fallback confound into any benchmark that reads it naively).
//
// atomicAdd is required, unlike the row-segmented kernel's plain per-row
// commit: a tail row has no row_ids partition guarantee -- it may also
// receive contributions from a kept BOO block (same row group, a different,
// denser column group) or from another tail entry.
template <typename scalar_t, int THREADS>
__global__ void __launch_bounds__(THREADS, 1536 / THREADS) boo_spmm_tail_kernel(
    const uint32_t tail_nnz,
    const uint32_t features,
    const uint32_t batch_size,
    const uint32_t rows,
    const uint32_t cols,
    const int32_t* __restrict__ tail_row,
    const int32_t* __restrict__ tail_col,
    const scalar_t* __restrict__ tail_val,
    const scalar_t* __restrict__ X,
    scalar_t* __restrict__ Y
) {
    const uint32_t batch_idx = blockIdx.y;
    if (batch_idx >= batch_size) return;

    const size_t x_base = static_cast<size_t>(batch_idx) * cols * features;
    const size_t y_base = static_cast<size_t>(batch_idx) * rows * features;
    const uint32_t stride = gridDim.x * blockDim.x;

    for (uint32_t k = blockIdx.x * blockDim.x + threadIdx.x; k < tail_nnz; k += stride) {
        const int32_t r = tail_row[k];
        const int32_t c = tail_col[k];
        const scalar_t v = tail_val[k];
        const scalar_t* __restrict__ xrow = X + x_base + static_cast<size_t>(c) * features;
        scalar_t* __restrict__ yrow = Y + y_base + static_cast<size_t>(r) * features;

        #pragma unroll 4
        for (uint32_t f = 0; f < features; ++f) {
            atomicAdd(&yrow[f], v * xrow[f]);
        }
    }
}


template <typename scalar_t>
void launch_boo_tail_spmm(
    const int tail_nnz, const int features, const int batch_size,
    const int rows, const int cols,
    const int32_t* tail_row, const int32_t* tail_col, const scalar_t* tail_val,
    const scalar_t* X, scalar_t* Y,
    cudaStream_t stream, int threads
) {
    if (tail_nnz == 0 || features == 0 || batch_size == 0) return;

    // Grid-stride, so CTA count is bounded by occupancy rather than by
    // tail_nnz -- the whole point of this kernel. OCCUPANCY_WAVES=32 is a
    // generous oversubscription factor for a kernel this light (no smem, no
    // barriers): enough resident waves to hide global-load/atomic latency
    // without the launch overhead of one CTA per tail element.
    constexpr int OCCUPANCY_WAVES = 32;
    int device = 0;
    cudaGetDevice(&device);
    int sm_count = 1;
    cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device);
    const int ctas_per_sm = max(1, 1536 / threads);
    const int occupancy_grid = sm_count * ctas_per_sm * OCCUPANCY_WAVES;
    const int work_grid = (tail_nnz + threads - 1) / threads;
    const dim3 grid(max(1, min(occupancy_grid, work_grid)), batch_size);

    #define BOO_TAIL_LAUNCH(T)                                                   \
        boo_spmm_tail_kernel<scalar_t, T><<<grid, dim3(T), 0, stream>>>(         \
            (uint32_t)tail_nnz, (uint32_t)features, (uint32_t)batch_size,        \
            (uint32_t)rows, (uint32_t)cols, tail_row, tail_col, tail_val, X, Y)

    if      (threads <= 128) BOO_TAIL_LAUNCH(128);
    else if (threads <= 256) BOO_TAIL_LAUNCH(256);
    else                     BOO_TAIL_LAUNCH(512);

    #undef BOO_TAIL_LAUNCH
}

template void launch_boo_tail_spmm<float>(
    const int, const int, const int, const int, const int,
    const int32_t*, const int32_t*, const float*,
    const float*, float*, cudaStream_t, int);

template void launch_boo_tail_spmm<double>(
    const int, const int, const int, const int, const int,
    const int32_t*, const int32_t*, const double*,
    const double*, double*, cudaStream_t, int);
