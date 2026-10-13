#include <cuda_runtime.h>
#include <cstdint>
#include <stdexcept>
#include <type_traits>
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

// Features per thread (FPT) for the row-segmented kernel. FPT=1 is the original
// one-(row, feature)-per-thread loop. FPT=2/4 give each thread a vector of
// adjacent features: one vector smem_X load plus one sh_nz_val/sh_nz_idx load
// per nonzero now feed FPT FMAs instead of one, attacking the MIO throttle
// measured on Boeing/pwtk (3 shared loads per FMA). Fewer threads per row
// (P/FPT) means a warp can span two rows when P/FPT < 32. Process-wide so a
// benchmark can interleave variants in one build; boo_matmul sets it per call.
static int g_boo_rowseg_fpt = 1;

void set_boo_rowseg_fpt(int fpt) {
    if (fpt != 1 && fpt != 2 && fpt != 4)
        throw std::invalid_argument("BOO features_per_thread must be 1, 2 or 4");
    g_boo_rowseg_fpt = fpt;
}

int get_boo_rowseg_fpt() { return g_boo_rowseg_fpt; }

// Feature-slab width for the row-segmented kernel (experiment parameterization).
// BOO_FEAT_BLOCK stays the default and the only width for nnzflat/diagnostic
// launches. Other widths exist only for float32/threads=256 to bound the number
// of instantiations; the launcher rejects anything else rather than falling back.
static int g_boo_feat_block = BOO_FEAT_BLOCK;

void set_boo_feat_block(int width) {
    if (width != 128 && width != 256 && width != 512 && width != 1024)
        throw std::invalid_argument("BOO feature-slab width must be 128, 256, 512 or 1024");
    g_boo_feat_block = width;
}

int get_boo_feat_block() { return g_boo_feat_block; }

template <typename scalar_t, int N> struct BooVec;
template <> struct BooVec<float, 1>  { using type = float; };
template <> struct BooVec<float, 2>  { using type = float2; };
template <> struct BooVec<float, 4>  { using type = float4; };
template <> struct BooVec<double, 1> { using type = double; };
template <> struct BooVec<double, 2> { using type = double2; };

__device__ __forceinline__ void boo_fma(float* a, float v, float x)    { a[0] += v * x; }
__device__ __forceinline__ void boo_fma(float* a, float v, float2 x)   { a[0] += v * x.x; a[1] += v * x.y; }
__device__ __forceinline__ void boo_fma(float* a, float v, float4 x) {
    a[0] += v * x.x; a[1] += v * x.y; a[2] += v * x.z; a[3] += v * x.w;
}
__device__ __forceinline__ void boo_fma(double* a, double v, double x)  { a[0] += v * x; }
__device__ __forceinline__ void boo_fma(double* a, double v, double2 x) { a[0] += v * x.x; a[1] += v * x.y; }


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

template <typename scalar_t, int THREADS, int P_LOG2, bool DIAG_RAW_STORE = false, int FPT = 1,
          int FEAT_BLOCK = BOO_FEAT_BLOCK>
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

    static_assert(FPT == 1 || FPT == 2 || FPT == 4, "FPT must be 1, 2 or 4");
    static_assert(FPT <= (1 << P_LOG2), "FPT must not exceed P");
    // A P tile must never straddle two slabs: base_f steps by P from the slab start.
    static_assert(FEAT_BLOCK % (1 << P_LOG2) == 0, "slab width must be a multiple of P");
    // FPT>1 reads smem_X as 8/16-byte vectors: start it on a 16-byte boundary.
    // The padding (<= 12 bytes) comes out of the arena's 16-byte slack.
    constexpr uint32_t X_ALIGN = FPT > 1 ? 16u / sizeof(scalar_t) : 1u;

    extern __shared__ __align__(16) char boo_smem_raw[];
    scalar_t* sh_nz_val = reinterpret_cast<scalar_t*>(boo_smem_raw);                // The Sparse Values
    scalar_t* smem_X = sh_nz_val + (max_nnz_blk + X_ALIGN - 1) / X_ALIGN * X_ALIGN; // The Staged X Tile
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

    const uint32_t feat_start = feat_slab * FEAT_BLOCK;
    if (feat_start >= features) return;
    const uint32_t feat_end = min(feat_start + FEAT_BLOCK, features);

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

        if constexpr (FPT > 1) {
            // --- STEP B+C, FPT>1: one thread per (row, FPT adjacent features) ---
            using vec_t = typename BooVec<scalar_t, FPT>::type;
            constexpr uint32_t LANES = P / FPT;  // threads per row
            for (uint32_t i = tx; i < n_r * LANES; i += THREADS) {
                const uint32_t r = i / LANES;
                const uint32_t f0 = (i % LANES) * FPT;

                const uint32_t begin = sh_row_start[r];
                const uint32_t end = sh_row_start[r + 1];

                scalar_t acc[FPT] = {};
                // #pragma unroll
                // for (int j = 0; j < FPT; ++j) acc[j] = scalar_t(0);

                for (uint32_t k = begin; k < end; ++k) {
                    const vec_t x = *reinterpret_cast<const vec_t*>(
                        smem_X + (sh_nz_idx[k] & col_mask) * P + f0);
                    boo_fma(acc, sh_nz_val[k], x); // acc[0..FPT-1] += sh_nz_val[k] * x[0..FPT-1]
                }

                const size_t y_row = y_batch_base + static_cast<size_t>(sh_row_ids[r]) * features;
                #pragma unroll
                for (int j = 0; j < FPT; ++j) {
                    const uint32_t global_f = base_f + f0 + j;
                    if (acc[j] != scalar_t(0) && global_f < features) {
                        if constexpr (DIAG_RAW_STORE) {
                            // Just for diagnostic timing: drops real cross-block accumulation, so output is wrong by construction.
                            Y[y_row + global_f] = acc[j];
                        } else {
                            atomicAdd(&Y[y_row + global_f], acc[j]);
                        }
                    }
                }
            }
        } else {
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
        }  // FPT == 1
        __syncthreads();
    }
}


// Kernel selection shared by launch_boo_spmm and boo_rowseg_launch_info, so the
// reported attributes are those of the kernel actually launched.
template <typename scalar_t>
using boo_kernel_t = decltype(&boo_spmm_rowseg_kernel<scalar_t, 256, 6>);

template <typename scalar_t, int T, int PL, int FB>
static boo_kernel_t<scalar_t> boo_rowseg_by_fpt(int fpt) {
    // FPT=4 is float-only; the double instantiation of that branch is never taken.
    constexpr int FPT_WIDE = sizeof(scalar_t) == 4 ? 4 : 2;
    if (fpt == 1) return boo_spmm_rowseg_kernel<scalar_t, T, PL, false, 1, FB>;
    if (fpt == 2) return boo_spmm_rowseg_kernel<scalar_t, T, PL, false, 2, FB>;
    return boo_spmm_rowseg_kernel<scalar_t, T, PL, false, FPT_WIDE, FB>;
}

template <typename scalar_t, int T, int PL>
static boo_kernel_t<scalar_t> boo_select_kernel_tp(bool row_seg, int fpt, int slab) {
    if (!row_seg) return boo_spmm_nnzflat_kernel<scalar_t, T, PL>;
    if constexpr (std::is_same_v<scalar_t, float> && T == 256) {
        switch (slab) {
            case 128:  return boo_rowseg_by_fpt<scalar_t, T, PL, 128>(fpt);
            case 256:  return boo_rowseg_by_fpt<scalar_t, T, PL, 256>(fpt);
            case 1024: return boo_rowseg_by_fpt<scalar_t, T, PL, 1024>(fpt);
            default: break;
        }
    }
    return boo_rowseg_by_fpt<scalar_t, T, PL, BOO_FEAT_BLOCK>(fpt);
}

template <typename scalar_t, int T>
static boo_kernel_t<scalar_t> boo_select_kernel_t(int p, bool row_seg, int fpt, int slab) {
    if      (p <= 8)   return boo_select_kernel_tp<scalar_t, T, 3>(row_seg, fpt, slab);
    else if (p <= 16)  return boo_select_kernel_tp<scalar_t, T, 4>(row_seg, fpt, slab);
    else if (p <= 32)  return boo_select_kernel_tp<scalar_t, T, 5>(row_seg, fpt, slab);
    else if (p <= 64)  return boo_select_kernel_tp<scalar_t, T, 6>(row_seg, fpt, slab);
    else               return boo_select_kernel_tp<scalar_t, T, 7>(row_seg, fpt, slab);
}

// Validates threads/p/FPT/slab and returns the kernel, the effective P (reduced
// to the smallest specialization covering F) and the slab width.
template <typename scalar_t>
static boo_kernel_t<scalar_t> boo_select_kernel(int threads, int& p, int features, bool row_seg,
                                                int& fpt, int& slab) {
    if (threads != 128 && threads != 256 && threads != 512)
        throw std::invalid_argument("BOO threads must be 128, 256 or 512");
    if (p != 8 && p != 16 && p != 32 && p != 64 && p != 128)
        throw std::invalid_argument("BOO p must be 8, 16, 32, 64 or 128");
    fpt = row_seg ? g_boo_rowseg_fpt : 1;
    if (fpt * static_cast<int>(sizeof(scalar_t)) > 16)
        throw std::invalid_argument("BOO features_per_thread=4 requires float32");
    slab = row_seg ? g_boo_feat_block : BOO_FEAT_BLOCK;
    if (slab != BOO_FEAT_BLOCK && (!std::is_same_v<scalar_t, float> || threads != 256))
        throw std::invalid_argument("BOO feature-slab widths other than 512 are instantiated "
                                    "only for float32 with threads=256");

    // Match an actual P specialization when sizing the arena. P=8 is the
    // smallest kernel; narrower inputs use padding, not a smaller allocation.
    while (p > 8 && p > features) p >>= 1;

    if      (threads <= 128) return boo_select_kernel_t<scalar_t, 128>(p, row_seg, fpt, slab);
    else if (threads <= 256) return boo_select_kernel_t<scalar_t, 256>(p, row_seg, fpt, slab);
    else                     return boo_select_kernel_t<scalar_t, 512>(p, row_seg, fpt, slab);
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
    int fpt = 1, slab = BOO_FEAT_BLOCK;
    const auto kernel = boo_select_kernel<scalar_t>(threads, p, features, row_seg, fpt, slab);
    if (num_blocks == 0 || features == 0 || batch_size == 0) return;

    const int scalar_bytes = static_cast<int>(sizeof(scalar_t));
    const int arena = row_seg
        ? boo_arena_bytes_rowseg(max_nr, max_nc, max_nnz_blk, p, scalar_bytes)
        : boo_arena_bytes(max_nr, max_nc, max_nnz_blk, p, scalar_bytes);

    const int feat_slabs = (features + slab - 1) / slab;
    const dim3 grid(num_blocks, batch_size, feat_slabs);

    // Only required above 48 KB/CTA, but set it guarded anyway: without
    // it an oversized arena fails at launch with a swallowed
    // cudaErrorInvalidValue rather than anything diagnosable.
    if (arena > 48 * 1024) {
        C10_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, arena));
    }
    kernel<<<grid, dim3(threads), arena, stream>>>(
        (uint32_t)num_blocks, (uint32_t)features, (uint32_t)batch_size,
        (uint32_t)rows, (uint32_t)cols,
        (uint32_t)max_nr, (uint32_t)max_nc, (uint32_t)max_nnz_blk,
        row_grp_ptr, row_ids, col_grp_ptr, col_ids,
        blk_row_grp, blk_col_grp, blk_nz_ptr, blk_shift,
        nz_idx, nz_val, X, Y, block_row_ptr, row_starts);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

BOORowsegLaunchInfo boo_rowseg_launch_info(int max_nr, int max_nc, int max_nnz_blk,
                                           int features, int threads, int p) {
    BOORowsegLaunchInfo info{};
    info.requested_p = p;
    info.threads = threads;
    int fpt = 1, slab = BOO_FEAT_BLOCK;
    const auto kernel = boo_select_kernel<float>(threads, p, features, true, fpt, slab);
    info.effective_p = p;
    info.fpt = fpt;
    info.feat_block = slab;
    info.grid_z = (features + slab - 1) / slab;
    info.arena_bytes = boo_arena_bytes_rowseg(max_nr, max_nc, max_nnz_blk, p, (int)sizeof(float));
    cudaFuncAttributes attr{};
    C10_CUDA_CHECK(cudaFuncGetAttributes(&attr, kernel));
    info.num_regs = attr.numRegs;
    info.local_size_bytes = (int)attr.localSizeBytes;
    info.static_shared_bytes = (int)attr.sharedSizeBytes;
    info.max_threads_per_block = attr.maxThreadsPerBlock;
    info.max_dynamic_shared_bytes = attr.maxDynamicSharedSizeBytes;
    info.max_active_blocks_per_sm = 0;
    int device = 0, optin = 0;
    C10_CUDA_CHECK(cudaGetDevice(&device));
    C10_CUDA_CHECK(cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, device));
    if (info.arena_bytes <= optin) {  // otherwise the launch itself would fail: report 0
        if (info.arena_bytes > 48 * 1024) {
            C10_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                info.arena_bytes));
        }
        C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &info.max_active_blocks_per_sm, kernel, threads, info.arena_bytes));
    }
    return info;
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
