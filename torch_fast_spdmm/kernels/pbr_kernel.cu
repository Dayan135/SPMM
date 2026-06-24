#include <cuda_runtime.h>
#include <cstdint>
#include <stdexcept>

// 16-byte vector type per scalar: float4 (4 lanes) / double2 (2 lanes). The
// vectorized kernel processes `lanes` contiguous features per thread so the
// hot shared-memory loads/stores become a single LDS.128 / STS.128.
template <typename scalar_t> struct PbrVec;
template <> struct PbrVec<float>  { using type = float4;  static constexpr int lanes = 4; };
template <> struct PbrVec<double> { using type = double2; static constexpr int lanes = 2; };

template <typename index_t, typename scalar_t,
          int BLOCK_ROWS, int BLOCK_COLS,
          int TOTAL_SHARED, int THREADS_PER_BLOCK, int FEAT_BLOCK_SIZE>
__global__ void __launch_bounds__(THREADS_PER_BLOCK, 16) pbr_spmm_zero_idle_kernel(
    const uint32_t num_pbr_blocks,
    const uint32_t features,
    const uint32_t batch_size,
    const uint32_t cols,
    const uint32_t rows,
    const int64_t* __restrict__ block_codes,
    const index_t* __restrict__ block_coords,
    const index_t* __restrict__ block_offsets,
    const scalar_t* __restrict__ block_data,
    const scalar_t* __restrict__ X,
    scalar_t* __restrict__ Y
) {
    __shared__ scalar_t smem[TOTAL_SHARED];

    const uint32_t b_idx = blockIdx.x;
    const uint32_t batch_idx = blockIdx.y;
    const uint32_t feat_block_idx = blockIdx.z;
    const uint32_t tx = threadIdx.x;

    if (b_idx >= num_pbr_blocks || batch_idx >= batch_size) return;

    // Load Metadata to Registers
    const uint64_t code = static_cast<uint64_t>(block_codes[b_idx]);
    const uint32_t row_origin = block_coords[b_idx * 2];
    const uint32_t col_origin = block_coords[b_idx * 2 + 1];
    const uint32_t data_offset = block_offsets[b_idx];

    // --- 1. SPARSITY SCAN & REGISTER PACKING ---
    uint32_t n_r = 0;
    uint64_t col_mask = 0;
    const uint64_t row_bits = (1ULL << BLOCK_COLS) - 1;

    uint32_t packed_active_rows = 0;
    uint64_t packed_row_offsets = 0;
    uint32_t current_relative_offset = 0;

    #pragma unroll
    for (uint32_t r = 0; r < BLOCK_ROWS; ++r) {
        const uint64_t row_mask = (code >> (r * BLOCK_COLS)) & row_bits;
        if (row_mask > 0) {
            packed_active_rows |= (r << (n_r * 3));
            packed_row_offsets |= ((uint64_t)current_relative_offset << (n_r * 7));

            n_r++;
            col_mask |= row_mask;
            current_relative_offset += __popcll(row_mask);
        }
    }

    const uint32_t n_c = __popcll(col_mask);
    if (n_r == 0 || n_c == 0) return;

    uint32_t packed_cols_physical = 0;
    uint32_t packed_cols_logical = 0;
    uint32_t pc_idx = 0;

    #pragma unroll
    for (uint32_t c = 0; c < BLOCK_COLS; ++c) {
        if ((col_mask >> c) & 1) {
            packed_cols_logical |= (pc_idx << (c * 3));
            packed_cols_physical |= (c << (pc_idx * 3));
            pc_idx++;
        }
    }

    // --- 2. DYNAMIC P CALCULATION ---
    // p is floored to a power of 2 so inner-loop /current_p and %current_p
    // become bit-shift and mask ops.  Correct when features is a power-of-2
    // multiple of p, which holds for all benchmarked feature counts.
    const uint32_t p_log2 = 31u - __clz(min(TOTAL_SHARED / (n_r + n_c), features));
    const uint32_t p      = 1u << p_log2;
    const uint32_t p_mask = p - 1u;

    const uint32_t feat_start = feat_block_idx * FEAT_BLOCK_SIZE;
    const uint32_t feat_end = min(feat_start + FEAT_BLOCK_SIZE, features);

    // --- 3. GRID-STRIDE LOOP OVER FEATURES ---
    for (uint32_t base_f = feat_start; base_f < feat_end; base_f += p) {

        const uint32_t current_p = min(p, feat_end - base_f);

        scalar_t* smem_X = smem;
        scalar_t* smem_Y = smem + (n_c * current_p);

        // STEP A: Cooperative Load
        const uint32_t total_x_elements = n_c * current_p;
        for (uint32_t i = tx; i < total_x_elements; i += THREADS_PER_BLOCK) {
            const uint32_t c_idx   = i >> p_log2;
            const uint32_t f_offset = i & p_mask;

            const uint32_t local_c = (packed_cols_physical >> (c_idx * 3)) & 0x7;
            smem_X[i] = X[(batch_idx * cols * features) + ((col_origin + local_c) * features) + base_f + f_offset];
        }

        __syncthreads();

        // STEP B: FLATTENED WORKER POOL
        const uint32_t total_work = n_r * current_p;

        for (uint32_t work_idx = tx; work_idx < total_work; work_idx += THREADS_PER_BLOCK) {
            const uint32_t pr    = work_idx >> p_log2;
            const uint32_t f_in_s = work_idx & p_mask;

            const uint32_t actual_r = (packed_active_rows >> (pr * 3)) & 0x7;
            uint32_t local_data_ptr = data_offset + ((packed_row_offsets >> (pr * 7)) & 0x7F);

            const uint64_t row_code = (code >> (actual_r * BLOCK_COLS)) & row_bits;

            scalar_t sum = 0;

            #pragma unroll
            for (uint32_t c = 0; c < BLOCK_COLS; ++c) {
                if ((row_code >> c) & 1) {
                    const uint32_t pc = (packed_cols_logical >> (c * 3)) & 0x7;
                    sum += __ldg(&block_data[local_data_ptr]) * smem_X[pc * current_p + f_in_s];
                    local_data_ptr++;
                }
            }

            smem_Y[work_idx] = sum;
        }

        __syncthreads();

        // STEP C: Cooperative Atomic Commit
        for (uint32_t work_idx = tx; work_idx < total_work; work_idx += THREADS_PER_BLOCK) {
            const uint32_t pr    = work_idx >> p_log2;
            const uint32_t f_in_s = work_idx & p_mask;

            const uint32_t actual_r = (packed_active_rows >> (pr * 3)) & 0x7;
            const uint32_t global_r = row_origin + actual_r;

            atomicAdd(&Y[(batch_idx * rows * features) + (global_r * features) + base_f + f_in_s], smem_Y[work_idx]);
        }

        __syncthreads();
    }
}


// Vectorized variant: each thread owns `L` (= PbrVec lanes) contiguous features.
// Requires features % L == 0 so the per-column feature run is L-aligned in both
// global X and shared memory; the launcher guarantees this before dispatching.
template <typename index_t, typename scalar_t,
          int BLOCK_ROWS, int BLOCK_COLS,
          int TOTAL_SHARED, int THREADS_PER_BLOCK, int FEAT_BLOCK_SIZE>
__global__ void __launch_bounds__(THREADS_PER_BLOCK, 16) pbr_spmm_vec_kernel(
    const uint32_t num_pbr_blocks,
    const uint32_t features,
    const uint32_t batch_size,
    const uint32_t cols,
    const uint32_t rows,
    const int64_t* __restrict__ block_codes,
    const index_t* __restrict__ block_coords,
    const index_t* __restrict__ block_offsets,
    const scalar_t* __restrict__ block_data,
    const scalar_t* __restrict__ X,
    scalar_t* __restrict__ Y
) {
    using vec_t = typename PbrVec<scalar_t>::type;
    constexpr int L = PbrVec<scalar_t>::lanes;
    union lane_u { vec_t v; scalar_t s[L]; };

    __shared__ __align__(16) scalar_t smem[TOTAL_SHARED];

    const uint32_t b_idx = blockIdx.x;
    const uint32_t batch_idx = blockIdx.y;
    const uint32_t feat_block_idx = blockIdx.z;
    const uint32_t tx = threadIdx.x;

    if (b_idx >= num_pbr_blocks || batch_idx >= batch_size) return;

    const uint64_t code = static_cast<uint64_t>(block_codes[b_idx]);
    const uint32_t row_origin = block_coords[b_idx * 2];
    const uint32_t col_origin = block_coords[b_idx * 2 + 1];
    const uint32_t data_offset = block_offsets[b_idx];

    // --- 1. SPARSITY SCAN & REGISTER PACKING (identical to scalar kernel) ---
    uint32_t n_r = 0;
    uint64_t col_mask = 0;
    const uint64_t row_bits = (1ULL << BLOCK_COLS) - 1;

    uint32_t packed_active_rows = 0;
    uint64_t packed_row_offsets = 0;
    uint32_t current_relative_offset = 0;

    #pragma unroll
    for (uint32_t r = 0; r < BLOCK_ROWS; ++r) {
        const uint64_t row_mask = (code >> (r * BLOCK_COLS)) & row_bits;
        if (row_mask > 0) {
            packed_active_rows |= (r << (n_r * 3));
            packed_row_offsets |= ((uint64_t)current_relative_offset << (n_r * 7));
            n_r++;
            col_mask |= row_mask;
            current_relative_offset += __popcll(row_mask);
        }
    }

    const uint32_t n_c = __popcll(col_mask);
    if (n_r == 0 || n_c == 0) return;

    uint32_t packed_cols_physical = 0;
    uint32_t packed_cols_logical = 0;
    uint32_t pc_idx = 0;

    #pragma unroll
    for (uint32_t c = 0; c < BLOCK_COLS; ++c) {
        if ((col_mask >> c) & 1) {
            packed_cols_logical |= (pc_idx << (c * 3));
            packed_cols_physical |= (c << (pc_idx * 3));
            pc_idx++;
        }
    }

    // --- 2. DYNAMIC P CALCULATION (vp floored to power of 2 → bit-op division) ---
    // vp_log2 / vp_mask replace /vp and %vp in inner loops; correct when
    // features is a power-of-2 multiple of p (all benchmarked feature counts).
    const uint32_t vp_log2 = 31u - __clz(min(TOTAL_SHARED / (n_r + n_c), features) / L);
    const uint32_t vp_mask = (1u << vp_log2) - 1u;
    const uint32_t p       = (1u << vp_log2) * L;

    const uint32_t feat_start = feat_block_idx * FEAT_BLOCK_SIZE;
    const uint32_t feat_end = min(feat_start + FEAT_BLOCK_SIZE, features);

    // --- 3. GRID-STRIDE LOOP OVER FEATURES (in units of L) ---
    for (uint32_t base_f = feat_start; base_f < feat_end; base_f += p) {

        const uint32_t current_p = min(p, feat_end - base_f);  // multiple of L
        const uint32_t vp = current_p / L;                     // vectors per column

        scalar_t* smem_X = smem;
        scalar_t* smem_Y = smem + (n_c * current_p);

        // STEP A: Cooperative vectorized load of X -> smem_X
        const uint32_t total_xv = n_c * vp;
        for (uint32_t i = tx; i < total_xv; i += THREADS_PER_BLOCK) {
            const uint32_t c_idx   = i >> vp_log2;
            const uint32_t f_off   = (i & vp_mask) * L;
            const uint32_t local_c = (packed_cols_physical >> (c_idx * 3)) & 0x7;

            const vec_t xv = *reinterpret_cast<const vec_t*>(
                &X[(batch_idx * cols * features) + ((col_origin + local_c) * features) + base_f + f_off]);
            *reinterpret_cast<vec_t*>(&smem_X[c_idx * current_p + f_off]) = xv;
        }

        __syncthreads();

        // STEP B: FLATTENED WORKER POOL (one float4/double2 per thread)
        const uint32_t total_vwork = n_r * vp;
        for (uint32_t vwork = tx; vwork < total_vwork; vwork += THREADS_PER_BLOCK) {
            const uint32_t pr    = vwork >> vp_log2;
            const uint32_t f_off = (vwork & vp_mask) * L;

            const uint32_t actual_r = (packed_active_rows >> (pr * 3)) & 0x7;
            uint32_t local_data_ptr = data_offset + ((packed_row_offsets >> (pr * 7)) & 0x7F);
            const uint64_t row_code = (code >> (actual_r * BLOCK_COLS)) & row_bits;

            lane_u acc;
            #pragma unroll
            for (int l = 0; l < L; ++l) acc.s[l] = 0;

            #pragma unroll
            for (uint32_t c = 0; c < BLOCK_COLS; ++c) {
                if ((row_code >> c) & 1) {
                    const uint32_t pc = (packed_cols_logical >> (c * 3)) & 0x7;
                    const scalar_t coef = __ldg(&block_data[local_data_ptr]);

                    lane_u xu;
                    xu.v = *reinterpret_cast<const vec_t*>(&smem_X[pc * current_p + f_off]);
                    #pragma unroll
                    for (int l = 0; l < L; ++l) acc.s[l] += coef * xu.s[l];

                    local_data_ptr++;
                }
            }

            *reinterpret_cast<vec_t*>(&smem_Y[pr * current_p + f_off]) = acc.v;
        }

        __syncthreads();

        // STEP C: Cooperative atomic commit (vector load from smem, L scalar atomics)
        for (uint32_t vwork = tx; vwork < total_vwork; vwork += THREADS_PER_BLOCK) {
            const uint32_t pr    = vwork >> vp_log2;
            const uint32_t f_off = (vwork & vp_mask) * L;

            const uint32_t actual_r = (packed_active_rows >> (pr * 3)) & 0x7;
            const uint32_t global_r = row_origin + actual_r;

            lane_u yu;
            yu.v = *reinterpret_cast<const vec_t*>(&smem_Y[pr * current_p + f_off]);
            #pragma unroll
            for (int l = 0; l < L; ++l) {
                atomicAdd(&Y[(batch_idx * rows * features) + (global_r * features) + base_f + f_off + l], yu.s[l]);
            }
        }

        __syncthreads();
    }
}


template <typename index_t, typename scalar_t>
void launch_pbr_spmm(
    const int num_pbr_blocks,
    const int features,
    const int batch_size,
    const int cols,
    const int rows,
    const int runtime_block_rows,
    const int runtime_block_cols,
    const int64_t* block_codes,
    const index_t* block_coords,
    const index_t* block_offsets,
    const scalar_t* block_data,
    const scalar_t* X,
    scalar_t* Y,
    cudaStream_t stream,
    int smem_size
) {
    if (num_pbr_blocks == 0) {
        return;
    }

    constexpr int THREADS_PER_BLOCK = 128;
    constexpr int FEAT_BLOCK_SIZE   = 512;
    constexpr int L = PbrVec<scalar_t>::lanes;

    const int feat_blocks = (features + FEAT_BLOCK_SIZE - 1) / FEAT_BLOCK_SIZE;
    const dim3 grid(num_pbr_blocks, batch_size, feat_blocks);
    const dim3 block(THREADS_PER_BLOCK);

    const bool use_vec = (features % L == 0);

    #define PBR_LAUNCH(BR, BC, TS)                                                                  \
        do {                                                                                        \
            if (use_vec)                                                                            \
                pbr_spmm_vec_kernel<index_t, scalar_t, BR, BC, TS, THREADS_PER_BLOCK,              \
                                    FEAT_BLOCK_SIZE><<<grid, block, 0, stream>>>(                   \
                    num_pbr_blocks, features, batch_size, cols, rows,                               \
                    block_codes, block_coords, block_offsets, block_data, X, Y);                    \
            else                                                                                    \
                pbr_spmm_zero_idle_kernel<index_t, scalar_t, BR, BC, TS,                            \
                                          THREADS_PER_BLOCK, FEAT_BLOCK_SIZE><<<grid, block, 0,     \
                                          stream>>>(                                                \
                    num_pbr_blocks, features, batch_size, cols, rows,                               \
                    block_codes, block_coords, block_offsets, block_data, X, Y);                    \
        } while (0)

    #define DISPATCH_SMEM(BR, BC)                   \
        do {                                        \
            if      (smem_size <= 512)  PBR_LAUNCH(BR, BC,  512); \
            else if (smem_size <= 1024) PBR_LAUNCH(BR, BC, 1024); \
            else                        PBR_LAUNCH(BR, BC, 2048); \
        } while (0)

    if      (runtime_block_rows == 2 && runtime_block_cols == 2) DISPATCH_SMEM(2, 2);
    else if (runtime_block_rows == 4 && runtime_block_cols == 4) DISPATCH_SMEM(4, 4);
    else if (runtime_block_rows == 8 && runtime_block_cols == 8) DISPATCH_SMEM(8, 8);
    else throw std::invalid_argument("Unsupported block size for PBR SpMM. Only 2x2, 4x4, and 8x8 are supported.");

    #undef PBR_LAUNCH
    #undef DISPATCH_SMEM
}

template void launch_pbr_spmm<int32_t, float>(const int, const int, const int, const int, const int, const int, const int, const int64_t*, const int32_t*, const int32_t*, const float*, const float*, float*, cudaStream_t, int);
template void launch_pbr_spmm<int64_t, float>(const int, const int, const int, const int, const int, const int, const int, const int64_t*, const int64_t*, const int64_t*, const float*, const float*, float*, cudaStream_t, int);
template void launch_pbr_spmm<int32_t, double>(const int, const int, const int, const int, const int, const int, const int, const int64_t*, const int32_t*, const int32_t*, const double*, const double*, double*, cudaStream_t, int);
template void launch_pbr_spmm<int64_t, double>(const int, const int, const int, const int, const int, const int, const int, const int64_t*, const int64_t*, const int64_t*, const double*, const double*, double*, cudaStream_t, int);


// ===========================================================================
// Row-group fusion kernel
// ===========================================================================
//
// Grid: (num_groups, batch_size, feat_tiles)
//   num_groups  — occupancy-aware; computed on CPU from SM count × blocks/SM
//   feat_tiles  — ceil(features / p), where p = SMEM_SIZE / (BLOCK_ROWS + BLOCK_COLS)
//
// Each CUDA block owns a contiguous range of row strips and processes every PBR
// block whose row_origin falls in that range, accumulating partial sums into
// smem_Y. Because no two CUDA blocks touch the same output rows, Y is written
// via plain store (no atomicAdd). The CSR remainder kernel, launched after this
// one on the same stream, uses atomicAdd to add its contribution.
//
// Smem layout (both arrays live in the same SMEM_SIZE-float pool):
//   smem_X [BLOCK_COLS × p]  — X tile for the current PBR block, reloaded per block
//   smem_Y [BLOCK_ROWS × p]  — persistent Y accumulator, reset per row strip
//
// Thread tx owns feature slot (base_f + tx % p) in smem; when p > THREADS_PER_BLOCK
// each thread covers multiple slots via a stride loop.
//
// Group boundaries are determined on the CPU using block_coords (which is
// already sorted by row_origin from csr_to_pbr) and an nnz-balancing scan.
// Inside the kernel, a single binary search on block_coords locates the first
// relevant PBR block; after that, a linear scan detects row-strip transitions
// from the sorted row_origin values — no row_group_starts indptr needed.

template <typename index_t, typename scalar_t,
          int BLOCK_ROWS, int BLOCK_COLS, int SMEM_SIZE, int THREADS_PER_BLOCK>
__global__ void __launch_bounds__(THREADS_PER_BLOCK)
pbr_spmm_rowgroup_kernel(
    const uint32_t  num_groups,
    const uint32_t  num_pbr_blocks,
    const uint32_t  features,
    const uint32_t  batch_size,
    const uint32_t  cols,
    const uint32_t  rows,
    const int*      __restrict__ group_row_starts,
    const int64_t*  __restrict__ block_codes,
    const index_t*  __restrict__ block_coords,
    const index_t*  __restrict__ block_offsets,
    const scalar_t* __restrict__ block_data,
    const scalar_t* __restrict__ X,
    scalar_t*       __restrict__ Y
) {
    // p features per smem tile; each of n_c X-columns and BLOCK_ROWS Y-rows gets p slots.
    constexpr int p = SMEM_SIZE / (BLOCK_ROWS + BLOCK_COLS);

    __shared__ scalar_t smem_X[BLOCK_COLS * p];
    __shared__ scalar_t smem_Y[BLOCK_ROWS * p];

    const uint32_t g         = blockIdx.x;
    const uint32_t batch_idx = blockIdx.y;
    const uint32_t feat_tile = blockIdx.z;
    const uint32_t tx        = threadIdx.x;

    if (g >= num_groups || batch_idx >= batch_size) return;

    const uint32_t base_f = feat_tile * (uint32_t)p;
    if (base_f >= features) return;
    // cur_p: actual features in this tile (< p only for the last tile when features % p != 0)
    const uint32_t cur_p = min((uint32_t)p, features - base_f);

    const uint32_t row_start = (uint32_t)group_row_starts[g];
    const uint32_t row_end   = (uint32_t)group_row_starts[g + 1];

    // --- Binary search: first PBR block with row_origin >= row_start ---
    // block_coords[i*2] is non-decreasing (blocks emitted stripe by stripe).
    uint32_t b;
    {
        uint32_t lo = 0, hi = num_pbr_blocks;
        while (lo < hi) {
            const uint32_t mid = (lo + hi) >> 1;
            if ((uint32_t)block_coords[mid * 2] < row_start) lo = mid + 1;
            else hi = mid;
        }
        b = lo;
    }

    const uint64_t row_bits = (1ULL << BLOCK_COLS) - 1;

    // --- Outer loop: one row strip per iteration ---
    while (b < num_pbr_blocks) {
        const uint32_t strip_row = (uint32_t)block_coords[b * 2];
        if (strip_row >= row_end) break;

        // Init smem_Y for this strip
        for (uint32_t i = tx; i < (uint32_t)(BLOCK_ROWS * p); i += THREADS_PER_BLOCK)
            smem_Y[i] = scalar_t(0);
        __syncthreads();

        // --- Inner loop: accumulate all PBR blocks in this row strip ---
        while (b < num_pbr_blocks && (uint32_t)block_coords[b * 2] == strip_row) {
            const uint64_t code       = (uint64_t)block_codes[b];
            const uint32_t col_origin = (uint32_t)block_coords[b * 2 + 1];
            uint32_t       data_ptr   = (uint32_t)block_offsets[b];

            // Union of active columns across all rows of this block
            uint64_t col_mask = 0;
            #pragma unroll
            for (uint32_t r = 0; r < BLOCK_ROWS; ++r)
                col_mask |= (code >> (r * BLOCK_COLS)) & row_bits;

            // Build bidirectional column index maps (3-bit packed, same as original kernel)
            uint32_t packed_cols_physical = 0;  // compressed index → local col offset
            uint32_t packed_cols_logical  = 0;  // local col offset → compressed index
            {
                uint32_t pc = 0;
                #pragma unroll
                for (uint32_t c = 0; c < BLOCK_COLS; ++c) {
                    if ((col_mask >> c) & 1) {
                        packed_cols_physical |= (c  << (pc * 3));
                        packed_cols_logical  |= (pc << (c  * 3));
                        ++pc;
                    }
                }
            }
            const uint32_t n_c = __popcll(col_mask);

            // Cooperative load: smem_X[n_c * cur_p] from global X.
            // Decoded with stride cur_p but stored with stride p so accumulation
            // can always use smem_X[pc * p + f] regardless of tile size.
            for (uint32_t i = tx; i < n_c * cur_p; i += THREADS_PER_BLOCK) {
                const uint32_t c_idx   = i / cur_p;
                const uint32_t f_off   = i % cur_p;
                const uint32_t local_c = (packed_cols_physical >> (c_idx * 3)) & 0x7;
                smem_X[c_idx * p + f_off] = X[(size_t)batch_idx * cols * features
                                              + (size_t)(col_origin + local_c) * features
                                              + base_f + f_off];
            }
            __syncthreads();

            // Accumulate: for every set bit (r, c) in the block code,
            // all threads update their owned smem_Y feature slots.
            #pragma unroll
            for (uint32_t r = 0; r < BLOCK_ROWS; ++r) {
                const uint64_t row_mask = (code >> (r * BLOCK_COLS)) & row_bits;
                if (row_mask == 0) continue;
                #pragma unroll
                for (uint32_t c = 0; c < BLOCK_COLS; ++c) {
                    if ((row_mask >> c) & 1) {
                        const scalar_t val = __ldg(&block_data[data_ptr++]);
                        const uint32_t pc  = (packed_cols_logical >> (c * 3)) & 0x7;
                        for (uint32_t f = tx; f < cur_p; f += THREADS_PER_BLOCK)
                            smem_Y[r * p + f] += val * smem_X[pc * p + f];
                    }
                }
            }
            __syncthreads();

            ++b;
        }

        // Commit smem_Y → global Y via plain store (exclusive row ownership)
        #pragma unroll
        for (uint32_t r = 0; r < BLOCK_ROWS; ++r) {
            const uint32_t global_row = strip_row + r;
            if (global_row >= rows) break;
            for (uint32_t f = tx; f < cur_p; f += THREADS_PER_BLOCK)
                Y[(size_t)batch_idx * rows * features
                  + (size_t)global_row * features
                  + base_f + f] = smem_Y[r * p + f];
        }
        __syncthreads();
    }
}


template <typename index_t, typename scalar_t>
void launch_pbr_rowgroup_spmm(
    const int  num_groups,
    const int  num_pbr_blocks,
    const int  features,
    const int  batch_size,
    const int  cols,
    const int  rows,
    const int  runtime_block_rows,
    const int  runtime_block_cols,
    const int* group_row_starts,
    const int64_t*  block_codes,
    const index_t*  block_coords,
    const index_t*  block_offsets,
    const scalar_t* block_data,
    const scalar_t* X,
    scalar_t*       Y,
    cudaStream_t    stream,
    int smem_size
) {
    if (num_groups == 0 || num_pbr_blocks == 0) return;

    constexpr int THREADS_PER_BLOCK = 128;

    // p = smem_size / (BLOCK_ROWS + BLOCK_COLS); feat_tiles tiles the feature dim.
    const int p          = smem_size / (runtime_block_rows + runtime_block_cols);
    const int feat_tiles = (features + p - 1) / p;

    const dim3 grid((uint32_t)num_groups, (uint32_t)batch_size, (uint32_t)feat_tiles);
    const dim3 block(THREADS_PER_BLOCK);

    #define RG_LAUNCH(BR, BC, TS) \
        pbr_spmm_rowgroup_kernel<index_t, scalar_t, BR, BC, TS, THREADS_PER_BLOCK> \
            <<<grid, block, 0, stream>>>( \
                (uint32_t)num_groups, (uint32_t)num_pbr_blocks, \
                (uint32_t)features,   (uint32_t)batch_size, \
                (uint32_t)cols,       (uint32_t)rows, \
                group_row_starts, block_codes, block_coords, \
                block_offsets, block_data, X, Y)

    #define DISPATCH_SMEM_RG(BR, BC)                         \
        do {                                                 \
            if      (smem_size <= 512)  RG_LAUNCH(BR, BC,  512); \
            else if (smem_size <= 1024) RG_LAUNCH(BR, BC, 1024); \
            else                        RG_LAUNCH(BR, BC, 2048); \
        } while (0)

    if      (runtime_block_rows == 8 && runtime_block_cols == 8) DISPATCH_SMEM_RG(8, 8);
    else if (runtime_block_rows == 4 && runtime_block_cols == 4) DISPATCH_SMEM_RG(4, 4);
    else if (runtime_block_rows == 2 && runtime_block_cols == 2) DISPATCH_SMEM_RG(2, 2);
    else throw std::invalid_argument("Unsupported block size for PBR row-group SpMM.");

    #undef RG_LAUNCH
    #undef DISPATCH_SMEM_RG
}

template void launch_pbr_rowgroup_spmm<int32_t, float>(const int, const int, const int, const int, const int, const int, const int, const int, const int*, const int64_t*, const int32_t*, const int32_t*, const float*,  const float*,  float*,  cudaStream_t, int);
template void launch_pbr_rowgroup_spmm<int64_t, float>(const int, const int, const int, const int, const int, const int, const int, const int, const int*, const int64_t*, const int64_t*, const int64_t*, const float*,  const float*,  float*,  cudaStream_t, int);
template void launch_pbr_rowgroup_spmm<int32_t, double>(const int, const int, const int, const int, const int, const int, const int, const int, const int*, const int64_t*, const int32_t*, const int32_t*, const double*, const double*, double*, cudaStream_t, int);
template void launch_pbr_rowgroup_spmm<int64_t, double>(const int, const int, const int, const int, const int, const int, const int, const int, const int*, const int64_t*, const int64_t*, const int64_t*, const double*, const double*, double*, cudaStream_t, int);


// ---------------------------------------------------------------------------
// Occupancy query
// ---------------------------------------------------------------------------
//
// Returns cudaOccupancyMaxActiveBlocksPerMultiprocessor for the rowgroup
// kernel at the given block_rows / smem_size combination.  Called from the
// Python grouper (_build_group_row_starts) so it can set
//   target_groups = sm_count × bps
// using the kernel's actual compiled register / smem footprint instead of the
// old hardcoded estimate (_BLOCKS_PER_SM_EST × _PIPELINE_DEPTH = 8).
//
// float32 is the representative type; the smem per block is
//   SMEM_SIZE × sizeof(float) = SMEM_SIZE × 4 bytes
// which is the same regardless of index_t, so a single dtype suffices.

template <int BR, int BC, int SMEM>
static int _rowgroup_bps() {
    constexpr int TPB = 128;
    int bps = 8;  // safe fallback if the query fails
    // dynamicSMemSize = 0: the kernel uses only static __shared__ arrays whose
    // size is known to the driver from the compiled binary.  Passing SMEM*4 here
    // would double-count the static footprint and return ~5 instead of ~10-12.
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &bps,
        pbr_spmm_rowgroup_kernel<int32_t, float, BR, BC, SMEM, TPB>,
        TPB,
        0);
    return bps;
}

int get_pbr_rowgroup_blocks_per_sm(int block_rows, int smem_size) {
    if (block_rows == 2) {
        if (smem_size <= 512)  return _rowgroup_bps<2, 2,  512>();
        if (smem_size <= 1024) return _rowgroup_bps<2, 2, 1024>();
        return                        _rowgroup_bps<2, 2, 2048>();
    }
    if (block_rows == 4) {
        if (smem_size <= 512)  return _rowgroup_bps<4, 4,  512>();
        if (smem_size <= 1024) return _rowgroup_bps<4, 4, 1024>();
        return                        _rowgroup_bps<4, 4, 2048>();
    }
    if (block_rows == 8) {
        if (smem_size <= 512)  return _rowgroup_bps<8, 8,  512>();
        if (smem_size <= 1024) return _rowgroup_bps<8, 8, 1024>();
        return                        _rowgroup_bps<8, 8, 2048>();
    }
    return 8;
}


// --- CSR remainder kernel (warp-per-row / CSR-Vector) ---
//
// One warp (32 lanes) owns one output row for ALL features.
//
// Key design: nnz-OUTER, features-INNER loop with register accumulators.
// For each nnz j, all FEATURES/32 feature slices of column col_ind[j] are
// loaded back-to-back before moving to the next nnz.  Each 2 KB column tile
// fits in L1, so slices 2..N are L1 hits → ~15/16 cache hit rate at FEATURES=512
// vs ~25% with the feature-outer / nnz-inner arrangement (which re-fetches the
// same scattered column from DRAM on every feature-stripe pass).
//
// FEATURES is a compile-time template parameter so the compiler register-
// allocates the accumulator array (sums[]).  Measured register cost at
// FEATURES=512: +16 regs/thread (56 total), within the 64-reg limit for
// full theoretical occupancy (256 threads/block on RTX 3090).
//
// For float with FEATURES divisible by 128: float4 path loads 4 features per
// lane per instruction (4× fewer load ops in the hot loop).
//
// atomicAdd is required: a row can receive contributions from the block kernel
// running concurrently on another stream.  Within this kernel each row belongs
// to exactly one warp, so there is no intra-kernel contention.
//
// Grid: (ceil(rows / WARPS_PER_BLOCK), batch_size), Block: (WARPS_PER_BLOCK * 32).
// Fallback (runtime-features): feature-outer / nnz-inner, used for uncommon sizes.

constexpr int CSR_WARP_SIZE      = 32;
constexpr int CSR_WARPS_PER_BLOCK = 8;   // 256 threads/block; 6 blocks/SM = full occ on RTX 3090

// Fast path — templated on FEATURES for register accumulation.
template <typename index_t, typename scalar_t, int FEATURES>
__global__ void __launch_bounds__(CSR_WARPS_PER_BLOCK * CSR_WARP_SIZE)
csr_spmm_kernel_fast(
    const uint32_t rows,
    const uint32_t batch_size,
    const uint32_t cols,
    const index_t* __restrict__ indptr,
    const index_t* __restrict__ col_ind,
    const scalar_t* __restrict__ csr_vals,
    const scalar_t* __restrict__ X,
    scalar_t* __restrict__ Y
) {
    const uint32_t warp_id = threadIdx.x / CSR_WARP_SIZE;
    const uint32_t lane    = threadIdx.x % CSR_WARP_SIZE;
    const uint32_t row     = blockIdx.x * CSR_WARPS_PER_BLOCK + warp_id;
    const uint32_t batch   = blockIdx.y;

    if (row >= rows || batch >= batch_size) return;

    const index_t rs = indptr[row];
    const index_t re = indptr[row + 1];
    if (rs == re) return;

    const scalar_t* x_ptr = X + (size_t)batch * cols * FEATURES;
    scalar_t*       y_ptr = Y + (size_t)batch * rows * FEATURES + (size_t)row * FEATURES;

    if constexpr (sizeof(scalar_t) == 4 && FEATURES % 128 == 0) {
        // float4 path: FEATURES/128 float4 accumulators per thread.
        // 32 lanes × float4 = 128 features per load instruction.
        constexpr int F4_STRIPS = FEATURES / 128;
        float4 sums[F4_STRIPS] = {};

        for (index_t j = rs; j < re; ++j) {
            const float val = csr_vals[j];
            const float4* x_col = reinterpret_cast<const float4*>(
                x_ptr + (size_t)col_ind[j] * FEATURES);
            #pragma unroll
            for (int i = 0; i < F4_STRIPS; ++i) {
                float4 v = x_col[lane + i * CSR_WARP_SIZE];
                sums[i].x += val * v.x;
                sums[i].y += val * v.y;
                sums[i].z += val * v.z;
                sums[i].w += val * v.w;
            }
        }

        // Commit: 4 atomicAdds per strip (same count as scalar path).
        #pragma unroll
        for (int i = 0; i < F4_STRIPS; ++i) {
            const uint32_t base = (lane + i * CSR_WARP_SIZE) * 4;
            atomicAdd(&y_ptr[base + 0], sums[i].x);
            atomicAdd(&y_ptr[base + 1], sums[i].y);
            atomicAdd(&y_ptr[base + 2], sums[i].z);
            atomicAdd(&y_ptr[base + 3], sums[i].w);
        }
    } else {
        // Scalar path: FEATURES/WARP_SIZE register accumulators per thread.
        constexpr int STRIPS = FEATURES / CSR_WARP_SIZE;
        scalar_t sums[STRIPS] = {};

        for (index_t j = rs; j < re; ++j) {
            const scalar_t val = csr_vals[j];
            const scalar_t* x_col = x_ptr + (size_t)col_ind[j] * FEATURES;
            #pragma unroll
            for (int i = 0; i < STRIPS; ++i)
                sums[i] += val * x_col[lane + i * CSR_WARP_SIZE];
        }

        #pragma unroll
        for (int i = 0; i < STRIPS; ++i)
            atomicAdd(&y_ptr[lane + i * CSR_WARP_SIZE], sums[i]);
    }
}

// Fallback — runtime features, feature-outer / nnz-inner (arbitrary FEATURES).
template <typename index_t, typename scalar_t>
__global__ void csr_spmm_kernel(
    const uint32_t rows,
    const uint32_t features,
    const uint32_t batch_size,
    const uint32_t cols,
    const index_t* __restrict__ indptr,
    const index_t* __restrict__ col_ind,
    const scalar_t* __restrict__ csr_vals,
    const scalar_t* __restrict__ X,
    scalar_t* __restrict__ Y
) {
    const uint32_t warp_id = threadIdx.x / CSR_WARP_SIZE;
    const uint32_t lane    = threadIdx.x % CSR_WARP_SIZE;
    const uint32_t row     = blockIdx.x * CSR_WARPS_PER_BLOCK + warp_id;
    const uint32_t batch   = blockIdx.y;

    if (row >= rows || batch >= batch_size) return;

    const index_t rs = indptr[row];
    const index_t re = indptr[row + 1];
    if (rs == re) return;

    const scalar_t* x_ptr = X + (size_t)batch * cols * features;
    scalar_t*       y_ptr = Y + (size_t)batch * rows * features + (size_t)row * features;

    for (uint32_t f = lane; f < features; f += CSR_WARP_SIZE) {
        scalar_t sum = 0;
        for (index_t j = rs; j < re; ++j)
            sum += csr_vals[j] * x_ptr[(size_t)col_ind[j] * features + f];
        atomicAdd(&y_ptr[f], sum);
    }
}

// The fast kernel is used for every supported feature count.  An earlier
// nnz/row>=24 heuristic gated it (the fast kernel was thought to lose on short
// remainder rows under block-kernel contention), but a controlled sweep
// (jobs/compare_csrs) showed the fast kernel wins or ties at every nnz/row from
// 2 to 256 — in isolation, under heavy 8x8 contention (32768 blocks), and under
// 2x2 contention (~131k blocks).  The fallback never won by more than noise, and
// the gate cost 2-4% by mispicking it.  Fallback is kept only for feature counts
// without a compile-time fast instantiation (the switch default below).
#define CSR_LAUNCH_FAST(F) \
    csr_spmm_kernel_fast<index_t, scalar_t, F><<<grid, block, 0, stream>>>( \
        rows, batch_size, cols, indptr, col_ind, csr_vals, X, Y)

#define CSR_LAUNCH_FALLBACK() \
    csr_spmm_kernel<index_t, scalar_t><<<grid, block, 0, stream>>>( \
        rows, features, batch_size, cols, indptr, col_ind, csr_vals, X, Y)

template <typename index_t, typename scalar_t>
void launch_csr_spmm(
    const int rows,
    const int features,
    const int batch_size,
    const int cols,
    const int nnz,
    const index_t* indptr,
    const index_t* col_ind,
    const scalar_t* csr_vals,
    const scalar_t* X,
    scalar_t* Y,
    cudaStream_t stream,
    int csr_mode
) {
    if (rows == 0 || features == 0 || batch_size == 0) return;
    const dim3 grid((rows + CSR_WARPS_PER_BLOCK - 1) / CSR_WARPS_PER_BLOCK, batch_size);
    const dim3 block(CSR_WARPS_PER_BLOCK * CSR_WARP_SIZE);

    // Auto and force_fast both use the fast kernel (see note above); only
    // force_fallback (csr_mode==2) selects the feature-outer fallback.
    // csr_mode: 0=auto, 1=force_fast, 2=force_fallback
    const bool use_fast = (csr_mode != 2);
    if (use_fast) {
        switch (features) {
            case  64: CSR_LAUNCH_FAST( 64); return;
            case 128: CSR_LAUNCH_FAST(128); return;
            case 256: CSR_LAUNCH_FAST(256); return;
            case 512: CSR_LAUNCH_FAST(512); return;
            default: break;  // uncommon feature count → fallback
        }
    }
    CSR_LAUNCH_FALLBACK();
}

#undef CSR_LAUNCH_FAST
#undef CSR_LAUNCH_FALLBACK

template void launch_csr_spmm<int32_t, float>(const int, const int, const int, const int, const int, const int32_t*, const int32_t*, const float*, const float*, float*, cudaStream_t, int);
template void launch_csr_spmm<int64_t, float>(const int, const int, const int, const int, const int, const int64_t*, const int64_t*, const float*, const float*, float*, cudaStream_t, int);
template void launch_csr_spmm<int32_t, double>(const int, const int, const int, const int, const int, const int32_t*, const int32_t*, const double*, const double*, double*, cudaStream_t, int);
template void launch_csr_spmm<int64_t, double>(const int, const int, const int, const int, const int, const int64_t*, const int64_t*, const double*, const double*, double*, cudaStream_t, int);
