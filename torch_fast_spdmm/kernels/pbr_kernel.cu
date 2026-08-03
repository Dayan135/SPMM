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
// Chunkcols kernel — bounded per-thread row-chunking (experimental)
// ===========================================================================
//
// Step B of pbr_spmm_zero_idle_kernel assigns one thread to a whole row's
// worth of active columns (up to BLOCK_COLS), so a block mixing dense and
// sparse rows lets some threads finish almost instantly while others do up
// to BLOCK_COLS multiply-adds -- a per-thread imbalance bounded only by
// BLOCK_COLS. This kernel caps that at CHUNK_K: a row with more than CHUNK_K
// active columns is split into ceil(row_nnz/CHUNK_K) chunks, each processed
// by its own thread. Chunks of the same row now write the same (row,
// feature) smem_Y slot, so — unlike zero_idle's single plain store — the
// Step B write must become atomicAdd, and smem_Y needs an explicit zero +
// sync before Step B (same pattern as the removed rowgroup kernel's
// persistent-accumulator flush).
//
// Rows are NOT compacted to n_r active slots here (unlike zero_idle) --
// smem_Y is sized BLOCK_ROWS*p and indexed by physical row directly. Simpler
// to get right for this first pass; costs a little smem headroom on
// row-sparse blocks (smaller p) versus zero_idle's compaction.
//
// Scalar-only (no vectorized/float4 variant) -- matches this project's
// convention for prior experimental kernel variants (rowgroup/colgroup).

template <typename index_t, typename scalar_t,
          int BLOCK_ROWS, int BLOCK_COLS,
          int TOTAL_SHARED, int THREADS_PER_BLOCK, int FEAT_BLOCK_SIZE>
__global__ void __launch_bounds__(THREADS_PER_BLOCK, 16) pbr_spmm_chunkcols_kernel(
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
    constexpr int CHUNK_K = 4;
    constexpr int MAX_CHUNKS = BLOCK_ROWS * ((BLOCK_COLS + CHUNK_K - 1) / CHUNK_K);

    __shared__ scalar_t smem[TOTAL_SHARED];

    const uint32_t b_idx = blockIdx.x;
    const uint32_t batch_idx = blockIdx.y;
    const uint32_t feat_block_idx = blockIdx.z;
    const uint32_t tx = threadIdx.x;

    if (b_idx >= num_pbr_blocks || batch_idx >= batch_size) return;

    const uint64_t code = static_cast<uint64_t>(block_codes[b_idx]);
    const uint32_t row_origin = block_coords[b_idx * 2];
    const uint32_t col_origin = block_coords[b_idx * 2 + 1];
    const uint32_t data_offset = block_offsets[b_idx];

    // --- 1. SPARSITY SCAN: per-row data-start offsets + chunk table ---
    const uint64_t row_bits = (1ULL << BLOCK_COLS) - 1;

    uint32_t row_start_offset[BLOCK_ROWS];
    uint8_t  chunk_row[MAX_CHUNKS];
    uint8_t  chunk_skip[MAX_CHUNKS];
    uint8_t  chunk_take[MAX_CHUNKS];
    uint32_t n_chunks = 0;
    uint64_t col_mask = 0;
    uint32_t current_relative_offset = 0;

    #pragma unroll
    for (uint32_t r = 0; r < BLOCK_ROWS; ++r) {
        const uint64_t row_mask = (code >> (r * BLOCK_COLS)) & row_bits;
        row_start_offset[r] = current_relative_offset;
        if (row_mask > 0) {
            col_mask |= row_mask;
            const uint32_t row_nnz = __popcll(row_mask);
            for (uint32_t skip = 0; skip < row_nnz; skip += CHUNK_K) {
                const uint32_t take = min((uint32_t)CHUNK_K, row_nnz - skip);
                chunk_row[n_chunks]  = (uint8_t)r;
                chunk_skip[n_chunks] = (uint8_t)skip;
                chunk_take[n_chunks] = (uint8_t)take;
                n_chunks++;
            }
            current_relative_offset += row_nnz;
        }
    }

    const uint32_t n_c = __popcll(col_mask);
    if (n_chunks == 0 || n_c == 0) return;

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

    // --- 2. DYNAMIC P CALCULATION --- (rows not compacted -> denominator uses BLOCK_ROWS)
    const uint32_t p_log2 = 31u - __clz(min(TOTAL_SHARED / (BLOCK_ROWS + n_c), features));
    const uint32_t p      = 1u << p_log2;
    const uint32_t p_mask = p - 1u;

    const uint32_t feat_start = feat_block_idx * FEAT_BLOCK_SIZE;
    const uint32_t feat_end = min(feat_start + FEAT_BLOCK_SIZE, features);

    for (uint32_t base_f = feat_start; base_f < feat_end; base_f += p) {
        const uint32_t current_p = min(p, feat_end - base_f);

        scalar_t* smem_X = smem;
        scalar_t* smem_Y = smem + (n_c * current_p);

        // STEP A: Cooperative Load (identical to zero_idle kernel)
        const uint32_t total_x_elements = n_c * current_p;
        for (uint32_t i = tx; i < total_x_elements; i += THREADS_PER_BLOCK) {
            const uint32_t c_idx   = i >> p_log2;
            const uint32_t f_offset = i & p_mask;
            const uint32_t local_c = (packed_cols_physical >> (c_idx * 3)) & 0x7;
            smem_X[i] = X[(batch_idx * cols * features) + ((col_origin + local_c) * features) + base_f + f_offset];
        }

        // Zero smem_Y -- Step B accumulates via atomicAdd (multiple chunks of
        // the same row target the same slot), unlike zero_idle's single store.
        for (uint32_t i = tx; i < (uint32_t)(BLOCK_ROWS * current_p); i += THREADS_PER_BLOCK)
            smem_Y[i] = scalar_t(0);

        __syncthreads();

        // STEP B: one thread per (chunk, feature) work-item
        const uint32_t total_work = n_chunks * current_p;

        for (uint32_t work_idx = tx; work_idx < total_work; work_idx += THREADS_PER_BLOCK) {
            const uint32_t chunk_idx = work_idx >> p_log2;
            const uint32_t f_in_s    = work_idx & p_mask;

            const uint32_t actual_r = chunk_row[chunk_idx];
            const uint32_t skip     = chunk_skip[chunk_idx];
            const uint32_t take     = chunk_take[chunk_idx];
            const uint64_t row_code = (code >> (actual_r * BLOCK_COLS)) & row_bits;

            uint32_t local_data_ptr = data_offset + row_start_offset[actual_r];
            uint32_t idx_in_row = 0;
            scalar_t sum = 0;

            #pragma unroll
            for (uint32_t c = 0; c < BLOCK_COLS; ++c) {
                if ((row_code >> c) & 1) {
                    if (idx_in_row >= skip && idx_in_row < skip + take) {
                        const uint32_t pc = (packed_cols_logical >> (c * 3)) & 0x7;
                        sum += __ldg(&block_data[local_data_ptr]) * smem_X[pc * current_p + f_in_s];
                    }
                    local_data_ptr++;
                    idx_in_row++;
                }
            }

            atomicAdd(&smem_Y[actual_r * current_p + f_in_s], sum);
        }

        __syncthreads();

        // STEP C: Cooperative Atomic Commit -- skip rows with no data
        const uint32_t total_flush = BLOCK_ROWS * current_p;
        for (uint32_t work_idx = tx; work_idx < total_flush; work_idx += THREADS_PER_BLOCK) {
            const uint32_t r      = work_idx >> p_log2;
            const uint32_t f_in_s = work_idx & p_mask;

            const uint64_t row_mask = (code >> (r * BLOCK_COLS)) & row_bits;
            if (row_mask == 0) continue;

            const uint32_t global_r = row_origin + r;
            atomicAdd(&Y[(batch_idx * rows * features) + (global_r * features) + base_f + f_in_s], smem_Y[work_idx]);
        }

        __syncthreads();
    }
}


template <typename index_t, typename scalar_t>
void launch_pbr_chunkcols_spmm(
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

    const int feat_blocks = (features + FEAT_BLOCK_SIZE - 1) / FEAT_BLOCK_SIZE;
    const dim3 grid(num_pbr_blocks, batch_size, feat_blocks);
    const dim3 block(THREADS_PER_BLOCK);

    #define CHUNKCOLS_LAUNCH(BR, BC, TS)                                                            \
        pbr_spmm_chunkcols_kernel<index_t, scalar_t, BR, BC, TS, THREADS_PER_BLOCK,                 \
                                  FEAT_BLOCK_SIZE><<<grid, block, 0, stream>>>(                      \
            (uint32_t)num_pbr_blocks, (uint32_t)features, (uint32_t)batch_size,                     \
            (uint32_t)cols, (uint32_t)rows,                                                         \
            block_codes, block_coords, block_offsets, block_data, X, Y)

    #define DISPATCH_SMEM_CHUNKCOLS(BR, BC)                              \
        do {                                                            \
            if      (smem_size <= 512)  CHUNKCOLS_LAUNCH(BR, BC,  512); \
            else if (smem_size <= 1024) CHUNKCOLS_LAUNCH(BR, BC, 1024); \
            else                        CHUNKCOLS_LAUNCH(BR, BC, 2048); \
        } while (0)

    if      (runtime_block_rows == 2 && runtime_block_cols == 2) DISPATCH_SMEM_CHUNKCOLS(2, 2);
    else if (runtime_block_rows == 4 && runtime_block_cols == 4) DISPATCH_SMEM_CHUNKCOLS(4, 4);
    else if (runtime_block_rows == 8 && runtime_block_cols == 8) DISPATCH_SMEM_CHUNKCOLS(8, 8);
    else throw std::invalid_argument("Unsupported block size for PBR chunkcols SpMM. Only 2x2, 4x4, and 8x8 are supported.");

    #undef CHUNKCOLS_LAUNCH
    #undef DISPATCH_SMEM_CHUNKCOLS
}

template void launch_pbr_chunkcols_spmm<int32_t, float>(const int, const int, const int, const int, const int, const int, const int, const int64_t*, const int32_t*, const int32_t*, const float*, const float*, float*, cudaStream_t, int);
template void launch_pbr_chunkcols_spmm<int64_t, float>(const int, const int, const int, const int, const int, const int, const int, const int64_t*, const int64_t*, const int64_t*, const float*, const float*, float*, cudaStream_t, int);
template void launch_pbr_chunkcols_spmm<int32_t, double>(const int, const int, const int, const int, const int, const int, const int, const int64_t*, const int32_t*, const int32_t*, const double*, const double*, double*, cudaStream_t, int);
template void launch_pbr_chunkcols_spmm<int64_t, double>(const int, const int, const int, const int, const int, const int, const int, const int64_t*, const int64_t*, const int64_t*, const double*, const double*, double*, cudaStream_t, int);


// ===========================================================================
// Nnzflat kernel — full nonzero-level flattening, column-sorted (experimental)
// ===========================================================================
//
// Where chunkcols bounds per-thread imbalance at CHUNK_K columns, this kernel
// removes it entirely: one thread per (nonzero, feature) work-item instead of
// one thread per row. Nonzeros are visited in COLUMN-major order (all rows
// sharing a column together) rather than the block's native row-major
// storage order, so nearby threads read the same smem_X column -- the
// locality property colgroup got from a host-side col-sort, reproduced here
// per-block with no host-side reordering. Every nonzero-feature pair gets
// its own atomicAdd into smem_Y -- a much larger atomic count than
// chunkcols. This is the direct test of whether finer-grained balance is
// worth its atomic cost: STALL_REASONS_NOATOMIC_ABLATION.md and
// STALL_REASONS_ROWGROUP_ATOMIC.md (prior investigation, see git history)
// found atomics already a proven major cost in this kernel family, and every
// attempt so far to reduce atomic frequency by coarsening (rowgroup) lost to
// an occupancy tax instead -- this kernel moves the opposite direction
// (finer atomics, no coarsening), so it's an open question, not an assumed
// win.
//
// Rows are NOT compacted (same simplification as chunkcols) -- smem_Y is
// BLOCK_ROWS*p, indexed by physical row.
//
// An earlier version of this kernel precomputed a per-block column-major
// "nonzero list" (nz_row[]/nz_ptr[]/nz_pc[], sized BLOCK_ROWS*BLOCK_COLS)
// once, then looked it up by nz_idx in Step B. Since nz_idx is a runtime
// value, those array reads couldn't be register-allocated and were
// confirmed (via ncu smsp__sass_inst_executed_op_local_{ld,st}.sum) to spill
// to local memory on every access, in every tested config -- up to 1.15B
// spill instructions/launch on dense matrices, with zero such spill in the
// `block` kernel family for comparison. This version replaces the array
// with an equivalent on-the-fly derivation, computed independently per
// (nz_idx, feature-tile-pass) access, entirely from `code` (already
// register-resident) via bounded #pragma-unrolled rank/select loops over
// BLOCK_COLS x BLOCK_ROWS candidates -- no addressable array exists, so no
// spill is possible. Trades the guaranteed local-memory round-trip for
// bounded ALU work; net effect to be measured empirically, not assumed.
//
// Scalar-only, no vectorized variant (see chunkcols).

template <typename index_t, typename scalar_t,
          int BLOCK_ROWS, int BLOCK_COLS,
          int TOTAL_SHARED, int THREADS_PER_BLOCK, int FEAT_BLOCK_SIZE>
__global__ void __launch_bounds__(THREADS_PER_BLOCK, 16) pbr_spmm_nnzflat_kernel(
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

    const uint64_t code = static_cast<uint64_t>(block_codes[b_idx]);
    const uint32_t row_origin = block_coords[b_idx * 2];
    const uint32_t col_origin = block_coords[b_idx * 2 + 1];
    const uint32_t data_offset = block_offsets[b_idx];

    // --- 1. SPARSITY SCAN: col_mask only (register scalar, no per-row array) ---
    const uint64_t row_bits = (1ULL << BLOCK_COLS) - 1;

    uint64_t col_mask = 0;
    #pragma unroll
    for (uint32_t r = 0; r < BLOCK_ROWS; ++r) {
        col_mask |= (code >> (r * BLOCK_COLS)) & row_bits;
    }

    const uint32_t n_c = __popcll(col_mask);
    if (n_c == 0) return;

    uint32_t packed_cols_physical = 0;
    uint32_t pc_idx = 0;

    #pragma unroll
    for (uint32_t c = 0; c < BLOCK_COLS; ++c) {
        if ((col_mask >> c) & 1) {
            packed_cols_physical |= (c << (pc_idx * 3));
            pc_idx++;
        }
    }

    // n_nz = total set bits in `code` = total nonzeros in the block (every
    // set bit is exactly one nonzero, regardless of enumeration order).
    const uint32_t n_nz = __popcll(code);

    // --- 1c. STATIC PER-THREAD nz_idx ASSIGNMENT, derived ONCE per launch ---
    // The first version of this rewrite (still deriving inline inside Step B's
    // flat (nz_idx, feature) work-item loop) eliminated the array/spill, but
    // regressed 56-187% across every tested config: nz_idx = work_idx >>
    // p_log2 depends only on tx/p_log2, not on base_f, so the SAME derivation
    // was being redone on every feature-tile pass -- the fix traded a
    // memory-bound bottleneck for a worse compute-bound one (ncu confirmed:
    // long_scoreboard_pct dropped 90-98% as designed, but math_pipe_throttle_pct
    // rose 700-5000% and became the new dominant cost).
    //
    // Since MAX_NZ = BLOCK_ROWS*BLOCK_COLS <= THREADS_PER_BLOCK for every
    // config this project uses (checked below), assigning nz_idx = tx (one
    // fixed nonzero per thread, for the WHOLE kernel invocation, computed
    // once here instead of once per work-item per pass) needs at most one
    // scalar set per thread -- still no array. block_data's value is also
    // hoisted here for the same reason (it doesn't depend on base_f either,
    // same redundant-reread issue the block_data-cache experiment targeted,
    // fixed here for free as a register instead of a shared-memory cache).
    // Step B below sequentially loops this thread's fixed nonzero over all
    // of current_p's features instead of splitting across threads --
    // trading Step B's thread-level parallelism (was up to THREADS_PER_BLOCK-wide,
    // now n_nz-wide) for zero re-derivation cost across passes. Net effect
    // to be measured empirically, not assumed.
    static_assert(BLOCK_ROWS * BLOCK_COLS <= THREADS_PER_BLOCK,
                  "static nz_idx=tx assignment requires MAX_NZ <= THREADS_PER_BLOCK");

    const bool has_nz = (tx < n_nz);
    uint32_t actual_r = 0, pc = 0;
    scalar_t val = scalar_t(0);

    if (has_nz) {
        uint32_t remaining = tx;
        uint32_t physical_c = 0;
        bool col_found = false;

        #pragma unroll
        for (uint32_t cand_pc = 0; cand_pc < BLOCK_COLS; ++cand_pc) {
            if (cand_pc < n_c && !col_found) {
                const uint32_t cand_c = (packed_cols_physical >> (cand_pc * 3)) & 0x7;

                uint32_t col_rows = 0;
                #pragma unroll
                for (uint32_t r = 0; r < BLOCK_ROWS; ++r)
                    col_rows |= ((uint32_t)((code >> (r * BLOCK_COLS + cand_c)) & 1ULL)) << r;

                const uint32_t cnt = __popc(col_rows);
                if (remaining < cnt) {
                    pc = cand_pc;
                    physical_c = cand_c;
                    col_found = true;

                    uint32_t rk = remaining;
                    bool row_found = false;
                    #pragma unroll
                    for (uint32_t r = 0; r < BLOCK_ROWS; ++r) {
                        if (!row_found && ((col_rows >> r) & 1u)) {
                            if (rk == 0) { actual_r = r; row_found = true; }
                            else { rk--; }
                        }
                    }
                } else {
                    remaining -= cnt;
                }
            }
        }

        const uint32_t local_data_ptr = data_offset +
            __popcll(code & ((1ULL << (actual_r * BLOCK_COLS + physical_c)) - 1));
        val = __ldg(&block_data[local_data_ptr]);
    }

    // --- 2. DYNAMIC P CALCULATION ---
    const uint32_t p_log2 = 31u - __clz(min(TOTAL_SHARED / (BLOCK_ROWS + n_c), features));
    const uint32_t p      = 1u << p_log2;
    const uint32_t p_mask = p - 1u;

    const uint32_t feat_start = feat_block_idx * FEAT_BLOCK_SIZE;
    const uint32_t feat_end = min(feat_start + FEAT_BLOCK_SIZE, features);

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

        for (uint32_t i = tx; i < (uint32_t)(BLOCK_ROWS * current_p); i += THREADS_PER_BLOCK)
            smem_Y[i] = scalar_t(0);

        __syncthreads();

        // STEP B: each thread with has_nz sequentially covers all current_p
        // features for its OWN fixed nonzero (actual_r/pc/val, derived once
        // above -- no re-derivation, no array). Parallelism here is n_nz-wide
        // (across threads) x current_p-wide (sequential per thread), instead
        // of the flat THREADS_PER_BLOCK-wide (nz, feature) split used before.
        if (has_nz) {
            for (uint32_t f_in_s = 0; f_in_s < current_p; ++f_in_s) {
                const scalar_t contribution = val * smem_X[pc * current_p + f_in_s];
                atomicAdd(&smem_Y[actual_r * current_p + f_in_s], contribution);
            }
        }

        __syncthreads();

        // STEP C: Cooperative Atomic Commit -- skip rows with no data
        const uint32_t total_flush = BLOCK_ROWS * current_p;
        for (uint32_t work_idx = tx; work_idx < total_flush; work_idx += THREADS_PER_BLOCK) {
            const uint32_t r      = work_idx >> p_log2;
            const uint32_t f_in_s = work_idx & p_mask;

            const uint64_t row_mask = (code >> (r * BLOCK_COLS)) & row_bits;
            if (row_mask == 0) continue;

            const uint32_t global_r = row_origin + r;
            atomicAdd(&Y[(batch_idx * rows * features) + (global_r * features) + base_f + f_in_s], smem_Y[work_idx]);
        }

        __syncthreads();
    }
}


template <typename index_t, typename scalar_t>
void launch_pbr_nnzflat_spmm(
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

    const int feat_blocks = (features + FEAT_BLOCK_SIZE - 1) / FEAT_BLOCK_SIZE;
    const dim3 grid(num_pbr_blocks, batch_size, feat_blocks);
    const dim3 block(THREADS_PER_BLOCK);

    #define NNZFLAT_LAUNCH(BR, BC, TS)                                                             \
        pbr_spmm_nnzflat_kernel<index_t, scalar_t, BR, BC, TS, THREADS_PER_BLOCK,                   \
                                FEAT_BLOCK_SIZE><<<grid, block, 0, stream>>>(                        \
            (uint32_t)num_pbr_blocks, (uint32_t)features, (uint32_t)batch_size,                     \
            (uint32_t)cols, (uint32_t)rows,                                                         \
            block_codes, block_coords, block_offsets, block_data, X, Y)

    #define DISPATCH_SMEM_NNZFLAT(BR, BC)                              \
        do {                                                          \
            if      (smem_size <= 512)  NNZFLAT_LAUNCH(BR, BC,  512); \
            else if (smem_size <= 1024) NNZFLAT_LAUNCH(BR, BC, 1024); \
            else                        NNZFLAT_LAUNCH(BR, BC, 2048); \
        } while (0)

    if      (runtime_block_rows == 2 && runtime_block_cols == 2) DISPATCH_SMEM_NNZFLAT(2, 2);
    else if (runtime_block_rows == 4 && runtime_block_cols == 4) DISPATCH_SMEM_NNZFLAT(4, 4);
    else if (runtime_block_rows == 8 && runtime_block_cols == 8) DISPATCH_SMEM_NNZFLAT(8, 8);
    else throw std::invalid_argument("Unsupported block size for PBR nnzflat SpMM. Only 2x2, 4x4, and 8x8 are supported.");

    #undef NNZFLAT_LAUNCH
    #undef DISPATCH_SMEM_NNZFLAT
}

template void launch_pbr_nnzflat_spmm<int32_t, float>(const int, const int, const int, const int, const int, const int, const int, const int64_t*, const int32_t*, const int32_t*, const float*, const float*, float*, cudaStream_t, int);
template void launch_pbr_nnzflat_spmm<int64_t, float>(const int, const int, const int, const int, const int, const int, const int, const int64_t*, const int64_t*, const int64_t*, const float*, const float*, float*, cudaStream_t, int);
template void launch_pbr_nnzflat_spmm<int32_t, double>(const int, const int, const int, const int, const int, const int, const int, const int64_t*, const int32_t*, const int32_t*, const double*, const double*, double*, cudaStream_t, int);
template void launch_pbr_nnzflat_spmm<int64_t, double>(const int, const int, const int, const int, const int, const int, const int, const int64_t*, const int64_t*, const int64_t*, const double*, const double*, double*, cudaStream_t, int);


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
