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
    // Divides by n_c alone (not n_r+n_c): smem holds only smem_X now (see Step B/C
    // merge below), so the whole TOTAL_SHARED buffer is available to it.
    const uint32_t p_log2 = 31u - __clz(min(TOTAL_SHARED / n_c, features));
    const uint32_t p      = 1u << p_log2;
    const uint32_t p_mask = p - 1u;

    const uint32_t feat_start = feat_block_idx * FEAT_BLOCK_SIZE;
    const uint32_t feat_end = min(feat_start + FEAT_BLOCK_SIZE, features);

    // --- 3. GRID-STRIDE LOOP OVER FEATURES ---
    for (uint32_t base_f = feat_start; base_f < feat_end; base_f += p) {

        const uint32_t current_p = min(p, feat_end - base_f);

        scalar_t* smem_X = smem;

        // STEP A: Cooperative Load
        const uint32_t total_x_elements = n_c * current_p;
        for (uint32_t i = tx; i < total_x_elements; i += THREADS_PER_BLOCK) {
            const uint32_t c_idx   = i >> p_log2;
            const uint32_t f_offset = i & p_mask;

            const uint32_t local_c = (packed_cols_physical >> (c_idx * 3)) & 0x7;
            smem_X[i] = X[(batch_idx * cols * features) + ((col_origin + local_c) * features) + base_f + f_offset];
        }

        __syncthreads();

        // STEP B+C (merged): each thread computes its own (row, feature) sum and
        // atomicAdds it directly from the register -- no smem_Y round trip. Step
        // B/C used to be split via smem_Y, but both loops visit the identical
        // work_idx sequence per thread (same thread writes and reads its own
        // slot, no cross-thread dependency), so the intermediate store/reload
        // and the syncthreads between them were unnecessary.
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

            const uint32_t global_r = row_origin + actual_r;
            atomicAdd(&Y[(batch_idx * rows * features) + (global_r * features) + base_f + f_in_s], sum);
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
    // Divides by n_c alone (not n_r+n_c): smem holds only smem_X now (see Step B/C
    // merge below), so the whole TOTAL_SHARED buffer is available to it.
    const uint32_t vp_log2 = 31u - __clz(min(TOTAL_SHARED / n_c, features) / L);
    const uint32_t vp_mask = (1u << vp_log2) - 1u;
    const uint32_t p       = (1u << vp_log2) * L;

    const uint32_t feat_start = feat_block_idx * FEAT_BLOCK_SIZE;
    const uint32_t feat_end = min(feat_start + FEAT_BLOCK_SIZE, features);

    // --- 3. GRID-STRIDE LOOP OVER FEATURES (in units of L) ---
    for (uint32_t base_f = feat_start; base_f < feat_end; base_f += p) {

        const uint32_t current_p = min(p, feat_end - base_f);  // multiple of L
        const uint32_t vp = current_p / L;                     // vectors per column

        scalar_t* smem_X = smem;

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

        // STEP B+C (merged): each thread computes its own float4/double2 sum and
        // atomicAdds it directly from the register -- no smem_Y round trip. Step
        // B/C used to be split via smem_Y, but both loops visit the identical
        // vwork sequence per thread (same thread writes and reads its own slot,
        // no cross-thread dependency), so the intermediate store/reload and the
        // syncthreads between them were unnecessary.
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

            const uint32_t global_r = row_origin + actual_r;
            #pragma unroll
            for (int l = 0; l < L; ++l) {
                atomicAdd(&Y[(batch_idx * rows * features) + (global_r * features) + base_f + f_off + l], acc.s[l]);
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
//   num_groups — occupancy-aware; sm_count × blocks_per_SM (from occupancy query)
//   feat_tiles — ceil(features / p), where p = SMEM_SIZE / (BLOCK_ROWS + BLOCK_COLS)
//
// Partitions PBR blocks by a host-precomputed group_offsets array (CSR-style,
// size num_groups+1): group g processes PBR blocks
//   [group_offsets[g], group_offsets[g+1])
// group_offsets is built on the host by _build_group_offsets (__init__.py) --
// either count-balanced (grouping='count', reproduces the original
// group_idx*group_size arithmetic exactly) or nnz-balanced (grouping='nnz',
// boundaries chosen so each group covers ~equal cumulative nnz via searchsorted
// over block_offsets -- see WHY_BLOCK_KERNEL_FAILS_TO_SCALE.md H4/nnz-balanced
// grouping experiment for why count-balance alone wasn't enough). No on-device
// scan either way -- each CUDA block just indexes group_offsets[group_idx].
//
// Because groups can span a row-strip boundary, two groups may write to the
// same output rows. Y is therefore written via atomicAdd (not plain store).
// However, smem_Y accumulates all blocks within the same strip before a single
// atomicAdd per feature slot — far fewer atomics than the block kernel
// (one per group-strip intersection vs. one per PBR block).
//
// Smem layout:
//   smem_X [BLOCK_COLS × p]  — X tile, reloaded per PBR block
//   smem_Y [BLOCK_ROWS × p]  — persistent Y accumulator, flushed at each strip change
//
// Stream: rowgroup uses atomicAdd so it can run concurrently with the CSR
// remainder kernel on a separate stream (same as the block kernel).

template <typename index_t, typename scalar_t,
          int BLOCK_ROWS, int BLOCK_COLS, int SMEM_SIZE, int THREADS_PER_BLOCK>
__global__ void __launch_bounds__(THREADS_PER_BLOCK)
pbr_spmm_rowgroup_kernel(
    const uint32_t  num_groups,
    const index_t*  __restrict__ group_offsets,
    const uint32_t  num_pbr_blocks,
    const uint32_t  features,
    const uint32_t  batch_size,
    const uint32_t  cols,
    const uint32_t  rows,
    const int64_t*  __restrict__ block_codes,
    const index_t*  __restrict__ block_coords,
    const index_t*  __restrict__ block_offsets,
    const scalar_t* __restrict__ block_data,
    const scalar_t* __restrict__ X,
    scalar_t*       __restrict__ Y
) {
    constexpr int p = SMEM_SIZE / (BLOCK_ROWS + BLOCK_COLS);

    __shared__ scalar_t smem_X[BLOCK_COLS * p];
    __shared__ scalar_t smem_Y[BLOCK_ROWS * p];

    const uint32_t group_idx = blockIdx.x;
    const uint32_t batch_idx = blockIdx.y;
    const uint32_t feat_tile = blockIdx.z;
    const uint32_t tx        = threadIdx.x;

    if (group_idx >= num_groups || batch_idx >= batch_size) return;

    const uint32_t base_f = feat_tile * (uint32_t)p;
    if (base_f >= features) return;
    const uint32_t cur_p = min((uint32_t)p, features - base_f);

    const uint32_t pbr_start = (uint32_t)group_offsets[group_idx];
    const uint32_t pbr_end   = (uint32_t)group_offsets[group_idx + 1];
    if (pbr_start >= num_pbr_blocks) return;

    const uint64_t row_bits = (1ULL << BLOCK_COLS) - 1;

    // Zero smem_Y; re-zeroed at each strip change after flush.
    for (uint32_t i = tx; i < (uint32_t)(BLOCK_ROWS * p); i += THREADS_PER_BLOCK)
        smem_Y[i] = scalar_t(0);
    __syncthreads();

    uint32_t current_strip = (uint32_t)block_coords[pbr_start * 2];

    for (uint32_t pbr_bi = pbr_start; pbr_bi < pbr_end; ++pbr_bi) {
        const uint32_t strip_row = (uint32_t)block_coords[pbr_bi * 2];

        // Strip change: flush smem_Y via atomicAdd, then re-zero.
        if (strip_row != current_strip) {
            #pragma unroll
            for (uint32_t r = 0; r < BLOCK_ROWS; ++r) {
                const uint32_t global_row = current_strip + r;
                if (global_row >= rows) break;
                for (uint32_t f = tx; f < cur_p; f += THREADS_PER_BLOCK)
                    atomicAdd(&Y[(size_t)batch_idx * rows * features
                                 + (size_t)global_row * features + base_f + f],
                              smem_Y[r * p + f]);
            }
            __syncthreads();
            for (uint32_t i = tx; i < (uint32_t)(BLOCK_ROWS * p); i += THREADS_PER_BLOCK)
                smem_Y[i] = scalar_t(0);
            __syncthreads();
            current_strip = strip_row;
        }

        const uint64_t code       = (uint64_t)block_codes[pbr_bi];
        const uint32_t col_origin = (uint32_t)block_coords[pbr_bi * 2 + 1];
        uint32_t       data_ptr   = (uint32_t)block_offsets[pbr_bi];

        uint64_t col_mask = 0;
        #pragma unroll
        for (uint32_t r = 0; r < BLOCK_ROWS; ++r)
            col_mask |= (code >> (r * BLOCK_COLS)) & row_bits;

        uint32_t packed_cols_physical = 0;
        uint32_t packed_cols_logical  = 0;
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

        for (uint32_t i = tx; i < n_c * cur_p; i += THREADS_PER_BLOCK) {
            const uint32_t c_idx   = i / cur_p;
            const uint32_t f_off   = i % cur_p;
            const uint32_t local_c = (packed_cols_physical >> (c_idx * 3)) & 0x7;
            smem_X[c_idx * p + f_off] = X[(size_t)batch_idx * cols * features
                                          + (size_t)(col_origin + local_c) * features
                                          + base_f + f_off];
        }
        __syncthreads();

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
    }

    // Flush the final strip.
    #pragma unroll
    for (uint32_t r = 0; r < BLOCK_ROWS; ++r) {
        const uint32_t global_row = current_strip + r;
        if (global_row >= rows) break;
        for (uint32_t f = tx; f < cur_p; f += THREADS_PER_BLOCK)
            atomicAdd(&Y[(size_t)batch_idx * rows * features
                         + (size_t)global_row * features + base_f + f],
                      smem_Y[r * p + f]);
    }
}


template <typename index_t, typename scalar_t>
void launch_pbr_rowgroup_spmm(
    const int  num_groups,
    const index_t*  group_offsets,
    const int  num_pbr_blocks,
    const int  features,
    const int  batch_size,
    const int  cols,
    const int  rows,
    const int  runtime_block_rows,
    const int  runtime_block_cols,
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

    const int p          = smem_size / (runtime_block_rows + runtime_block_cols);
    const int feat_tiles = (features + p - 1) / p;

    const dim3 grid((uint32_t)num_groups, (uint32_t)batch_size, (uint32_t)feat_tiles);
    const dim3 block(THREADS_PER_BLOCK);

    #define RG_LAUNCH(BR, BC, TS) \
        pbr_spmm_rowgroup_kernel<index_t, scalar_t, BR, BC, TS, THREADS_PER_BLOCK> \
            <<<grid, block, 0, stream>>>( \
                (uint32_t)num_groups,     group_offsets, \
                (uint32_t)num_pbr_blocks, (uint32_t)features,   \
                (uint32_t)batch_size,     (uint32_t)cols,        \
                (uint32_t)rows,                                  \
                block_codes, block_coords, block_offsets, block_data, X, Y)

    #define DISPATCH_SMEM_RG(BR, BC)                              \
        do {                                                      \
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

template void launch_pbr_rowgroup_spmm<int32_t, float>(const int, const int32_t*, const int, const int, const int, const int, const int, const int, const int, const int64_t*, const int32_t*, const int32_t*, const float*,  const float*,  float*,  cudaStream_t, int);
template void launch_pbr_rowgroup_spmm<int64_t, float>(const int, const int64_t*, const int, const int, const int, const int, const int, const int, const int, const int64_t*, const int64_t*, const int64_t*, const float*,  const float*,  float*,  cudaStream_t, int);
template void launch_pbr_rowgroup_spmm<int32_t, double>(const int, const int32_t*, const int, const int, const int, const int, const int, const int, const int, const int64_t*, const int32_t*, const int32_t*, const double*, const double*, double*, cudaStream_t, int);
template void launch_pbr_rowgroup_spmm<int64_t, double>(const int, const int64_t*, const int, const int, const int, const int, const int, const int, const int, const int64_t*, const int64_t*, const int64_t*, const double*, const double*, double*, cudaStream_t, int);


// ---------------------------------------------------------------------------
// Occupancy query — rowgroup kernel
// ---------------------------------------------------------------------------
// dynamicSMemSize=0: kernel uses only static __shared__ arrays.

template <int BLOCK_ROW, int BLOCK_COL, int SMEM>
static int _rowgroup_bps() {
    constexpr int THREADS_PER_BLOCK = 128;
    int bps = 8;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &bps,
        pbr_spmm_rowgroup_kernel<int32_t, float, BLOCK_ROW, BLOCK_COL, SMEM, THREADS_PER_BLOCK>,
        THREADS_PER_BLOCK,
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


// ===========================================================================
// Column-group fusion kernel
// ===========================================================================
//
// Grid: (num_groups, batch_size, feat_tiles)
//   Same geometry as rowgroup; num_groups = sm_count × bps (occupancy query).
//
// Input arrays are pre-sorted by (col_origin, row_origin) at .to('cuda') time
// (col_codes, col_coords, col_offsets, col_data).
//
// Each CUDA block processes PBR blocks [pbr_start, pbr_end) in column-major
// order.  When col_origin changes, smem_X is reloaded — amortising the global
// X load across all blocks that share the same col band.  smem_X holds ALL
// BLOCK_COLS columns (not just active ones), so no packed-cols indirection is
// needed in the inner loop and smem_X is reusable even when different blocks
// in the same col band have different active-column patterns.
//
// smem_Y is zeroed and flushed (via atomicAdd) once per PBR block, because
// consecutive blocks in the same col-band have different row_origins and
// therefore write to different output rows.
//
// Smem layout (identical footprint to rowgroup):
//   smem_X [BLOCK_COLS × p]  — X tile, reloaded only on col_origin change
//   smem_Y [BLOCK_ROWS × p]  — per-block Y accumulator, flushed each block
//
// Sync structure (3 per block, minimum):
//   sync_A — after optional smem_X reload; also covers previous flush's smem_Y reads
//   sync_B — after smem_Y zero
//   sync_C — after smem_Y accumulation (before flush reads)
//
// current_col is initialised to ~0u so the very first block always triggers
// an smem_X reload, which sync_A then covers.

template <typename index_t, typename scalar_t,
          int BLOCK_ROWS, int BLOCK_COLS, int SMEM_SIZE, int THREADS_PER_BLOCK>
__global__ void __launch_bounds__(THREADS_PER_BLOCK)
pbr_spmm_colgroup_kernel(
    const uint32_t  num_groups,
    const index_t*  __restrict__ group_offsets,  // see rowgroup kernel: count- or nnz-balanced
    const uint32_t  num_pbr_blocks,
    const uint32_t  features,
    const uint32_t  batch_size,
    const uint32_t  cols,
    const uint32_t  rows,
    const int64_t*  __restrict__ col_codes,     // col-sorted block codes
    const index_t*  __restrict__ col_coords,    // col-sorted: [row0,col0, row1,col1, ...]
    const index_t*  __restrict__ col_offsets,   // col-sorted start offsets into col_data
    const scalar_t* __restrict__ col_data,      // col-sorted block data
    const scalar_t* __restrict__ X,
    scalar_t*       __restrict__ Y
) {
    constexpr int p = SMEM_SIZE / (BLOCK_ROWS + BLOCK_COLS);

    __shared__ scalar_t smem_X[BLOCK_COLS * p];
    __shared__ scalar_t smem_Y[BLOCK_ROWS * p];

    const uint32_t group_idx = blockIdx.x;
    const uint32_t batch_idx = blockIdx.y;
    const uint32_t feat_tile = blockIdx.z;
    const uint32_t tx        = threadIdx.x;

    if (group_idx >= num_groups || batch_idx >= batch_size) return;

    const uint32_t base_f = feat_tile * (uint32_t)p;
    if (base_f >= features) return;
    const uint32_t cur_p = min((uint32_t)p, features - base_f);

    const uint32_t pbr_start = (uint32_t)group_offsets[group_idx];
    const uint32_t pbr_end   = (uint32_t)group_offsets[group_idx + 1];
    if (pbr_start >= num_pbr_blocks) return;

    const uint64_t row_bits = (1ULL << BLOCK_COLS) - 1;

    // ~0u forces a reload on the very first block.
    uint32_t current_col = ~0u;

    for (uint32_t pbr_bi = pbr_start; pbr_bi < pbr_end; ++pbr_bi) {
        const uint32_t col_origin = (uint32_t)col_coords[pbr_bi * 2 + 1];
        const uint32_t row_origin = (uint32_t)col_coords[pbr_bi * 2];

        // Reload smem_X when col_origin changes (always on first block).
        if (col_origin != current_col) {
            for (uint32_t i = tx; i < (uint32_t)(BLOCK_COLS * cur_p); i += THREADS_PER_BLOCK) {
                const uint32_t c_idx = i / cur_p;
                const uint32_t f_off = i % cur_p;
                smem_X[c_idx * p + f_off] = X[(size_t)batch_idx * cols * features
                                               + (size_t)(col_origin + c_idx) * features
                                               + base_f + f_off];
            }
            current_col = col_origin;
        }
        // sync_A: smem_X visible + previous flush's smem_Y reads are complete.
        __syncthreads();

        // Zero smem_Y for this block.
        for (uint32_t i = tx; i < (uint32_t)(BLOCK_ROWS * p); i += THREADS_PER_BLOCK)
            smem_Y[i] = scalar_t(0);
        // sync_B: smem_Y is zero.
        __syncthreads();

        const uint64_t code     = (uint64_t)col_codes[pbr_bi];
        uint32_t       data_ptr = (uint32_t)col_offsets[pbr_bi];

        // Accumulate: smem_Y += val × smem_X using direct column index c.
        // No packed_cols indirection needed — smem_X holds all BLOCK_COLS columns.
        #pragma unroll
        for (uint32_t r = 0; r < BLOCK_ROWS; ++r) {
            const uint64_t row_mask = (code >> (r * BLOCK_COLS)) & row_bits;
            if (row_mask == 0) continue;
            #pragma unroll
            for (uint32_t c = 0; c < BLOCK_COLS; ++c) {
                if ((row_mask >> c) & 1) {
                    const scalar_t val = __ldg(&col_data[data_ptr++]);
                    for (uint32_t f = tx; f < cur_p; f += THREADS_PER_BLOCK)
                        smem_Y[r * p + f] += val * smem_X[c * p + f];
                }
            }
        }
        // sync_C: smem_Y is complete before flush reads.
        __syncthreads();

        // Flush smem_Y → Y via atomicAdd.
        #pragma unroll
        for (uint32_t r = 0; r < BLOCK_ROWS; ++r) {
            const uint32_t global_row = row_origin + r;
            if (global_row >= rows) break;
            for (uint32_t f = tx; f < cur_p; f += THREADS_PER_BLOCK)
                atomicAdd(&Y[(size_t)batch_idx * rows * features
                             + (size_t)global_row * features + base_f + f],
                          smem_Y[r * p + f]);
        }
        // No sync here — sync_A of the next iteration covers this flush.
    }
}


template <typename index_t, typename scalar_t>
void launch_pbr_colgroup_spmm(
    const int  num_groups,
    const index_t*  group_offsets,
    const int  num_pbr_blocks,
    const int  features,
    const int  batch_size,
    const int  cols,
    const int  rows,
    const int  runtime_block_rows,
    const int  runtime_block_cols,
    const int64_t*  col_codes,
    const index_t*  col_coords,
    const index_t*  col_offsets,
    const scalar_t* col_data,
    const scalar_t* X,
    scalar_t*       Y,
    cudaStream_t    stream,
    int smem_size
) {
    if (num_groups == 0 || num_pbr_blocks == 0) return;

    constexpr int THREADS_PER_BLOCK = 128;

    const int p          = smem_size / (runtime_block_rows + runtime_block_cols);
    const int feat_tiles = (features + p - 1) / p;

    const dim3 grid((uint32_t)num_groups, (uint32_t)batch_size, (uint32_t)feat_tiles);
    const dim3 block(THREADS_PER_BLOCK);

    #define CG_LAUNCH(BR, BC, TS) \
        pbr_spmm_colgroup_kernel<index_t, scalar_t, BR, BC, TS, THREADS_PER_BLOCK> \
            <<<grid, block, 0, stream>>>( \
                (uint32_t)num_groups,     group_offsets, \
                (uint32_t)num_pbr_blocks, (uint32_t)features,   \
                (uint32_t)batch_size,     (uint32_t)cols,        \
                (uint32_t)rows,                                  \
                col_codes, col_coords, col_offsets, col_data, X, Y)

    #define DISPATCH_SMEM_CG(BR, BC)                              \
        do {                                                      \
            if      (smem_size <= 512)  CG_LAUNCH(BR, BC,  512); \
            else if (smem_size <= 1024) CG_LAUNCH(BR, BC, 1024); \
            else                        CG_LAUNCH(BR, BC, 2048); \
        } while (0)

    if      (runtime_block_rows == 8 && runtime_block_cols == 8) DISPATCH_SMEM_CG(8, 8);
    else if (runtime_block_rows == 4 && runtime_block_cols == 4) DISPATCH_SMEM_CG(4, 4);
    else if (runtime_block_rows == 2 && runtime_block_cols == 2) DISPATCH_SMEM_CG(2, 2);
    else throw std::invalid_argument("Unsupported block size for PBR col-group SpMM.");

    #undef CG_LAUNCH
    #undef DISPATCH_SMEM_CG
}

template void launch_pbr_colgroup_spmm<int32_t, float>(const int, const int32_t*, const int, const int, const int, const int, const int, const int, const int, const int64_t*, const int32_t*, const int32_t*, const float*,  const float*,  float*,  cudaStream_t, int);
template void launch_pbr_colgroup_spmm<int64_t, float>(const int, const int64_t*, const int, const int, const int, const int, const int, const int, const int, const int64_t*, const int64_t*, const int64_t*, const float*,  const float*,  float*,  cudaStream_t, int);
template void launch_pbr_colgroup_spmm<int32_t, double>(const int, const int32_t*, const int, const int, const int, const int, const int, const int, const int, const int64_t*, const int32_t*, const int32_t*, const double*, const double*, double*, cudaStream_t, int);
template void launch_pbr_colgroup_spmm<int64_t, double>(const int, const int64_t*, const int, const int, const int, const int, const int, const int, const int, const int64_t*, const int64_t*, const int64_t*, const double*, const double*, double*, cudaStream_t, int);


// ---------------------------------------------------------------------------
// Occupancy query — colgroup kernel
// ---------------------------------------------------------------------------

template <int BLOCK_ROW, int BLOCK_COL, int SMEM>
static int _colgroup_bps() {
    constexpr int THREADS_PER_BLOCK = 128;
    int bps = 8;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &bps,
        pbr_spmm_colgroup_kernel<int32_t, float, BLOCK_ROW, BLOCK_COL, SMEM, THREADS_PER_BLOCK>,
        THREADS_PER_BLOCK,
        0);
    return bps;
}

int get_pbr_colgroup_blocks_per_sm(int block_rows, int smem_size) {
    if (block_rows == 2) {
        if (smem_size <= 512)  return _colgroup_bps<2, 2,  512>();
        if (smem_size <= 1024) return _colgroup_bps<2, 2, 1024>();
        return                        _colgroup_bps<2, 2, 2048>();
    }
    if (block_rows == 4) {
        if (smem_size <= 512)  return _colgroup_bps<4, 4,  512>();
        if (smem_size <= 1024) return _colgroup_bps<4, 4, 1024>();
        return                        _colgroup_bps<4, 4, 2048>();
    }
    if (block_rows == 8) {
        if (smem_size <= 512)  return _colgroup_bps<8, 8,  512>();
        if (smem_size <= 1024) return _colgroup_bps<8, 8, 1024>();
        return                        _colgroup_bps<8, 8, 2048>();
    }
    return 8;
}


// ===========================================================================
// Tile ("block of blocks") kernel
// ===========================================================================
//
// Grid: (num_tiles, batch_size, feat_tiles)
//   num_tiles — one CUDA block per tile; fully data-determined (no CPU scan
//               needed at launch time -- tile_offsets.size(0)-1 from Python).
//
// A tile groups up to TILE_ROWS x TILE_COLS PBR blocks that together touch at
// most TILE_ROWS distinct row-bands and TILE_COLS distinct col-bands. Unlike
// rowgroup/colgroup (which reuse only smem_Y or only smem_X, one band at a
// time, sequentially), this kernel holds ALL of the tile's row-bands and
// col-bands resident in smem simultaneously, so smem_X is loaded ONCE and
// smem_Y is flushed ONCE per tile regardless of block count -- 2 syncthreads
// total per tile, vs ~2-3 per PBR block for rowgroup/colgroup.
//
// Which PBR blocks land in which tile (and at which local row/col slot) is
// decided entirely on the host by a pluggable strategy (see _TILE_STRATEGIES
// in __init__.py: 'grid' partitions by fixed row/col-band windows; 'greedy'
// clusters by density to also catch non-grid-aligned reuse). This kernel is
// agnostic to that choice -- it only consumes the resulting contract:
//   block_codes/block_offsets/block_data — tile-ordered (like block/rowgroup)
//   tile_offsets      [num_tiles+1]        — CSR range into the block_* arrays
//   tile_row_origins  [num_tiles*TILE_ROWS]  — absolute row origin per row-slot, -1 = unused
//   tile_col_origins  [num_tiles*TILE_COLS]  — absolute col origin per col-slot, -1 = unused
//   tile_row_local/tile_col_local            — per (tile-ordered) block, its row/col slot
//
// Smem layout:
//   smem_X [TILE_COLS*BLOCK_COLS × p]  — all col-slots, loaded once
//   smem_Y [TILE_ROWS*BLOCK_ROWS × p]  — all row-slots, zeroed once, flushed once
//
// v1 only supports BLOCK_ROWS=BLOCK_COLS=8 (see launch_pbr_tile_spmm) to keep
// the tile_shape x smem_tier template instantiation count bounded.

template <typename index_t, typename scalar_t,
          int BLOCK_ROWS, int BLOCK_COLS, int TILE_ROWS, int TILE_COLS,
          int SMEM_SIZE, int THREADS_PER_BLOCK>
__global__ void __launch_bounds__(THREADS_PER_BLOCK)
pbr_spmm_tile_kernel(
    const uint32_t  num_tiles,
    const uint32_t  features,
    const uint32_t  batch_size,
    const uint32_t  cols,
    const uint32_t  rows,
    const int64_t*  __restrict__ block_codes,       // tile-ordered
    const index_t*  __restrict__ block_offsets,     // tile-ordered, CSR offsets into block_data
    const scalar_t* __restrict__ block_data,        // tile-ordered
    const index_t*  __restrict__ tile_offsets,      // [num_tiles+1]
    const index_t*  __restrict__ tile_row_origins,  // [num_tiles*TILE_ROWS], -1 = unused
    const index_t*  __restrict__ tile_col_origins,  // [num_tiles*TILE_COLS], -1 = unused
    const uint8_t*  __restrict__ tile_row_local,    // per tile-ordered block
    const uint8_t*  __restrict__ tile_col_local,    // per tile-ordered block
    const scalar_t* __restrict__ X,
    scalar_t*       __restrict__ Y
) {
    constexpr int p = SMEM_SIZE / (TILE_ROWS * BLOCK_ROWS + TILE_COLS * BLOCK_COLS);

    __shared__ scalar_t smem_X[TILE_COLS * BLOCK_COLS * p];
    __shared__ scalar_t smem_Y[TILE_ROWS * BLOCK_ROWS * p];
    __shared__ index_t  s_row_origin[TILE_ROWS];
    __shared__ index_t  s_col_origin[TILE_COLS];

    const uint32_t tile_idx  = blockIdx.x;
    const uint32_t batch_idx = blockIdx.y;
    const uint32_t feat_tile = blockIdx.z;
    const uint32_t tx        = threadIdx.x;

    if (tile_idx >= num_tiles || batch_idx >= batch_size) return;

    const uint32_t base_f = feat_tile * (uint32_t)p;
    if (base_f >= features) return;
    const uint32_t cur_p = min((uint32_t)p, features - base_f);

    const index_t blk_start = tile_offsets[tile_idx];
    const index_t blk_end   = tile_offsets[tile_idx + 1];

    const uint64_t row_bits = (1ULL << BLOCK_COLS) - 1;

    if (tx < (uint32_t)TILE_ROWS) s_row_origin[tx] = tile_row_origins[tile_idx * TILE_ROWS + tx];
    if (tx < (uint32_t)TILE_COLS) s_col_origin[tx] = tile_col_origins[tile_idx * TILE_COLS + tx];

    // Load smem_X: every col-slot's full BLOCK_COLS x cur_p band, once. Slots
    // with no data in this tile (-1 origin) are left uninitialized -- no
    // block's col_local ever indexes an unused slot, so it's never read.
    for (uint32_t i = tx; i < (uint32_t)(TILE_COLS * BLOCK_COLS * cur_p); i += THREADS_PER_BLOCK) {
        const uint32_t slot_and_c = i / cur_p;
        const uint32_t f_off      = i % cur_p;
        const uint32_t col_slot   = slot_and_c / BLOCK_COLS;
        const uint32_t c_in_block = slot_and_c % BLOCK_COLS;
        const index_t  col_origin = tile_col_origins[tile_idx * TILE_COLS + col_slot];
        if (col_origin >= 0) {
            smem_X[slot_and_c * p + f_off] = X[(size_t)batch_idx * cols * features
                                                + (size_t)(col_origin + c_in_block) * features
                                                + base_f + f_off];
        }
    }

    // Zero smem_Y, once.
    for (uint32_t i = tx; i < (uint32_t)(TILE_ROWS * BLOCK_ROWS * p); i += THREADS_PER_BLOCK)
        smem_Y[i] = scalar_t(0);

    // Covers: smem_X fully loaded, smem_Y fully zeroed, s_row/col_origin visible.
    __syncthreads();

    // Accumulate every PBR block in the tile. No sync between blocks: smem_X
    // is never rewritten mid-tile (unlike rowgroup/colgroup, which reload it
    // per block and so must sync between iterations), and smem_Y[row_slot*
    // BLOCK_ROWS+r][f] is always touched by the same thread (f mod
    // THREADS_PER_BLOCK) regardless of which block contributes to it, so
    // there's no cross-thread write hazard to guard against either.
    for (index_t bi = blk_start; bi < blk_end; ++bi) {
        const uint64_t code     = (uint64_t)block_codes[bi];
        uint32_t       data_ptr = (uint32_t)block_offsets[bi];
        const uint32_t row_slot = tile_row_local[bi];
        const uint32_t col_slot = tile_col_local[bi];

        #pragma unroll
        for (uint32_t r = 0; r < BLOCK_ROWS; ++r) {
            const uint64_t row_mask = (code >> (r * BLOCK_COLS)) & row_bits;
            if (row_mask == 0) continue;
            const uint32_t smem_row = row_slot * BLOCK_ROWS + r;
            #pragma unroll
            for (uint32_t c = 0; c < BLOCK_COLS; ++c) {
                if ((row_mask >> c) & 1) {
                    const scalar_t val = __ldg(&block_data[data_ptr++]);
                    const uint32_t smem_col = col_slot * BLOCK_COLS + c;
                    for (uint32_t f = tx; f < cur_p; f += THREADS_PER_BLOCK)
                        smem_Y[smem_row * p + f] += val * smem_X[smem_col * p + f];
                }
            }
        }
    }

    // smem_Y fully accumulated before the flush reads it.
    __syncthreads();

    // Flush smem_Y -> Y once per tile, guarded per row-slot by the origin
    // sentinel (an atomicAdd of 0 is still a real atomic RMW, worth skipping
    // for tiles -- especially 'greedy'-built ones -- with unused row-slots).
    #pragma unroll
    for (uint32_t r_slot = 0; r_slot < (uint32_t)TILE_ROWS; ++r_slot) {
        const index_t row_origin = s_row_origin[r_slot];
        if (row_origin < 0) continue;
        #pragma unroll
        for (uint32_t r = 0; r < BLOCK_ROWS; ++r) {
            const uint32_t global_row = (uint32_t)row_origin + r;
            if (global_row >= rows) break;
            const uint32_t smem_row = r_slot * BLOCK_ROWS + r;
            for (uint32_t f = tx; f < cur_p; f += THREADS_PER_BLOCK)
                atomicAdd(&Y[(size_t)batch_idx * rows * features
                             + (size_t)global_row * features + base_f + f],
                          smem_Y[smem_row * p + f]);
        }
    }
}


template <typename index_t, typename scalar_t>
void launch_pbr_tile_spmm(
    const int  num_tiles,
    const int  features,
    const int  batch_size,
    const int  cols,
    const int  rows,
    const int  runtime_block_rows,
    const int  runtime_block_cols,
    const int  tile_rows,
    const int  tile_cols,
    const int64_t*  block_codes,
    const index_t*  block_offsets,
    const scalar_t* block_data,
    const index_t*  tile_offsets,
    const index_t*  tile_row_origins,
    const index_t*  tile_col_origins,
    const uint8_t*  tile_row_local,
    const uint8_t*  tile_col_local,
    const scalar_t* X,
    scalar_t*       Y,
    cudaStream_t    stream,
    int smem_size
) {
    if (num_tiles == 0) return;
    if (runtime_block_rows != 8 || runtime_block_cols != 8)
        throw std::invalid_argument("PBR tile SpMM currently only supports block_rows=block_cols=8.");

    constexpr int THREADS_PER_BLOCK = 128;
    constexpr int BR = 8, BC = 8;

    #define TILE_LAUNCH(TR, TC, TS)                                                            \
        do {                                                                                   \
            constexpr int p      = TS / (TR * BR + TC * BC);                                   \
            const int feat_tiles = (features + p - 1) / p;                                     \
            const dim3 grid((uint32_t)num_tiles, (uint32_t)batch_size, (uint32_t)feat_tiles);  \
            const dim3 block(THREADS_PER_BLOCK);                                               \
            pbr_spmm_tile_kernel<index_t, scalar_t, BR, BC, TR, TC, TS, THREADS_PER_BLOCK>      \
                <<<grid, block, 0, stream>>>(                                                   \
                    (uint32_t)num_tiles, (uint32_t)features, (uint32_t)batch_size,              \
                    (uint32_t)cols, (uint32_t)rows,                                             \
                    block_codes, block_offsets, block_data,                                     \
                    tile_offsets, tile_row_origins, tile_col_origins,                           \
                    tile_row_local, tile_col_local, X, Y);                                      \
        } while (0)

    #define DISPATCH_TILE_SMEM(TR, TC)                                \
        do {                                                          \
            if      (smem_size <= 512)  TILE_LAUNCH(TR, TC,  512);   \
            else if (smem_size <= 1024) TILE_LAUNCH(TR, TC, 1024);   \
            else                        TILE_LAUNCH(TR, TC, 2048);   \
        } while (0)

    if      (tile_rows == 1 && tile_cols == 1) DISPATCH_TILE_SMEM(1, 1);
    else if (tile_rows == 1 && tile_cols == 2) DISPATCH_TILE_SMEM(1, 2);
    else if (tile_rows == 2 && tile_cols == 1) DISPATCH_TILE_SMEM(2, 1);
    else if (tile_rows == 1 && tile_cols == 4) DISPATCH_TILE_SMEM(1, 4);
    else if (tile_rows == 4 && tile_cols == 1) DISPATCH_TILE_SMEM(4, 1);
    else if (tile_rows == 2 && tile_cols == 2) DISPATCH_TILE_SMEM(2, 2);
    else if (tile_rows == 2 && tile_cols == 4) DISPATCH_TILE_SMEM(2, 4);
    else if (tile_rows == 4 && tile_cols == 2) DISPATCH_TILE_SMEM(4, 2);
    else if (tile_rows == 4 && tile_cols == 4) DISPATCH_TILE_SMEM(4, 4);
    else if (tile_rows == 8 && tile_cols == 8) DISPATCH_TILE_SMEM(8, 8);
    else throw std::invalid_argument("Unsupported tile shape for PBR tile SpMM.");

    #undef TILE_LAUNCH
    #undef DISPATCH_TILE_SMEM
}

template void launch_pbr_tile_spmm<int32_t, float>(const int, const int, const int, const int, const int, const int, const int, const int, const int, const int64_t*, const int32_t*, const float*,  const int32_t*, const int32_t*, const int32_t*, const uint8_t*, const uint8_t*, const float*,  float*,  cudaStream_t, int);
template void launch_pbr_tile_spmm<int64_t, float>(const int, const int, const int, const int, const int, const int, const int, const int, const int, const int64_t*, const int64_t*, const float*,  const int64_t*, const int64_t*, const int64_t*, const uint8_t*, const uint8_t*, const float*,  float*,  cudaStream_t, int);
template void launch_pbr_tile_spmm<int32_t, double>(const int, const int, const int, const int, const int, const int, const int, const int, const int, const int64_t*, const int32_t*, const double*, const int32_t*, const int32_t*, const int32_t*, const uint8_t*, const uint8_t*, const double*, double*, cudaStream_t, int);
template void launch_pbr_tile_spmm<int64_t, double>(const int, const int, const int, const int, const int, const int, const int, const int, const int, const int64_t*, const int64_t*, const double*, const int64_t*, const int64_t*, const int64_t*, const uint8_t*, const uint8_t*, const double*, double*, cudaStream_t, int);


// ---------------------------------------------------------------------------
// Occupancy query — tile kernel
// ---------------------------------------------------------------------------
// dynamicSMemSize=0: kernel uses only static __shared__ arrays (the same
// caveat already hit once for rowgroup -- see LAST_EXPERIMENTS.md).

template <int TR, int TC, int BR, int BC, int SMEM>
static int _tile_bps() {
    constexpr int THREADS_PER_BLOCK = 128;
    int bps = 8;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &bps,
        pbr_spmm_tile_kernel<int32_t, float, BR, BC, TR, TC, SMEM, THREADS_PER_BLOCK>,
        THREADS_PER_BLOCK,
        0);
    return bps;
}

// block_rows is accepted for signature symmetry with get_pbr_rowgroup/colgroup_blocks_per_sm;
// v1 only supports 8x8 PBR blocks for the tile kernel (see launch_pbr_tile_spmm), so it's unused.
int get_pbr_tile_blocks_per_sm(int tile_rows, int tile_cols, int block_rows, int smem_size) {
    (void)block_rows;
    constexpr int BR = 8, BC = 8;

    if (tile_rows == 1 && tile_cols == 1) {
        if (smem_size <= 512)  return _tile_bps<1, 1, BR, BC,  512>();
        if (smem_size <= 1024) return _tile_bps<1, 1, BR, BC, 1024>();
        return                        _tile_bps<1, 1, BR, BC, 2048>();
    }
    if (tile_rows == 1 && tile_cols == 2) {
        if (smem_size <= 512)  return _tile_bps<1, 2, BR, BC,  512>();
        if (smem_size <= 1024) return _tile_bps<1, 2, BR, BC, 1024>();
        return                        _tile_bps<1, 2, BR, BC, 2048>();
    }
    if (tile_rows == 2 && tile_cols == 1) {
        if (smem_size <= 512)  return _tile_bps<2, 1, BR, BC,  512>();
        if (smem_size <= 1024) return _tile_bps<2, 1, BR, BC, 1024>();
        return                        _tile_bps<2, 1, BR, BC, 2048>();
    }
    if (tile_rows == 1 && tile_cols == 4) {
        if (smem_size <= 512)  return _tile_bps<1, 4, BR, BC,  512>();
        if (smem_size <= 1024) return _tile_bps<1, 4, BR, BC, 1024>();
        return                        _tile_bps<1, 4, BR, BC, 2048>();
    }
    if (tile_rows == 4 && tile_cols == 1) {
        if (smem_size <= 512)  return _tile_bps<4, 1, BR, BC,  512>();
        if (smem_size <= 1024) return _tile_bps<4, 1, BR, BC, 1024>();
        return                        _tile_bps<4, 1, BR, BC, 2048>();
    }
    if (tile_rows == 2 && tile_cols == 2) {
        if (smem_size <= 512)  return _tile_bps<2, 2, BR, BC,  512>();
        if (smem_size <= 1024) return _tile_bps<2, 2, BR, BC, 1024>();
        return                        _tile_bps<2, 2, BR, BC, 2048>();
    }
    if (tile_rows == 2 && tile_cols == 4) {
        if (smem_size <= 512)  return _tile_bps<2, 4, BR, BC,  512>();
        if (smem_size <= 1024) return _tile_bps<2, 4, BR, BC, 1024>();
        return                        _tile_bps<2, 4, BR, BC, 2048>();
    }
    if (tile_rows == 4 && tile_cols == 2) {
        if (smem_size <= 512)  return _tile_bps<4, 2, BR, BC,  512>();
        if (smem_size <= 1024) return _tile_bps<4, 2, BR, BC, 1024>();
        return                        _tile_bps<4, 2, BR, BC, 2048>();
    }
    if (tile_rows == 4 && tile_cols == 4) {
        if (smem_size <= 512)  return _tile_bps<4, 4, BR, BC,  512>();
        if (smem_size <= 1024) return _tile_bps<4, 4, BR, BC, 1024>();
        return                        _tile_bps<4, 4, BR, BC, 2048>();
    }
    if (tile_rows == 8 && tile_cols == 8) {
        if (smem_size <= 512)  return _tile_bps<8, 8, BR, BC,  512>();
        if (smem_size <= 1024) return _tile_bps<8, 8, BR, BC, 1024>();
        return                        _tile_bps<8, 8, BR, BC, 2048>();
    }
    return 8;
}


// ===========================================================================
// Chunk kernel — diagnostic-only, isolates the "grouping criterion" variable
// ===========================================================================
//
// WHY_BLOCK_KERNEL_FAILS_TO_SCALE.md found `block` launches one CUDA block
// per naturally-occurring PBR block (grid.x == num_pbr_blocks) while cuSPARSE
// partitions work by nnz count -- ~5x fewer, denser launched units on
// diag_b0.3. rowgroup/colgroup also reduce CTA count, but they ALSO change
// the flush pattern (smem_Y persists across a group, atomicAdd once per
// strip/never) and add extra __syncthreads, confounding "fewer CTAs helped"
// with "the reuse logic helped/hurt".
//
// This kernel changes ONLY the grouping: each CUDA block processes the PBR
// blocks in [group_offsets[group_idx], group_offsets[group_idx+1]) via a
// runtime loop (deliberately NOT #pragma-unrolled -- H4 found a compile-time
// unrolled CHUNK made things worse as it grew, via rising register pressure
// and a 7.4x DRAM-traffic blowup for identical total work; a runtime loop
// avoids that confound). Each iteration's body is byte-for-byte the same
// per-block work `pbr_spmm_vec_kernel` already does (own smem_X/smem_Y
// carve-out, own load, own atomicAdd flush -- zero reuse across iterations).
//
// group_offsets is built host-side by _build_group_offsets (__init__.py),
// either count-balanced (grouping='count' -- H4's original chunk_size
// behavior, reproduced via a fixed-stride offsets array) or nnz-balanced
// (grouping='nnz' -- the new experiment: does balancing by nnz instead of
// block count, with everything else held identical, close the gap to
// cuSPARSE that count-balancing alone could not?).

template <typename index_t, typename scalar_t,
          int BLOCK_ROWS, int BLOCK_COLS,
          int TOTAL_SHARED, int THREADS_PER_BLOCK, int FEAT_BLOCK_SIZE>
__global__ void __launch_bounds__(THREADS_PER_BLOCK, 16) pbr_spmm_chunk_kernel(
    const uint32_t num_groups,
    const index_t* __restrict__ group_offsets,
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

    const uint32_t group_idx = blockIdx.x;
    const uint32_t batch_idx = blockIdx.y;
    const uint32_t feat_block_idx = blockIdx.z;
    const uint32_t tx = threadIdx.x;

    if (group_idx >= num_groups || batch_idx >= batch_size) return;

    const uint32_t group_start = (uint32_t)group_offsets[group_idx];
    const uint32_t group_end   = (uint32_t)group_offsets[group_idx + 1];

    // Every thread in the CTA takes the same branch at each of these guards
    // (b_idx depends only on blockIdx/loop state; code/n_r/n_c are the same
    // value for every thread), so the __syncthreads() calls below stay
    // uniform across the block despite the `continue`s -- no divergence/
    // deadlock risk. Deliberately no #pragma unroll -- see comment above.
    #pragma unroll 1
    for (uint32_t b_idx = group_start; b_idx < group_end; ++b_idx) {
        if (b_idx >= num_pbr_blocks) continue;

        const uint64_t code = static_cast<uint64_t>(block_codes[b_idx]);
        const uint32_t row_origin = block_coords[b_idx * 2];
        const uint32_t col_origin = block_coords[b_idx * 2 + 1];
        const uint32_t data_offset = block_offsets[b_idx];

        // --- SPARSITY SCAN & REGISTER PACKING (identical to pbr_spmm_vec_kernel) ---
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
        if (n_r == 0 || n_c == 0) continue;

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

        const uint32_t vp_log2 = 31u - __clz(min(TOTAL_SHARED / (n_r + n_c), features) / L);
        const uint32_t vp_mask = (1u << vp_log2) - 1u;
        const uint32_t p       = (1u << vp_log2) * L;

        const uint32_t feat_start = feat_block_idx * FEAT_BLOCK_SIZE;
        const uint32_t feat_end = min(feat_start + FEAT_BLOCK_SIZE, features);

        for (uint32_t base_f = feat_start; base_f < feat_end; base_f += p) {
            const uint32_t current_p = min(p, feat_end - base_f);
            const uint32_t vp = current_p / L;

            scalar_t* smem_X = smem;
            scalar_t* smem_Y = smem + (n_c * current_p);

            const uint32_t total_xv = n_c * vp;
            for (uint32_t i = tx; i < total_xv; i += THREADS_PER_BLOCK) {
                const uint32_t c_idx = i >> vp_log2;
                const uint32_t f_off = (i & vp_mask) * L;
                const uint32_t local_c = (packed_cols_physical >> (c_idx * 3)) & 0x7;

                const vec_t xv = *reinterpret_cast<const vec_t*>(
                    &X[(batch_idx * cols * features) + ((col_origin + local_c) * features) + base_f + f_off]);
                *reinterpret_cast<vec_t*>(&smem_X[c_idx * current_p + f_off]) = xv;
            }

            __syncthreads();

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
}

template <typename index_t, typename scalar_t>
void launch_pbr_chunk_spmm(
    const int num_groups,
    const index_t* group_offsets,
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
    if (num_groups == 0 || num_pbr_blocks == 0) return;

    constexpr int THREADS_PER_BLOCK = 128;
    constexpr int FEAT_BLOCK_SIZE   = 512;

    const int feat_blocks = (features + FEAT_BLOCK_SIZE - 1) / FEAT_BLOCK_SIZE;
    const dim3 grid((uint32_t)num_groups, (uint32_t)batch_size, (uint32_t)feat_blocks);
    const dim3 block(THREADS_PER_BLOCK);

    #define CHUNK_LAUNCH(BR, BC, TS)                                                              \
        pbr_spmm_chunk_kernel<index_t, scalar_t, BR, BC, TS, THREADS_PER_BLOCK,                    \
                              FEAT_BLOCK_SIZE><<<grid, block, 0, stream>>>(                        \
            (uint32_t)num_groups, group_offsets, (uint32_t)num_pbr_blocks,                         \
            (uint32_t)features, (uint32_t)batch_size, (uint32_t)cols, (uint32_t)rows,              \
            block_codes, block_coords, block_offsets, block_data, X, Y)

    #define DISPATCH_SMEM_CHUNK(BR, BC)                        \
        do {                                                   \
            if      (smem_size <= 512)  CHUNK_LAUNCH(BR, BC,  512); \
            else if (smem_size <= 1024) CHUNK_LAUNCH(BR, BC, 1024); \
            else                        CHUNK_LAUNCH(BR, BC, 2048); \
        } while (0)

    if      (runtime_block_rows == 2 && runtime_block_cols == 2) DISPATCH_SMEM_CHUNK(2, 2);
    else if (runtime_block_rows == 4 && runtime_block_cols == 4) DISPATCH_SMEM_CHUNK(4, 4);
    else if (runtime_block_rows == 8 && runtime_block_cols == 8) DISPATCH_SMEM_CHUNK(8, 8);
    else throw std::invalid_argument("Unsupported block size for PBR chunk SpMM. Only 2x2, 4x4, and 8x8 are supported.");

    #undef CHUNK_LAUNCH
    #undef DISPATCH_SMEM_CHUNK
}

template void launch_pbr_chunk_spmm<int32_t, float>(const int, const int32_t*, const int, const int, const int, const int, const int, const int, const int, const int64_t*, const int32_t*, const int32_t*, const float*, const float*, float*, cudaStream_t, int);
template void launch_pbr_chunk_spmm<int64_t, float>(const int, const int64_t*, const int, const int, const int, const int, const int, const int, const int, const int64_t*, const int64_t*, const int64_t*, const float*, const float*, float*, cudaStream_t, int);
template void launch_pbr_chunk_spmm<int32_t, double>(const int, const int32_t*, const int, const int, const int, const int, const int, const int, const int, const int64_t*, const int32_t*, const int32_t*, const double*, const double*, double*, cudaStream_t, int);
template void launch_pbr_chunk_spmm<int64_t, double>(const int, const int64_t*, const int, const int, const int, const int, const int, const int, const int, const int64_t*, const int64_t*, const int64_t*, const double*, const double*, double*, cudaStream_t, int);


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
