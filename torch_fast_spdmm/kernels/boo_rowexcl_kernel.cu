#include "boo_rowexcl_kernels.hpp"
#include <c10/cuda/CUDAException.h>

namespace {
constexpr int THREADS = 128;
constexpr int MAX_ROWS = 32;

template<int SLAB>
__global__ void boo_rowexcl_kernel(BOORowExclusiveView a, int features,
                                  const float* __restrict__ x, float* __restrict__ y) {
    extern __shared__ float tile[];
    constexpr int ITEMS = MAX_ROWS * SLAB / THREADS;
    float sum[ITEMS] = {};
    const int unit = blockIdx.x, group = a.unit_group[unit];
    const int begin = a.unit_begin[unit], nr = a.unit_end[unit] - begin;
    const int local_begin = begin - a.row_groups[group];
    const int feature_begin = blockIdx.y * SLAB;
    // Every thread executes every barrier, including feature/row tail threads.
    for (int chunk = a.group_chunks[group]; chunk < a.group_chunks[group + 1]; ++chunk) {
        const int cp = a.chunk_cols[chunk];
        const int nc = a.chunk_cols[chunk + 1] - cp;
        for (int i = threadIdx.x; i < nc * SLAB; i += THREADS) {
            const int f = feature_begin + i % SLAB;
            tile[i] = f < features ? x[(int64_t)a.column_ids[cp + i / SLAB] * features + f] : 0.f;
        }
        __syncthreads();
        for (int position = a.chunk_blocks[chunk]; position < a.chunk_blocks[chunk + 1]; ++position) {
            const int b = a.block_ids[position];
            const int mask = (1 << a.shifts[b]) - 1;
            const int base = a.block_xbase[position];
#pragma unroll
            for (int item = 0; item < ITEMS; ++item) {
                const int i = threadIdx.x + item * THREADS;
                const int row = i / SLAB, lane = i % SLAB;
                if (row < nr && feature_begin + lane < features) {
                    const int rp = a.block_rows[b] + local_begin + row;
                    for (int z = a.row_starts[rp]; z < a.row_starts[rp + 1]; ++z) {
                        const int col = a.coordinates[z] & mask;
                        sum[item] = fmaf(a.values[z], tile[(base + col) * SLAB + lane], sum[item]);
                    }
                }
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int item = 0; item < ITEMS; ++item) {
        const int i = threadIdx.x + item * THREADS;
        const int row = i / SLAB, f = feature_begin + i % SLAB;
        if (row < nr && f < features) {
            const int global_row = a.row_ids[begin + row];
            for (int z = a.rem_ptr[global_row]; z < a.rem_ptr[global_row + 1]; ++z)
                sum[item] = fmaf(a.rem_values[z], x[(int64_t)a.rem_cols[z] * features + f], sum[item]);
            // Also writes exact zeros for structurally empty rows. No atomics,
            // initialization kernel, or shared partial-output allocation.
            y[(int64_t)global_row * features + f] = sum[item];
        }
    }
}

template<int S>
void resources(int bytes, int* regs, int* active) {
    cudaFuncAttributes a;
    C10_CUDA_CHECK(cudaFuncGetAttributes(&a, boo_rowexcl_kernel<S>));
    *regs = a.numRegs;
    C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(active, boo_rowexcl_kernel<S>, THREADS, bytes));
}
}

void launch_boo_rowexcl(const BOORowExclusiveView& a, int units, int features,
                       int slab, int bytes, const float* x, float* y, cudaStream_t stream) {
    if (!units) return;
    const dim3 grid(units, (features + slab - 1) / slab);
    if (slab == 16) boo_rowexcl_kernel<16><<<grid, THREADS, bytes, stream>>>(a, features, x, y);
    else if (slab == 32) boo_rowexcl_kernel<32><<<grid, THREADS, bytes, stream>>>(a, features, x, y);
    else boo_rowexcl_kernel<64><<<grid, THREADS, bytes, stream>>>(a, features, x, y);
}
void boo_rowexcl_resources(int slab, int bytes, int* registers, int* active_ctas) {
    if (slab == 16) resources<16>(bytes, registers, active_ctas);
    else if (slab == 32) resources<32>(bytes, registers, active_ctas);
    else resources<64>(bytes, registers, active_ctas);
}
