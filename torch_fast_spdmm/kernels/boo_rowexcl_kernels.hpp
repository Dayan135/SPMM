#pragma once
#include <cuda_runtime.h>
#include <cstdint>

// Auxiliary schedule only: values and packed coordinates remain in BOO.
struct BOORowExclusiveView {
    const int *unit_group, *unit_begin, *unit_end;
    const int *group_chunks, *chunk_blocks, *block_ids, *block_xbase;
    const int *chunk_cols, *column_ids, *block_rows, *row_starts;
    const int *row_groups, *row_ids, *rem_ptr, *rem_cols;
    const uint8_t* shifts;
    const uint16_t* coordinates;
    const float *values, *rem_values;
};
void launch_boo_rowexcl(const BOORowExclusiveView&, int units, int features,
                       int slab, int smem_bytes, const float* x, float* y, cudaStream_t);
void boo_rowexcl_resources(int slab, int smem_bytes, int* registers, int* active_ctas);
