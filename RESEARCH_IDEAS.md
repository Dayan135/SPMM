# Research ideas

Ideas raised while reading the implementation together. These are hypotheses,
not established performance improvements. Record measurements and decisions here
when an idea is evaluated.

## 1. Precompute per-block row boundaries in the CPU builder

Status: implemented and measured negative -- see BOO_OPTIMIZATION.md's
"Stages 3 and 4 implementation" section. Isolated from bucketing (one-bucket
plan with precomputed rows vs without), it's a ~2% slowdown across all 36
tested cells, no exceptions: the extra global-memory traffic for the stored
boundary table outweighs the saved per-block binary search, as this
writeup's own risk note anticipated. Not promoted.

### Observation

`boo_spmm_rowseg_kernel` constructs `sh_row_start` by binary-searching the
row-major `sh_nz_idx` array. It does this once per CUDA block, before the
feature-tile loop. The table is reused across tiles, but reconstructed for each
matmul, batch item, and feature slab.

The BOO struct does not explicitly store these boundaries. `row_grp_ptr` indexes
group members in `row_ids`; `blk_nz_ptr` locates a whole block's nonzeros. Neither
describes the distribution of nonzeros among local rows within that block.
Boundaries must be per block because blocks sharing a row group can have
different per-row nonzero counts.

Example from our walkthrough:

```text
sh_nz_idx    = [0, 2, 5, 7], shift = 2
local rows  = [0, 0, 1, 1]
sh_row_start = [0, 2, 4]
```

### Proposed change and hypothesis

Compute row boundaries after row-major ordering in `csr_to_boo`. Store flattened
per-block tables with enough offset information to locate each table, stage them
on the GPU, and cooperatively load them into `sh_row_start` instead of performing
binary searches.

Hypothesis: removing repeated boundary searches reduces kernel time enough to
outweigh the additional global-memory traffic. This is not yet measured.

The current searches read indices already staged in shared memory. Precomputation
adds persistent host/GPU storage and transfer work. With int32 boundaries, table
storage alone is `4 * sum_b(n_r(b) + 1)` bytes: 132 bytes for a 32-row block,
excluding table-location metadata. The first comparison should retain the same
shared row-start table and synchronization so it isolates boundary construction.

### Experiment

- Compare the existing binary-search variant with a precomputed-table variant in
  the same build, using identical matrices, clustering, small-block policy,
  nonzero order, launch parameters, and numeric types.
- Interleave timing rounds in the same process. Measure both kernel time and
  complete matmul time; report variability and the amount of work sent to CSR
  or COO. Record extra storage, preprocessing time, and GPU staging time.
- Include sparse and dense blocks, imbalanced row populations, multiple group
  sizes, batches, and feature counts spanning multiple tiles and slabs.
- Verify identical row boundaries and correct multiplication, including empty
  local rows and blocks with contributions to rows also handled by CSR or COO.
  Keep the column-major kernel path working; the row-segmented path requires
  row-major entries.
- Run compilation and tests on the remote GPU machine, as required by CLAUDE.md.

### Relevant code

- `torch_fast_spdmm/kernels/boo_kernel.cu`: `boo_spmm_rowseg_kernel`
- `torch_fast_spdmm/boo_matrix/boo_matrix.hpp`: `boo_matrix_t`
- `torch_fast_spdmm/boo_matrix/boo_matrix.cpp`: `csr_to_boo`
- `torch_fast_spdmm/torch_fast_spdmm/boo.py`: GPU metadata staging and dispatch
- `torch_fast_spdmm/bindings/bindings_boo.cu`: CUDA binding

Results and decision: pending.

## 2. Separate Step B computation from Step C output updates

Status: proposed; not implemented or benchmarked.

### Observation and proposed change

`boo_spmm_rowseg_kernel` currently combines B and C: each `(local row, feature)`
work item accumulates its partial sum in a register and immediately issues a
global `atomicAdd` to Y. The proposed variant finishes computation into a shared
output tile first, then flushes that tile to global Y in a separate loop.

For each feature tile:

1. Step A: load the required X tile into shared memory, as today.
2. Step B: compute contributions using the shared sparse data and X tile; write
   the results into `smem_Y[local_row * P + feature]`.
3. Synchronize the block to establish a completed shared output tile.
4. Step C: use a separate loop to flush `smem_Y` to global Y with `atomicAdd`.
5. Synchronize before shared buffers are reused for the next tile.

For the first experiment, preserve the current row-segmented work assignment and
register accumulation within each work item, then store its completed sum into
shared memory. Every output-tile entry has one writer, so this requires neither
shared atomics nor a separate zeroing pass, provided every entry is assigned,
including zero sums and empty rows. This isolates separating B and C from changes
to the accumulation algorithm. Global atomics remain necessary because other BOO
blocks and the CSR/COO paths can contribute to the same output element. Preserve
the existing feature bounds check and zero-sum skip in the flush loop.

### Hypothesis and costs to measure

Hypothesis: separating arithmetic from global atomic updates improves scheduling
enough to offset staging the output in shared memory. This is not yet measured;
it does not inherently reduce the number of global atomics.

The straightforward implementation adds `max_nr * P * sizeof(scalar_t)` shared
bytes per CUDA block, plus shared stores/loads, another work loop, and a phase
barrier. The larger allocation may reduce occupancy. Update the launch allocation
and reported memory/occupancy estimates when implementing the variant.

### Existing related implementation and experiment

`boo_spmm_nnzflat_kernel` already separates B and C through `smem_Y`, but assigns
work per `(nonzero, feature)` and uses shared atomics. It is a related reference,
not an isolated test of this proposal: comparing it with the row-segmented kernel
also changes work distribution and accumulation.

Compare fused and separated row-segmented variants in the same build with
identical input formats and launch parameters. Interleave measurements, report
kernel and full matmul time, shared-memory use and occupancy, and inspect whether
atomic stalls change. Verify empty rows, zero sums, partial feature tiles,
multiple tiles/slabs, batches, and overlapping contributions from BOO and CSR/COO.
Run compilation and tests on the remote GPU machine.

Results and decision: pending.
