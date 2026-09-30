# Row-exclusive BOO

Phase A is complete: three separate RTX 3090 jobs, all 84 workloads, and 9,576 successful SciPy FP64 reference checks including F=100/200/520. The implementation removes output initialization and atomics, but it is slower overall than both direct cuSPARSE and current BOO in this experiment.

| N | Row-exclusive speedup vs direct cuSPARSE, C++ | Row-exclusive speedup vs current BOO, full API |
|---:|---:|---:|
| 16,384 | 0.388× [0.387, 0.401] | 0.797× [0.783, 0.832] |
| 65,536 | 0.324× [0.319, 0.325] | 0.448× [0.435, 0.455] |
| 262,144 | 0.306× [0.305, 0.307] | 0.370× [0.367, 0.371] |

Each entry is the median and range of three job-level geometric means over 28 workloads. Higher is better; below one means row-exclusive is slower. Each family uses its best fixed configuration selected by the median across jobs. Current BOO includes allocation, output zeroing, and its remainder path; row-exclusive includes allocation in the API comparison. Both C++ arms use preallocated output and CUDA events around the execution call. These scopes are reported separately.

No workload beats direct cuSPARSE by at least 1.3× in every job; 58 lose by that margin. Against current BOO at API level, five win and 41 lose by that margin. Removing atomics alone does not make this implementation competitive across these workloads. Differences smaller than 1.3× are treated as inconclusive rather than wins.

## Implementation and validation

Measured source is `f65e4b2257e2bf1a97923923251576810a3be4bd`. The [Phase A report](BOO_ROWEXCL_PHASE_A.md) records binary/source hashes, nodes, job IDs, exact configurations, timing, every workload's median/range, and links to all raw configurations and samples.

The BOO format and HEM grouping are retained. Each 128-thread CTA owns a slice of at most 32 rows in one row group and a disjoint feature slab (16/32/64). It accumulates all block contributions in registers, adds CSR-remainder contributions, and stores every owned output exactly once, including empty rows. There is no output memset or atomic instruction in the row-exclusive kernel.

Scheduling splits oversized groups only by rows. The requested cap is 2,048 nonzeros per CTA; the effective cap rises to the maximum single-row degree when necessary, since a row cannot be split under this ownership rule. A CTA stages the union of its row group's column groups within an 8 KiB shared-memory budget, enlarged if needed for a single column group, and otherwise iterates through chunks. Split row slices currently stage the original group's union, which can duplicate loading across slices. All these structural choices, per-CTA nnz histograms, chunk counts, register counts, and resident-CTA bounds are recorded in the CSVs.

GPU tests passed: 281 passed, 12 skipped across row-exclusive, existing BOO, and direct-cuSPARSE tests (job 21734489). Tests cover poisoned preallocated output, empty rows/matrices, rectangular shapes, feature tails, forced row splitting/chunking, and CSR-only remainder execution. All 84 benchmark workloads passed in all three jobs; every HEM target and slab also passed the 126 extra workload cells at F=100/200/520 in job 1.

## Profiling

[Six Nsight Compute captures](BOO_ROWEXCL_NCU.md) completed in job 21766000 on ise-pheno-10 at N=262,144, F=512, using identical HEM groups for each paired mode. Row-exclusive achieved 66.27–66.35% occupancy and 1.50–1.73% `no_instruction` stalls. Current BOO achieved 99.63–99.69% occupancy on FEM and 33.03% on diag_b1.0; its corresponding stalls were 1.24–1.27% and 0.61%.

At slab 32, row-exclusive uses 64 registers/thread and has a resource bound of eight resident 128-thread CTAs per SM, or 66.7% of the RTX 3090's 48 warps. Measured occupancy is close to that bound. These captures do not show severe instruction-fetch starvation, but FEM loses occupancy relative to current BOO. The current design therefore avoids a collapse in achieved occupancy while still losing performance. It does not establish that ownership alone solves the earlier rowgroup performance problem.

Nearly all FEM row units require multiple X chunks at the selected k=32/slab=32 (99.45% natural, 99.24% permuted), whereas diag_b1.0 at k=64 requires none. Repeated block traversal, staging, and barriers remain plausible costs; these counters do not establish a causal breakdown. Profiler durations are diagnostic and are not substituted for benchmark timings. Raw captures and per-kernel counters are retained.

## Phase B status

Merged the other agent's symmetric harness and report at `c2ccde58c8d6849ebdc2b86956032516538f86fa` into `feat/boo-rowexcl` only. A separate adapter imports its unchanged PBR/BOO/diagnostic registry and adds row-exclusive. Validation and the three-replicate Phase B run are pending. All real methods use C++ CUDA events and preallocated output; the intentionally incorrect no-atomic/no-zero diagnostic remains separately labeled and excluded from correctness/performance rankings. Row-exclusive sweep entries share an output buffer with serialized calls to keep the full sweep within GPU memory; no rebinding or allocation is timed.
