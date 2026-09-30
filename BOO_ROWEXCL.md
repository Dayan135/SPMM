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

## Profiling and Phase B status

The six requested Nsight Compute captures are running in job 21766000. They compare row-exclusive with current BOO on identical HEM groups for fem2d, fem2d_permuted, and diag_b1.0 at N=262,144, F=512. Performance numbers above are unprofiled; diagnostic counter results will be added separately.

Phase B is pending the other agent's pushed `CUSPARSE_DIRECT_SYMMETRIC.md` and symmetric harness. A fetch on 2026-09-30 found no commit for that path in the available remote branches. This report does not claim to have run the five-family C++ comparison yet. All work is isolated on `feat/boo-rowexcl`; no other branch or shared cluster checkout is modified.
