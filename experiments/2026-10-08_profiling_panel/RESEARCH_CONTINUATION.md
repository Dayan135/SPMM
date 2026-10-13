# BOO research continuation — 2026-10-08

Target workload, confirmed by the user: a fixed sparsity pattern reused across
many SpMM calls, with changing dense inputs. Changing sparse values should not require
reclustering; value-update costs should be measured separately if relevant.
This is a proposed plan, not a new performance result.

## Research questions and existing evidence

The next phase should answer two questions:

1. Which measurable matrix properties predict a BOO advantage over a strong CSR
   implementation, including CSR after the same reordering?
2. On those matrices, which part of BOO's execution consumes the remaining time,
   and which format or kernel change reduces that cost?

The [latest clustering experiment](../2026-10-06_clustering/CLUSTERING_EXPERIMENT.md) supports starting
with actual group sizes around 32–64. This is a useful starting range, not a
universal optimum: some FEM cells prefer 16, and HEM/Graclus targets 128/256
sometimes produce the same 64-row groups. Deduplicate identical labels before
treating them as separate configurations.

At N=1,048,576 and F=512, the best measured BOO time on natural FEM is 43.91 ms,
versus 27.81 ms for torch.sparse.mm. On permuted FEM it is 45.38 ms, versus
53.93 ms. These single-session results indicate locality recovery, but do not
establish an advantage over well-ordered CSR. Backend differences are generally
small; another broad backend sweep is a lower priority than profiling and
real-matrix coverage. See [the saved table](../2026-10-06_clustering/plots/xl/best_per_backend.csv).

The [matched direct-cuSPARSE experiment](../2026-09-29_direct_cusparse_symmetric/CUSPARSE_DIRECT_SYMMETRIC.md) already
separates Python-call and C++ execution scopes. Carry that stronger baseline
forward. The latest clustering tables use torch.sparse.mm as their baseline;
their reported speedups should retain that qualification.

Previous negative results constrain the next experiments:

- Stored int32 row boundaries slowed the existing format by about 2%.
- Multi-grid size bucketing regressed across the tested cells.
- Skipping stream/event setup for pure BOO had no consistent benefit.
- The earlier group-shape/thread/tile sweep did not establish a robust win.

These results are documented in [BOO_OPTIMIZATION.md](../2026-09-23_boo_optimization/BOO_OPTIMIZATION.md).
Revisit them only when a new representation or measured workload difference
changes the relevant cost. The proposed separation of rowseg computation and
output flushing in [RESEARCH_IDEAS.md](../../RESEARCH_IDEAS.md) remains untested; it
adds shared storage and barriers without reducing global atomic count, so it
should follow evidence of a scheduling problem.

## 1. Establish a small profiling panel

Begin with six synthetic cases at N=262,144, F=128/512:

- Natural and permuted FEM: isolate ordering sensitivity.
- diag_b0.3 and diag_b1.0: vary density at fixed diagonal-block geometry.
- diag_skewed: exercise dense blocks plus sparse remainder.
- pattern_row_skewed: exercise row imbalance.

Use the current row-major rowseg implementation, feasible-first clustering,
auto launch, 256 threads, P=64 where it fits, and remainder threshold 4. Save
and reuse the labels. Compare actual group sizes 32/64, recording the selected
P; use k=16 as a control where earlier FEM results motivate it. Add a few small
and 1M cases after the panel identifies a bottleneck.

Collect three distinct views:

| View | Purpose | Measurements |
|---|---|---|
| Nsight Systems | Explain complete-call cost | Host gaps, output initialization, launches, BOO/remainder overlap and joins |
| Nsight Compute | Explain individual kernel execution | DRAM/L2 traffic, shared-memory dependencies/conflicts, register use/spills, occupancy, eligible warps, barriers and instruction mix |
| Unprofiled interleaved timing | Decide whether a change helps | Complete-call time and matched preallocated C++ execution time |

Profile the baseline and BOO on identical A/X. Treat stall percentages as
diagnostic evidence, not a direct decomposition of elapsed time. Profiler replay
and cache control can change execution; use unprofiled timing for performance
claims. [NVIDIA's profiling guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)
describes these effects. The repository's validated stall metric set in
`optimizer/performance.py` should remain separate from additional counter
collections, consistent with its recorded replay sensitivity.

Reuse `benchmark.profile_boo`, `benchmark.profile_torch_grid`, and the timed
bindings, but align their defaults/flags with the selected baseline. In
particular, `profile_boo` defaults to identity/keep/threshold=0 and does not
expose all the current sweep controls. `optimizer.quick` currently profiles
PBR; it is not yet a BOO profiling workflow. Extend the drivers to accept saved
labels and real-matrix inputs rather than profiling a different configuration.

Deliverable: a short bottleneck report for each case, with a ranked hypothesis
and the evidence needed to test it. Choose the first kernel experiment from
that report.

## 2. Benchmark real matrices early

Start with six matrices selected for different mechanisms, before choosing a
kernel redesign. Matrix roles below are experiment hypotheses; structural
statistics must verify them after loading.

| Matrix | Source classification | Reason to include |
|---|---|---|
| [Boeing/pwtk](https://sparse.tamu.edu/Boeing/pwtk) | Symmetric structural stiffness problem | Test engineering block structure beyond synthetic FEM |
| [Schmid/thermal2](https://sparse.tamu.edu/Schmid/thermal2) | Symmetric unstructured thermal FEM | Test a large, relatively sparse mesh problem |
| [AMD/G2_circuit](https://sparse.tamu.edu/AMD/G2_circuit) | Symmetric circuit simulation | Test low average row occupancy |
| [Williams/webbase-1M](https://sparse.tamu.edu/Williams/webbase-1M) | Weighted directed graph | Test asymmetry and measure hub/degree effects |
| [SNAP/roadNet-CA](https://sparse.tamu.edu/SNAP/roadNet-CA) | Undirected road graph, binary values | Test very sparse graph structure; report binary-value specialization separately |
| [LPnetlib/lp_cre_b](https://sparse.tamu.edu/LPnetlib/lp_cre_b) | Rectangular linear programming problem | Test independent row/column grouping |

Use the original full matrices; do not synthesize larger sizes by replication.
Record source URLs, file/content hashes, shape, stored entries, numerical nnz,
dtype and preprocessing. Canonicalize duplicates and define an explicit-zero
policy consistently across methods: pwtk contains explicit zeros. Expand
triangular storage according to the file's symmetry declaration, while
preserving the actual asymmetry of directed/rectangular matrices. Do not
symmetrize a matrix to make clustering convenient.

Initially use F=32/128/512, FP32 with an independent higher-precision reference,
and k=32/64. Test identity and one fixed clustering backend throughout; compare
other backends only on representative cases. Keep the existing CPU-label/GPU-
timing separation. Expand to 15–20 matrices and add unseen matrix families for
final validation once the first panel exposes useful structural regimes.

Include these baselines:

- torch.sparse.mm, for continuity with the latest plots.
- Direct cuSPARSE with supported CSR algorithms and reusable descriptors,
  workspace and preprocessing, following the existing matched-scope harness.
- cuSPARSE after the exact same BOO group-label permutation. Where appropriate,
  add RCM; row and column permutations must be separate for rectangular inputs.
- Existing PBR as an internal format control on a manageable configuration set.

For reordered CSR, include X gathering and Y restoration when external callers
require the original coordinates. Also report steady-state time when the
application can keep its features in reordered coordinates. For BOO, explicitly
state that row_ids/col_ids provide the equivalent mapping during execution.

Report individual matrices and family-level results as well as aggregate
speedups. Selection of a best configuration from a sweep is an oracle result;
a dispatcher must be assessed on held-out matrices.

## 3. Characterize matrices in terms of execution cost

Collect inexpensive CSR statistics before clustering, then block statistics for
each candidate grouping. Preserve distributions, not just maxima or means.

| Property | Cost or opportunity it may explain |
|---|---|
| Row/column degree quantiles, empty rows, degree skew | Row imbalance and hub-driven similarity-graph cost |
| Symmetry, bandwidth/locality, support overlap | Ordering and reuse opportunities |
| Actual group sizes and per-block nnz/density quantiles | Useful work per CTA and shared-memory pressure |
| Active rows/columns per block relative to full group sizes | Idle row work and unnecessary X staging |
| Sum of full column-group sizes over retained blocks | Logical X loads in the current kernel |
| Sum of active rows over retained blocks; block incidence per row | Output partial sums and potential atomic cost |
| Remainder fraction and its row distribution | Hybrid execution and overlap costs |
| Row occupancy distribution inside each block | Rowseg imbalance and alternative work assignment |
| Repeated masks/patterns, dense blocks and small repeated subblocks | Bitmap, implicit-index or dense-tile alternatives |
| Total format bytes, arena size and measured occupancy | Compression versus resource constraints |

`boo_info.x_loads` currently sums full column-group sizes, including columns a
particular block might not use. `y_writes` similarly sums full row-group sizes;
rowseg skips zero sums, so it is an opportunity count rather than an exact
atomic instruction count. Add active-support statistics to avoid conflating
these quantities.

For FP32, useful first-order quantities are:

```
useful_flops = 2 * nnz * F
logical_X_bytes = 4 * F * sum_b(ncols_in_group_b)
sparse_payload_bytes = 6 * retained_nnz * batch * ceil(F / 512)
potential_output_updates = F * sum_b(active_rows_b)
```

The sparse payload is loaded once per CTA, hence once per batch item and
512-feature slab. These quantities are not DRAM traffic: cache reuse, shared
memory and atomic behavior require counters. Model the remainder separately.

Example from the latest 1M natural-FEM result, HEM k=32/F=512: 37,650,496 nnz,
7,706,772 logical X-row loads and 272,714 BOO blocks. Sparse indices occupy
75.3 MB; logical X staging is 15.78 GB. Saving one index byte per nnz would save
37.7 MB, about 0.24% of that logical X volume. This motivates profiling X reuse
and shared resources before assuming index bandwidth dominates.

Fit a small interpretable cost model using measured traffic, output splitting,
CTA count and imbalance. Evaluate by held-out matrix/family, including feature
width and hardware as inputs. A useful eventual output is a cheap policy that
chooses CSR, reordered CSR or BOO and selects a small BOO parameter set. Charge
the policy for any clustering/sampling it actually performs.

## 4. Index and representation experiments

### First candidate: local CSR with uint8 columns

A 32x32 combined coordinate needs 10 bits; a 64x64 coordinate needs 12. Native
12-bit scalar loads are unnecessary for compression: packed bitstreams can be
read using wider words and decoded. Coalescing operates across threads, so a
wider load does not imply every index must occupy that many bits on disk or in
device memory. [NVIDIA's CUDA guide](https://docs.nvidia.com/cuda/archive/13.1.0/cuda-programming-guide/02-basics/writing-cuda-kernels.html)
describes the 32-byte global-memory transaction model.

However, the more natural first experiment for the current rowseg kernel is
to eliminate the stored row coordinate altogether:

```
local_row_ptr: uint16[nrows + 1] per block
local_col:     uint8[nnz] per block
values:        float32[nnz]
```

The row is known from the row segment; column values 0–63 fit in one byte.
Restrict the first prototype to actual groups <=64 on both axes, so the largest
local nnz offset is 4096 and fits uint16, including the end sentinel. Keep a
wider-format fallback for other shapes and the column-major path. For dense
256x256 blocks, the 65536 end sentinel would overflow uint16; never generalize
the offset type solely from the coordinate type.

For z nonzeros and r rows, structural payload changes from `2*z` to
`z + 2*(r+1)` bytes, before table-location metadata and alignment. It saves
space when `z > 2*(r+1)`; an additional int32 table-location entry changes that
condition to `z > 2*(r+1)+4`. Compare fixed-stride versus variable-length table
storage using actual total bytes, including padding.

At r=64, z=512: 1024 index bytes become 642 bytes before extra location metadata.
With values included, 3072 bytes become 2690 bytes: a 12.4% sparse-payload
reduction, not a 12.4% total-SpMM traffic reduction. In the dense limit, the
per-nnz value+index payload approaches 6 to 5 bytes.

This differs from the negative stored-boundary experiment: it replaces the
coordinate representation, instead of adding int32 boundaries while retaining
the 16-bit coordinates. Test three same-build controls: original coordinates
and searches, original coordinates plus compact boundaries, and compact
boundaries plus uint8 columns. A one-bucket execution-plan wrapper previously
added overhead; avoid attributing that wrapper cost to a new format.

Measure kernel/full-call time, all persistent bytes, load sectors, shared-memory
bytes, occupancy and decoder instruction cost. A smaller format may matter
through a resource threshold even when its DRAM saving is small.

### Follow-up representations, conditional on matrix statistics

| Representation | Storage opportunity | Main experimental concern |
|---|---|---|
| Packed 12-bit combined coordinates | 25% smaller index array; 6 to 5.5 bytes per FP32 nonzero before other metadata | Extraction/alignment overhead; unpack once per CTA versus repeatedly |
| Row counts plus uint8 columns | For <=64 columns, one byte per row count instead of two-byte row offsets | Cooperative prefix sum and synchronization |
| One 64-bit mask per row for <=64 columns | 8*r structural bytes; smaller than 2*z when z>4*r | Bit enumeration/popcount and rowseg decode cost |
| Dense tiles with implicit coordinates | Remove indices and exploit dense arithmetic | Padded zero work; numerical precision if using tensor cores |
| Per-pattern dictionary or repeated small-block encoding | Amortize structural metadata over recurring masks | Pattern frequency, dictionary lookup and added dispatch cost |

Start each experiment with one representation per run. A per-block hybrid may
eventually make sense, but separate launches or branching can consume its gain;
the prior negative bucketing result makes this a measured design choice.

## 5. Kernel experiments selected by profiling

| Evidence | First targeted experiment | Cost to track |
|---|---|---|
| X load latency / repeated staging dominates | Vectorized aligned loads, then Ampere cp.async pipelining | Register pressure, double-buffer shared memory and barriers |
| Many staged columns are unused in individual blocks | Block-local active-column compaction | Extra ID maps, build cost and loss of group-map sharing |
| Shared loads / instructions dominate | Several feature accumulators per thread; reuse each sparse value/index across features | Registers, spills and reduced occupancy |
| Long local rows / high work imbalance | Split long row segments and reduce cooperatively, preserving coalesced feature access | Reduction overhead and additional partial sums |
| Output atomic/initialization cost dominates | One CTA owns a row-group/feature tile and traverses its column blocks | Reduced grid parallelism, longer CTA tails and register/shared storage |
| Metadata load / arena constraint dominates | uint8-column local CSR experiment above | Boundary traffic, decoder work and actual occupancy thresholds |

The current mapping already distributes each row across feature threads; a
generic "warp per row" label alone does not specify a new algorithm. Define
which dimension is split, where partial sums live and how they are reduced.
Relevant prior work includes [Yang et al.'s load-balancing and coalescing study](https://arxiv.org/abs/1803.08601)
and [GE-SpMM's sparse-row reuse design](https://arxiv.org/abs/2007.03179).

For row ownership, include contributions that currently go to the remainder in
the ownership design, or explicitly synchronize and combine them. Plain stores
are correct only when the output has a single owner. The existing raw-store
diagnostic loses contributions; its speedup is not an achievable performance
claim. Prefer profiling cases with modest and high block-row fanout before
investing in this redesign, given the earlier unsuccessful PBR grouping work.

Ampere supports asynchronous global-to-shared copies; see [NVIDIA's tuning guide](https://docs.nvidia.com/cuda/archive/12.5.1/ampere-tuning-guide/index.html).
Dense-tile tensor-core work is a later branch if sufficiently dense blocks are
common and reduced-precision arithmetic is acceptable. Report accuracy and
zero-work inflation. Arbitrary sparsity does not satisfy the hardware 2:4
pattern automatically; [NVIDIA describes that constraint here](https://developer.nvidia.com/blog/sparsity-in-int8-training-workflow-and-best-practices-for-tensorrt-acceleration).

## 6. Improve clustering objectives after characterization

The existing backends share a support-overlap graph. Replacing its partitioner
does not necessarily address a poor execution-cost objective. Inspect:

- Degree bias from raw shared-column counts; compare normalized overlap only
  if degree-skewed matrices reveal a problem.
- The fraction of support excluded by the degree-1024 hub guard and the effect
  of top-32 neighbor truncation. These approximations may matter on real graphs.
- A grouping objective that trades X reuse against output splitting, row
  imbalance and arena size, rather than optimizing edge cut alone.
- Cheap alternatives for preprocessing, such as support sketches or sampled
  overlaps, if full similarity construction dominates amortization.

Fix the kernel and format during an objective comparison. Report both final
SpMM time and clustering cost. Hypergraph modeling is a longer-term option
when the measured cost model shows pairwise overlap misses relevant reuse.

## Execution order and decision gates

1. Assemble the six-case profiling panel and matched baselines; add the real-
   matrix loader and first six matrices. Reuse saved labels.
2. Collect structural distributions and profiles. Select one bottleneck to
   attack; keep the natural/permuted controls throughout.
3. Evaluate uint8 local CSR and the highest-priority kernel change separately,
   with unchanged labels, then test their combination only if both justify it.
4. Validate promising changes on additional, held-out matrices and independent
   GPU allocations. Revisit clustering objectives if the profiles support it.
5. Build a cheap selection policy only after the measurements support one.

All CUDA compilation, correctness tests, sanitizer runs and GPU benchmarks run
on the remote GPU machine, per CLAUDE.md. The initial plan requires no GPU jobs.

For each experiment, preserve source/binary hashes, labels, A/X hashes, actual
groups, selected P, remainder path, per-call samples and environment versions.
Use paired interleaved measurements and at least three independent allocations
for claims worth promoting. Check correctness before timing, including empty
rows, remainder overlap, boundary dimensions, batches, multiple slabs and
floating-point cancellation. Use sanitizer checks for changed memory or
synchronization behavior.

Choose tuning configurations on development matrices; assess selection on
held-out matrices. A proposed practical gate is a reproducible >=10% complete-
call improvement in the targeted regime, with regressions quantified and a
fallback for other regimes. Smaller improvements can qualify if independent
runs establish them and implementation cost is low. This is a prospective
decision rule, not a claim that 10% differences are automatically significant.

Report preprocessing, construction, transfer and any per-call reorder/update
costs. If the per-call saving is positive, estimate amortization as

```
calls_to_break_even = ceil(extra_one_time_cost / per_call_saving)
```

Use the matched baseline's setup costs in the difference. If per-call saving is
nonpositive, there is no finite break-even point. Record negative results with
their structural regime and mechanism, so later work can distinguish a new
hypothesis from repeating an already unsuccessful experiment.
