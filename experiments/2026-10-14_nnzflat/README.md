# BOO nnzflat research plan — 2026-10-14

Status: **executed and confirmed** (2026-10-14). nnzflat with FPT1/2 does not
become competitive: the best nnzflat setting is slower than tuned rowseg and
direct cuSPARSE on every input across three allocations, and FPT2 is a confirmed
nnzflat regression. No default changed. Results: [REPORT.md](REPORT.md); the
plan below is unchanged; see [Execution record](#execution-record) for paths,
jobs and deviations.

Investigate whether distributing nonzero work across more CUDA threads makes
BOO nnzflat competitive, especially with small feature tiles P and large BOO
blocks. Add features per thread (FPT) **1 and 2** and measure their interaction
with P, thread count, block size and row imbalance.

## Questions and hypotheses

Distinguish two sizes: **T** is CUDA threads per block; **k** is the requested
maximum row/column group size in the BOO format. Each BOO block launches one
CTA per batch/feature slab. A larger k does not itself add CUDA threads.

| Hypothesis | Controlled comparison | Evidence against it |
|---|---|---|
| More CUDA threads reduce serial work per thread | T128/256/512 on the same BOO object, P and FPT | Lower work per thread but slower complete calls, fewer resident CTAs, or more inactive lanes |
| Small P improves the distribution of nonzero work and reduces tile memory | P8/16/32 versus P64/128 at fixed T/k/FPT | Extra passes/barriers, shared atomic pressure or poorer memory access outweigh savings |
| Large BOO blocks supply enough work and reuse to benefit nnzflat | k32/64/128; k16 is a small-block control in the controlled panel | Sparse staging, shared-memory limits, atomic contention or too few CTAs dominate |
| FPT2 saves sparse-index/value loads and addressing | FPT2 versus FPT1 at identical P/T/labels | Fewer independent work items, register pressure, bank conflicts or atomic scheduling outweigh reuse |
| nnzflat benefits from uneven row lengths | Balanced versus skewed synthetic blocks with equal total nnz, dimensions and labels | Shared atomic contention costs more than rowseg's imbalance |

Small P reduces work **per tile**, not total arithmetic. With z nonzeros in
one BOO block and R=FPT, Step B has `z * (P/R)` work items per full tile.
Its maximum loop-trip count per thread is `ceil(z * P / (R*T))`, and its
active-thread fraction in the first iteration is `min(1, z * P / (R*T))`.
Each item performs R multiplies and R shared atomic updates. Across F valid
features, mathematical multiply count remains z*F. Feature-tail padding can
add executed work; measure it separately.

At F512/slab512, P8/16/32/64/128 require 64/32/16/8/4 tile passes. The
existing nnzflat path has three barriers per pass plus initial sparse staging.
FPT2 halves the Step B work-item count, but keeps the shared atomic update
count for full tiles. Thus FPT2 deliberately trades some thread parallelism
for reuse; it need not win in the small-P regime.

With P8/FPT2, a warp can span eight nonzeros. If these share a row, shared
atomic contention may increase. Test and profile this rather than assuming
more nonzeros per warp means more useful parallelism.

## Implementation required before measurement

Work on **BOO** `boo_spmm_nnzflat_kernel`, not the PBR nnzflat variants.
Current `boo_matmul(..., row_seg=False, features_per_thread=2)` is rejected,
and the C++ selector forces nnzflat FPT1. This plan requires actual kernel
and dispatch support, not just a harness parameter.

1. Keep the original FPT1 kernel as a reference. Add an nnzflat FPT template
   with specializations 1 and 2. For R=2, use `LANES=P/2`, map each work item
   to `(nonzero, adjacent feature pair)`, decode its row/column and load its
   value once, then issue two scalar shared atomic adds into `smem_Y`.
   Use an aligned float2 shared X load for the initial R=2 implementation.
   Keep X staging, output zeroing/flush, barriers and sparse staging unchanged
   to isolate Step B. FPT2 does not imply a vector atomic operation.
2. Expose FPT1/2 for explicit `row_seg=False` in Python/C++ and the timing
   wrappers. Use an explicit nnzflat setting; do not silently ignore FPT or
   repurpose rowseg's setter without updating its API meaning. Set the desired
   mode at every call so interleaved rowseg/nnzflat/FPT comparisons are correct.
   nnzflat FPT4 remains unsupported in this experiment.
3. Add nnzflat launch introspection using the same selected specialization
   and arena as the launcher: requested/effective P, T, FPT, slab, grid,
   registers, static/dynamic shared memory, local bytes and active CTAs/SM.
   Existing `boo_rowseg_launch_info` is not suitable for nnzflat. Audit actual
   kernel names with Nsight Systems or a CUDA profiler, including R=2.
4. Primary measurements use float32. Preserve existing float64/FPT1 support;
   either implement and validate float64/FPT2 or reject it explicitly.
   Verify float2 alignment for P, shared regions and feature pairs; guard
   partial features and keep barrier participation uniform.

No shared-atomic elimination, warp aggregation, changed flush assignment,
slab tuning, bucketing or launch-bound changes in the initial experiment.
If a later ablation changes one, give it a separate named arm.

## Fixed conditions and feasibility

Primary settings: float32, batch1, seed0, slab512, row-major nonzeros,
explicit `row_seg=False`, auto launch, csr_mode0, remainder threshold4,
existing zero/duplicate policy. Reuse canonical matrices and saved labels.
Do not cluster, build, stage or prepare a plan inside the timed call.
Current prepared execution plans require rowseg; use the unbucketed nnzflat
path and check that the harness does not silently substitute rowseg.

At fixed input/method/k, build and stage **one BOO object** for the entire
P/T/FPT comparison. Freeze group labels, max_block_nnz constraints and
remainder policy across launch settings. Across k, the representation changes;
report block count, actual group sizes, nnz/block and remainder fraction.
Changing k therefore measures a representation-plus-kernel interaction.

For float32, the current launch arena is:

```text
S = 6 * max_nnz_blk
  + 4 * (max_nr + max_nc)
  + 4 * P * (max_nr + max_nc)
  + 16 bytes
```

FPT2 does not reduce this arena. All CTAs receive the allocation based on
global maxima, even if most blocks are much smaller. Small P reduces the
X/Y tile term but cannot fix excessive sparse staging. A full 128x128 block
already needs 98,304 bytes for values/indices alone, before tiles and IDs.
Check the actual device and specialization limits, including >48 KiB opt-in.

Retain every requested cell as measured, an identical-label alias, unsupported
with a reason, or failed pending repair. Record the nnzflat arena and actual
P; never lower k/P/T, alter labels, cap nnz, or route extra work to remainder
to make a requested cell fit. Verify feasible occupancy, not just byte size.
Do not interpret rowseg's k128 regression as proof about nnzflat.

## Stage 0 — correctness and unchanged-work control

Build and test on a remote GPU node in a new isolated snapshot, following
[CLAUDE.md](../../CLAUDE.md). Preserve the shared extension and existing runs.

- Validate both FPT variants against CPU float64 multiplication of the
  original matrix with the established row-wise numerical bounds. Atomic
  accumulation order can change; bitwise agreement is not required.
- Exercise all P/T specializations, several loop strides per thread, balanced
  and heavily skewed rows, duplicate destinations/cancellation, explicit zeros,
  irregular rectangular groups, empty rows, and blocks sharing output rows.
- Include all-BOO, mixed BOO+CSR/COO and remainder-only paths, batches and
  nondefault streams. Check F={1,7,9,33,65,129,511,513,1025} for padding and
  slab tails; add an input with >48 KiB shared-memory opt-in.
- Preserve column-major FPT1 and test new FPT2 on both nonzero orders.
  Run relevant BOO/FPT regressions and the small sanitizer subset remotely;
  extend the tests to cover new FPT2 accesses, races and synchronization.
- Compare original FPT1 with the new FPT1 at identical work and inspect compiled
  code. Carry an identical-work timing control through subsequent stages.
  Resolve a systematic control regression before interpreting FPT2 gains.

## Stage 1 — controlled mechanism panel

Generate two blocked synthetic patterns at N16384 with seed0 and fixed
contiguous row/column labels: balanced local row lengths and highly skewed
local row lengths. Match dimensions, total nnz, values distribution and total
nnz in each base 16x16 block. Use the same two matrices throughout the k sweep;
coarsen labels to k32/64/128 rather than regenerating A for each k. Report the
actual skew, density and nnz/block after grouping. Avoid unintended duplicate
entries or differing zero policies. Ensure enough blocks to exercise many SMs.

Sweep both patterns at:

| Axis | Values |
|---|---|
| k, maximum row/column group size | 16, 32, 64, 128 |
| P, feature tile width | 8, 16, 32, 64, 128 |
| T, CUDA threads per block | 128, 256, 512 |
| FPT | 1, 2 |
| F, dense feature width | 128, 512 |

This is **480 requested nnzflat configurations** before feasibility accounting.
Choose matched sparsity low enough that at least some k128/P8 cases fit;
retain dense/oversized cases as explicit feasibility controls in validation.
Compare rowseg at the same P/T/labels/FPT where supported, and a rowseg
P64/T256/FPT1/2 anchor. Extra baseline calls are outside the 480 count.

This stage tests T and P without clustering variability, then tests k without
changing the input matrix. It also isolates the row-imbalance mechanism better
than comparing unrelated real matrices.

## Stage 2 — focused application screen

Use eight input/labeling arms. Generated inputs use N262144; real inputs retain
native dimensions. Freeze labeling choices before timings:

| Input | Labeling method | Role |
|---|---|---|
| Boeing/pwtk | identity | Strong reuse/instruction-pressure case |
| Williams/webbase-1M | identity | Sparse irregular/atomic regression control |
| SNAP/roadNet-CA | METIS | Sparse blocks and remainder control |
| Generated fem2d | HEM | Naturally ordered FEM |
| Generated fem2d_permuted | HEM | Same family after locality destruction |
| Generated diag_b1.0 | identity | Dense-block and shared-memory limit control |
| Generated diag_dd0.05 | identity | Sparse noisy diagonal |
| Generated diag_skewed | identity | Skew/remainder stress case |

Reuse and hash-check existing labels where possible; generate missing labels
with the same grouping rules. Identity target sizes may produce identical
labels: keep aliases, not fabricated independent observations.

Cross **k={32,64,128}, P={8,16,32,64,128}, T={128,256,512}, FPT={1,2},
F={128,512}**: **1,440 requested nnzflat configurations**. Small-P/big-k
is the target; P64/128 and k32 are required controls, not presumed losers.
The earlier rowseg result against a thread sweep does not settle nnzflat's T
question, since nnzflat has a different work assignment and shared arena.

For each input/k/F, remeasure rowseg P64/T256/FPT1/2 on the same A/X and BOO
object. For leading nnzflat candidates, also measure rowseg at their exact
P/T/FPT. Report both comparisons: same-setting kernel effect and independently
tuned performance. If a winning labeling is worth revisiting, treat a second
method or column-major nnzflat comparison as a separate follow-up, holding A,
membership, values and all other launch settings fixed. Do not expand to a
four-method clustering sweep initially.

## Timing and profiling protocol

Use the canonical loader/reference and matched complete-call scopes from the
existing research harnesses. Measure allocating/zeroing Y, main BOO execution,
remainder and join costs. CUDA timing must cover all streams and completion.
Kernel-only timing is a diagnostic alongside complete-call time.

Use at least five warmups and 20 repeats per arm. Independently shuffle arm
order for every repeat with recorded seeds; do not use rotation of one fixed
sequence. Pair comparisons on the same allocation/A/X. Use bounded parameter
groups when the full grid is too slow, retaining repeated common anchors and
explicitly marking comparisons that cross groups/processes. Store repeat,
position, predecessor, raw time, GPU/environment and source/extension hashes.

Profile a small subset after unprofiled timing: small versus large P, T256
versus T512, FPT1 versus FPT2, balanced versus skewed blocks, and a large-k
gain/regression or feasibility boundary. Keep the other axes identical.
Collect:

- actual grid/CTA count, active lanes and Step B work/loop-trip estimates;
- registers, spills/local bytes, dynamic shared memory, resident CTAs/SM,
  achieved occupancy and resource limits;
- executed sparse-index/value/X loads, arithmetic and atomic instructions,
  shared bank conflicts/transactions, barrier counts/stalls, short/long
  scoreboard, MIO/ALU pressure, DRAM/L2 traffic and global atomic traffic;
- block/row nnz distributions (p50/p95/max), skew within each block, active
  columns versus staged columns, BOO/remainder fractions and per-phase times.

Inspect SASS for actual float2 loads and scalar shared atomics. Record exact
counter names and denominators; distinguish shared atomics from global output
updates and requests from sectors. Profiler duration and high stall percentages
alone do not establish a runtime gain or a cause. If FPT2 wins but its mechanism
is unresolved, add a scalar-load FPT2 arm at identical mapping to separate
vector loading from index/value reuse; do not mix this into the primary grid.

## Stage 3 — independent confirmation and decision

Freeze finalists after screening and before confirmation. On each input at
each F, retain the best feasible nnzflat candidate, its identical P/T/k FPT
counterpart, and the best screened alternative with a different P or k.
Also retain a matched T256/T512 comparison where more threads helped, the
rowseg anchors, and identical-work control. Keep pwtk, webbase, roadNet,
natural/permuted FEM, dense/noisy diagonal and the skew control even if nnzflat
loses. Bound the confirmation manifest explicitly rather than retuning it.

Remeasure these arms on **three independent RTX3090 allocations** with per-repeat
shuffles. Include F256 and generated N1048576 as held-out feature/scale checks
for frozen candidate settings; failures or infeasibility remain visible.
For practical performance claims, measure all four supported direct cuSPARSE
algorithms with matched complete-call scope in these allocations, recording
resource exclusions. Keep original-coordinate calls as the primary comparison;
any resident reordered-coordinate timings need separate labels.

Report paired median ratios per allocation, raw dispersion and the
identical-work control spread. A useful candidate should show a repeatable
complete-call improvement with consistent direction across the three allocations
and an effect larger than the observed control variation. Mark small or mixed
effects unresolved; 20 within-allocation repeats are not 20 independent GPUs.
Treat >5% confirmed regressions as material for any proposed general setting.

Answer each hypothesis separately, even if nnzflat still loses to rowseg.
If nnzflat wins only for large/skewed blocks, propose a documented manual
setting or subsequent selection experiment. Do not promote a universal default
from a post-hoc winner. Record incremental setup/storage/selection costs and
approximate break-even call counts for applications with repeated SpMM.

## Execution constraints and deliverables

Execution is a later step; this request produces the plan. For that step, use
a new isolated source/build/result directory and unique drivers. All CUDA
compilation, correctness tests and timings run remotely. GPU Slurm jobs use
one RTX3090, at most 60 GB memory, no explicit CPU-count request and
OMP_NUM_THREADS=1; cap concurrency at two GPU jobs and four CPU clustering
jobs. Verify current cluster requirements before submitting. Preserve historical
results, user edits and the shared installation; log job IDs and failures.

Separate CPU preprocessing from GPU execution using distinct Slurm jobs:

- **CPU jobs:** generate/load canonical inputs, compute clustering labels,
  construct BOO data, compute CPU reference results and prepare validation
  artifacts. Use the CPU partition and its normal QoS, with CPU/memory requests
  appropriate to the task. Save artifacts with input/label/configuration hashes.
  Schedule analysis and plotting on CPU resources or locally as appropriate.
- **GPU jobs:** consume the completed, hash-validated artifacts, stage data,
  run CUDA correctness checks, benchmarks and profiling. Submit with dependencies
  on successful CPU preparation. Keep lightweight loading/validation on the
  allocated CPUs, but do not run expensive clustering or CPU reference generation
  while holding a GPU. Chunk validation artifacts if needed to stay within the
  GPU job's memory limit. CUDA builds and GPU tests still require GPU nodes.

This separation follows the previous HPC Cluster IT warning: the October 13
GPU array was cancelled after requesting **16 CPUs and 128 GB per GPU job**.
Use **at most 60 GB** and omit both `--cpus-per-task` and `--cpus-per-gpu` for
GPU submissions, allowing Slurm's default CPU allocation per GPU. Do not carry
CPU preprocessing resource requests over to GPU jobs. See the
[incident record](../2026-10-13_eran_fpt/README.md#recovery-after-cluster-cancellation).

Deliver an audited manifest accounting for 480 controlled plus 1,440 application
screening requests, with baseline/confirmation counts separate. Retain summary
tables and a report answering the five hypotheses. Produce P×T heatmaps for
each k/FPT, paired FPT2/FPT1 plots, k curves annotated with actual arena/block
statistics, and nnzflat/rowseg/direct comparisons. Archive raw samples, labels,
full profiler captures and logs with checksums under the
[archive policy](../../RESULTS_ARCHIVE.md). No performance conclusion until the
numerical and actual-launch audits pass.

Related evidence: [original rowseg comparison](../../docs/reference/BOO_FORMAT.md#13-optimizing-step-b--the-row-segmented-kernel),
[tile review](../2026-10-12_tile_optimization/TILE_REVIEW.md),
[current FPT screening](../2026-10-13_eran_fpt/analysis/screening_report.md), and
[BOO correctness contract](../../docs/reference/BOO_TESTING.md).

## Execution record

Report and hypothesis answers: [REPORT.md](REPORT.md). Tables and figures:
[analysis/](analysis/) (coverage audits, per-stage paired tables, confirmation
pairs/verdicts/controls, profiler counters, setup costs, Stage 0 launch/SASS/nsys
audits). Per-task summaries and manifests:
`tests/torch_fast_spdmm/benchmark/results/nnzflat_20261014/`.

**Implementation** (uncommitted, in the working tree):
`boo_spmm_nnzflat_fpt_kernel<scalar_t, T, P_LOG2, FPT>` (FPT 1/2, float32 and
float64) beside the unchanged original `boo_spmm_nnzflat_kernel` in
`torch_fast_spdmm/kernels/boo_kernel.cu`; process-wide `set_boo_nnzflat_fpt` /
`set_boo_nnzflat_reference` and `boo_nnzflat_launch_info` (bindings_boo.cu);
`BOOTimedSpMM(..., nnzflat_fpt, nnzflat_reference)`; `boo_matmul(...,
features_per_thread, nnzflat_impl)` applies every rowseg/nnzflat setting at each
call. Defaults are unchanged: `row_seg=False` with FPT1 still launches the
original kernel, and nnzflat FPT4 is rejected. Tests:
`tests/torch_fast_spdmm/test_boo_nnzflat_fpt.py` (one assertion in
`test_boo_fpt.py` changed from "nnzflat FPT2 rejected" to "nnzflat FPT4
rejected"). Drivers: `benchmark/nnzflat_common.py`, `benchmark/bench_nnzflat.py`,
`benchmark/profile_nnzflat.py`, `scripts/research_panel/nnzflat_{job.sh,prep.py,
manifest.py,stage0.py,launch_audit.py,analyze.py,report.py,profile.sh,
profile_report.py}`.

**Remote:** isolated snapshot `/home/dayanb/research/nnzflat-20261014`
(working tree incl. uncommitted edits at `1b7d052`; `benchmark_source.json`),
run root `.../nnzflat_run`. The shared installation and earlier run roots were
not modified. All Stage 1–3 results used extension SHA-256
`9d4318d5149ce732a5dc38d8c3e0bd9a507491561069b956563cb0d21214f1cd` and source
hash `218e28bb8c67de18a196b2a7bae30b62b36f0a52b417d5c5b189d1fef2ba50a0`
(torch 2.10.0+cu128, CUDA 12.8, RTX 3090, driver 580.126.20). Labels are
hash-validated byte copies of `/home/dayanb/research/eran-fpt-20261013/labels`;
the Stage 1 synthetic labels are new contiguous groupings.

| Slurm job | Role | Resources | Outcome |
|---|---|---|---|
| 22955017, 22955088, 22955142 | Stage 0 attempts | GPU 1×3090, 48 GB, default 6 CPU | Failed in post-test audit scripts (import order; duplicate dict key; nsys name spelling); all tests passed in 22955088/22955142. Logs in `logs/failed_attempts/` |
| 22955020/21, 22955089/90, 22955143/44 | Prep / manifest | — | Cancelled by the failed Stage 0 dependency |
| 22955297 | Stage 0: build, tests, sanitizers, launch/nsys/SASS audits | GPU | Completed |
| 22955298 (0–14, ≤4 concurrent) | CPU preparation: inputs, labels, BOO digests, float64 references | cpu, 2 CPU, 32 GB | 15/15 completed |
| 22955299 | Manifests (480 + 1,440) | cpu | Completed |
| 22955316 | Stage 1 screen | GPU | Completed |
| 22955317 (0–7, ≤2 concurrent) | Stage 2 screen | GPU | 8/8 completed |
| 22959249 | Stage 1 profiling (13 configs, nsys + ncu full) | GPU | Completed |
| 22959545 | Stage 2 nsys launch audit | GPU | 16/16 cells match |
| 22959547 (0–2, ≤2 concurrent) | Confirmation allocations 1–3 (cs-pheno-09/10/08) | GPU, 48 GB | 3/3 completed |
| 22959548 | Stage 3 nsys launch audit | GPU | 34/34 cells match |
| 22959568 | Stage 2 profiling (12 configs) | GPU | Completed |
| 22965802 (0–2) | Supplementary roadNet F512 with direct baselines, fresh processes | GPU | 3/3 completed |

The coverage audit and finalist freeze (`nnzflat_analyze.py audit|freeze`) ran
on the login node: a few seconds of CSV reading. The confirmation manifest
`manifests/stage3.json` was frozen by its documented rule before any confirmation
job ran.

**Archive:** `raw-results-nnzflat-20261014.tar.gz` (68 MB, 978 files: raw
samples, labels, BOO/prep artifacts, synthetic inputs, ncu reports, nsys traces,
launch audits, Stage 0 logs/SASS, job logs), SHA-256
`b4d9931491c26eff755779d6262d02885126a47e67ddc70c3f2494502117e62f`, at
`nnzflat_run/` on the cluster and in `local-archives/`. The float64 reference
arrays (66 GB) and deterministic generated inputs are excluded; their SHA-256s
are in `artifacts/prep/*.json` and regenerate with `nnzflat_job.sh prep`.
