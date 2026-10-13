# Focused BOO continuation — 2026-10-12

**Status: the profiling, P×FPT and slab continuation below is complete.**
The three-allocation confirmation and reproduction details are in
[TILE_EXPERIMENTS.md](experiments/2026-10-12_tile_optimization/TILE_EXPERIMENTS.md). The original design below is retained
for provenance; these screens should not be rerun as pending work.

## Next priority after confirmation

Test coalesced output atomics while preserving the existing FPT computation.
Compare the current and experimental writes at identical labels/P/FPT/slab,
starting with webbase and diag_dd0.05 as regression controls, plus pwtk, dense
diagonal and permuted FEM as gain controls. Keep slab512 and use P64/FPT2,
P64/FPT4 and P128/FPT4. Validate row boundaries, partial warps, feature tails,
zero accumulators, multiple feature slabs and remainder coexistence. Account
for the cost of redistributing accumulators and destinations across lanes.

Record total atomic requests/sectors and the exact denominator of each ratio.
The reported 4/8/~16 sector amplification is
`lts__t_sectors_op_red.sum / l1tex__t_requests_pipe_lsu_mem_global_op_red.sum`.
In the webbase profiles, sectors per *L2* RED request remain 4, while total L2
requests increase 1×/2×/4× at FPT1/2/4. Reducing this amplification is an
intermediate diagnostic goal; success requires improved complete-call time,
including shuffle/register costs, and preserved gains on dense-block controls.

Use per-repeat shuffled order plus an identical-work control and three
independent allocations for finalists. Use explicit expected-no-regression
comparisons against FPT2 for webbase/diag_dd0.05, without requiring either
matrix to beat cuSPARSE to establish a successful kernel ablation.

The older `bench_spmm.round_orders` still rotates one seeded sequence. Replace
or explicitly bypass that method before using those harnesses for further
performance claims; retain historical samples and flag their small effects.
See [the review](experiments/2026-10-12_tile_optimization/TILE_REVIEW.md) for confirmed decisions and metric definitions.

## Completed experiment design

Execute through [the profiling-agent prompt](scripts/research_panel/PROFILING_CONTINUATION_TASK.md).
**Status 2026-10-12: executed; results in [TILE_EXPERIMENTS.md](experiments/2026-10-12_tile_optimization/TILE_EXPERIMENTS.md).**
Target workload: one fixed sparsity pattern reused across many SpMM calls.
The full FPT sweep is complete; this plan narrows the next work to understanding
and improving the row-segmented kernel. Existing results are in
[the analysis](experiments/2026-10-09_full_fpt/plots/ANALYSIS.md).

## 1. Profile the actual FPT variants

At F512, compare FPT1/2/4 with P64/slab512 on six fixed-label cases:

| Input | Saved labels | Reason |
|---|---|---|
| Boeing/pwtk | identity k32 | Strong real FPT gain; prior instruction-pressure profile |
| Williams/webbase-1M | identity k32 | FPT4 regression control |
| SNAP/roadNet-CA | METIS k64 | Weak FPT response; sparse blocks/remainder |
| generated diag_b1.0, N262144 | identity k32 | Dense blocks, strong FPT4 gain |
| generated fem2d, N262144 | HEM k32 | Local ordering; still loses to direct CSR |
| generated fem2d_permuted, N262144 | HEM k32 | Same matrix family with a BOO win |

Use unprofiled interleaved complete-call timings to decide performance. Use
Nsight Systems to locate initialization, main-kernel, remainder, launch and join
costs; use separate Nsight Compute passes for stalls, occupancy/memory, and
instruction mix. Include registers/thread, local-memory spills, allocated shared
bytes/block, occupancy limits from registers/shared memory/threads, achieved
occupancy, shared transactions/bank conflicts, DRAM/L2 traffic, barriers and
row divergence. Inspect compiled instructions for actual vector shared loads.

Record matrix/input/label/source/binary hashes and actual runtime settings.
The current real profiler defaults to FPT1; it must be extended or wrapped to
profile explicit FPT variants. Identify a limiting resource using measured
counters plus a controlled timing comparison; a high stall percentage alone
is insufficient.

## 2. Sweep P jointly with FPT

Keep threads256, slab512, batch1, labels, values, row-major rowseg, auto launch
and remainder threshold4 fixed. Screen P={16,32,64,128} × FPT={1,2,4} on the six
cases at F128/512; validate every pair. Add P8 only if the profiles motivate a
smaller tile. Record infeasible arena sizes instead of silently changing P.

Earlier P experiments predate FPT2/4. The full FPT panel used automatic P32
at F32 and P64 at F128/512, so the joint interaction is untested. P changes the
X tile, feature-loop/barrier count and shared-memory occupancy; P/FPT changes
warp-to-row assignment. Compare against both FPT1/P64 and FPT2/P64. Record the
rowseg arena actually passed to the launch, not an estimate from the older
kernel with a shared output tile. Include F130 for partial-tile correctness.

## 3. Feature slabs and F1024

Use selected P/FPT pairs on pwtk and generated dense diagonal and
natural/permuted FEM. Test slab widths {128,256,512,1024} at F512/1024.
Extend to roadNet/webbase if they fit and their profiles indicate parallelism,
staging or long-block duration could be limiting.

At F1024, these widths launch 8/4/2/1 slabs per BOO block. Smaller slabs expose
more independent work but repeat sparse staging. At a fixed P, slab width does
not change the shared-memory arena. Use P<=slab and slab divisible by P so a
P tile cannot cross a slab boundary. Validate F1000/1025 and use F128 as a
partition-equivalence control with slab>=128.

Use source-hashed builds or a small compile-time specialization dispatch. Hold
P/FPT fixed while comparing slab widths, then check the best two widths with
more than one P. Avoid the full P×FPT×slab cross-product before screening.

## 4. Confirm winners and assess practical benefit

Repeat finalists on three independent RTX3090 allocations, remeasuring the
current anchor and direct cuSPARSE in each. Include generated N1048576 as a
scale control where memory permits, and a held-out sparse noisy diagonal case.
Compare against the strongest measured direct algorithm in the same cell; fill
missing baseline coverage by testing fewer direct plans per process with repeated
BOO anchors. Count 20 timing samples as within-allocation repeats only.

Record complete-call gains, regressions, setup/selection costs and break-even
call counts. Keep default changes separate from selecting experimental winners.

## Deferred ideas, with conditions for revisiting them

- Sweep threads128/256/512 for finalists only if register/shared limits or
  P/FPT row divergence point to a launch-shape issue.
- Compare vector vs scalar X loads at fixed FPT if compiled instructions or
  shared-memory counters leave the source of the FPT gain unresolved.
- Measure batch>1 if Nsight shows insufficient work to fill the GPU. A fair
  baseline needs native cuSPARSE batching and separate-call ALG3 measurements;
  the current direct wrapper only accepts 2D X.
- Active-column compaction or a different local index format needs evidence
  that unused staging or index instructions remain limiting after tile tuning.

No new broad clustering sweep is needed. Reuse the existing matrix cache and
labels, and carry forward both the win and regression controls.
