Work in `/Users/dayan/local/research/SPMM-boo`.

Execute the focused BOO continuation in `NEXT_EXPERIMENTS.md`: explain the FPT
speedups/regressions through profiling, then screen P and feature-slab settings
and independently confirm useful gains. This is an execution task: implement
necessary drivers, validate them, run the experiments and deliver reports/plots.
The workload reuses one sparse matrix pattern across many SpMM calls. Start with
batch1. Do not substitute a new clustering sweep for kernel analysis.

Read `CLAUDE.md`, `NEXT_EXPERIMENTS.md`,
`experiments/2026-10-09_full_fpt/FULL_FPT_BENCHMARK.md`,
`experiments/2026-10-09_full_fpt/plots/{README.md,ANALYSIS.md}` and the relevant kernel/bindings
before editing. Inspect `git status`/`git diff`: this shared worktree contains
uncommitted research and user edits. Preserve them. Do not commit or push.

## Completed evidence and resources

The full sweep completed 104 CPU and 104 GPU shards, with 1,836 measured BOO
combinations and 36 explicitly duplicate-label combinations. At F512, BOO wins
on pwtk, dense diagonal and permuted FEM but still loses on natural FEM and the
other five real matrices. FPT2 has no >5% regressions across 612 measured cells;
FPT4 has larger dense-block gains but 29 >5% regressions. Independent timing
samples within a shard are not independent GPU allocations.

Local raw/derived data:
- `tests/torch_fast_spdmm/benchmark/results/full_fpt_20261009/`
- `experiments/2026-10-09_full_fpt/plots/{best_configs.csv,paired_fpt.csv,fpt_statistics.csv}`
- `tests/torch_fast_spdmm/benchmark/results/fpt_handoff_20261009/profiles/`
  for earlier real FPT1 traces/instruction mix.
- `tests/torch_fast_spdmm/benchmark/results/research_panel_20261008/`
  for earlier synthetic FPT1 traces.

SSH alias `bgu-slurm`; all compilation and tests run remotely. Use an isolated
new remote snapshot, in-place build, and PYTHONPATH to that snapshot; do not
replace the shared installed extension. Request an explicit RTX3090 with
`--partition=rtx3090 --account=erant --qos=erant --gres=gpu:rtx_3090:1`.
Activate `fem`; set CUDA_HOME and runtime/compiler paths per CLAUDE.md.
Import torch before importing the CUDA extension directly.

Existing remote snapshots are read-only inputs for you:
- `/home/dayanb/research/boo-full-fpt-20261009`: complete source/build,
  `labels/`, `sweep.json`, `benchmark_source.json`, and `results/task_NNN/`.
- `/home/dayanb/research/real-matrices-20261008`: canonical `matrix_cache/`
  and earlier real labels/results.
- `/home/dayanb/research/real-matrices-fpt-20261008`: validated FPT build.
- `/home/dayanb/research/boo-profile-20261008-a`: baseline synthetic labels/traces.
- `/home/dayanb/research/boo-fpt-control-20261009`: focused FPT control.

The full snapshot has isolated PyMETIS dependencies in `deps/`; include it only
if your imports require it. The validated experimental extension SHA256 was
`bc688ff876317ddee01a888d5de6d9ac08b7ea87a331090e03021daaa0ff607c`.
Do not assume the current local kernel still hashes identically: inspect user
edits and build the exact source you intend to measure. Preserve that source
and distinguish its new measurements from the historical ones.

## Scope and implementation

Prefer new drivers/tests/scripts under unique names, such as
`benchmark/profile_tiles.py`, `benchmark/bench_tiles.py`,
`benchmark/test_tiles.py` and `scripts/boo_tile_experiments/`. Reuse the canonical
loader, saved-label validation, independent numerical reference and timing
helpers rather than reimplementing matrix semantics. Preserve directed and
rectangular input behavior and the existing explicit-zero policy (keep).

The current `benchmark/profile_real.py` does not pass features_per_thread, so
it profiles FPT1. Extend via a new driver or minimal reviewed parameter plumbing
and prove the requested variant is the one launched. `scripts/research_panel/
full_sweep.py` illustrates registering generated CSR in the real-panel harness.
The synthetic and real label archives have different naming/metadata schemas;
use their matching loaders. Prefer the unified full-sweep labels for this work.

Kernel/build changes needed to vary `BOO_FEAT_BLOCK` are in scope. Keep changes
limited to experiment parameterization and required correctness fixes. Preserve
FPT1/2/4 and default slab512. Do not change kernel defaults or rewrite the sparse
format in this task. Avoid wholesale edits to the external real-agent sources
(`real_matrices.py`, `bench_real.py`, `profile_real.py`, `test_real_matrices.py`)
and to existing profiling/plotting scripts when a new adapter suffices.

1. Build and validate a current baseline, then reproduce the selected cases.
   At F512, fixed P64/slab512/threads256/batch1 use:
   - Boeing/pwtk, identity_k32.
   - Williams/webbase-1M, identity_k32.
   - SNAP/roadNet-CA, metis_k64.
   - generated diag_b1.0, N262144, identity_k32.
   - generated fem2d and fem2d_permuted, N262144, hem_k32.
   Load the exact saved labels, never cluster in the GPU timing phase. Use
   row-major rowseg, auto launch, remainder threshold4, seed0, float32.
   Reproduce FPT1/2/4 in interleaved timing cells with matching original-order
   cuSPARSE baselines. Later add pwtk hem_k32 if needed to confirm its full-sweep
   winner rather than generalizing an identity result to a different labeling.

2. Profile those variants and identify the actual bottleneck in each case.
   Validate first, warm up, capture one call with CUDA profiler start/stop.
   Keep Nsight Systems, the established `optimizer/performance.py:METRICS`
   stall collection, occupancy/memory collection, and instruction mix in
   separate invocations. Do not append unrelated counters to METRICS; repository
   notes document replay effects. Use OCCUPANCY_METRICS/installed Nsight sections
   where available and record unavailable counters instead of guessing.

   Collect and explain:
   - Nsight Systems: main kernel, memset, remainder, actual overlap, stream joins
     and launch/host gaps. Do not sum overlapping kernels as full-call latency.
   - Shared memory: actual dynamic bytes/CTA, estimated residency, executed loads/
     stores, wavefronts/bank conflicts, MIO/short-scoreboard and barrier stalls.
     Rowseg has no shared Y tile. X staging is n_c×P, including unused group
     columns; use per-block active-column support to quantify logical waste.
   - Registers: registers/thread, theoretical occupancy limit due to registers,
     local-memory traffic/spills, achieved occupancy, active blocks/SM and other
     limits (shared memory, warps/threads, launch bounds). High achieved occupancy
     does not by itself prove efficient execution or identify a bottleneck.
   - Memory/compute: DRAM/L2 traffic/throughput, long-scoreboard stalls, FMA and
     instruction counts, and eligible/issued warps where available.
   - Work mapping: P/FPT lanes per row; row-length variation when a warp spans
     multiple rows. Inspect compiled instructions to check whether float2/float4
     source loads become vector shared-memory loads. Distinguish evidence from
     an inference about the cause of a speedup.

   Produce one short diagnosis per case with supporting counters and a
   controlled timing comparison. Profile a validated direct algorithm selected
   from same-cell timings; do not assume DEFAULT or ALG3 is always best.

3. Execute the P×FPT screen in NEXT_EXPERIMENTS.md.
   P={16,32,64,128}, FPT={1,2,4}, F128/512 on these fixed-label cases. Keep slab512
   and threads256 fixed. Record requested and effective P: the launcher reduces
   P when P>F. Validate all variants and report infeasible ones rather than
   quietly falling back. Check the exact rowseg arena/launch attributes;
   `bench_clustering.feature_tile` estimates the older shared-output arena and
   should not define feasibility for this explicit rowseg sweep. Reuse one BOO
   object across variants where possible and reset the process-wide FPT setting
   before every BOOTimedSpMM.run/timed_run invocation. Never run setting-changing
   calls concurrently. Report matched complete-call and C++ scopes separately.
   Add F130 correctness; P8 or thread sweeps require a concrete profiler reason.

4. Execute a focused slab/F1024 screen using the leading P/FPT pairs.
   Start with pwtk, dense diagonal and natural/permuted FEM; widths
   {128,256,512,1024}, features512/1024. Keep P<=slab and slab divisible by P.
   If compiling separate builds, record hashes, repeat the slab512 anchor on
   the same allocation and distinguish sequential-process from interleaved
   comparisons. A small template-specialization dispatch in one binary is an
   option if it preserves consistent launch/configuration plumbing. Do not
   accidentally profile slab512 while timing another variant. At fixed P the
   shared arena should not grow with slab size: grid.z and metadata reloads do.
   Validate F1000/F1025; include F128 as a partition-equivalence control.
   Add roadNet/webbase only if memory and the profiles justify it. Inspect both
   partial-slab guards and memory allocation/headroom, especially at F1024.

5. Validate and independently repeat finalists.
   Run meaningful tests for new settings, plus existing FPT/BOO correctness
   suites on the GPU; use compute-sanitizer on representative new boundary and
   alignment cases when kernel code changes. DIAG_RAW_STORE must be false in
   benchmarked/correctness variants. Keep build flags consistent.
   Use at least 5 warmups and 20 randomized/interleaved timing repeats; record
   raw samples, median and spread. Confirm finalists on three independent
   RTX3090 allocations, repeating baseline anchors/direct comparisons in each.
   Add N1048576 generated scale checks and one held-out sparse noisy diagonal
   case, reusing labels rather than rerunning expensive clustering.

   All four direct algorithms should be checked when feasible. If plans cannot
   coexist in GPU memory, run smaller groups of direct algorithms with repeated
   BOO anchors in each process/cell. Report missing baseline coverage. A pooled
   fastest baseline across allocations is only a sensitivity check, not a
   paired comparison. Never present profiler time as the performance result.

## Other ideas: evaluate only if the evidence calls for them

The main task is profiling, P tuning, slab tuning and confirmation. Use the
profiles to recommend one next step, without starting a broad additional sweep:
- A small threads128/256/512 sweep if launch resource limits explain a loss.
- A scalar-vs-vector-load ablation at fixed FPT if instruction evidence leaves
  vectorization's contribution unclear; preserve identical arithmetic.
- Active-column compaction if measured staging/traffic remains limiting. Logical
  unused-X fractions alone are insufficient to justify an implementation.
- Batch>1 if underfilled grids dominate. Keep one shared A and separate X/Y;
  compare native batched cuSPARSE and separate-call ALG3 (current DirectSpMM is
  2D-only). This needs its own baseline work and is deferred here.
- Setup-time selection among original CSR and tuned BOO for fixed patterns.
  Estimate setup/selection cost and amortization from measured differences; do
  not infer a universal density threshold from these few cases.

## Deliverables

Save raw profiler exports, metadata, correctness logs, per-allocation timing
samples and reproduction commands in a new results directory. Include:
- A diagnosis table: case, limiting cost, evidence, uncertainty, next action.
- P×FPT timing heatmaps with arena/occupancy/spill information.
- Slab/F1024 timings and grid/staging effects at fixed P/FPT.
- Gains versus both FPT2/P64/slab512 and strongest measured cuSPARSE; regressions
  and unsupported variants included. Distinguish tuning from held-out results.
- Independent-repeat summary and setup break-even where a gain warrants it.
- A concise handoff naming edited files, remote roots/job IDs, completed vs
  pending work, and one evidence-supported next experiment.

Proceed autonomously within this scope. Do not stop after a plan or after
submitting jobs; monitor, collect and analyze them, repairing concrete failures.
Report a genuine resource/access blocker or a required scope expansion clearly.
