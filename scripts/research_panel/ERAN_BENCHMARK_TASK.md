# Agent task: complete the FPT comparison for Eran

Work in `/Users/dayan/local/research/SPMM-boo`.

Execute the benchmark and analysis portion of
[`docs/ERAN_MEETING_PLAN.md`](../../docs/ERAN_MEETING_PLAN.md). Implement the
necessary harness, validate it remotely, run and monitor the experiments,
retrieve results, and deliver a report plus presentation figures. Continue
through collection and analysis; job submission alone is not completion.

This task measures the current FPT implementation. Kernel optimization,
changing defaults, resizing real matrices, and the output-atomic experiment
in `NEXT_EXPERIMENTS.md` are outside scope. Do not rerun the completed P/slab
optimization as pending work. Do not commit, push, or overwrite historical runs.

## Read first

- `CLAUDE.md` for remote build/test and toolchain requirements.
- `docs/ERAN_MEETING_PLAN.md` for hypotheses, grid, comparisons and slide order.
- `RESULTS_ARCHIVE.md` for raw-data recovery.
- `experiments/2026-10-09_full_fpt/FULL_FPT_BENCHMARK.md` and
  `experiments/2026-10-09_full_fpt/plots/ANALYSIS.md`.
- `experiments/2026-10-12_tile_optimization/TILE_REVIEW.md` for confirmed
  findings and timing/counter qualifications.
- `scripts/research_panel/full_sweep.py`, `benchmark/bench_real.py`,
  `benchmark/bench_tiles.py` and `benchmark/tiles_common.py` under
  `tests/torch_fast_spdmm/` for the latter three paths.

Inspect Git status/diffs and preserve user edits. Prefer new uniquely named
drivers and result directories; keep historical manifests and plots intact.

## Exact scope

- All six existing SuiteSparse inputs at native dimensions.
- All ten existing full-sweep synthetic profiles at N=262144 and N=1048576.
- F={128,256,512}; method={identity,hem,graclus,metis}; k={32,64,128};
  FPT={1,2,4}.
- Primary grid: float32, batch1, P64, threads256, slab512, rowmajor/rowseg,
  auto launch, remainder threshold4, seed0, explicit zeros kept.
- 2,808 requested BOO complete-call configurations, before duplicate labelings,
  infeasibility, baselines and independent confirmation. Account for every
  requested configuration as measured, an explicit duplicate, or unsupported
  with a concrete reason. Failures remain visible until repaired or explained.
- Missing historical coverage: F256/k32/64 and k128 at all three F values.
  Remeasure overlapping cells under the improved timing protocol so the final
  table does not mix historical and new allocations as paired observations.

At fixed P64, record infeasibility rather than substituting another tile.
Optional smaller-P rescue measurements must be labeled separately. Keep actual
group sizes and identical-label target aliases explicit.

## Remote resources

These paths are historical hints; verify availability before relying on them:

- SSH alias: `bgu-slurm`.
- `/home/dayanb/research/boo-full-fpt-20261009`: saved full-sweep labels and
  historical source/results.
- `/home/dayanb/research/real-matrices-20261008/matrix_cache`: canonical inputs.
- `/home/dayanb/research/clustering-20261006/xl/labels`: older synthetic k128
  label candidates, with a different schema requiring explicit validation.
- The full-sweep snapshot has isolated dependencies in `deps/`.

Create a new isolated remote snapshot. Build in place on a GPU node and set
PYTHONPATH to that snapshot; preserve the shared installed extension. Activate
`fem`, set CUDA_HOME/compiler/runtime paths per CLAUDE.md, and import torch
before directly importing the extension. All compilation and tests run remotely.
Use RTX3090 Slurm settings from the existing drivers:
`--partition=rtx3090 --account=erant --qos=erant --gres=gpu:rtx_3090:1`.
For every GPU submission, request at most 60 GB of memory and omit both
`--cpus-per-task` and `--cpus-per-gpu`; the cluster applies its default CPU
allocation per GPU. The HPC Cluster IT team cancelled an earlier GPU array
that requested 16 CPUs and 128 GB per task. Use one OpenMP thread in GPU jobs
unless the default allocation and measured workload justify another value.
CPU-only clustering uses the available CPU partition and normal QoS.
Follow the established bounded concurrency: at most two GPU tasks and four CPU
clustering tasks concurrently; do not assume those resources are immediately
available. Record job IDs and the new snapshot path.

## Execution

1. Implement a parameterized adapter for the requested grid, reusing canonical
   loading, saved-label checks, float64 CPU reference and row-wise numerical
   bounds from the real-panel harness. Keep directed/rectangular semantics.
   Use independent seeded shuffles per timing repeat, as in bench_tiles;
   bench_real's existing rotation order is insufficient. Record order mode,
   seed, sample position and predecessor. Verify the actual launched FPT/P/slab.
2. Validate the exact snapshot on a GPU node. Cover all backends and new
   F256/k128 cells with small smoke inputs, plus feature tails and feasible/
   infeasible launch accounting. Run the relevant existing FPT, BOO and backend
   checks; no kernel rewrite is needed. Save build/source hashes and test logs.
3. Validate and reuse saved labels by matrix hash, zero policy, seed, grouping
   rules and schema. Generate missing labels on CPU nodes. Deduplicate identical
   labels while preserving requested aliases and actual group sizes.
4. Run the full screening grid. Start with known win/regression cases to catch
   protocol problems before expanding. Pair FPT1/2/4 on one BOO object. Compare
   method/k on identical A/X in the same allocation where feasible; otherwise
   randomize process order, repeat shared anchors, and label the comparison's
   allocation limitations. Never cluster inside the GPU timing phase.
5. Benchmark original-order PyTorch CSR and all four supported direct cuSPARSE
   algorithms with matched complete-call timing. Include reordered CSR using
   BOO's own labels: original-coordinate results include gather/restore; resident
   reordered-coordinate results are separately labeled. Split memory-heavy
   direct plans into processes with repeated BOO anchors when necessary, and
   record missing algorithm coverage. Do not equate pooled timings to paired
   timings or profiler duration to unprofiled complete-call latency.
6. Predeclare the focused confirmation panel after screening, before confirmation
   measurements: pwtk, webbase, roadNet and natural/permuted FEM plus dense/noisy
   diagonal at both synthetic sizes. Cover F128/256/512 and k32/64/128; use
   identity plus the leading clustering method per input, adding the runner-up
   wherever claiming a backend ranking. Keep selections fixed across three
   independent allocations. Include FPT1/2/4, strongest direct baselines and an
   identical-work control. At F<=512, P64/FPT4/slab512 vs slab1024 is the existing
   control candidate; verify both perform identical work.
7. Retrieve and analyze results. Use at least five warmups and 20 shuffled
   repeats. Distinguish within-allocation IQR from variation across allocations.
   Treat effects comparable to control spread as unresolved. Keep exploration
   and confirmation separate. Estimate incremental setup break-even only where
   runtime improves, with CPU-host and setup-measurement limitations explicit.

If a stage fails, preserve diagnostics, repair the specific issue, and retry in
a fresh output directory. A focused-panel confirmation does not establish all
method/input interactions throughout the full grid. Report actual completion
and resource/access blockers without fabricating missing results.

## Deliverables and completion

Create a dated experiment folder with a README/report and selected figures:

- Coverage/accounting table and exact manifest.
- Paired FPT gain plot at fixed matrix/F/method/k/P.
- BOO-versus-direct complete-call speedup heatmap, separate real/synthetic views.
- Runtime-versus-k curves for identity/HEM/Graclus/METIS at fixed FPT, with
  separate views for tuned practical winners.
- Synthetic N262k-to-N1M comparison, recording nnz/row and noise-scaling caveats.
- Actual group-size, block/remainder and staging statistics for representative
  wins and losses; natural/permuted FEM and reordered-CSR comparisons.
- Independent-confirmation table and timing-control spread.
- A concise meeting narrative following the seven-slide plan, with figures
  linked and unsupported/missing cases explicit.

Retain scripts, metadata, configuration summaries and selected figures in the
worktree. Archive raw samples/arrays and record location/checksum according to
RESULTS_ARCHIVE.md; do not prune originals before verifying the new archive.
Finish with edited files, remote roots/job IDs, correctness outcomes, measured
coverage, unresolved exclusions and links to the report/figures. Update the
experiment registry and meeting plan to point to the completed evidence.
