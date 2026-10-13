# BOO profiling panel — 2026-10-08

Status (2026-10-09): the baseline synthetic timing and profiling runs completed.
The real-matrix agent implemented an opt-in features-per-thread (FPT) kernel;
its completed real panel is reviewed below. Paired synthetic FPT controls completed
in an isolated snapshot. No default kernel change is promoted.

The separate real-matrix track has a [handoff prompt](../../scripts/research_panel/REAL_MATRIX_TASK.md).
It owns the real-matrix loader/benchmark. This track owns the synthetic panel,
structural measurements, profiler orchestration and reporting.

## Implementation

- [research_panel.py](../../tests/torch_fast_spdmm/benchmark/research_panel.py): CPU label
  preparation, matched GPU timing and one-call profiling using saved labels.
- [panel_structure.py](../../tests/torch_fast_spdmm/benchmark/panel_structure.py): exact
  active row/column counts, row imbalance, output fanout and metadata byte models.
- [job.sh](../../scripts/research_panel/job.sh): isolated in-place build, tests, smoke,
  labels, timing and Nsight runs. [Usage](../../scripts/research_panel/README.md).
- [report.py](../../scripts/research_panel/report.py): timing tables and extraction of
  kernel, GPU activity and counter records from raw profiler outputs.

The existing `bench_spmm.measure_cell` accepts optional `diagnostics=False`,
so this panel omits the intentionally incorrect raw-store variants. Existing
callers retain their previous default. The panel recomputes speedup ratios
within matching timing scopes and labels the existing C++ BOO harness's legacy
fork/join explicitly. API BOO uses auto launch.

Direct profiling selects the fastest validated full-call cuSPARSE algorithm
for the exact matrix/input hashes in the completed timing panel. It does not
assume DEFAULT is the strongest implementation.

## Validation completed

- GPU build and BOO/direct-cuSPARSE/focused tests: **215 passed, 12 skipped**.
- Final focused harness tests, including added direct-baseline selection:
  **11 passed**. These overlap the earlier focused tests; do not add the counts.
- Smoke: N=1024, FEM and diag_skewed, F=32/128, identity/HEM k=32:
  **52 configuration/workload rows passed numerical validation**.
- [Saved smoke report](../../tests/torch_fast_spdmm/benchmark/results/research_panel_20261008/report.md),
  [raw results](../../tests/torch_fast_spdmm/benchmark/results/research_panel_20261008/smoke).

The smoke structural check found 50.93% of staged X rows unused in their
individual BOO blocks for HEM/FEM/k=32. This is a structural observation at
N=1024, not measured DRAM waste or proof that compaction speeds up SpMM.

## Remote execution

Isolated snapshot: `/home/dayanb/research/boo-profile-20261008-a` on `bgu-slurm`.
The extension was built in place without reinstalling the shared environment.

| Job | Purpose | Last observed status |
|---|---|---|
| 22397102 | Build and correctness | Completed |
| 22397121 | CPU labels for six profiles, N=262144, identity/HEM, k=32/64 | Completed |
| 22397122 | GPU smoke | Completed |
| 22397190 | Full timing panel, F=128/512, final focused tests | Completed: 204 validated rows |
| 22397192 | Nsight subset, dependent on successful timing | Completed |

These are submission/checkpoint states, not live status. Check `squeue`/`sacct`
and result metadata before reporting completion. GPU output directories must
be new; failed runs retain their partial artifacts.

The timing panel covers natural/permuted FEM, diag_b0.3, diag_b1.0,
diag_skewed and pattern_row_skewed. The initial profiler subset covers natural
FEM, diag_b1.0 and pattern_row_skewed at F=512, HEM k=32, for BOO, torch and
direct cuSPARSE. Nsight Systems, stall counters and memory/occupancy counters
run in separate processes. Unprofiled timings decide performance; profiler
durations diagnose behavior.

Results reside under `<snapshot>/results`. After completion, retrieve the
panel CSV/JSON files, profiler CSVs and SQLite traces and run:

```bash
python scripts/research_panel/report.py tests/torch_fast_spdmm/benchmark/results/research_panel_20261008
```

Next analysis: establish whether active-column waste translates into memory
traffic or stalls; compare it with shared-memory dependencies, row imbalance,
occupancy and output accumulation. Select one kernel experiment from that
evidence, then validate on the real-matrix panel and independent allocations.

## Completed evidence and FPT follow-up — 2026-10-09

The baseline panel validated **204/204 rows**, with the final focused harness
suite passing **11 tests**. The report linked above now contains the full panel,
23 extracted Nsight Systems kernel records and 1,081 Nsight Compute records.
At F=512 the best full-call BOO configuration beats the strongest measured direct
cuSPARSE baseline by 1.079x on permuted FEM, 1.033x on dense diagonal, and 1.034x
on row-skewed input. Natural FEM remains 1.85x slower. These are post-hoc best
configurations in one allocation, not independently reproduced gains.

For HEM k32, F512, synthetic FEM has 26.6% MIO-throttle stalls and dense diagonal
44.8%, consistent with investigating shared-load instruction pressure. Row-skewed
input instead has 40.0% long-scoreboard stalls and 2.1% MIO stalls. Structural
unused-X fractions are respectively 54.4%, 0%, and 20.0%; these counts do not
measure actual DRAM traffic. This supports testing FPT across different bottlenecks.

The real-matrix agent's FPT job **22400141 completed successfully**, covering all
72 cells. Raw artifacts were copied read-only from the external snapshot into
[the local handoff directory](../../tests/torch_fast_spdmm/benchmark/results/fpt_handoff_20261009/real_panel).
Paired complete-call timings at F512:

| Matrix / labels | FPT1 ms | FPT2 ms | FPT4 ms | Best measured original-order direct ms |
|---|---:|---:|---:|---:|
| pwtk / identity k32 | 11.208 | 7.320 | 6.703 | 7.923 (ALG2) |
| roadNet-CA / HEM k32 | 28.759 | 27.728 | 28.534 | 21.990 (DEFAULT only) |
| webbase-1M / identity k32 | 16.796 | 16.279 | 19.024 | 8.259 (ALG3) |

FPT4 yields **1.67x over FPT1 and 1.18x over direct cuSPARSE on pwtk**. It is
**13.3% slower than FPT1 on webbase** in the displayed case. RoadNet barely
responds, agreeing with its sparse blocks and memory-latency-heavy profile.
The roadNet cell lacks other cuSPARSE algorithms because of memory limits.
Use raw CSV timings when reading the upstream report: its FPT table's final
ratio heading is reversed relative to its values. Independent allocations are
still required before promoting any performance claim into a default policy.

The synthetic driver now accepts `--boo-fpt 1 2 4`, reuses identical saved labels,
and interleaves all variants with direct baselines. BOOTimedSpMM wrappers reset
the process-wide FPT selector before **every** run/timed_run, preventing another
variant from silently changing the C++ measurement. Tests cover this selector
behavior and all three variants on a feature-tail case.

Control snapshot: `/home/dayanb/research/boo-fpt-control-20261009`, using the exact
validated FPT binary (SHA256 recorded in `benchmark_source.json`). It does not
modify either external agent snapshot or the shared installation. The control
panel fixes HEM k32, N262144, F128/512 and covers natural/permuted FEM, dense
diagonal and row-skewed input. Job **22417146** completed on an explicitly requested RTX 3090.
Job 22417139 failed at the binary-check import before tests; the script now loads
PyTorch first. Job 22417145 was cancelled because the generic GPU request was
assigned a different GPU model. Neither produced comparison results.

Next decision: examine paired synthetic FPT gains, then independently replicate
pwtk and representative regressions. Keep FPT opt-in. Active-column compaction
is a separate candidate for roadNet-like structure; the present evidence does
not establish its benefit. The external agent's kernel and real-matrix sources
have not been edited by this profiling follow-up.

## Completed synthetic FPT controls

Job 22417146 passed **114 tests** and validated **120/120 timing rows** across
all eight cells. The [paired report](../../tests/torch_fast_spdmm/benchmark/results/fpt_control_20261009/results/report.md)
contains API and C++ scopes, raw timings and structural counts. At F512, HEM k32:

| Synthetic input | FPT1 ms | FPT4 ms | FPT1 / FPT4 | Best direct / FPT4 |
|---|---:|---:|---:|---:|
| Natural FEM | 11.351 | 7.024 | 1.616 | 0.882 |
| Permuted FEM | 11.580 | 7.972 | 1.453 | 1.568 |
| Dense diagonal | 13.689 | 5.740 | 2.385 | 2.353 |
| Row-skewed | 1.651 | 1.754 | 0.941 | 0.959 |

Ratios above use complete-call measurements from the same control allocation.
FPT4 is 6.2% slower on row-skewed input, while FPT2 improves it by only 1.3%.
Natural FEM still loses to the strongest measured direct baseline despite a
large BOO-relative gain. Gains on dense diagonal and FEM, alongside the small
or negative row-skewed response, agree with the earlier profiling distinction.
This is supporting evidence for the optimization mechanism, not a replacement
for profiling the optimized kernel or independently repeating the measurements.

The next experiment should replicate pwtk, dense diagonal, FEM, webbase and
row-skewed cases on independent allocations before designing an FPT selector.
For the fixed-pattern workload, per-matrix setup-time tuning is a candidate:
compare FPT1/2/4 and original CSR, then amortize that choice over repeated calls.
Do not infer a universal density threshold from this small panel. The global
FPT selector also requires care with concurrent callers before production use.
