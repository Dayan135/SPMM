# Eran FPT comparison — 2026-10-13

Status: screening and its coverage audit are complete; independent confirmation
is running. The confirmation result is pending until all three allocations and
the final report finish.

The [meeting plan](../../docs/ERAN_MEETING_PLAN.md) and
[execution task](../../scripts/research_panel/ERAN_BENCHMARK_TASK.md) define the
questions and requested scope. The primary grid has 26 inputs × 4 grouping
methods × 3 target sizes × 3 feature widths × 3 FPT values = 2,808 requested
BOO cells. It fixes P64, 256 threads, slab512, float32, batch1, seed0,
rowmajor/rowseg, auto launch and remainder threshold4. Infeasible cells retain
their requested P64 and an explicit reason. Identity target sizes that create
identical labels are recorded as aliases.

The new driver is [full_sweep.py](../../scripts/research_panel/full_sweep.py),
using the canonical real-panel loader, saved-label validation, float64 CPU
reference and row-wise numerical bounds. The complete-call timing order uses
independent seeded shuffles per repeat. Samples record repeat, position,
predecessor and order seed. All four direct cuSPARSE algorithms, original-order
PyTorch CSR and reordered CSR are requested in each cell subject to explicit
resource accounting. Five warmups and 20 repeats are configured for screening.

Remote snapshot: `/home/dayanb/research/eran-fpt-20261013`.
New run root: `/home/dayanb/research/eran-fpt-20261013/eran_run`.
The snapshot reused hash-validated k32/k64 labels from the October 9 sweep and
updated its source to the October 12 tile version before building in place.
The shared installed extension was not modified.

| Slurm job | Purpose | Outcome |
|---|---|---|
| 22737171 | Build and GPU preflight | Completed 0:0 |
| 22737522 | CPU clustering array, 0–103, at most 4 concurrent | 104/104 completed |
| 22737555 | Original GPU screening array | Cancelled after eight shards completed |
| 22741262 | Original coverage audit | Exited 1 on incomplete coverage, as designed |
| 22741962 | Original confirmation allocations | Did not run after audit failure |
| 22742147 | Original confirmation finalizer | Did not run |

## Recovery after cluster cancellation

The HPC Cluster IT team cancelled the original GPU array `22737555` because
each job requested 16 CPUs and 128 GB of memory. Their instruction is to
request no more than 60 GB for a GPU job and to use the default CPU allocation
per GPU. Eight GPU shards (indices 0–6 and 8) completed successfully before
cancellation. The partial results from interrupted indices 7 and 9 are retained
under `eran_run/interrupted/`. All 104 CPU clustering shards finished 0:0.
The original finalizer `22741262` correctly exited 1 for incomplete coverage,
so its dependent confirmation jobs did not run. The interrupted result folders
and logs remain as incident records. Slurm captured the recovery finalizer's
script at submission time: it wrote the verified complete archive as
`raw-results-20261013.tar.gz`, replacing the earlier partial archive of the
same name. The interrupted folders themselves were preserved separately.

The recovery GPU array uses **one RTX 3090, 56 GB, and no explicit CPU request**
per task; Slurm applies its default `CpusPerTres=gpu:6`. GPU runner OpenMP
threading is set to one. Slurm's submitted job record was inspected to verify
these settings. At most two recovery GPU tasks run concurrently. Completed
shards are reused; only indices `7,9-103` were resubmitted.

| Slurm job | Recovery role | Outcome |
|---|---|---|
| 22758802 | GPU screening recovery, indices 7 and 9–103 | 96/96 completed |
| 22758838 | Coverage audit, figures and raw archive | Completed |
| 22758871 | Three independent GPU confirmation allocations, 56 GB and default CPU allocation | Allocation 1 timed out at six hours; allocation 2 completed; allocation 3 continues |
| 22758872 | Original confirmation finalizer | Cancelled after allocation 1 timed out |
| 22885574 | Full rerun of confirmation allocation 1, 56 GB, default CPU allocation, ten-hour limit | Running |
| 22885736 | Confirmation table and archive | Queued after allocation 1 rerun and allocation 3 terminate; exits nonzero if any allocation is incomplete |

Both GPU runners now set `OMP_NUM_THREADS=1`; the CPU clustering phase retains
eight OpenMP threads. The source and extension measured by the recovered jobs
are the same isolated snapshot as the first eight successful shards; only the
runner's CPU-thread setting changed.

The preflight passed 151 relevant tests. Four-backend k128 smoke cases at
F35/128/256/257 produced 48 numerically valid BOO rows. Profiler inspection
verified the requested kernel specialization for all 48 rows. The resulting
extension SHA-256 is
`37644508ea5fb3d92d17d68c2937c79cc282bd776ffce0b5f75897fe120d098c`.
The preflight log and raw smoke results remain in the remote run root; earlier
failed smoke attempts are retained there for diagnosis.

Historical timings are not treated as paired observations with this new sweep.

## Completed screening evidence

The [screening audit and figures](analysis/screening_report.md) account for all
**2,808 requested BOO cells**: 2,628 measured, 180 aliases of identical saved
labels, zero unsupported and zero missing. All 104 task statuses are complete.
The [aggregate report](../../tests/torch_fast_spdmm/benchmark/results/eran_fpt_20261013/aggregate/report.md),
[configuration table](../../tests/torch_fast_spdmm/benchmark/results/eran_fpt_20261013/aggregate/summary.csv)
and [coverage table](analysis/coverage.csv) are retained locally. The
[paired FPT figure](analysis/paired_fpt.pdf), [real-input speedup](analysis/speedup_real.pdf),
and [synthetic-input speedup](analysis/speedup_synthetic.pdf) are ready for
review; the screening report links the k and size figures.

Across independently measured cells at F512, FPT2/FPT4 geometric-mean gains
over FPT1 are 1.112×/1.158× on real inputs and 1.199×/1.257× on synthetic
inputs. FPT2 has no regression greater than 5% in this grid; FPT4 has such
regressions, so it remains a tuned option. These are one-allocation screening
statistics, equally weighting configurations. The focused three-allocation
confirmation is needed for decisions and small method rankings.

The final screening archive is `eran_run/raw-results-20261013.tar.gz`; its
checksum is in `eran_run/raw-results-20261013.sha256`. A copy is kept under
`local-archives/` (ignored by Git); see [the archive policy](../../RESULTS_ARCHIVE.md).

### Interpretation of the screening data

These statements are exploratory: one GPU allocation per matrix/method shard,
and winner configurations were selected after observing the screen. Ratios
below compare complete calls; an original-order direct baseline is measured
in the same shard and labeling cell.

- **FPT2 is the stable improvement in this grid.** Across 876 independently
  measured matrix/F/method/k comparisons, its geometric-mean gain over FPT1 is
  1.173× and none loses more than 5% (worst ratio 0.996). For F512 alone, the
  gains are 1.112× on real inputs and 1.199× on synthetic inputs. Geomeans
  weight configurations equally, so these are not application-weighted gains.
- **FPT4 has a larger upside and a measurable downside.** Its geometric-mean
  gain over FPT1 is 1.230×, but 46/876 comparisons lose more than 5%; the
  worst ratio is 0.842. FPT4 is selected for 42/78 post-hoc matrix/F winners,
  FPT2 for the other 36. Selection by matrix structure remains necessary.
- **The new k128 target does not help this P64 grid.** Among 792 k128 cases
  with distinct labels and comparable k64 measurements at the same matrix,
  method, F and FPT, all are more than 5% slower than k64; the median ratio
  `time(k64) / time(k128)` is 0.525. The remaining identical-label target
  aliases are not counted as independent observations. None of the 78 selected
  matrix/F winners uses k128: 48 use k32 and 30 use k64. For example, pwtk/HEM
  at F512/FPT4 takes 6.53, 7.89 and 17.89 ms at k32/64/128. Its recorded
  shared-memory arena grows from about 14 to 64 KiB. The arena and block-size
  changes suggest possible resource effects; this screen does not isolate their
  contribution to latency.
- **Grouping is valuable when ordering destroys locality.** At N=1M/F512,
  permuted FEM with identity labels takes 186.8 ms in BOO; HEM k32 takes
  32.2 ms. The HEM grouping reduces its BOO block count from 2,121,680 to
  272,629. Original-order direct cuSPARSE takes 51.7 ms, giving grouped BOO
  a 1.61× advantage. Reordered cuSPARSE takes 27.0 ms with resident reordered
  coordinates, or 36.7 ms when X gather and Y restore occur each call. Thus
  BOO's practical advantage depends on the caller's coordinate contract.
- **BOO does not broadly beat CSR.** The post-hoc best BOO wins 31/78 matrix/F
  cases against the fastest measured same-cell original-order direct baseline;
  all 31 wins have all four direct algorithms available. pwtk is the only one
  of six real inputs with a win, reaching 1.20× at F512. Dense synthetic
  diagonal reaches 2.78× at N=1M/F512. Natural FEM remains slower than direct
  (0.85× at N=1M/F512), and webbase is about 0.50×. The best-target method
  counts are descriptive, not proof that one clustering algorithm dominates:
  methods ran in different screening shards, and several close rankings may
  shift on independent allocations.

Clustering, construction, and staging cost is outside the GPU call. The
[winner table](analysis/screened_winners.csv) includes approximate incremental
setup and break-even calls, but those CPU times come from different hosts and
omit some deployment costs. The confirmation panel tests whether the selected
runtime differences persist across independent allocations.

## Confirmation timeout and repair

Allocation 1 of job `22758871` completed 20 of 22 panel tasks before hitting
its six-hour limit during the 1M sparse noisy diagonal. Its partial result tree
is preserved as `confirm/alloc1_timeout_22758871/`. Allocation 2 completed all
22 tasks and its timing control. Allocation 3 is still running. Allocation 1
is being remeasured from the start under job `22885574` with a ten-hour limit;
this keeps each reported allocation a single independent GPU session. The
replacement finalizer `22885736` will wait for that rerun and allocation 3,
archive their outputs, and require all three statuses to be complete.

The confirmation selection is frozen by
[eran_confirm_manifest.py](../../scripts/research_panel/eran_confirm_manifest.py)
only after all screening shards complete: pwtk, webbase, roadNet, and natural/
permuted FEM plus dense/noisy diagonal at both synthetic sizes. It selects the
fastest nonidentity method at F512 in screening (falling back to F256/F128
only if P64 has no feasible F512 candidate), then measures that method and
identity at every F and k on three independent allocations. The
[confirmation driver](../../scripts/research_panel/eran_confirm.py) also times
an identical-work P64/FPT4/slab512-vs-1024 control on representative cases.
The method arms share an allocation but run sequentially, so small cross-method
effects remain sensitive to drift.
