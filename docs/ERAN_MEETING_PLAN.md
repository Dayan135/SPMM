# Plan for the Eran meeting — 2026-10-13

Status: benchmark execution started on 2026-10-13. This plan was written
before the new run. The [dated experiment record](../experiments/2026-10-13_eran_fpt/README.md)
tracks the new measurements and their completion status; retain the historical
figures below as provisional until the new coverage audit and confirmation finish.

The benchmark work can be delegated using
[the execution task](../scripts/research_panel/ERAN_BENCHMARK_TASK.md).

The question to answer is: **when does increasing features per thread improve
BOO SpMM, how does grouping affect that gain, and when should we keep CSR?**
The target workload is a fixed sparsity pattern reused for many calls, with
resident GPU inputs, batch 1 and float32. Present runtime first, then the cost
of obtaining that runtime.

## Where we stand

| Evidence | Coverage | What it establishes / limits |
|---|---|---|
| [October 9 full FPT sweep](../experiments/2026-10-09_full_fpt/FULL_FPT_BENCHMARK.md) | 26 inputs: 10 synthetic profiles at each of N=262,144/1,048,576, plus 6 native real matrices. Identity/HEM/Graclus/METIS; k=32/64; F=32/128/512; FPT=1/2/4. | 1,836 measured BOO complete-call configurations and 36 explicit duplicate-label combinations. Broad coverage, but one allocation per matrix/method shard and rotation-ordered timing. Small effects and rankings remain provisional. |
| [October 12 tile review](../experiments/2026-10-12_tile_optimization/TILE_REVIEW.md) and [detailed results](../experiments/2026-10-12_tile_optimization/TILE_EXPERIMENTS.md) | Focused P × FPT and slab screens; selected F512/F1024 cases confirmed on three independent RTX 3090 allocations with shuffled timing order. | Stronger evidence for wins and regressions on the repeated cases. P64/slab512 is the general anchor; no global default was changed. |
| [October 6 clustering sweep](../experiments/2026-10-06_clustering/CLUSTERING_EXPERIMENT.md) | Four synthetic profiles, both large N values, F128/512, group targets 8–256. | Includes k128, but predates the latest FPT sweep and uses a different baseline protocol. Useful context and reusable labels subject to validation; do not merge its timings into a new factorial. |

In the full sweep, FPT2 has no >5% regression against FPT1 across 612 measured
matrix/F/method/k cells. FPT4 has 29 such regressions. At F512 the geometric
mean speedups against FPT1 are 1.122×/1.141× for FPT2/FPT4 on real inputs and
1.232×/1.280× on synthetic inputs. These are historical exploratory statistics,
equally weighting configurations; see [the analysis](../experiments/2026-10-09_full_fpt/plots/ANALYSIS.md).

Confirmed complete-call comparisons at F512:

| Case | Confirmed BOO variant | Direct cuSPARSE time / BOO time |
|---|---|---:|
| pwtk, identity k32 | P64/FPT4 | 1.16–1.20× |
| dense diagonal, N262k / N1M | P128/FPT4 | 2.65–2.67× / 2.74–2.81× |
| permuted FEM, N262k / N1M | P64/FPT4 | 1.54–1.55× / 1.56–1.58× |
| natural FEM, N262k / N1M | P64/FPT4 | 0.80–0.82× / 0.84–0.87× |
| webbase | P64/FPT2 | about 0.50× |
| roadNet, METIS k64 | P64/FPT4 | about 0.67× |

Ratios above one favor BOO. These are selected configurations, not averages or
controlled group-size comparisons. On webbase, FPT4 achieves only 0.86× the
FPT2 speed; on the held-out sparse noisy diagonal, 0.92×. FPT2 is a conservative
candidate and FPT4 a conditional choice. P128's small improvement over P64 on
dense diagonal overlaps the identical-work control spread; its large win over
CSR does not establish a large tile-size effect.

## Proposed benchmark grid

| Dimension | Proposed values |
|---|---|
| Synthetic size | N=262,144 and N=1,048,576 |
| Real size | Native dimensions, recorded as rows × columns |
| Feature width F | 128, 256, 512 |
| Clustering | HEM, Graclus, METIS, plus identity as the control |
| Group target k | 32, 64, 128 on both axes |
| Features per thread FPT | 1, 2, 4 |
| Fixed kernel settings | P64, threads256, slab512, rowmajor/rowseg, auto launch, remainder threshold4 |

F, FPT, k and P are distinct: F is the dense width, FPT is the work assigned to
one thread, k is a maximum grouping target, and P is the kernel feature tile.
Keep P fixed to isolate the requested effects. If P64 is infeasible, mark that
configuration unsupported in the primary grid. Any smaller-P rescue belongs in
a separate practical-results table with its actual P recorded.

Use the existing 10 synthetic profiles for coverage: fem2d, fem2d_permuted,
diag_dd0.017/0.033/0.05, diag_b0.1/0.3/1.0, diag_skewed and
pattern_diag_stripe. Lead the presentation with natural/permuted FEM, dense
diagonal, and sparse noisy diagonal; retain the other profiles in the appendix.
Do not change their generators while filling this grid. A separate constant
nnz/row scaling control can be added later: fixed noise density in existing
profiles makes noise nnz grow approximately 16× when N grows 4×.

Use all six existing real inputs. Their retained dimensions are:

| Matrix | Rows × columns |
|---|---:|
| Boeing/pwtk | 217,918 × 217,918 |
| Schmid/thermal2 | 1,228,045 × 1,228,045 |
| AMD/G2_circuit | 150,102 × 150,102 |
| Williams/webbase-1M | 1,000,005 × 1,000,005 |
| SNAP/roadNet-CA | 1,971,281 × 1,971,281 |
| LPnetlib/lp_cre_b | 9,648 × 77,137 |

Do not resize real matrices to fit the synthetic N labels. If exact-size real
subsets are required, define and report them as derived workloads; a principal
submatrix or sampled subgraph changes the structure and answers a different
question. Keep the small circuit and rectangular LP as structural controls,
with explicit dimensions, rather than calling every real case large.

With 26 inputs and identity included, the proposed grid has
26 × 4 × 3 × 3 × 3 = **2,808 requested BOO configurations**, before numerical
duplicates, infeasibility, baselines and replication. The three requested
clustering methods account for 2,106; identity adds 702.

The October 9 sweep overlaps 1,248 requested configurations
(F128/512 × k32/64), of which 1,224 were independently measured and 24 were
explicit duplicate-label combinations. Missing coverage is **1,560 requested
configurations**: 624 at F256/k32/64 and 936 at k128/all three F values.
Historical overlap is coverage evidence, not a replacement for fresh paired
timing under the improved protocol.

## Execution order

1. **Prepare the meeting from existing evidence now.** Build the F128/F512
   story using the full-sweep figures, and use the confirmed report for repeated
   cases. Mark F256 and the latest-FPT/k128 interaction as pending.
2. **Prepare one parameterized benchmark snapshot.** Adapt
   [full_sweep.py](../scripts/research_panel/full_sweep.py), whose groups and
   features are currently hardcoded. Use the real-panel validation/schema for
   both input families and the independently shuffled order already available
   in [bench_tiles.py](../tests/torch_fast_spdmm/benchmark/bench_tiles.py).
   Include launch verification and an identical-work timing control. Smoke-test
   the new F256/k128 cells, all backends, feature tails and memory exclusions
   on a GPU node before the large sweep.
3. **Reuse the CPU work.** Recover saved arrays from
   [the archive](../RESULTS_ARCHIVE.md) or the remote label cache. Validate matrix
   hash, canonicalization/explicit-zero policy, seed, grouping rules and label
   schema. Compute missing k128 labels on CPU nodes, outside GPU allocations.
   Hash-identical labels are one grouping with multiple requested targets;
   keep those target aliases explicit without counting them as independent
   evidence.
4. **Screen the full requested grid with fresh timing.** Start with pwtk,
   webbase, natural/permuted FEM, dense diagonal and sparse noisy diagonal to
   verify expected behavior, then complete all 26 inputs. Use one labeled build
   for FPT1/2/4 and the same A/X across method/k comparisons. Interleave method/k
   arms in the same allocation where feasible; if memory forces separate
   processes, randomize their order and repeat shared anchors. An FPT comparison
   within one shard does not by itself make comparisons across shards paired.
5. **Confirm a predeclared focused panel on three independent allocations.**
   Include all three F values and group targets on pwtk, webbase, roadNet and
   natural/permuted FEM plus dense/noisy diagonal at both synthetic sizes.
   Select the identity and the screened leading clustering method for each
   input before confirmation; add the runner-up wherever claiming a backend
   ranking. Hold configurations fixed across allocations. Keep the screen and
   confirmation separate, and include losses as well as wins. A global claim
   covering every method/input interaction requires replicating that broader
   grid, not extrapolating from this focused panel.
6. **Generate the final figures and archive the samples.** Retain the manifest,
   hardware/backend versions, source/build hashes, summaries, exclusions and
   figures in Git. Record the raw-sample archive and checksum. Report scope,
   selection and replication explicitly.

For new timings, use at least the established five warmups and 20 repeats,
independently shuffled each repeat, with order seeds recorded. Report allocation
medians and their range; distinguish within-allocation IQR from variation across
allocations. Treat differences comparable to the timing control as unresolved.

The primary metric is complete-call GPU elapsed time in original coordinates,
including output initialization, BOO main kernel, remainder and joins. Record
setup separately. Kernel-only profiling is explanatory evidence. Do not divide
a complete BOO call by a kernel-only cuSPARSE time.

Benchmark all four direct cuSPARSE algorithms when supported. If they do not fit
together, split them across processes with repeated BOO anchors and record the
coverage. Include reordered CSR under the same label permutation: its
original-coordinate comparison includes X gather and Y restore. Show resident
reordered-coordinate timing separately for callers that can keep that layout.
This distinguishes a BOO-format advantage from recovery of lost ordering/locality.

## What to show: seven slides, about 20 minutes

| Slide | Main question | Figure / content |
|---|---|---|
| 1. Problem and contribution | What changed in the kernel? | A simple FPT1/2/4 lane sketch; fixed-pattern workload; one thread reuses sparse values/indices across multiple features. Explain F, k and P once. |
| 2. Experimental scope | Which comparisons are actually measured? | Grid above; native real dimensions; coverage and confirmation status. |
| 3. FPT effect | Does FPT help with the same grouping? | Paired FPT1-time / FPT2-or-4-time plot, split real/synthetic and F. Fixed method/k/P. Show median/geomean, tail regressions and support count. |
| 4. Practical performance | Does faster BOO beat the strongest CSR? | Per-matrix speedup heatmap at F128/256/512, real and synthetic separated. Label post-hoc best selections and confirmed cases; add original/reordered CSR comparisons for FEM. |
| 5. Clustering and group target | When does reorganization pay, and does k128 help? | Runtime vs k=32/64/128 at fixed FPT/P, lines for identity/HEM/Graclus/METIS, faceted by F and synthetic N. Add actual group-size and remainder statistics beside the representative cases. |
| 6. Explain the gains and losses | Why is FPT4 conditional? | Dense diagonal/pwtk versus webbase: fewer executed shared loads, then increased output RED traffic. Use the corrected counter definitions in the tile review; counters suggest the mechanism, not a measured decomposition of elapsed time. |
| 7. Decision and next experiment | What should we do next? | Conservative FPT2 candidate, conditional FPT4, CSR on losing cases; preprocessing break-even; proposed coalesced-output ablation at fixed labels/P/FPT/slab. |

Use existing [paired-FPT](../experiments/2026-10-09_full_fpt/plots/02_paired_fpt.png),
[overview](../experiments/2026-10-09_full_fpt/plots/01_speedup_overview.png),
[clustering](../experiments/2026-10-09_full_fpt/plots/03_clustering.png),
[group-size](../experiments/2026-10-09_full_fpt/plots/04_group_size.png) and
[scaling](../experiments/2026-10-09_full_fpt/plots/06_size_scaling.png) figures
for the first version. Preserve their historical scope and caveats. Replace
them with the requested-grid plots after the new run; never interpolate F256.

Keep the two main speedup definitions visible:

- **FPT gain:** time(FPT1) / time(FPT2 or FPT4), holding input/F/method/k/P fixed.
- **BOO competitiveness:** strongest supported direct complete-call time /
  BOO complete-call time in the same measurement cell.

For the clustering plot, hold FPT and k fixed when discussing the effect of
method. Put separately tuned per-method winners in a second practical-selection
view. Similarly, hold FPT fixed for group-size comparisons before showing the
best FPT per k. Report actual mean/max row and column group sizes, block count,
nnz/block, remainder fraction and staged-X use; k is an upper bound, not a
guarantee of groups of exactly that size.

Show representative matrix sparsity plots before/after grouping for natural
and permuted FEM. They make the structural explanation easier to follow than
many additional bar charts. Put the full parameter tables, backend definitions,
support exclusions and additional profiles in the appendix. Here Graclus means
the repository's iterated PyG CPU greedy matching implementation.

For amortization, use incremental setup relative to CSR:
ceil((setup_BOO − setup_CSR) / (time_CSR − time_BOO)) when BOO is faster and
incremental setup is positive. Include grouping, construction, staging and
selection, plus relevant permutation/value-update costs. There is no runtime
break-even when BOO loses per call. Existing setup estimates are approximate;
CPU host differences also limit historical backend setup comparisons.

## Recommendation for the discussion

We already have a defensible result to show. Completing F256 and k128 makes the
requested comparison complete, but the broad sweep alone will not establish a
universal best backend or default. Present structure-dependent wins and losses,
confirm the comparisons on which decisions depend, then discuss
[coalesced output atomics](../NEXT_EXPERIMENTS.md) as the next kernel experiment.
Keep that kernel change separate from the benchmark that documents the current
FPT implementation.
