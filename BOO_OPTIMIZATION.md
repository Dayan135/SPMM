# BOO optimization experiments

The 2026-09-23 comparison is the historical baseline. Each experiment below also
includes an in-process legacy baseline on the same seeded matrix and dense input;
absolute timings from different jobs are not used to attribute improvements.

## Order and decision gates

1. **Feasible HEM partners and pure-BOO launch path.** Test each change alone and
   together. Keep the old behavior selectable to identify regressions.
2. **Joint parameter sweep.** Independently vary row/column group sizes, feature
   tile P, thread count, and CSR remainder threshold. Report every configuration;
   retrospective best configurations are not advertised as an automatic selector.
3. **Shared-memory buckets.** Compare a few block-size classes against one global
   allocation, including extra launch overhead and preprocessing/storage costs.
4. **Precomputed row boundaries.** Compare loading stored per-block boundaries
   against building them by binary search, retaining the same accumulation kernel.

Larger output-ownership changes follow profiling evidence. They are not part of
the first implementation. No experimental option is promoted to a default before
correctness and paired timing results have been reviewed.

## Measurement contract

All builds, tests and sanitizer runs use the remote RTX 3090 environment from
`CLAUDE.md`, in an isolated source/build directory. All timings are milliseconds.
Use 5 warmup calls and 20 individually measured calls per configuration, rotating
execution order. Preserve every sample, medians, quartiles, mean/std/min/max,
conversion/staging costs, source/binary hashes, and matrix/input hashes in local
CSV files and the `private-dayan/torch-fast-spdmm` W&B project.

Each configuration must match multiplication of the original matrix in CPU
float64 before timing. CUDA tests cover alternate streams, strided temporary
inputs, batching, empty cases, both floating dtypes, and CSR/COO mixed paths.
Format objects are reused across launch-only variants; recorded build/staging
costs describe that shared format and must not be summed once per launch variant.

## Stage 1 implementation

`csr_to_boo(..., hem_feasible_first=True)` filters HEM graph edges violating size
or nnz caps before selecting mutual preferred partners. The baseline selects
partners first and then rejects infeasible matches. A regression example uses
group costs `[30,30,2,2]`, cap 32, and edge weights AB=10, AC=9, BD=8: the
baseline finds no merge, while the candidate accepts AC and BD.

`boo_matmul(..., launch_mode='auto')` launches pure BOO directly on the caller's
current CUDA stream. Output initialization, contiguous copies, computation and
consumers remain stream-ordered. Global atomics remain necessary between BOO
blocks. Inputs containing CSR remainder or COO tail retain the original
three-stream fork/join. `launch_mode='legacy'` preserves the baseline exactly.

Both options remain opt-in while measurements are pending. The binary exports
separate legacy/auto entry points with the same argument lists, so old low-level
callers retain their existing signatures.

**`hem_feasible_first` resolution (2026-09-23):** the `[30,30,2,2]` toy example
undersold this. Directly comparing `cluster_axes(..., feasible_first=False)`
against `True` on all 14 benchmark profiles at N=65536, target=32 (CPU-only,
no GPU needed -- clustering has no CUDA dependency):

| profile | blocks, legacy | blocks, feasible_first | ratio | row-group median, legacy |
|---|---:|---:|---:|---:|
| diag_b0.3 | 244,688 | 4,096 | **59.7x** | 1 |
| diag_b0.1 | 66,957 | 4,274 | **15.7x** | 1 |
| fem2d / fem2d_permuted | 18,662 / 18,622 | 16,804 / 16,767 | 1.11x | 28 |
| pattern_row_skewed | 3,587 | 3,226 | 1.11x | 1 |
| diag_b1.0, diag_skewed, all scattered/pattern_diag_stripe/corner_cluster/checkerboard | -- | -- | 1.00x (no effect) | -- |

On `diag_b0.1`/`diag_b0.3` (a 64-wide diagonal band at 10-30% density, i.e.
wider than the target=32 group size but not so dense every row is maximally
connected) legacy HEM matching nearly collapses: median row-group size 1 --
most rows never merge with anyone. Mechanism: within a partially-dense band,
many row pairs have similar-strength ties, so a preferred pair sitting at the
size cap gets rejected and neither side looks at a feasible alternative in the
same round -- not a rare edge case, the dominant outcome for this structure.
Fully-dense bands (diag_b1.0/diag_skewed, density 1.0) don't hit this: every
pair is comparably maximal, so the vectorized matching still finds *a* match
each round even when the preferred one is capped out. The effect is a no-op to
+11% fewer blocks on the other 12 profiles -- never worse in this suite.

**Promoted to default** (`csr_to_boo(..., hem_feasible_first=True)` as of this
commit) on that basis: large, structure-specific win, zero measured downside.
`launch_mode` stays opt-in (see "Validation and results" below -- measured no
effect, even isolated).

```bash
python -m benchmark.bench_boo_opt --phase first \
  --sizes 16384 65536 --features 128 512 \
  --profiles fem2d fem2d_permuted diag_b0.3 diag_skewed scattered_1x8 pattern_diag_stripe \
  --out results/boo-first
```

This compares all four combinations of feasible filtering and launch mode for
each k in 8/16/32/64, plus the eight PBR configurations and the CSR baseline.

## Stage 2 design

```bash
python -m benchmark.bench_boo_opt --phase parameters \
  --sizes 65536 --features 128 512 \
  --profiles fem2d_permuted diag_b0.3 diag_skewed scattered_1x8 \
  --out results/boo-parameters
```

Grid: group shapes 16×16, 32×32, 64×64, 16×64, 64×16, 32×64; remainder thresholds
4/16; threads 128/256/512; P=32/64/128. Feasible HEM filtering and auto launch
are enabled. Each workload also includes the original k=32/256-thread/P=64 BOO
baseline, PBR configurations and CSR. Configuration records include every choice.

**Result (`results/boo-parameters-20260923`, 109 BOO configs × 8 workloads,
remote job 21641843):** no swept combination beats a `hem_feasible_first=True,
launch_mode='auto'` baseline (unchanged k=32/threads=256/P=64/threshold=4) by a
reproducible margin. Best geomean was +1.8% (`min16` threshold, k=32 otherwise)
with a per-cell range of 0.92–1.09× -- inside run-to-run noise at 20 repeats,
not a structural win. **Rectangular shapes paired with P=128 are a clear
regression**, not noise: 16×64/32×64 groups at P=128 measured 0.51–0.64×
geomean (up to ~50% slower) across every cell, both thresholds, all three
thread counts -- plausibly shared-memory/occupancy pressure from staging a
64-wide column tile at P=128 simultaneously. None of group shape, threshold,
threads, or P are promoted; the existing k=32/threads=256/P=64/threshold=4
defaults stand, now with `hem_feasible_first=True` layered on top.

The real story in this data is `hem_feasible_first` itself, and it isn't the
uniform ~7% Stage 1's aggregate geomean suggested: comparing
`boo_original_k32` (the pre-Stage-1 defaults) against the same k=32 config
with only `hem_feasible_first`+`launch_mode='auto'` added, **`diag_b0.3` at
N=65536 improves ~2.6-3.1×** (1.08→0.37ms at f128, 3.39→1.30ms at f512) while
`diag_skewed`, `fem2d_permuted`, and `scattered_1x8` move by low single
digits either way. Stage 1's ~7% geomean was an average over profiles where
the effect ranges from "irrelevant" to "3x" -- `diag_b0.3`'s block-diagonal
density profile likely produces many groups sized near the target cap,
exactly the "two same-size groups prefer each other, both rejected, neither
looks at a feasible alternative" pathology `hem_feasible_first` fixes (see
Stage 1 implementation notes above). Not yet confirmed by inspecting the
actual cluster-size distribution; worth a follow-up if `hem_feasible_first`
is promoted to default, to understand which matrix structures it helps and
by how much before writing that into the general story.

## Stages 3 and 4 implementation

`prepare_boo_cuda(boo, bucketed=True, precomputed_rows=True, p=64)` returns an
explicit execution plan. Pass it to `boo_matmul(..., launch_mode='auto',
execution_plan=plan, p=64)`. The rowmajor format and its GPU staging remain
unchanged; a plan becomes invalid after moving/re-staging the matrix.

Bucketing splits each block's estimated shared-memory requirement at its median
and 90th percentile. Stable physical reordering makes each bucket contiguous;
each of at most three grids receives its own row/column/nnz maxima. All grids
accumulate into the same output. Extra launches and the ordinary CSR/COO stream
join are inside the timed call. Local row/column groups are not compacted.

Precomputed boundaries store `n_r+1` int32 local nnz offsets per block, plus an
int32 pointer per block and sentinel. The kernel loads these into the same shared
table used by binary search, preserving accumulation and synchronization. Empty
local rows and packed indices above 32767 are covered by regression tests.

Both options are independently selectable. The paired grid includes the ordinary
auto call, a one-bucket plan without stored boundaries, each optimization alone,
and both together for k=32/64. The plan baseline exposes Python/binding overhead.
CPU plan construction, transfer time, additional device bytes, bucket counts and
bucket maxima are recorded separately. Bucketed plans currently duplicate block
storage to retain the baseline format; these costs are included in the report.

```bash
python -m benchmark.bench_boo_opt --phase plans \
  --sizes 16384 65536 262144 --features 128 512 \
  --profiles fem2d fem2d_permuted diag_b0.3 diag_skewed scattered_1x8 pattern_diag_stripe \
  --out results/boo-plans
```

The kernel now accepts optional row-table pointers. Therefore each later run has
its own same-binary baseline; differences between absolute timings from the
stage-1 binary and the plan binary are not attributed to an optimization.

## Validation and results

Stage 1 remote job 21628759: 1,201 tests passed, 13 expected skips. Each of
memcheck, racecheck and synccheck passed all 32 marked tests, with zero errors
or race hazards.

Stage 1 paired timings (`results/boo_optimization_20260923/first`, 384 BOO
configurations, threshold=4): `hem_feasible_first=True` is a real win, geomean
~7% faster than `False` (0.465-0.468ms vs 0.503-0.505ms). `launch_mode='auto'`
vs `'legacy'` showed only a ~0.5% difference in that run, but threshold=4
means `small_block_policy='remainder'` spills some nonzeros to the CSR stream
on almost every workload, so `csr_nnz` was rarely 0 and the pure-BOO fast path
(`PURE_FAST_PATH`) rarely triggered -- that comparison wasn't actually
isolating the fast path.

A follow-up `--phase fastpath` run (`results/boo-fastpath-20260923`,
`small_block_threshold=0` so `remainder_nnz` is confirmed 0 in every one of
the 24 paired cells, i.e. `PURE_FAST_PATH` genuinely engages) found the same
result holds even when isolated: geomean `auto`/`legacy` ratio 0.997, with
per-cell ratios scattered 0.92-1.02x and no consistent direction, including at
N=16384 where fixed per-call overhead should matter most if the 3-stream/
4-event fork-join were the bottleneck. **Idea 6 (skip stream/event setup for
pure-BOO calls) does not measurably help** -- BOO's weaker small-matrix
results have a different cause. `launch_mode='auto'` and `hem_feasible_first`
remain independently opt-in; only `hem_feasible_first` is worth carrying
forward as a default candidate.

Plan validation and timings (stages 3/4) pending.
