# Interpretation and next experiments

**Follow-up:** [TILE_EXPERIMENTS.md](../../2026-10-12_tile_optimization/TILE_EXPERIMENTS.md) now contains
three-allocation confirmations for selected cases. Its timing-order controls
show that small effects/rankings from this rotation-ordered full sweep are
provisional. Keep the data below as historical evidence; use the confirmed
report for repeated cases and [TILE_REVIEW.md](../../2026-10-12_tile_optimization/TILE_REVIEW.md) for the
atomic-counter definitions and next experiment.

These are complete-call results from one allocation per (matrix, clustering)
shard. FPT comparisons are paired inside each cell; clustering comparisons
cross allocations. Treat small differences as provisional. The full factorial
contains 612 measured matrix/F/clustering/k cells, each with FPT1/2/4; explicit
duplicate labels account for the other 12 cells. Duplicate observations are
excluded from the FPT distribution statistics.

## What the sweep establishes

**FPT2 is the strongest conservative candidate.** Across all 612 measured cells,
its worst observed speedup over FPT1 is 0.997 (about 0.3% slower), and none
regress by more than 5%. On real matrices the geometric-mean speedup grows from
1.073 at F32 to 1.122 at F512. FPT4 has larger gains on dense-block cases, but
29/612 cells regress by more than 5%; its worst ratio 0.843 means 18.7% slower.
No FPT1 configuration is the fastest selected winner among the 78 matrix/F
cases. This supports independently validating FPT2 as a default candidate and
retaining FPT4 as a tuned option; it does not justify unconditional FPT4.

**The main cuSPARSE wins survive a stronger baseline check.** Best BOO wins
30/78 matrix/F cases against its paired direct baseline, and 29/78 against the
fastest original-order direct timing observed across all shards. That second
comparison mixes allocations and is only a sensitivity check. Of the 71 winner
cells where all four direct algorithms were measured, BOO wins 28. The only
win lost under pooling is 262k permuted FEM at F128, originally just 2.7%.
At F512, pwtk remains 1.205x faster using the pooled baseline; dense diagonal
remains 2.594x at 262k and 2.574x at 1M. Permuted FEM remains 1.526x and 1.620x.
Missing algorithms are still a limitation even when pooled observations help.

**Clustering should be selected, not assumed beneficial.** At F512, after
choosing k/FPT separately for each method:

- 1M permuted FEM: identity 197.040 ms versus HEM 31.747 ms (6.21x). Block
  count falls from 2,121,680 to 272,629, with full stored-entry coverage in both.
- thermal2: identity 46.412 ms versus HEM 27.597 ms (1.68x). Graclus gives
  27.843 ms, very close; its recorded clustering time is 13.73 s versus HEM's
  60.61 s. Original CSR still wins on this matrix.
- 1M diag_dd0.05: identity 11.662 ms versus Graclus 22.306 ms. Identity k64
  has 16,384 blocks and 93.85% entry coverage; Graclus k32 has 200,804 blocks
  and 88.94% coverage. This is evidence of a poor resulting partition, not
  evidence that more expensive clustering necessarily yields better SpMM.
- roadNet: METIS is the fastest BOO option at F512, but still loses to CSR.

These selected-config comparisons change k and sometimes FPT as well as the
clustering method. They measure the achievable configurations in this grid,
not a controlled causal effect of clustering alone. See paired_fpt.csv for
comparisons that hold clustering and k fixed.

**Neither k32 nor k64 dominates.** Sparse diagonal winners often use k64/FPT2;
dense diagonal winners use k32/FPT4. The group-size plot holds matrix, F and
method fixed while optimizing FPT independently for each k. Reported ties for
HEM/Graclus diagonal stripes reflect genuinely identical saved labels at both
k values, not separately timed evidence.

**Size changes both workload and structure for noisy inputs.** Going from 262k
to 1M multiplies dimension by four. The repository fixes off-diagonal noise
density, so its expected noise count grows by sixteen. Consequently, diag_skewed
is not a constant-nnz-per-row scaling experiment. The size plot shows runtime
ratios without calling all departures from 4x a kernel scaling failure.

## What to do next

The concrete `P` × FPT and `BOO_FEAT_BLOCK` experiment designs are in
[NEXT_EXPERIMENTS.md](../../../NEXT_EXPERIMENTS.md).

1. Repeat a focused panel on three independent GPU allocations: pwtk, dense
   diagonal, natural/permuted FEM, webbase and sparse noisy diagonal. Keep
   FPT1/2/4 paired and cover k32/64. Include the winning labels and identity.
   This distinguishes large transferable gains from small selection effects.
2. For memory-limited F512 cells, run fewer direct algorithms per process while
   repeating BOO anchors in each cell. This fills missing baseline coverage
   without comparing unrelated timing scopes.
3. For fixed-pattern repeated calls, evaluate setup-time selection among original
   CSR, BOO FPT2 and BOO FPT4. First prune expensive clustering on already-local
   matrices. Measure selection/setup cost and amortization, then assess the
   policy on held-out matrices; do not fit a density threshold to this panel.
4. Keep the next structural kernel experiment focused on low-occupancy blocks
   and unused staged X (roadNet-like cases). FPT addresses repeated instruction
   work; the weak response there suggests a different bottleneck. Profile before
   choosing compaction or a new index representation.

Reproduce plots and derived CSVs from the repository root:

```bash
MPLCONFIGDIR=/private/tmp/spmm-mpl conda run -n fem python scripts/research_panel/plot_full_sweep.py
```

No kernel/default changes or new GPU jobs were made during this analysis.
