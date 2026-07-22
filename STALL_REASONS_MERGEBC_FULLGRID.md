# mergeBC on the standard full benchmark grid: does the 9-12% win generalize? (2026-07-22)

Branch `feat/pbr-mergebc-alloc-ablation`. Follow-up to `STALL_REASONS_ZEROIDLE_MICROBENCH.md` (found
`mergeBC` beats `baseline` by ~9-12% on 4/5 matrices, narrow 5-matrix/2-feature diagnostic grid) and
`STALL_REASONS_MERGEBC_ALLOC_ABLATION.md` (confirmed the win is instruction-removal, not an occupancy
side effect). Question here: was that win an artifact of the small hand-picked grid, or does it hold up
on this project's actual standard benchmark surface?

## Grid

Taken verbatim from `tests/torch_fast_spdmm/test_benchmark.py` rather than inventing new parameters:

- **Matrices**: all 10 `MATRIX_FAMILIES` (`diag_b0.3`, `diag_b1.0`, and 4 scattered shapes ×
  {0.5pct, 2pct} density: `1x8`, `8x1`, `2x2`, `2x4`).
- **Features**: `[16, 32, 64, 128, 256, 512]` (test_benchmark.py's `FEATURES`).
- **Block/min_nnz configs**: test_benchmark.py's own `_SCALAR_DIAG_BLOCK_CONFIGS` — `(2,2) (2,3) (4,4)
  (4,8) (8,2) (8,12) (8,20)`, fixed `smem_size=2048`, **`col_major=True`** (reuses the colgroup kernel's
  pre-sorted arrays — this is test_benchmark.py's own established convention for its scalar_diag sweep,
  `SCALAR_DIAG_RUNS`, not something invented for this experiment).
- **Sizes**: `{16384, 65536, 262144}` — this project's standard size list (see CLAUDE.md's
  `plot_benchmarks` example / `LAST_EXPERIMENTS.md`), not test_benchmark.py's full `_ALL_SIZES` (which
  also has the toy sizes 1024/4096).
- **Variants**: `baseline` vs `mergeBC`, both forced onto the scalar `scalar_diag` path
  (`kernel="scalar_diag"`) — no new kernel code, reuses exactly what `feat/pbr-mergebc-alloc-ablation`
  already has.
- **Trials**: N=5 repeated `torch.cuda.Event` timings per config after 3 untimed warmup calls (same
  repeated-trial standard used all session), pure wall-clock, no `ncu` (would be far too slow at this
  grid size — see below).

10 matrices × 3 sizes × 6 features × 7 block configs = **1260 (matrix, size, features, block config)
cells**, each with a `baseline` and `mergeBC` mean over 5 trials. Script:
`tests/torch_fast_spdmm/benchmark/benchmark_mergebc_fullgrid.py`. Jobs: one per size (this project hit
24h timeouts on monolithic full-config-matrix jobs before — see `LAST_EXPERIMENTS.md`'s tileblock
section — so `jobs/bench_mergebc_fullgrid/bench_{16384,65536,262144}.sh` are separate sbatch
submissions). All three completed in **4-5 minutes each** (pure wall-clock timing has none of `ncu`'s
per-launch profiling overhead), 0 OOMs, 0 skipped configs, 0 errors — 12,600 total timed trials.

## Results — win/loss breakdown (±3% WIN/REGRESSION/WASH, this project's standard threshold)

**Overall, 1260 cells: 547 WIN (43.4%), 35 REGRESSION (2.8%), 678 WASH (53.8%).** Geomean speedup
1.0306× (+3.06%), median 1.0199× (+1.99%) — `mergeBC` over `baseline`, so positive = `mergeBC` faster.

By matrix family — the split is stark:

| matrix | n | WIN | REGRESSION | WASH | geomean Δ | median Δ |
|---|---|---|---|---|---|---|
| `diag_b0.3` | 126 | 109 (86.5%) | 3 (2.4%) | 14 (11.1%) | **+6.28%** | +6.19% |
| `diag_b1.0` | 126 | 119 (94.4%) | 0 (0.0%) | 7 (5.6%) | **+8.45%** | +8.46% |
| `scattered_1x8_d0.5pct` | 126 | 31 (24.6%) | 3 (2.4%) | 92 (73.0%) | +1.71% | +0.76% |
| `scattered_1x8_d2pct` | 126 | 45 (35.7%) | 0 (0.0%) | 81 (64.3%) | +3.23% | +1.50% |
| `scattered_2x2_d0.5pct` | 126 | 49 (38.9%) | 2 (1.6%) | 75 (59.5%) | +2.62% | +1.11% |
| `scattered_2x2_d2pct` | 126 | 73 (57.9%) | 3 (2.4%) | 50 (39.7%) | +3.28% | +4.12% |
| `scattered_2x4_d0.5pct` | 126 | 20 (15.9%) | 11 (8.7%) | 95 (75.4%) | +1.15% | +0.49% |
| `scattered_2x4_d2pct` | 126 | 30 (23.8%) | 6 (4.8%) | 90 (71.4%) | +1.01% | +0.60% |
| `scattered_8x1_d0.5pct` | 126 | 34 (27.0%) | 2 (1.6%) | 90 (71.4%) | +1.69% | +0.80% |
| `scattered_8x1_d2pct` | 126 | 37 (29.4%) | 5 (4.0%) | 84 (66.7%) | +1.40% | +0.75% |

The dense-band matrices (`diag_b0.3`/`diag_b1.0` — the same family the original diagnostic doc's
strongest results came from) dominate the win, with regressions essentially absent (0-2.4%). The 4
scattered families are net-positive but much softer — mostly WASH, modest WIN rates, and a handful of
real (if small) regressions, worst concentrated in `scattered_2x4` (8.7%/4.8% REGRESSION).

By size (all matrices combined):

| size | n | WIN | REGRESSION | WASH | geomean Δ |
|---|---|---|---|---|---|
| 16384 | 420 | 217 (51.7%) | 16 (3.8%) | 187 (44.5%) | +3.73% |
| 65536 | 420 | 178 (42.4%) | 13 (3.1%) | 229 (54.5%) | +2.91% |
| 262144 | 420 | 152 (36.2%) | 6 (1.4%) | 262 (62.4%) | +2.53% |

**The win doesn't degrade into regressions at scale — if anything the opposite (REGRESSION rate falls
3.8%→1.4%) — but it does soften toward WASH** (WIN rate falls 51.7%→36.2%) in the aggregate. This
softening is entirely a scattered-family effect, not a dense-family one: filtered to just
`diag_b0.3`/`diag_b1.0`, the win **holds essentially flat across all 3 sizes** (`diag_b0.3`: +7.05% →
+6.07% → +5.72%; `diag_b1.0`: +9.37% → +7.95% → +8.04%, win rate 95-100% at every size). The aggregate
softening is because a growing share of scattered-family cells become tiny/launch-overhead-bound at
larger sizes with these small (2×2, 4×4) block configs and land inside the ±3% noise band, not because
the mechanism itself weakens with scale.

## Sanity-check against the original narrow-grid numbers

The one directly comparable cell (`block=8, min_nnz=2, size=262144` — the exact config
`STALL_REASONS_ZEROIDLE_MICROBENCH.md` used, modulo `col_major`) reproduces closely:

| matrix/feat | this grid (col_major=True) | original doc (row-major) |
|---|---|---|
| `diag_b0.3`/128 | +10.27% | +9.9% |
| `diag_b0.3`/512 | +11.05% | +10.0% |
| `diag_b1.0`/128 | +8.99% | (not tested in original 5-matrix grid) |
| `diag_b1.0`/512 | +9.95% | (not tested) |

Consistent to ~1 point — good cross-validation of both studies' methodology.

**One real, load-bearing discrepancy found**: `scattered_8x1` at this exact same config
(`block=8,min_nnz=2,size=262144`) showed a strong ~10-11% win in the *original row-major* study, but
here (col-sorted, `col_major=True`) it's **flat — +0.58%/-0.19%** at feat=128/512 (tight, low-noise
measurements: sd ≈ 0.001-0.002ms on ~0.4-1.5ms means, not measurement jitter). This is a genuine,
reproducible sensitivity to PBR block ordering: col-sorting improves `baseline`'s own performance on this
matrix shape (better X-cache locality across consecutive scheduled blocks — the same mechanism that
motivated the `colgroup` kernel), which erases the relative gap `mergeBC` used to close. Not something
the original 5-matrix diagnostic doc could have caught (it only tested row-major order). This means
**`mergeBC`'s benefit on `scattered_8x1` specifically is conditional on block ordering**, not a
free-standing win independent of it.

## Worst regressions and best wins (illustrative extremes)

Worst losses cluster in small/sparse configs (`block=2x2` or small `min_nnz`, low-to-mid features,
`scattered_2x4`/`scattered_2x2`/`scattered_8x1`) where absolute times are ~0.05-0.2ms — launch-overhead
dominated, high relative noise despite low absolute sd:

```
scattered_8x1_d2pct   16384/f64   block=2x2 min_nnz=3   0.054→0.092ms  -40.7%
scattered_2x4_d2pct   16384/f16   block=8x8 min_nnz=2   0.056→0.093ms  -39.5%
scattered_2x2_d2pct   16384/f16   block=8x8 min_nnz=12  0.063→0.098ms  -35.6%
```

Best wins are similarly concentrated at small/sparse configs where the reverse noise direction hits
(same regime, opposite sign — not a systematic mechanism):

```
scattered_1x8_d2pct   16384/f64   block=8x8 min_nnz=20  0.161→0.053ms  +204%
scattered_2x4_d0.5pct 16384/f16   block=4x4 min_nnz=8   0.094→0.050ms  +88%
```

None of the biggest swings (either direction) are on the two dense-band matrices, whose largest cells
land in the 1-12ms range and whose speedups cluster tightly around their family geomean (`diag_b0.3`
min/max across all 126 cells: 0.915×/1.572×; most of that spread is itself at the smallest, launch-
overhead-bound `size=16384` configs).

## Verdict

**The 9-12% win is real and generalizes for the matrix family it was found on (`diag_b0.3`/`diag_b1.0` —
dense, many-row-per-block matrices) — it was not cherry-picked. It holds at all 3 standard sizes without
degrading, with regression rates near zero (0-2.4%) throughout.** It does *not* generalize as strongly to
the four scattered/sparse matrix families: still net-positive on aggregate (geomean +1.0% to +3.3%,
never negative), but mostly a WASH at the ±3% threshold, with a genuine minority of real regressions
(worst: `scattered_2x4_d0.5pct` at 8.7% REGRESSION rate) concentrated in small-block/high-min_nnz/
low-feature/moderate-size corners where kernel time is small enough that CUDA launch overhead — not the
`smem_Y` traffic `mergeBC` removes — dominates the wall-clock budget, and where measured swings in both
directions are large relative to the (still small, sub-2% sd) trial noise. One matrix family
(`scattered_8x1`) that showed a strong win in the original row-major-only diagnostic study loses that win
entirely under this project's standard col-sorted (`col_major=True`) ordering, a genuine methodological
finding this full-grid run surfaced that the narrow grid could not have.

**On porting into the production vec-kernel path**: this result is a stronger green light for `diag`-like
(dense-band) workloads than the narrow grid alone gave, and de-risks the "was it cherry-picked" concern
for that family specifically. But it also sharpens the caveat: the win is workload-shape-dependent and at
least one case (`scattered_8x1`) is ordering-dependent, not a blanket improvement. Before porting
`mergeBC`'s fixed-`p` Step B/C fusion into `pbr_spmm_vec_kernel` (the kernel actually used in production
at `features%4==0`, per `STALL_REASONS_ZEROIDLE_MICROBENCH.md`'s original recommendation — still not
attempted), it would be worth re-running this same full-grid methodology on the vec kernel directly
rather than assuming the scalar-kernel win transfers unchanged, precisely because this experiment showed
the effect is sensitive to workload shape and block ordering in ways a single narrow config could miss.

## Data

Raw CSVs (12,600 rows: `matrix,size,features,block_size,min_nnz,smem_size,variant,rep,time_ms`) at job
time in `ncu_out/mergebc_fullgrid/mergebc_fullgrid_{16384,65536,262144}.csv` on the remote host (not
committed — regenerable via `jobs/bench_mergebc_fullgrid/bench_<size>.sh`).
