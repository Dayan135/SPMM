# Warp stall-reason breakdown across matrix shapes, sizes, and feature counts

**Status: extends H6's stall-reason verification (see `WHY_BLOCK_KERNEL_FAILS_TO_SCALE.md` H6 and
`LAST_EXPERIMENTS.md` Experiment 5) beyond its single data point (`diag_b0.3`, N=262144, f512) to a
5-matrix × 2-size × 2-feature grid, `block` vs cuSPARSE.** H6's original claim — that `block`'s
redundant per-thread instruction volume "directly explains" its SM-throughput gap vs cuSPARSE — was
already corrected once (the stall-verified causes are `mio_throttle`/atomicAdd contention and
`barrier`/`__syncthreads()`, not the ALU pipe). This document checks whether *that* corrected finding
itself generalizes across matrix shapes, or was itself specific to `diag_b0.3`. **It does not fully
generalize** — see Finding 2.

## Method

- Jobs: `jobs/profile_stall_reasons/profile_20260714.sh` (diag matrices + first scattered pass) and
  `jobs/profile_stall_reasons/rerun_scattered_20260714.sh` (scattered matrices, rerun after fixing a
  matrix-generation bug — see Caveats).
- Metrics: identical set to H6's original stall-reason run —
  `smsp__warp_issue_stalled_{barrier,drain,long_scoreboard,math_pipe_throttle,mio_throttle,
  no_instruction,not_selected,selected,short_scoreboard,wait}_per_warp_active.pct`,
  `sm__pipe_{alu,fma}_cycles_active.avg.pct_of_peak_sustained_elapsed`,
  `sm__warps_active.avg.pct_of_peak_sustained_active`, plus `gpu__time_duration.sum` and
  `sm__throughput.avg.pct_of_peak_sustained_elapsed` for context.
- Matrices: `diag_b0.1`, `diag_b0.3` (64-wide diagonal band, no off-diagonal noise, diag_density
  0.1/0.3), `scattered_1x8`, `scattered_8x1`, `scattered_2x2` (8×8-PBR-block windows with a single
  populated sub-pattern of the given shape — see `profile_pbr.py`'s `_MATRIX_PROFILES`).
- Sizes: 1024, 262144. Features: 128, 512. Kernel: `block` (PBR) vs cuSPARSE (`torch.sparse.mm`,
  `csrmm_alg2_kernel`).
- Raw ncu CSVs: `ncu_out/stall_reasons/` (untracked, local only, same convention as the original H6
  `ncu_out/stall_analysis/`).

### Caveat: scattered-matrix density bug, fixed mid-run

`profile_pbr.py`'s scattered-matrix generator originally used a hardcoded `density=1e-5`, which
produces **zero populated blocks at N=1024** (verified failure: all 12 `scattered_*_1024_*` runs in
the first pass errored out with `ValueError: ... produced 0 blocks`). Fixed by scaling density as
`0.005 * 1024 / size` (0.5% at N=1024, matching `test_benchmark.py`'s existing `_sd()` convention),
then **reran all scattered matrices at both sizes** so every scattered result in this doc uses the
same, consistent density formula (`4c67d78`, `1b4d7b8`). `diag_*` results are unaffected (diag
density is size-independent already) and were not rerun.

### Caveat: N=1024 results are directional only

At N=1024, both kernels complete in **5–60 microseconds** (see the appendix table) — the `block`
grid is only a few hundred CTAs and SM occupancy for `block` bottoms out at 5–9% for the scattered
matrices. This is too short/too small a kernel for ncu's per-SM stall-reason sampling to be reliable,
and it shows: e.g. `scattered_1x8_1024_f128`'s cuSPARSE row reports `long_scoreboard=56.5%` while its
262144 counterpart reports `89.7%` — a huge swing that reflects sampling noise at tiny scale, not a
real trend with size. **All conclusions below are drawn from the N=262144 rows only.** The N=1024
numbers are included in the appendix for completeness since they were explicitly requested, but
should not be used to draw conclusions.

## Results (N=262144, the reliable rows)

| matrix | features | kernel | time (ms) | SM% | `mio_throttle` | `barrier` | `long_scoreboard` | `short_scoreboard` | `math_pipe_throttle` | ALU util | warps active |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `diag_b0.1` | 128 | block    | 2.013  | 54.83 | 10.77 | 19.87 | 24.90 | 12.09 | 2.12 | 25.71 | 88.19 |
| `diag_b0.1` | 128 | cusparse | 0.686  | 31.39 | 0.60  | 0.00  | 85.66 | 2.53  | 0.15 | 11.94 | 72.52 |
| `diag_b0.1` | 512 | block    | 8.021  | 33.06 | 17.66 | 25.29 | 19.26 | 13.00 | 0.95 | 18.46 | 90.63 |
| `diag_b0.1` | 512 | cusparse | 2.765  | 30.99 | 2.54  | 0.00  | 84.08 | 3.54  | 0.14 | 11.79 | 73.25 |
| `diag_b0.3` | 128 | block    | 3.336  | 44.25 | 17.84 | 14.10 | 20.48 | 12.91 | 1.75 | 21.18 | 89.36 |
| `diag_b0.3` | 128 | cusparse | 1.026  | 58.38 | 0.27  | 0.00  | 77.91 | 2.86  | 0.33 | 20.84 | 75.91 |
| `diag_b0.3` | 512 | block    | 12.986 | 30.28 | **21.32** | **20.11** | 17.57 | 12.83 | 0.83 | 17.24 | 91.00 |
| `diag_b0.3` | 512 | cusparse | 4.101  | 58.19 | 0.21  | 0.00  | 78.24 | 2.83  | 0.32 | 20.77 | 76.02 |
| `scattered_1x8` | 128 | block    | 0.129 | 49.66 | **0.02** | 25.39 | 46.99 | 3.21  | 2.15 | 25.78 | 87.73 |
| `scattered_1x8` | 128 | cusparse | 0.135 | 17.54 | 0.01  | 0.00  | 89.70 | 1.61  | 0.17 | 9.34  | 72.56 |
| `scattered_1x8` | 512 | block    | 0.496 | 26.56 | **0.29** | 36.58 | 49.91 | 2.19  | 0.42 | 16.80 | 90.05 |
| `scattered_1x8` | 512 | cusparse | 0.486 | 19.25 | 1.18  | 0.00  | 86.36 | 3.17  | 0.10 | 10.25 | 74.36 |
| `scattered_2x2` | 128 | block    | 0.091 | 67.49 | 0.41  | 29.47 | 27.42 | 5.15  | 3.25 | 30.03 | 85.79 |
| `scattered_2x2` | 128 | cusparse | 0.082 | 19.65 | 0.76  | 0.00  | 78.35 | 5.23  | 0.40 | 13.18 | 68.31 |
| `scattered_2x2` | 512 | block    | 0.354 | 30.32 | 22.56 | 9.90  | 22.58 | 13.82 | 1.09 | 15.87 | 88.32 |
| `scattered_2x2` | 512 | cusparse | 0.309 | 21.15 | 1.60  | 0.00  | 79.76 | 7.37  | 0.24 | 14.21 | 70.80 |
| `scattered_8x1` | 128 | block    | 0.288 | 38.18 | 17.97 | 21.12 | 16.06 | 14.38 | 1.42 | 16.52 | 87.78 |
| `scattered_8x1` | 128 | cusparse | 0.117 | 27.74 | **21.55** | 0.00  | 36.19 | 14.11 | 0.62 | 14.83 | 67.34 |
| `scattered_8x1` | 512 | block    | 1.090 | 26.30 | 22.33 | 25.34 | 11.65 | 13.48 | 0.64 | 13.78 | 90.02 |
| `scattered_8x1` | 512 | cusparse | 0.446 | 28.10 | **21.59** | 0.00  | 37.44 | 14.52 | 0.52 | 15.02 | 71.78 |

## Finding 1: the mio_throttle + barrier mechanism generalizes across diag densities

`diag_b0.1` (sparser 10%-density band) reproduces `diag_b0.3`'s qualitative profile closely: both
show `block` dominated by `mio_throttle` (10.8–21.3%) and `barrier` (14.1–25.3%), both far above
cuSPARSE's near-zero values for the same two reasons, and both show cuSPARSE staying
`long_scoreboard`-dominated (78–86%). The magnitudes shift a bit with density and feature count, but
the *mechanism* — `block`'s per-(row,feature) `atomicAdd` Y-flush contending on the shared-mem/atomic
pipe, plus `__syncthreads()` waits, while cuSPARSE just waits on DRAM — holds across both densities.
This was the expected result and is not new; it confirms H6's corrected finding wasn't a one-off
artifact of the specific 0.3 density.

## Finding 2: the mechanism is NOT universal — it depends on block occupancy shape, not just density

`scattered_1x8` breaks the pattern completely. `block`'s `mio_throttle` collapses to **0.02–0.29%**
— essentially zero, the same order of magnitude as cuSPARSE's own near-zero value — while `barrier`
(25–37%) and especially `long_scoreboard` (47–50%) take over as the dominant stalls. This is a
qualitatively different bottleneck than `diag_b0.1`/`diag_b0.3`: `block` on `scattered_1x8` behaves
more like a memory-latency-bound kernel (elevated `long_scoreboard`) with synchronization overhead on
top, not an atomic-contention-bound one.

`scattered_8x1` and `scattered_2x2`, on the other hand, largely reproduce the diag-like
`mio_throttle`-heavy profile (17.97–22.56% at N=262144, comparable to `diag_b0.3`'s 17.84–21.32%) —
`scattered_2x2`'s f128 case is the one partial exception (`mio_throttle`=0.41%, more like
`scattered_1x8`), suggesting the effect can also depend on feature count, not just block shape, at
least at this matrix's scale.

**Working hypothesis** (correlational, not independently isolated by an ablation the way H4/H5 were):
`mio_throttle` tracks how many *distinct output rows* a single PBR block's `atomicAdd` flush touches.
`diag`'s dense 64-wide band and `scattered_8x1`'s 8-populated-rows-per-block pattern both write many
rows per block per flush; `scattered_1x8`'s 1-populated-row-per-block pattern writes only one row via
`atomicAdd` regardless of how many columns are packed into it, so there is far less atomic-pipe
traffic to contend on. This has **not** been verified with a dedicated ablation (the same discipline
H4/H5 used to confirm CTA-count and nnz-balance mechanisms) — it is offered as the most likely
explanation given the data, not a confirmed cause.

## Finding 3: mio_throttle is not `block`-specific — cuSPARSE shows it too, on scattered_8x1

The clearest new result: **cuSPARSE's own `mio_throttle` is 21.55%/21.59%** on `scattered_8x1` at
both feature counts — the same order of magnitude as `block`'s 17.97%/22.33% there, and wildly out of
line with cuSPARSE's near-zero `mio_throttle` everywhere else in this table (0.01–2.54%). cuSPARSE's
`long_scoreboard` also drops sharply on this matrix (36–37%, vs 77–90% elsewhere), consistent with
`mio_throttle` displacing it as a stall cause rather than adding to a fixed total.

This means the original H6 framing — `mio_throttle`/atomicAdd-pipe contention as a `block`-kernel
implementation weakness relative to a "clean" cuSPARSE — does not hold universally. For at least this
one matrix shape (many rows populated per structural block, scattered across the matrix), **cuSPARSE
itself pays the same atomic/shared-mem-pipe contention cost.** That points to `mio_throttle` being at
least partly a property of the *workload* — narrow-column, many-row scattered writes forcing any
row-accumulating SpMM algorithm through contended atomic/shared-mem paths — rather than solely a
`block`-specific inefficiency to fix in isolation.

## Updated bottom line

H6's corrected mechanism (`mio_throttle` + `barrier` as `block`'s dominant, stall-verified causes,
not the ALU pipe) is real and reproduces on a second diag density, but **is matrix-shape-dependent,
not universal**: it disappears almost entirely on `scattered_1x8` (replaced by
`long_scoreboard`+`barrier`), and on the one matrix shape where it's strongest for cuSPARSE too
(`scattered_8x1`), it isn't even unique to `block`. Any future work targeting `mio_throttle`
specifically (e.g. reducing `atomicAdd` contention in `block`'s Y-flush) should expect the benefit to
be **matrix-shape-dependent** — largest on dense/many-rows-per-block matrices like `diag_*` and
`scattered_8x1`, close to irrelevant on single-row-per-block matrices like `scattered_1x8`, where the
real lever would instead be `long_scoreboard`/`barrier` reduction.

## Appendix: N=1024 raw data (directional only — see caveat above)

| matrix | features | kernel | time (µs) | SM% | `mio_throttle` | `barrier` | `long_scoreboard` | warps active |
|---|---|---|---|---|---|---|---|---|
| `diag_b0.1` | 128 | block    | 13.15 | 34.36 | 4.19  | 15.53 | 18.25 | 61.37 |
| `diag_b0.1` | 128 | cusparse | 10.62 | 7.97  | 0.00  | 0.00  | 47.55 | 7.89  |
| `diag_b0.1` | 512 | block    | 40.29 | 25.66 | 14.00 | 24.09 | 19.44 | 70.84 |
| `diag_b0.1` | 512 | cusparse | 12.06 | 28.40 | 0.13  | 0.00  | 49.03 | 24.51 |
| `diag_b0.3` | 128 | block    | 18.62 | 30.89 | 9.63  | 12.53 | 18.66 | 62.09 |
| `diag_b0.3` | 128 | cusparse | 10.18 | 23.26 | 0.03  | 0.00  | 47.82 | 18.96 |
| `diag_b0.3` | 512 | block    | 60.48 | 25.93 | 17.38 | 19.32 | 18.78 | 70.63 |
| `diag_b0.3` | 512 | cusparse | 17.09 | 55.79 | 2.07  | 0.00  | 62.91 | 68.30 |
| `scattered_1x8` | 128 | block    | 5.09  | 5.19 | 0.00 | 19.85 | 25.04 | 8.32 |
| `scattered_1x8` | 128 | cusparse | 16.93 | 0.53 | 0.00 | 0.00  | 56.52 | 7.39 |
| `scattered_1x8` | 512 | block    | 8.64  | 5.95 | 0.00 | 28.23 | 30.07 | 8.33 |
| `scattered_1x8` | 512 | cusparse | 19.90 | 1.79 | 0.00 | 0.00  | 58.35 | 6.91 |
| `scattered_2x2` | 128 | block    | 4.58  | 5.39 | 0.00 | 21.54 | 18.60 | 8.32 |
| `scattered_2x2` | 128 | cusparse | 20.70 | 0.31 | 0.00 | 0.00  | 34.71 | 5.89 |
| `scattered_2x2` | 512 | block    | 5.89  | 7.22 | 0.00 | 4.03  | 27.53 | 8.19 |
| `scattered_2x2` | 512 | cusparse | 19.78 | 1.24 | 0.00 | 0.00  | 35.50 | 6.36 |
| `scattered_8x1` | 128 | block    | 5.66  | 7.70 | 0.00 | 13.84 | 16.84 | 8.19 |
| `scattered_8x1` | 128 | cusparse | 17.79 | 0.68 | 0.00 | 0.00  | 21.37 | 6.36 |
| `scattered_8x1` | 512 | block    | 12.10 | 9.33 | 0.00 | 32.30 | 14.76 | 8.28 |
| `scattered_8x1` | 512 | cusparse | 12.86 | 3.84 | 0.00 | 0.00  | 21.22 | 7.07 |

Note `block`'s `mio_throttle` is exactly 0.00 for every scattered matrix at N=1024 (even
`scattered_8x1`, which shows the diag-like `mio_throttle`-heavy profile at N=262144) — consistent
with too few concurrent CTAs at this size for atomic-pipe contention to build up at all, another
reason these rows are directional only.
