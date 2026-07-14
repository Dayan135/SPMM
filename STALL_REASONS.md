# Warp stall-reason breakdown across matrix shapes, sizes, and feature counts

Extends H6's stall-reason check (`WHY_BLOCK_KERNEL_FAILS_TO_SCALE.md` H6, previously verified only on
`diag_b0.3` @ N=262144/f512) across 5 matrices × 2 sizes × 2 feature counts, `block` vs cuSPARSE.

## What the metrics mean

Each is "% of cycles a warp was stalled, unable to issue, for this reason" (`smsp__warp_issue_stalled_*`),
or a pipe/occupancy percentage:

- **`mio_throttle`** — stalled because the shared-memory/atomic execution pipe is saturated. In
  `block`, this is the pipe `atomicAdd` (the Y-accumulation flush) and shared-memory ops both go
  through — high value = atomic/smem contention.
- **`barrier`** — stalled at a `__syncthreads()` waiting for other threads in the block to catch up.
  High value = lots of time synchronizing, not computing.
- **`long_scoreboard`** — stalled waiting on a long-latency global-memory (DRAM) load. High value is
  *normal/healthy* for a memory-bound kernel like SpMM — it means the kernel is limited by fetching
  data, not by an internal bottleneck.
- **`warps active`** (`sm__warps_active`, occupancy) — % of the SM's warp slots that had a resident
  warp on average. Not a stall reason — a measure of how full the SM was kept, independent of whether
  those warps were stalled or issuing.
- (Table also has `short_scoreboard` = same as above but shared-memory/fast latency, and
  `math_pipe_throttle` = stalled specifically on the ALU/FMA pipe, included because it's the metric
  that directly disproves H6's original ALU-pipe-bottleneck claim.)

## Method

Jobs: `jobs/profile_stall_reasons/profile_20260714.sh` + `rerun_scattered_20260714.sh`. Matrices:
`diag_b0.1`/`diag_b0.3` (64-wide diagonal band), `scattered_1x8`/`scattered_8x1`/`scattered_2x2`
(single populated sub-pattern per 8×8 PBR block). Sizes 1024/262144, features 128/512, `block` (PBR)
vs cuSPARSE. Raw CSVs: `ncu_out/stall_reasons/` (untracked, local only).

**N=1024 caveat:** kernels finish in 5–60µs there (see appendix) — too short/small (`block` occupancy
bottoms out at 5–9%) for reliable stall sampling. All conclusions below use N=262144 only.

## Data (N=262144)

| matrix | feat | kernel | time (ms) | SM% | `mio_throttle` | `barrier` | `long_scoreboard` | warps active |
|---|---|---|---|---|---|---|---|---|
| `diag_b0.1` | 128 | block    | 2.013  | 54.83 | 10.77 | 19.87 | 24.90 | 88.19 |
| `diag_b0.1` | 128 | cusparse | 0.686  | 31.39 | 0.60  | 0.00  | 85.66 | 72.52 |
| `diag_b0.1` | 512 | block    | 8.021  | 33.06 | 17.66 | 25.29 | 19.26 | 90.63 |
| `diag_b0.1` | 512 | cusparse | 2.765  | 30.99 | 2.54  | 0.00  | 84.08 | 73.25 |
| `diag_b0.3` | 128 | block    | 3.336  | 44.25 | 17.84 | 14.10 | 20.48 | 89.36 |
| `diag_b0.3` | 128 | cusparse | 1.026  | 58.38 | 0.27  | 0.00  | 77.91 | 75.91 |
| `diag_b0.3` | 512 | block    | 12.986 | 30.28 | **21.32** | **20.11** | 17.57 | 91.00 |
| `diag_b0.3` | 512 | cusparse | 4.101  | 58.19 | 0.21  | 0.00  | 78.24 | 76.02 |
| `scattered_1x8` | 128 | block    | 0.129 | 49.66 | **0.02** | 25.39 | 46.99 | 87.73 |
| `scattered_1x8` | 128 | cusparse | 0.135 | 17.54 | 0.01  | 0.00  | 89.70 | 72.56 |
| `scattered_1x8` | 512 | block    | 0.496 | 26.56 | **0.29** | 36.58 | 49.91 | 90.05 |
| `scattered_1x8` | 512 | cusparse | 0.486 | 19.25 | 1.18  | 0.00  | 86.36 | 74.36 |
| `scattered_2x2` | 128 | block    | 0.091 | 67.49 | 0.41  | 29.47 | 27.42 | 85.79 |
| `scattered_2x2` | 128 | cusparse | 0.082 | 19.65 | 0.76  | 0.00  | 78.35 | 68.31 |
| `scattered_2x2` | 512 | block    | 0.354 | 30.32 | 22.56 | 9.90  | 22.58 | 88.32 |
| `scattered_2x2` | 512 | cusparse | 0.309 | 21.15 | 1.60  | 0.00  | 79.76 | 70.80 |
| `scattered_8x1` | 128 | block    | 0.288 | 38.18 | 17.97 | 21.12 | 16.06 | 87.78 |
| `scattered_8x1` | 128 | cusparse | 0.117 | 27.74 | **21.55** | 0.00  | 36.19 | 67.34 |
| `scattered_8x1` | 512 | block    | 1.090 | 26.30 | 22.33 | 25.34 | 11.65 | 90.02 |
| `scattered_8x1` | 512 | cusparse | 0.446 | 28.10 | **21.59** | 0.00  | 37.44 | 71.78 |

## Conclusions

1. **H6's mechanism generalizes across diag densities.** `diag_b0.1` reproduces `diag_b0.3`:
   `block` dominated by `mio_throttle`(11–22%)+`barrier`(14–25%), cuSPARSE clean and
   `long_scoreboard`-bound (78–86%, the healthy memory-bound signature).

2. **It is not universal — it depends on block occupancy shape.** On `scattered_1x8` (one populated
   row per PBR block), `block`'s `mio_throttle` collapses to ~0% — no atomic contention at all.
   Instead `block` is `long_scoreboard`(47–50%)+`barrier`(25–37%)-dominated, a different bottleneck
   entirely. `scattered_8x1` (8 populated rows per block) and `scattered_2x2`/f512 reproduce the
   diag-like `mio_throttle`-heavy profile; `scattered_2x2`/f128 doesn't. Working hypothesis
   (correlational, not ablation-verified): `mio_throttle` tracks how many distinct output rows a
   block's `atomicAdd` flush touches per launch — many rows (diag, `scattered_8x1`) → high
   contention; one row (`scattered_1x8`) → none.

3. **mio_throttle is not `block`-specific.** On `scattered_8x1`, cuSPARSE's own `mio_throttle` is
   **21.55–21.59%** — same order of magnitude as `block`'s 17.97–22.33% there, and nowhere near
   cuSPARSE's near-zero value everywhere else. So for this matrix shape, atomic/shared-mem-pipe
   contention is a property of the *workload* (narrow-column, many-row scattered writes), not a
   `block`-only design flaw.

**Bottom line:** the `mio_throttle`/`barrier` fix target from H6 is real but matrix-shape-dependent —
worth pursuing for dense/many-rows-per-block matrices (`diag_*`, `scattered_8x1`), close to irrelevant
for single-row-per-block matrices (`scattered_1x8`), where `long_scoreboard`/`barrier` reduction would
be the real lever instead.

## Appendix: N=1024 raw data (directional only)

| matrix | feat | kernel | time (µs) | SM% | `mio_throttle` | `barrier` | `long_scoreboard` | warps active |
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

`block`'s `mio_throttle` is exactly 0.00 for every scattered matrix at N=1024 — even `scattered_8x1`,
which shows the diag-like `mio_throttle`-heavy profile at N=262144 — consistent with too few
concurrent CTAs at this size for atomic contention to build up.
