# Warp stall-reason breakdown across matrix shapes, sizes, and feature counts

Extends H6's stall-reason check (`WHY_BLOCK_KERNEL_FAILS_TO_SCALE.md` H6, previously verified only on
`diag_b0.3` @ N=262144/f512) across 5 matrices × 2 sizes × 2 feature counts, `block` vs cuSPARSE.

## What the metrics mean

Each is "% of cycles a warp was stalled, unable to issue, for this reason" (`smsp__warp_issue_stalled_*`),
or a pipe/occupancy percentage:

- **`mio_throttle`** — stalled because the shared-memory execution pipe (MIO) is saturated.
  **Correction (2026-07-14, verified via `ncu --query-metrics`):** global `atomicAdd` is on a
  *separate* pipe (`_pipe_lsu`), not MIO — only `block`'s `smem_X`/`smem_Y` loads and stores drive
  this metric, not the Y-accumulation `atomicAdd` itself. High value = shared-memory instruction
  pressure, not atomic contention.
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

## Attempted fix (2026-07-14, branch `feat/pbr-block-stall-fix`): negative result

### Hypothesis

`block`'s Step B (compute into `smem_Y`) and Step C (`atomicAdd` from `smem_Y`) iterate the identical
per-thread work sequence — the same thread writes and later reads its own `smem_Y` slot, with no
cross-thread dependency. Reading the kernel source suggested the `__syncthreads()` between them and
the entire `smem_Y` store/reload were unnecessary: merge the two loops so each thread `atomicAdd`s
directly from its own register, drop `smem_Y`, and let `smem_X` use the whole shared-memory buffer
(growing the feature-tile width `p`, since only one buffer needs to fit now instead of two).

Verified first via `ncu --query-metrics` that global `atomicAdd` is on the LSU pipe, not MIO — so this
targets real MIO-classified traffic (the now-corrected glossary above), not a misattributed atomic
cost.

### Result: regression, not a win

Implemented in `pbr_spmm_vec_kernel`/`pbr_spmm_zero_idle_kernel`. All 3704 correctness tests pass
unchanged (pure refactor, same accumulated sum). Re-profiled the identical N=262144 stall-reason grid
before vs. after:

| matrix/feat | time before→after | SM% before→after | `mio_throttle` before→after | `barrier` before→after |
|---|---|---|---|---|
| `diag_b0.1`/128 | 2.013→2.005ms (−0.4%) | 54.8→52.9 | 10.8→**13.6** | 19.9→18.8 |
| `diag_b0.1`/512 | 8.021→8.191ms (+2.1%) | 33.1→26.9 | 17.7→**26.5** | 25.3→20.2 |
| `diag_b0.3`/128 | 3.336→3.351ms (+0.5%) | 44.3→41.8 | 17.8→**22.3** | 14.1→15.7 |
| `diag_b0.3`/512 | 12.99→13.73ms (+5.7%) | 30.3→24.1 | 21.3→**26.7** | 20.1→24.0 |
| `scattered_1x8`/128 | 0.129→0.135ms (+4.5%) | 49.7→47.7 | 0.02→0.05 | 25.4→22.4 |
| `scattered_1x8`/512 | 0.496→0.496ms (~0%) | 26.6→22.1 | 0.29→**11.7** | 36.6→23.5 |
| `scattered_2x2`/128 | 0.091→0.091ms (~0%) | 67.5→65.6 | 0.41→0.69 | 29.5→26.2 |
| `scattered_2x2`/512 | 0.354→0.354ms (~0%) | 30.3→28.4 | 22.6→27.6 | 9.9→10.3 |
| `scattered_8x1`/128 | 0.288→0.288ms (~0%) | 38.2→35.4 | 18.0→24.8 | 21.1→19.7 |
| `scattered_8x1`/512 | 1.090→1.125ms (+3.2%) | 26.3→22.1 | 22.3→**28.6** | 25.3→23.7 |

`mio_throttle` **increased** in 9 of 10 cases (sometimes drastically — `scattered_1x8`/512 went from
0.29% to 11.7%), `sm__throughput` **decreased in all 10**, and wall-clock time is flat-to-worse
everywhere (never meaningfully better). `barrier` did drop in most cases as predicted, but the MIO
regression outweighed it.

### Why: growing `p` backfired

Removing `smem_Y` freed shared memory, so `p` (features handled per iteration before the next
`__syncthreads()`) roughly doubled for blocks where `n_r ≈ n_c`. Total shared-memory instruction count
across a full kernel invocation should still have gone down (fewer iterations, and the `smem_Y`
round-trip is gone entirely) — but each surviving iteration now issues about twice as many
back-to-back `smem_X` load/read instructions with no synchronization point in between to let the MIO
issue queue drain. The barrier count dropping did remove real synchronization overhead, but batching
more shared-memory traffic into fewer, larger uninterrupted bursts made the MIO queue fill up more
often per unit time than before — a genuine case of "less total work, worse instantaneous pipe
pressure." This is exactly the kind of effect flagged as an open, unresolved question during plan
review (whether removing synchronization changes contention for the better or worse) — it went the
other way.

### Disposition

Reverted (`torch_fast_spdmm/kernels/pbr_kernel.cu` restored to the pre-fix version) on
`feat/pbr-block-stall-fix`; the branch is kept for the record but not merged. No production kernel
change resulted from this investigation — `mio_throttle`/`barrier` reduction remains an open problem.
A follow-up that was **not** attempted here (flagged for future work if revisited): cap `p` at its old
value even with `smem_Y` removed, to isolate whether the store/reload removal helps in isolation from
the iteration-count/burst-size change it was bundled with in this attempt.

## Three more attempts, in parallel (2026-07-14): all negative/wash

Asked "why doesn't cuSPARSE have this problem" first: `ncu`'s `sm__sass_inst_executed_op_shared.sum`
shows cuSPARSE executes **zero** shared-memory instructions on every matrix tested — it reads `X`
straight from global memory (L1/L2 cache), never staging into shared memory the way `block` does.
That reframes the problem: `mio_throttle` isn't a bug in `block`, it's the structural cost of `block`'s
core design choice (stage `X` in shared memory so multiple rows in a PBR block can reuse it) vs.
cuSPARSE's choice (no staging, trust the cache). Three follow-up attempts were run in parallel, each
on its own branch, each independently built/tested/profiled on the remote GPU:

| branch | idea | verdict |
|---|---|---|
| `feat/pbr-smemy-capped-p` | Redo the `smem_Y` merge from the attempt above, but keep `p` capped at its old value (isolate the store/reload removal from the burst-size growth that sank the first attempt) | **Negative.** `mio_throttle`/`barrier` got *worse* on `diag_*` (the matrices the fix targeted), same direction as the reverted attempt, smaller magnitude. Scattered-matrix "wins" don't move wall-clock time (all deltas within ±2%, noise floor). |
| `feat/pbr-adaptive-smem-bypass` | Skip `smem_X` staging entirely for low-reuse PBR blocks (`n_r<=1`, e.g. `scattered_1x8`), read `X` via `__ldg` straight from global — literally cuSPARSE's approach, applied selectively | **Wash.** `barrier` collapses to exactly 0.00% in *every* config tested — including `diag_b0.3`, which never triggers this bypass at all (0% of its blocks have `n_r<=1`), so this looks like a compiler/SASS side effect of adding the branch, not the bypass logic itself; mechanism unconfirmed. SM% improves substantially everywhere (up to +30pts on `scattered_1x8`), but `mio_throttle`/`long_scoreboard` absorb the freed cycles — wall-clock time ranges −5.0% to +12.9%, no config wins clearly. |
| `feat/pbr-warp-sync` | Make shared-memory reuse warp-local instead of block-wide (per-warp `smem_X`/`smem_Y` sub-buffers, `__syncwarp()` instead of `__syncthreads()`) | **Wash.** `barrier` drops to exactly 0.00% in all 10 configs (this one *by design*, not a mystery — no block-wide sync remains). `mio_throttle`/`long_scoreboard` again absorb the freed cycles; net time averages +1.1%, ranging −5.0% to +12.1%. |

All three pass full correctness (3704/3704 tests) and all three show the *same qualitative pattern*:
whichever named stall reason is targeted, it goes down (often to exactly zero) — but the freed cycles
get reallocated to `mio_throttle` and/or `long_scoreboard` rather than shortening the critical path,
so wall-clock time never moves reliably in either direction. `scattered_1x8` under the adaptive-bypass
attempt is the clearest illustration: SM utilization nearly doubles (26.6%→45.8% at f512) while time
stays flat — consistent with `block` being fundamentally memory-latency-bound on this matrix already
(`long_scoreboard`=49.9% at baseline), so improving issue-slot efficiency doesn't help when DRAM
latency, not issue throughput, is the actual critical path.

**Updated bottom line:** across four independent kernel-modification attempts now (the original
`smem_Y` merge plus these three), suppressing `block`'s named stall reasons one at a time has not
produced a single reliable wall-clock win. This suggests `mio_throttle`/`barrier` reduction, while a
real and correctly-diagnosed phenomenon, is not on its own a productive lever for closing the gap to
cuSPARSE — the kernel appears to have enough other latent latency (`long_scoreboard`, or contention
this session's tooling doesn't isolate) that clearing one bottleneck just exposes the next one at
roughly the same wall-clock cost. None of the four branches are recommended for merge; all are kept
for the record. Full per-approach data: `STALL_REASONS_SMEMY_CAPPED.md` (branch
`feat/pbr-smemy-capped-p`), `STALL_REASONS_ADAPTIVE_BYPASS.md` (branch
`feat/pbr-adaptive-smem-bypass`), `STALL_REASONS_WARP_SYNC.md` (branch `feat/pbr-warp-sync`).
