# Warp-local smem reuse: does replacing `barrier` with `__syncwarp()` actually help?

Branch `feat/pbr-warp-sync`. Follows the same before/after stall-reason methodology as
`STALL_REASONS.md`, applied to the ONE change this branch makes: `pbr_spmm_zero_idle_kernel` and
`pbr_spmm_vec_kernel` (`kernel=block`) used one block-wide `smem_X`/`smem_Y` buffer shared by all 4
warps in the CTA, requiring a full `__syncthreads()` per step (load, compute, flush). This redesigns
reuse to be warp-local instead.

## What was actually implemented

The full idea from the task brief, not a scaled-back version: each feature tile (`current_p` wide) is
split into 4 disjoint sub-slices, one per warp, splitting **on the feature axis** rather than rows —
each warp gets its own private `smem_X`/`smem_Y` sub-buffer sized `current_p/4` and does its own
load → compute → atomicAdd-flush using only `__syncwarp()`, never `__syncthreads()`. Splitting on
features (not rows) was a deliberate choice over the row-partition sketched as the "starting design"
in the task brief: since every warp still needs the *same* `n_c` columns (just for a narrower slice
of features), there is no column ever needed by more than one warp, so this sidesteps the row-split's
"reuse spans warps" problem entirely — **zero redundant reloading**, not even the redundant-load
trade-off the brief anticipated as the likely honest cost. When `current_p` is narrower than 4 warps'
worth (rare — only shows up for small `features`/`smem_size` combinations, not in the profiled grid),
only warp 0 handles the whole unsplit tile, matching the original code exactly on 32 threads instead
of 128. Total shared-memory footprint is unchanged (`(n_r+n_c)*current_p`, same as before), so no
static-smem-budget risk. Both kernels changed identically; `rowgroup`/`colgroup`/`chunk`/`tile` were
not touched.

Correctness: full suite passes, **3704/3704** (`pytest tests/torch_fast_spdmm/test_pbr.py
tests/torch_fast_spdmm/test_csr.py -q`).

## Method

Job: `jobs/profile_stall_reasons/profile_warpsync_20260714.sh` — identical matrix/size/feature grid
and metric list to `jobs/profile_stall_reasons/profile_20260714.sh` (`STALL_REASONS.md`'s source job),
profiling only `kernel=block` (cuSPARSE is untouched by this branch; its numbers are unchanged from
`STALL_REASONS.md`, not re-profiled). Raw CSVs: `ncu_out/stall_reasons_warpsync/` (untracked, local
only). All 128/512-feature configs on this branch dispatch to `pbr_spmm_vec_kernel` (`features % 4 ==
0` for float), so every row below is the vectorized kernel — see `STALL_REASONS.md` for the metric
glossary.

**N=1024 caveat carried over unchanged:** too short/small for reliable stall sampling (occupancy
5–10%) — appendix only, directional.

## Data (N=262144) — before (`STALL_REASONS.md`) vs after (this branch)

| matrix | feat | time before (ms) | time after (ms) | Δtime | `mio_throttle` before→after | `barrier` before→after | `long_scoreboard` before→after | SM% before→after |
|---|---|---|---|---|---|---|---|---|
| `diag_b0.1` | 128 | 2.013 | 2.039 | +1.3% | 10.77→12.86 | 19.87→**0.00** | 24.90→32.43 | 54.83→66.22 |
| `diag_b0.1` | 512 | 8.021 | 7.803 | -2.7% | 17.66→25.26 | 25.29→**0.00** | 19.26→29.77 | 33.06→42.36 |
| `diag_b0.3` | 128 | 3.336 | 3.343 | +0.2% | 17.84→20.42 | 14.10→**0.00** | 20.48→30.20 | 44.25→47.89 |
| `diag_b0.3` | 512 | 12.986 | 12.338 | -5.0% | 21.32→28.84 | 20.11→**0.00** | 17.57→26.54 | 30.28→34.07 |
| `scattered_1x8` | 128 | 0.129 | 0.128 | -0.8% | 0.02→0.16 | 25.39→**0.00** | 46.99→49.42 | 49.66→72.65 |
| `scattered_1x8` | 512 | 0.496 | 0.496 | +0.0% | 0.29→1.24 | 36.58→**0.00** | 49.91→70.77 | 26.56→45.92 |
| `scattered_2x2` | 128 | 0.091 | 0.102 | **+12.1%** | 0.41→0.14 | 29.47→**0.00** | 27.42→24.53 | 67.49→85.25 |
| `scattered_2x2` | 512 | 0.354 | 0.348 | -1.7% | 22.56→25.53 | 9.90→**0.00** | 22.58→26.35 | 30.32→32.15 |
| `scattered_8x1` | 128 | 0.288 | 0.285 | -1.0% | 17.97→22.83 | 21.12→**0.00** | 16.06→25.46 | 38.18→42.04 |
| `scattered_8x1` | 512 | 1.090 | 1.178 | **+8.1%** | 22.33→28.84 | 25.34→**0.00** | 11.65→23.93 | 26.30→26.56 |

Average Δtime across the 10 configs: **+1.1%** (noise-level; not a systematic win or loss).

## Conclusions

1. **The mechanism worked exactly as designed: `barrier` goes to zero, everywhere, with no
   exceptions.** All 10 configs drop from double-digit `barrier` stall (9.9–36.6%) to a flat **0.00%**.
   `__syncwarp()` genuinely does not register on `smsp__warp_issue_stalled_barrier` the way
   `__syncthreads()` does — this part of the hypothesis is unambiguously confirmed.

2. **But the freed-up cycles mostly got reabsorbed by `mio_throttle` and `long_scoreboard`, not
   converted to wall-clock speedup.** In 8 of 10 configs, `mio_throttle` and/or `long_scoreboard` grew
   by roughly the same order of magnitude that `barrier` shrank (e.g. `diag_b0.3`/f512: `barrier`
   -20.11pp, but `mio_throttle` +7.52pp and `long_scoreboard` +8.97pp; `scattered_1x8`/f512: `barrier`
   -36.58pp, `long_scoreboard` +20.86pp). SM throughput (`SM%`) rose substantially in most cases too
   (e.g. `scattered_1x8`/f128: 49.66→72.65) — the GPU is issuing more instructions per unit time
   post-change, consistent with warp-local sub-buffers doing more, smaller-granularity shared-memory
   transactions (32-lane LDS/STS/atomic ops instead of 128-lane ones) rather than doing less total
   work.

3. **Net wall-clock effect: a wash, not a win.** Average Δtime over the 10 real (N=262144) configs is
   +1.1%, with a spread from -5.0% (`diag_b0.3`/f512, mild win) to +12.1% (`scattered_2x2`/f128, mild
   regression). No configuration shows a decisive improvement; two configurations
   (`scattered_2x2`/f128 and `scattered_8x1`/f512) are measurably worse. `scattered_1x8` (the one
   shape where `mio_throttle` was already ~0% before the change — H6/`STALL_REASONS.md`'s "single row
   per block" case) is the closest to a clean win: `barrier` drops to 0 and both feature counts hold
   time flat or slightly improve, even though `long_scoreboard` rises there too — plausibly because
   that shape was already latency-bound with slack to absorb it, not throughput-bound.

**Bottom line:** this is the same pattern this project has now seen twice (H4's chunk-size regression,
the reverted Step B/C smem_Y merge): isolating one named stall reason and directly suppressing it
(`barrier` → 0 via `__syncwarp()`) does not, on its own, produce a wall-clock win, because the warp
scheduler was already using those "stalled" barrier cycles productively (interleaving other warps'
work), and removing the barrier just exposes a different bottleneck (`mio_throttle`/
`long_scoreboard`) at roughly the same total cost. The redesign is correct (3704/3704 tests pass) and
achieves its literal stated goal (0% barrier stall), but does not deliver a net performance
improvement on this profiling grid.

## Appendix: N=1024 raw data (directional only, before → after)

| matrix | feat | time before (µs) | time after (µs) | `mio_throttle` before→after | `barrier` before→after | `long_scoreboard` before→after |
|---|---|---|---|---|---|---|
| `diag_b0.1` | 128 | 13.15 | 14.0 | 4.19→4.96 | 15.53→**0.00** | 18.25→25.25 |
| `diag_b0.1` | 512 | 40.29 | 42.0 | 14.00→19.39 | 24.09→**0.00** | 19.44→29.85 |
| `diag_b0.3` | 128 | 18.62 | 19.0 | 9.63→12.20 | 12.53→**0.00** | 18.66→27.79 |
| `diag_b0.3` | 512 | 60.48 | 58.0 | 17.38→22.89 | 19.32→**0.00** | 18.78→28.82 |
| `scattered_1x8` | 128 | 5.09 | 5.0 | 0.00→0.00 | 19.85→**0.00** | 25.04→31.88 |
| `scattered_1x8` | 512 | 8.64 | 9.0 | 0.00→0.00 | 28.23→**0.00** | 30.07→33.90 |
| `scattered_2x2` | 128 | 4.58 | 5.0 | 0.00→0.00 | 21.54→**0.00** | 18.60→28.55 |
| `scattered_2x2` | 512 | 5.89 | 6.0 | 0.00→0.00 | 4.03→**0.00** | 27.53→28.53 |
| `scattered_8x1` | 128 | 5.66 | 6.0 | 0.00→0.00 | 13.84→**0.00** | 16.84→26.55 |
| `scattered_8x1` | 512 | 12.10 | 11.0 | 0.00→0.00 | 32.30→**0.00** | 14.76→42.51 |

Same qualitative story at this (unreliable-sampling) size: `barrier`→0 everywhere, `long_scoreboard`
consistently absorbs most of it, times move by only a few µs either way — consistent with, not
contradicting, the N=262144 verdict above.
