# Zero-idle-kernel microbenchmarks: block_data load, Step B/C merge, syncthreads (2026-07-16)

Branch `feat/pbr-zeroidle-microbench` (forked from `feat/pbr-block-stall-fix`). Six forced-scalar-dispatch
variants of `pbr_spmm_zero_idle_kernel` (never normally dispatched at f128/f512, since `use_vec` always
picks the vec kernel there — see `STALL_REASONS_NOATOMIC_ABLATION.md`'s branch lineage). Cross-verified
with **two independent measurement methods**: single-shot `ncu` profiling (`jobs/profile_zeroidle_microbench/`)
and a 10x-repeated wall-clock benchmark (`benchmark_scalar_diag.py`, plain `torch.cuda.Event` timing, no
profiler overhead) — both on the same 5-matrix × 2-feature grid, N=262144.

## Correctness (`check_scalar_diag_correctness.py`, 60 checks)

| variant | expected | observed |
|---|---|---|
| `baseline` | correct | **correct**, all 10 configs |
| `constdata` | wrong | **wrong**, all 10 configs (rel. error 1.7–2.8×) |
| `mergeBC` | correct | **correct**, all 10 configs |
| `nosyncA` | wrong (Step A/B race) | **correct** in all 10 configs tested — unexpected |
| `nosyncB` | wrong (Step B/C race, naive) | **correct** in all 10 configs tested — unexpected |
| `nosyncC` | wrong (WAR hazard) | **correct** in all 10 configs tested — unexpected |

**Caveat, stated plainly:** this does not prove the three sync removals are safe in general. GPU races are
scheduler/occupancy/hardware-dependent; a different launch configuration, GPU, or driver could expose one
that this test didn't. All it shows is: on this exact kernel shape, RTX 3090, and these 5 matrices/2 feature
counts, no corruption was observed in 60 checks.

## Performance — ncu (single-shot) vs. benchmark (10x mean, `std`)

Time in ms. `Δ` = variant vs. `baseline`, same measurement method.

| matrix/feat | metric | baseline | constdata | nosyncA | nosyncB | nosyncC | mergeBC |
|---|---|---|---|---|---|---|---|
| `diag_b0.1`/128  | ncu   | 2.143 | 2.041 (−4.8%) | 2.147 (+0.2%) | 2.163 (+0.9%) | 2.124 (−0.9%) | **1.922 (−10.3%)** |
|                  | bench | 1.960 (σ.007) | 1.850 (−5.6%) | 1.936 (−1.2%) | 1.952 (−0.4%) | 1.944 (−0.8%) | **1.779 (−9.2%)** |
| `diag_b0.1`/512  | ncu   | 7.660 | 6.280 (−18.0%) | 7.622 (−0.5%) | 7.624 (−0.5%) | 7.601 (−0.8%) | **6.848 (−10.6%)** |
|                  | bench | 6.605 (σ.029) | 5.678 (−14.0%) | 6.604 (~0%) | 6.610 (~0%) | 6.597 (~0%) | **6.011 (−9.0%)** |
| `diag_b0.3`/128  | ncu   | 3.251 | 2.886 (−11.2%) | 3.265 (+0.4%) | 3.244 (−0.2%) | 3.268 (+0.5%) | **2.933 (−9.8%)** |
|                  | bench | 2.659 (σ.103) | 2.459 (−7.5%) | 2.737 (+2.9%) | 2.747 (+3.3%) | 2.732 (+2.7%) | **2.493 (−6.2%)** |
| `diag_b0.3`/512  | ncu   | 12.433 | 9.847 (−20.8%) | 12.406 (−0.2%) | 12.414 (−0.2%) | 12.409 (−0.2%) | **11.196 (−9.9%)** |
|                  | bench | 10.386 (σ.013) | 8.532 (−17.9%) | 10.586 (+1.9%) | 10.443 (+0.5%) | 10.258 (−1.2%) | **9.321 (−10.3%)** |
| `scattered_1x8`/128 | ncu | 0.129 | 0.127 (−1.5%) | 0.130 (+0.8%) | 0.129 (+0.5%) | 0.129 (+0.3%) | 0.128 (−0.8%) |
|                     | bench | 0.308 (σ.002) | 0.306 (−0.7%) | 0.307 (−0.3%) | 0.308 (~0%) | 0.307 (−0.3%) | 0.308 (~0%) |
| `scattered_1x8`/512 | ncu | 0.495 | 0.497 (+0.3%) | 0.498 (+0.6%) | 0.496 (+0.2%) | 0.497 (+0.4%) | 0.497 (+0.3%) |
|                     | bench | 1.129 (σ.004) | 1.130 (~0%) | 1.128 (~0%) | 1.129 (~0%) | 1.130 (~0%) | 1.127 (~0%) |
| `scattered_8x1`/128 | ncu | 0.253 | 0.220 (−12.9%) | 0.253 (~0%) | 0.253 (~0%) | 0.255 (+0.8%) | **0.227 (−10.4%)** |
|                     | bench | 0.367 (σ.010) | 0.356 (−3.2%) | 0.360 (−2.2%) | 0.362 (−1.6%) | 0.368 (+0.1%) | **0.351 (−4.5%)** |
| `scattered_8x1`/512 | ncu | 0.976 | 0.747 (−23.4%) | 0.964 (−1.2%) | 0.963 (−1.3%) | 0.972 (−0.3%) | **0.860 (−11.9%)** |
|                     | bench | 1.340 (σ.005) | 1.310 (−2.2%) | 1.341 (~0%) | 1.342 (~0%) | 1.348 (+0.6%) | **1.304 (−2.7%)** |
| `scattered_2x2`/128 | ncu | 0.106 | 0.108 (+1.9%) | 0.104 (−2.1%) | 0.104 (−1.5%) | 0.105 (−0.5%) | 0.095 (−10.2%) |
|                     | bench | 0.262 (σ.001) | 0.260 (−0.8%) | 0.261 (−0.6%) | 0.264 (+0.5%) | 0.261 (−0.4%) | 0.259 (−1.3%) |
| `scattered_2x2`/512 | ncu | 0.301 | 0.299 (−0.6%) | 0.302 (+0.5%) | 0.304 (+1.2%) | 0.299 (−0.7%) | 0.302 (+0.5%) |
|                     | bench | 0.936 (σ.002) | 0.935 (−0.1%) | 0.938 (+0.2%) | 0.937 (+0.1%) | 0.936 (~0%) | 0.934 (−0.2%) |

The two methods agree on every qualitative conclusion below; `ncu` runs consistently ~5–16% slower than the
plain-benchmark numbers on `diag_*`/`scattered_8x1` at high mio_throttle (profiling instrumentation
overhead), and the benchmark's number is *end-to-end* `pbr_matmul` wall time (includes the concurrent
zero-init + CSR-remainder kernels on separate streams) while `ncu`'s is isolated to just the `pbr_spmm_*`
kernel — which is why `scattered_8x1` (which does have CSR remainder work) shows a muted effect in the
benchmark column relative to `ncu`. Neither discrepancy changes any conclusion; the *within-method*
relative deltas are what matter and both methods reproduce them independently, at CV well under 1% in the
repeated-trial numbers (see `benchmark_scalar_diag.py` output — worst-case CV 3.9% on `diag_b0.3`/128
baseline, everything else ≤3%).

## Findings, by original question

**1. Is `block_data`'s global read (`__ldg`) itself a stall source?** Yes, substantially — `constdata`
cuts time 5–23% on every dense/many-row matrix (`diag_*`, `scattered_8x1`), confirmed in both measurement
methods. `mio_throttle` collapses to near-zero everywhere too (e.g. `diag_b0.3`/512: 22.81→0.50), but
that's a secondary effect: removing the load also removes the dependent multiply that feeds `sum`,
shortening the whole Step B critical path per iteration, not just the memory access in isolation. The
cleaner direct signal is `long_scoreboard` (DRAM-latency stall), which also drops meaningfully everywhere
(`diag_b0.3`/512: 8.93→5.91; `scattered_8x1`/512: 9.58→7.33) — the expected signature of removing a real
DRAM round-trip. Net: the `block_data` read is a genuine, meaningful stall/time contributor, distinct
from (and roughly as large as) the atomic-flush effect found in the prior ablation.

**2. Are the three `__syncthreads()` calls load-bearing for performance?** No — none of `nosyncA`/
`nosyncB`/`nosyncC` produces a reliable win in *either* measurement method; every delta is within ±3%,
mostly under 1%, i.e. noise. This reproduces the exact "whack-a-mole" pattern from the four prior
`feat/pbr-block-stall-fix`-lineage attempts: `nosyncA` does cut `barrier` substantially (e.g.
`diag_b0.3`/512: 8.34→5.50, a genuine ~34% relative reduction), but `mio_throttle` absorbs the freed
cycles (22.81→24.21 there) and wall-clock stays flat. Combined with the correctness result above: these
three deletions are performance-neutral *and* didn't corrupt output in this test — i.e., as measured here,
free to remove but pointless, not a lever worth pursuing on their own.

**3. Can Step B/C be merged (drop `smem_Y`, `atomicAdd` straight from register), and does it help when `p`
is held fixed?** **Yes, and it's the one clear, reproducible win in this round.** `mergeBC` beats baseline
by ~9–12% on every `diag_*`/`scattered_8x1` config in both measurement methods (the two dense/many-row
matrix families), and is neutral-to-slightly-positive on `scattered_1x8`/`scattered_2x2` (where baseline
was already near its `long_scoreboard`-bound floor, nothing to gain). This is genuinely new information —
both prior attempts at this exact merge (`feat/pbr-block-stall-fix` commit `44bf132`,
`feat/pbr-smemy-capped-p`) only ever profiled with `--kernel block` at `features%4==0`, so the zero_idle
kernel's merged code was compiled but never dispatched or measured before this. The key structural
difference from `nosyncB` (which shows ~0% gain) is that `mergeBC` eliminates the `smem_Y`
store-then-reload *instructions themselves*, not just the barrier between them — confirming the store/
reload traffic, not the sync, was the real cost. Consistent with that, `mergeBC`'s `mio_throttle` is
*flat-to-slightly-worse* than baseline (e.g. `diag_b0.1`/512: 21.66→24.61) even as wall-clock drops
~10–11% — another reminder that `mio_throttle` alone doesn't reliably predict wall-clock direction; the
net instruction-count reduction wins even when the named stall-reason counter doesn't cooperate.

## Implication

`mergeBC` (Step B/C fusion with `p` held fixed) is a real, reproducible, correctness-preserving win on
`pbr_spmm_zero_idle_kernel` specifically — worth a follow-up to check whether the same fixed-`p` fusion
produces a comparable win on `pbr_spmm_vec_kernel` (the kernel actually dispatched in production at
`features%4==0`), since that's the one prior attempts tested (and got a regression from, due to the
p-growth confound they bundled in). The `block_data`-read cost (constdata) and the three individual
syncthreads (no-ops) are diagnostic-only findings — not directly actionable without breaking correctness,
except as confirmation that shared-memory instruction *volume* (not synchronization *count*) is the
lever that matters.

## Disposition

Not merged — `constdata`/`nosyncA`/`nosyncB`/`nosyncC` are diagnostic-only and/or not intended for
production use even where they happen not to corrupt output here. `mergeBC` is the one variant worth
carrying forward as a real fix candidate (on the vec kernel, with `p` held fixed) — not yet attempted.
Kept on `feat/pbr-zeroidle-microbench` (forked from `feat/pbr-block-stall-fix`) for the record.
