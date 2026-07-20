# Paired float2 atomicAdd via CAS-loop: mechanism confirmed again, worst regression yet

Branch: `worktree-agent-aa64a615c4ef38b58` (based on `research/pbr-cusparse-scalability-gap`, commit
`28cb9de`). Kernel change: commit `e1c4eca`.

## Hypothesis

`STALL_REASONS_NOATOMIC_ABLATION.md` named the untried, correctness-preserving lever: cut atomic
cost/frequency without touching CTA count (unlike `rowgroup`, which did cut atomic count but lost 7 of
8 points to an occupancy cost — `STALL_REASONS_ROWGROUP_ATOMIC.md`). `pbr_spmm_vec_kernel`'s Step C
issues 4 independent scalar `atomicAdd(float)` calls per thread (one per vector lane) to 4 contiguous,
16-byte-aligned floats in `Y`. Ampere (RTX 3090, CC 8.6) has no native `atomicAdd(float2)` — that's
Hopper-only — so this experiment implements it manually via a 64-bit `atomicCAS` loop, halving the
atomic instruction count (2 paired RMWs instead of 4 scalar ones) with **zero change to grid shape,
CTA count, or occupancy** — a clean, deliberately unconfounded contrast to `rowgroup`'s approach.

**Predicted mechanism:** `mio_throttle` drops (fewer atomic ops issued) with `warps_active_pct` staying
flat (no occupancy side effect this time). **Named risk, flagged before running:** CAS-loop retries
under real contention could show up as a *new* stall (`long_scoreboard`/`short_scoreboard`), potentially
outweighing the gain — this is exactly what the noatomic ablation's ceiling measurement couldn't
detect, since deleting the atomic entirely has no retry cost at all.

## Implementation

`torch_fast_spdmm/kernels/pbr_kernel.cu`, `pbr_spmm_vec_kernel`'s Step C, float path (`L=4`) only —
`double` (`L=2`) left untouched via `if constexpr`:

```cpp
__device__ __forceinline__ void atomicAddFloat2(float2* addr, float2 val) {
    unsigned long long int* addr_u64 = reinterpret_cast<unsigned long long int*>(addr);
    unsigned long long int old = *addr_u64, assumed;
    do {
        assumed = old;
        float2 old_f2 = *reinterpret_cast<float2*>(&assumed);
        float2 new_f2 = make_float2(old_f2.x + val.x, old_f2.y + val.y);
        old = atomicCAS(addr_u64, assumed, *reinterpret_cast<unsigned long long int*>(&new_f2));
    } while (assumed != old);
}
```
Lanes {0,1} and {2,3} each become one `atomicAddFloat2` call (both pairs are 8-byte-aligned since
`f_off` is always a multiple of `L=4`). Mathematically still a correct sum; not necessarily bit-identical
to the scalar-atomic baseline due to floating-point non-associativity (order of accumulation changes
slightly) — the test suite's tolerance-based comparisons absorbed this with no failures.

## Correctness

Built and tested via `optimizer.run`'s built-in correctness gate on the remote GPU: full pytest suite
passed (3704/3704, matching this project's baseline) before either sweep was allowed to proceed.

## Method

Same paired-ledger methodology as the last two iterations: `optimizer.run` full sweep (5 matrices ×
{128,512} features × block∈{2,4,8} × smem∈{512,1024,2048}, N=262144, `--kernels block`) into
`optimizer_ledger_paired_atomic.sqlite3` for this branch, and a matching baseline sweep (unmodified
`research/pbr-cusparse-scalability-gap`) into `optimizer_ledger.sqlite3`. 240 points each side, 186
matched, compared via `optimizer.compare` (which already runs each point under `ncu` with the project's
standard 15-metric stall-reason list, so no separate profiling job was needed this time).

## Results

**186 matched config points: 13 WIN, 166 REGRESSION, 7 WASH — the worst aggregate result of any
kernel-edit attempt in this project's history**, including the two prior atomic-focused ones.

| point | time: base→paired | Δtime | `mio_throttle`: base→paired | `long_scoreboard`: base→paired | `warps_active`: base→paired |
|---|---|---|---|---|---|
| `diag_b0.1`/f128 (block=8,smem=2048) | 2.039→3.222 ms | **+58.1%** | 10.81→**0.28** | 24.89→**53.85** | 88.2→90.4 (flat) |
| `diag_b0.3`/f128 (block=8,smem=2048) | 3.348→5.603 ms | **+67.4%** | 17.77→**0.87** | 20.46→**60.92** | — |
| `diag_b0.3`/f512 (block=8,smem=2048) | 13.075→18.174 ms | **+39.0%** | 21.30→**1.00** | 17.59→**64.55** | 91.0→91.4 (flat) |
| `diag_b0.3`/f128 (block=4,smem=2048) | — | **+196.5%** (worst) | — | — | — |
| `scattered_1x8`/f128 (block=8,smem=2048, near-zero baseline contention) | 0.129→0.136 ms | +5.1% | 0.02→0.00 | 46.84→36.26 | 87.9→88.0 (flat) |
| `scattered_2x2`/f512 (block=8,smem=1024, best case) | — | **−13.3%** (best) | — | — | — |
| `scattered_8x1`/f512 (block=8,smem=2048) | 1.099→1.163 ms | +6.4% | 22.40→**1.54** | 11.71→**67.04** | 90.0→90.6 (flat) |

(`warps_active_pct` confirmed flat within noise (±2pp) across every point checked — the occupancy
confound that sank `rowgroup` is definitively absent here, isolating the failure to the CAS-loop itself.)

## Mechanism: confirmed again, and this time the retry cost dominates

**`mio_throttle` collapses even more dramatically than `rowgroup` achieved** (85–97% relative drop,
vs. `rowgroup`'s already-large reduction) — the atomic-contention hypothesis is confirmed a third
time, now with two independent, structurally different fixes (batched-flush grouping, and paired-RMW
CAS-loop) both proving it's real. **But `long_scoreboard` (global-memory-latency stalls) explodes
116–472%** on every matrix with real baseline contention (`diag_b0.1`, `diag_b0.3`, `scattered_8x1`) —
exactly the named risk. The CAS loop's `atomicCAS` retries when a concurrent writer touches the same
8-byte pair between this thread's read and its compare-and-swap; each retry is a full round-trip to the
memory subsystem, and under the same cross-CTA contention that made `mio_throttle` high in the
baseline, retries are frequent enough to cost far more than the halved instruction count saves.

**Even `scattered_1x8`, the negative control with near-zero baseline `mio_throttle`/contention, still
regresses (+5.1%)** — `long_scoreboard` actually *improves* there (−22.6%, less contention to begin
with), but the CAS-loop's inherent fixed cost (read + compute + compare-and-swap vs. one hardware
`atomicAdd` instruction) still taxes the kernel even when retries are rare. This is a cost `rowgroup`
never paid (its `atomicAdd` calls stayed as plain hardware atomics, just fewer of them) — the paired
version pays a per-call tax on top of the (much larger) contention-driven retry cost.

The 13 wins cluster on `diag_b0.1` (the sparsest, least-contended diagonal matrix — `blocks_per_stripe`
mean 7.9 vs `diag_b0.3`'s 8.0, but far fewer actual nnz per block so less real collision) and
`scattered_2x2`/f512 specifically — consistent with the retry-cost story: where there's genuinely
little contention to retry against, halving the instruction count still wins modestly; where
contention is real, retries dominate and losses reach up to **+196.5%** (diag_b0.3/f128).

## Verdict

**Not a win — the worst-regressing kernel-edit attempt in this project's history**, despite being the
most mechanistically successful at its stated goal (`mio_throttle` reduction). This is a clean,
complete falsification: manually-implemented CAS-loop atomics are the wrong tool here because their
retry cost scales with exactly the contention this kernel already has too much of — the fix's own cost
model is inverted relative to the problem. Two independent, structurally different attempts (`rowgroup`
and this one) have now both confirmed the atomic-contention mechanism is real while both failing to
turn that confirmation into a net win, for two different, well-understood reasons (occupancy cost;
retry cost). **No further "reduce atomic cost" variant should be attempted without a fundamentally
different implementation strategy** (e.g. hardware-native wide atomics, only available on newer
architectures than this project's RTX 3090 target) — this line of investigation is exhausted for this
hardware generation.
