# mergeBC allocation-ablation: is the win instructions or occupancy? (2026-07-21)

Branch `feat/pbr-mergebc-alloc-ablation` (forked from `feat/pbr-zeroidle-microbench`). Follows up on
`STALL_REASONS_ZEROIDLE_MICROBENCH.md`'s `mergeBC` finding (~9-12% win on `diag_*`/`scattered_8x1` from
fusing Step B/C — computing the per-(row,feature) partial sum and `atomicAdd`ing it straight to global
`Y`, instead of writing it to `smem_Y` and reading it back before the atomic commit).

## Motivating objection

Eran flagged that `mergeBC` changes two things relative to true baseline (`pbr_spmm_zero_idle_kernel`,
real `smem_Y` read/write) at once: (a) it removes the `smem_Y` store/reload *instructions* themselves,
and (b) since `smem_Y` is no longer touched, it might also *free shared-memory allocation*, independently
improving occupancy (more concurrent blocks/SM) — a mechanism this project already confirmed in the
opposite direction on `feat/pbr-cache-block-data` (adding a new static `__shared__` buffer there regressed
wall-clock via `launch__occupancy_limit_shared_mem` / `launch__waves_per_multiprocessor` shifting).

## The control: `mergeBC_alloc`

New `scalar_diag` variant (`variant=6` in `launch_pbr_spmm_scalar_diag`,
`torch_fast_spdmm/kernels/pbr_kernel.cu`): identical fused compute+atomicAdd to `mergeBC`, but keeps an
explicit (dead) `smem_Y` pointer declared at the same offset/extent baseline's `smem_Y` uses, and — like
`mergeBC` already did — computes `p` via the same `TOTAL_SHARED / (n_r + n_c)` divisor as baseline (not
`TOTAL_SHARED / n_c`, which would let `p` grow). Numerically identical to `mergeBC` (same computation,
same atomicAdd target).

**Note on what this control can and can't show:** `__shared__ scalar_t smem[TOTAL_SHARED]` is a
compile-time-sized static array — `TOTAL_SHARED` is a template int fixed by `smem_size` at dispatch time,
and its reservation does not depend on which runtime-computed index range within it is actually
written/read. `mergeBC` already used the same `TOTAL_SHARED` and the same `p` divisor as baseline, so by
inspection the two were already going to reserve identical static shared memory — `mergeBC_alloc` mainly
makes that explicit in source and lets ncu confirm it empirically rather than by inspection alone.

## Correctness

`check_scalar_diag_correctness.py` (extended with `mergeBC_alloc`, `EXPECTED_CORRECT` now includes it),
70 checks total (5 matrices × 2 features × 7 variants), same N=262144/block=8/min_nnz=2 grid: **`baseline`
and `mergeBC` and `mergeBC_alloc` all correct in all 10 configs each** (max abs/rel error ~1e-6/1e-7,
floating-point-accumulation-order noise only). The 30 "unexpected" flags are all on
`nosyncA`/`nosyncB`/`nosyncC` — pre-existing, already documented in `STALL_REASONS_ZEROIDLE_MICROBENCH.md`
as "correct in this test but not proven safe in general"; unrelated to this change.

`python -m pytest test_pbr.py test_csr.py -v` (run from `tests/torch_fast_spdmm/`): **3704 passed, 1
pre-existing unrelated warning, in 21.4s** — no regressions. These are the two correctness suites per
CLAUDE.md's file table (`test_pbr.py`: PBR correctness, `test_csr.py`: CSR-naive correctness); their
combined count (3608 + 96 = 3704) exactly matches the "3704 passed" figure quoted in
`STALL_REASONS_CACHE_BLOCK_DATA.md`, confirming this is the right correctness-gate scope.

Two other files under `tests/torch_fast_spdmm/` were deliberately **not** run as part of a blanket
`pytest tests/torch_fast_spdmm/ -v`:
- `test_sparse_block_mat.py` fails to *collect* — `In test_block_mat_sanity: function uses no argument
  'block_density'` — on both this branch and unmodified `feat/pbr-zeroidle-microbench`; confirmed
  pre-existing and unrelated to this change.
- `test_benchmark.py` is the GPU **benchmark** suite (pytest-benchmark timing + W&B logging, per
  CLAUDE.md's file table), not a correctness gate — `--collect-only` shows **79,200** parametrized items
  on this branch (its `SCALAR_DIAG_RUNS`/colmajor-sweep grid has grown considerably since
  `STALL_REASONS_CACHE_BLOCK_DATA.md`). Two attempts at a blanket `pytest tests/torch_fast_spdmm/ -v` run
  timed out (45 min partial progress, then a full 2:00:00 at only ~7% through `test_benchmark.py`,
  `test_pbr_spmm` specifically) before this was diagnosed — not a hang, deadlock, or a bug in
  `mergeBC_alloc`: at the observed throughput a full run would take on the order of a day, because
  `pytest-benchmark`'s calibrated multi-round timing multiplies out over 79,200 parametrizations.
  `mergeBC_alloc` is not wired into `test_benchmark.py`'s `SCALAR_DIAG_RUNS` (only `baseline`/`mergeBC`
  are, pre-existing); its correctness is covered by `check_scalar_diag_correctness.py` above instead.

## Method

Same 5-matrix × 2-feature × N=262144/block=8/min_nnz=2 grid as `STALL_REASONS_ZEROIDLE_MICROBENCH.md`, 3
variants (`baseline`, `mergeBC_alloc`, `mergeBC`), **N=5 repeated `ncu` runs per (matrix, features,
variant, pass)** for mean/sd instead of a single point value. Two separate `ncu` invocations per rep
(never mixed — see `STALL_REASONS.md`'s methodology note): an occupancy/launch-config pass and the
project's standard stall-reason `METRICS` list (`optimizer/performance.py`). Driver:
`tests/torch_fast_spdmm/benchmark/profile_pbr.py --kernel scalar_diag --scalar-variant <variant>` (already
supported the mechanism; only its `--scalar-variant` choices list needed extending). Job:
`jobs/profile_mergebc_alloc_ablation/profile_20260721_rep.sh` (300 `ncu` invocations, 0 `FAILED` entries,
completed in 23 min on an RTX 3090).

## Results — full data (mean ± sd over 5 reps)

| matrix/feat | variant | time (ms) | Δ vs baseline | long_scoreboard % | mio_throttle % | barrier % | occ_limit_smem | waves/SM |
|---|---|---|---|---|---|---|---|---|
| diag_b0.1/128 | baseline | 2.1089±0.0089 | — | 11.89±0.04 | 7.36±0.00 | 5.69±0.01 | 11 | 287.86 |
| diag_b0.1/128 | mergeBC_alloc | 1.8844±0.0008 | **-10.6%** | 17.20±0.01 | 9.35±0.01 | 5.79±0.02 | 11 | 287.85 |
| diag_b0.1/128 | mergeBC | 1.8836±0.0007 | **-10.7%** | 17.20±0.01 | 9.34±0.01 | 5.77±0.00 | 11 | 287.86 |
| diag_b0.1/512 | baseline | 7.6128±0.0240 | — | 9.62±0.01 | 21.67±0.01 | 8.29±0.01 | 11 | 287.83 |
| diag_b0.1/512 | mergeBC_alloc | 6.8010±0.0029 | **-10.7%** | 15.58±0.01 | 24.63±0.01 | 8.16±0.01 | 11 | 287.80 |
| diag_b0.1/512 | mergeBC | 6.8052±0.0141 | **-10.6%** | 15.58±0.00 | 24.63±0.01 | 8.16±0.01 | 11 | 287.83 |
| diag_b0.3/128 | baseline | 3.2201±0.0003 | — | 11.44±0.01 | 11.49±0.01 | 5.75±0.01 | 11 | 290.63 |
| diag_b0.3/128 | mergeBC_alloc | 2.9003±0.0002 | **-9.9%** | 17.62±0.01 | 14.54±0.01 | 6.25±0.02 | 11 | 290.63 |
| diag_b0.3/128 | mergeBC | 2.9000±0.0004 | **-9.9%** | 17.61±0.00 | 14.54±0.01 | 6.24±0.01 | 11 | 290.63 |
| diag_b0.3/512 | baseline | 12.3834±0.0436 | — | 8.90±0.00 | 22.82±0.01 | 8.35±0.02 | 11 | 290.63 |
| diag_b0.3/512 | mergeBC_alloc | 11.1037±0.0015 | **-10.3%** | 14.90±0.00 | 26.23±0.01 | 8.64±0.02 | 11 | 290.63 |
| diag_b0.3/512 | mergeBC | 11.1395±0.0471 | **-10.0%** | 14.90±0.00 | 26.22±0.01 | 8.65±0.02 | 11 | 290.63 |
| scattered_1x8/128 | baseline | 0.1301±0.0010 | — | 52.91±0.26 | 0.33±0.01 | 2.89±0.02 | 11 | 23.25 |
| scattered_1x8/128 | mergeBC_alloc | 0.1289±0.0006 | -0.9% | 57.79±0.35 | 0.23±0.01 | 2.35±0.06 | 11 | 23.25 |
| scattered_1x8/128 | mergeBC | 0.1288±0.0005 | -0.9% | 57.82±0.27 | 0.22±0.01 | 2.31±0.01 | 11 | 23.25 |
| scattered_1x8/512 | baseline | 0.4973±0.0010 | — | 73.13±0.17 | 0.24±0.01 | 3.35±0.07 | 11 | 23.25 |
| scattered_1x8/512 | mergeBC_alloc | 0.4962±0.0011 | -0.2% | 77.07±0.11 | 0.15±0.00 | 2.76±0.08 | 11 | 23.25 |
| scattered_1x8/512 | mergeBC | 0.4964±0.0010 | -0.2% | 77.11±0.21 | 0.16±0.00 | 2.88±0.16 | 11 | 23.25 |
| scattered_8x1/128 | baseline | 0.2510±0.0001 | — | 9.83±0.02 | 14.71±0.02 | 7.71±0.03 | 11 | 23.25 |
| scattered_8x1/128 | mergeBC_alloc | 0.2245±0.0001 | **-10.5%** | 17.11±0.04 | 16.66±0.05 | 9.30±0.05 | 11 | 23.25 |
| scattered_8x1/128 | mergeBC | 0.2244±0.0001 | **-10.6%** | 17.12±0.03 | 16.62±0.04 | 9.36±0.09 | 11 | 23.25 |
| scattered_8x1/512 | baseline | 0.9603±0.0007 | — | 9.79±0.03 | 25.32±0.02 | 8.75±0.07 | 11 | 23.25 |
| scattered_8x1/512 | mergeBC_alloc | 0.8539±0.0001 | **-11.1%** | 16.55±0.04 | 28.09±0.06 | 9.50±0.12 | 11 | 23.25 |
| scattered_8x1/512 | mergeBC | 0.8547±0.0021 | **-11.0%** | 16.54±0.03 | 28.15±0.06 | 9.50±0.14 | 11 | 23.25 |
| scattered_2x2/128 | baseline | 0.1035±0.0002 | — | 21.17±0.44 | 1.08±0.01 | 4.33±0.02 | 11 | 23.25 |
| scattered_2x2/128 | mergeBC_alloc | 0.0945±0.0002 | **-8.7%** | 26.94±0.13 | 1.04±0.01 | 3.52±0.02 | 11 | 23.25 |
| scattered_2x2/128 | mergeBC | 0.0944±0.0001 | **-8.8%** | 27.07±0.13 | 1.05±0.01 | 3.52±0.01 | 11 | 23.25 |
| scattered_2x2/512 | baseline | 0.3026±0.0011 | — | 35.94±0.55 | 8.95±0.27 | 5.58±0.09 | 11 | 23.25 |
| scattered_2x2/512 | mergeBC_alloc | 0.3003±0.0010 | -0.8% | 51.18±0.57 | 7.38±0.12 | 5.96±0.33 | 11 | 23.25 |
| scattered_2x2/512 | mergeBC | 0.3021±0.0025 | -0.2% | 50.74±0.60 | 7.60±0.29 | 5.67±0.46 | 11 | 23.25 |

(time = `gpu__time_duration.sum` from the stall-reason ncu pass, mean±sd over 5 reps; `occ_limit_smem` =
`launch__occupancy_limit_shared_mem` in blocks/SM, `waves/SM` = `launch__waves_per_multiprocessor`, both
from the separate occupancy pass, mean over 5 reps — sd omitted, sub-0.1% jitter throughout.)

Full occupancy-pass metrics (`launch__shared_mem_per_block_static`, `launch__occupancy_limit_regs`,
`_warps`, `_blocks`, `launch__registers_per_thread`) are **bit-identical across all three variants in
every one of the 10 configs**: `smem_static=8192 B`, `occ_limit_smem=11`, `occ_limit_regs=12`,
`occ_limit_warps=12`, `occ_limit_blocks=16`, `regs_per_thread=40`. Shared memory is the binding occupancy
limiter here (11 < 12 < 16), and it is identically 11 blocks/SM for `baseline`, `mergeBC_alloc`, and
`mergeBC` alike — the static `__shared__ scalar_t smem[TOTAL_SHARED]` reservation never changed size, in
any variant, at any config. `waves_per_multiprocessor` differences between variants are ≤0.05 out of
~23-291 (noise floor — comparable to the spread across `mergeBC`'s own 5 reps of the same variant).

## Verdict: mergeBC's win is (a), not (b) — the allocation was never freed

**`mergeBC_alloc` recovers essentially all of `mergeBC`'s win, everywhere.** On the 8 configs with a real
effect to measure (`diag_b0.1`, `diag_b0.3`, `scattered_8x1` — where `mergeBC`'s delta is -8.8% to
-11.1%), `mergeBC_alloc`'s delta matches `mergeBC`'s to within 0.1-0.3 percentage points every time
(e.g. `diag_b0.3`/512: mergeBC_alloc -10.3% vs mergeBC -10.0%; `scattered_8x1`/512: -11.1% vs -11.0%) —
recovering 94-108% of `mergeBC`'s delta by ratio, well inside the ~0.5% run-to-run sd on these
measurements. On `scattered_1x8` and `scattered_2x2`/512, both variants show the same near-zero,
noise-level delta as each other (`mergeBC` itself was already flat there in the original zeroidle
microbench doc — nothing to gain, nothing lost).

This directly answers Eran's question: **mechanism (b) (freed shared-memory allocation → better
occupancy) was never in play for this kernel.** `launch__occupancy_limit_shared_mem` and
`launch__waves_per_multiprocessor` are identical, config-for-config, across `baseline`/`mergeBC_alloc`/
`mergeBC` — the static `smem[TOTAL_SHARED]` array's declared size doesn't depend on which byte range a
given kernel body happens to touch, and `mergeBC` was already using the same `TOTAL_SHARED` and the same
`p`-divisor as baseline before this ablation existed. `mergeBC_alloc` mostly *confirms in ncu numbers*
what was true by construction; it does not reveal a previously-hidden occupancy shift, because there
wasn't one to find. The entire ~9-12% win is attributable to mechanism (a) — removing the `smem_Y`
store-then-reload instructions and the associated `__syncthreads()` (the second one, immediately after
Step B/C, is skipped: only one `__syncthreads()` remains inside the loop instead of two) — consistent
with `mio_throttle` rising (not falling) in `mergeBC`/`mergeBC_alloc` relative to baseline even as
wall-clock drops, and `long_scoreboard` rising too (fewer total instructions issued per unit time means
the same absolute latency-stall cycles are a larger *percentage* of a shorter kernel — a ratio artifact,
not a sign the kernel got slower on that axis).

**Disposition:** confirms `STALL_REASONS_ZEROIDLE_MICROBENCH.md`'s original attribution stands as stated.
No occupancy confound found in this specific case, unlike `feat/pbr-cache-block-data`'s (opposite-direction)
confirmed case. `mergeBC_alloc` is diagnostic-only, kept for the record alongside `mergeBC`; the original
recommendation (worth carrying `mergeBC`'s fixed-`p` Step B/C fusion forward to `pbr_spmm_vec_kernel`,
not yet attempted) is unchanged by this result.
