# Caching `block_data` in shared memory across the `base_f` loop: negative (whack-a-mole)

Branch: `feat/pbr-cache-block-data` (based on `origin/research/pbr-cusparse-scalability-gap`, commit `206c2d1`).

## Hypothesis

In both `pbr_spmm_zero_idle_kernel` and `pbr_spmm_vec_kernel`, Step B re-reads `block_data` via `__ldg`
inside the `for (base_f = feat_start; base_f < feat_end; base_f += p)` grid-stride loop. `block_data`'s
values (and the `local_data_ptr` sequence used to read them) are a pure function of `data_offset`,
`packed_row_offsets`, and the block's populated columns — none of which depend on `base_f`. So at
`features=512` with `p=128` (4 iterations), every thread re-reads the same `block_data` values from
global memory 4× over. Prior art (`feat/pbr-zeroidle-microbench`'s `constdata` ablation) showed removing
this `__ldg` cuts time 5–23% and reduces `long_scoreboard` meaningfully — but that ablation was
diagnostic-only (drops correctness, since it substitutes constant data). This experiment tests the
correctness-preserving version: cache `block_data` into shared memory **once**, before the `base_f`
loop, and read from shared memory instead of re-issuing `__ldg` on every iteration.

Predicted mechanism: `long_scoreboard` (global-memory latency stalls) drops in proportion to the number
of `base_f` iterations skipped; wall-clock time improves, especially at `features=512` (4 iterations)
more than `features=128` (1–2 iterations, less redundancy to remove).

## Code change

Applied to both kernels in `torch_fast_spdmm/kernels/pbr_kernel.cu`:

- Added `__shared__ scalar_t s_block_data[BLOCK_ROWS * BLOCK_COLS]` (statically sized, ≤64 elements for
  the largest 8×8 PBR block — small, compile-time-bounded, no per-thread register tracking needed).
- After the Step-A sparsity scan (which produces `total_nnz = current_relative_offset`), all threads
  cooperatively stage `block_data[data_offset .. data_offset+total_nnz)` into `s_block_data` via a
  grid-stride loop over `total_nnz`, followed by one `__syncthreads()`.
- Inside the `base_f` loop, `local_data_ptr` is now block-relative (no `+ data_offset`) and Step B reads
  `s_block_data[local_data_ptr]` instead of `__ldg(&block_data[local_data_ptr])`.

Net effect: replaces `(features/p)` global reads per nnz-entry with exactly 1 global read (staged into
shared memory) + 1 extra `__syncthreads()` per kernel launch. Commit `8388e4a`.

## Correctness

Built and tested in an isolated worktree (`.claude/worktrees/agent-afe064b20d91cd9d1`) via
`optimizer.run`'s automated correctness gate: `pytest tests/torch_fast_spdmm/` → **3704 passed**, 1
pre-existing unrelated warning, on commit `8388e4ab0953163ef35c41868494c440d772a3ea`. Same accumulated
sum as baseline, no regressions.

## Method

Full automated sweep via `optimizer/` (Phase 0 tooling), not a one-off `ncu` script:

```
python -m optimizer.run --matrices diag_b0.1 diag_b0.3 scattered_1x8 scattered_8x1 scattered_2x2 \
    --sizes 262144 --features 128 512 --kernels block \
    --git-ref feat/pbr-cache-block-data --db optimizer_ledger_cache_block_data.sqlite3
```

Baseline swept the same grid against `research/pbr-cusparse-scalability-gap` (unmodified kernel) into a
separate ledger, `optimizer_ledger.sqlite3`, to avoid the config-key dedup gap (see `optimizer/run.py`'s
`--db` docstring). Compared with `python -m optimizer.compare --baseline optimizer_ledger.sqlite3
--candidate optimizer_ledger_cache_block_data.sqlite3`, which reports wall-clock delta plus the same
7-metric stall-reason table as every prior hand-run comparison in `STALL_REASONS.md`.

240 candidate points measured (block ∈ {2,4,8} × min_nnz × smem_size ∈ {512,1024,2048} × 5 matrices ×
{128,512} features); 186 matched a baseline point (the rest are `min_nnz > block` combinations the grid
enumerates but the kernel legitimately skips on both sides — pre-existing, unrelated to this change).

## Results — aggregate verdict

**186 matched config points: 3 WIN, 47 REGRESSION, 136 WASH.** Split by feature count:

| features | WIN | REGRESSION | WASH |
|---|---|---|---|
| 128 | 1 | 41 | 51 |
| 512 | 2 | 6 | 85 |

Split by block size (both feature counts combined) — the effect is strongly block-size-dependent:

| block | f128 | f512 |
|---|---|---|
| 2 | 24 REGRESSION | 1 REGRESSION, 23 WASH |
| 4 | 15 REGRESSION, 21 WASH | 5 REGRESSION, 31 WASH |
| 8 | 2 REGRESSION, 30 WASH, 1 WIN | 31 WASH, 2 WIN |

This matches the predicted mechanism's shape, but not its sign: `block=2` PBR blocks have at most 4
nnz entries to cache (2×2), so there's almost nothing to save by caching, yet every launch still pays
the extra `__syncthreads()` — pure overhead, 100% regression at f128. `block=8` blocks (≤64 nnz, more
redundant reads eliminated) come closest to neutral, but even there the gain never clears the 3%
WIN threshold except in 3 isolated points.

## Results — reference config (`block=8, smem=2048`, the `STALL_REASONS.md` baseline slice)

| matrix/feat | time before→after | Δtime | `long_scoreboard` before→after | `barrier` before→after |
|---|---|---|---|---|---|
| `diag_b0.1`/128 | 2.039→2.013 ms | −1.3% | 24.78→**20.80** (−4.0) | 19.82→**27.31** (+7.5) |
| `diag_b0.1`/512 | 8.010→8.002 ms | −0.1% | 19.29→**13.97** (−5.3) | 25.32→**29.12** (+3.8) |
| `diag_b0.3`/128 | 3.354→3.372 ms | +0.6% | 20.49→**13.93** (−6.6) | 14.09→**22.60** (+8.5) |
| `diag_b0.3`/512 | 12.998→12.921 ms | −0.6% | 17.57→**10.65** (−6.9) | 20.11→**25.79** (+5.7) |
| `scattered_1x8`/128 | 130.1→129.2 µs | −0.7% | 46.85→44.50 (−2.4) | 26.04→25.71 (−0.3) |
| `scattered_1x8`/512 | 497.5→497.1 µs | −0.1% | 51.10→54.09 (+3.0) | 34.95→31.70 (−3.3) |
| `scattered_2x2`/128 | 92.1→96.2 µs | +4.4% | 27.43→**22.26** (−5.2) | 29.37→**34.67** (+5.3) |
| `scattered_2x2`/512 | 353.5→351.2 µs | −0.6% | 22.44→**17.22** (−5.2) | 9.89→**21.89** (+12.0) |
| `scattered_8x1`/128 | 286.3→288.1 µs | +0.6% | 16.07→**9.98** (−6.1) | 21.20→**33.23** (+12.0) |
| `scattered_8x1`/512 | 1098.8→1088.4 µs | −0.9% | 11.75→**5.15** (−6.6) | 25.34→**32.59** (+7.2) |

(all percentages in %; time as measured by `gpu__time_duration.sum`.)

## Verdict: mechanism confirmed, whack-a-mole again

The predicted mechanism **is real and directly visible**: `long_scoreboard` drops in every single
reference-config row, by 2.4–6.9 percentage points — global-memory latency stalls genuinely go down,
consistent with prior art's `constdata` ablation. But `barrier_pct` absorbs essentially all of the
freed time, rising 3.8–12.0 points in the same rows — the mandatory `__syncthreads()` after staging
`s_block_data` costs almost exactly what the eliminated redundant reads saved. Net wall-clock time is
flat to slightly negative across the board (−1.3% to +4.4%), never clearing the ±3% WIN/REGRESSION
threshold at this reference config — same "whack-a-mole" pattern as all four prior kernel-edit attempts
in `STALL_REASONS.md`, just with a different pair of stall reasons trading places (`long_scoreboard`
↔ `barrier` instead of `mio_throttle` ↔ something else).

**Not recommended for merge.** The extra `__syncthreads()` needed to safely publish the shared-memory
cache costs about as much as the redundant global reads it removes, and for small PBR blocks
(`block=2`) it's a clear net loss with no offsetting benefit. A version that avoided the barrier (e.g.
warp-local caching within a warp's own nnz subset, no cross-warp staging) might change this tradeoff,
but that's a distinct, more invasive hypothesis, not a variant of this one.
