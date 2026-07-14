# Isolating the smem_Y store/reload removal from the p-growth effect: negative/mixed result

Branch: `feat/pbr-smemy-capped-p` (based on `origin/feat/pbr-nnzgroup`).

Follow-up to `STALL_REASONS.md`'s "Attempted fix (2026-07-14)" section, which merged `block`'s Step B
(compute into `smem_Y`) and Step C (`atomicAdd` from `smem_Y`) into a single loop **and** let `p` grow
into the shared memory freed by removing `smem_Y`. That attempt regressed `mio_throttle`/`sm__throughput`
because the larger `p` produced fewer but much bigger uninterrupted bursts of shared-memory traffic
between `__syncthreads()`s. That experiment flagged an unresolved question: does removing the
`smem_Y` store/reload help *on its own*, separate from the burst-size change it was bundled with?

This experiment answers that question directly.

## Code change

Applied to both `pbr_spmm_zero_idle_kernel` and `pbr_spmm_vec_kernel` in
`torch_fast_spdmm/kernels/pbr_kernel.cu`:

- Removed the `smem_Y` shared-memory array entirely.
- Merged what were Step B (write per-thread sum into `smem_Y`) and Step C (read `smem_Y`, `atomicAdd`
  into global `Y`) into one loop: each thread now computes its sum/vector accumulator in a register and
  `atomicAdd`s directly from it. The `__syncthreads()` that previously separated B and C is gone.
- **`p` (`p_log2`/`vp_log2`) is UNCHANGED** — still `floor_pow2(min(TOTAL_SHARED / (n_r + n_c), features))`,
  the same divisor as before the change. `smem_X` therefore only occupies `n_c * current_p` of the
  `TOTAL_SHARED` buffer — strictly less than the full buffer, since the freed `smem_Y` space is left
  unused on purpose. Iteration count and per-iteration `smem_X` burst size are byte-for-byte identical
  to the pre-change kernel; the *only* difference from the original kernel is: no `smem_Y` array, no
  store/reload round trip, no middle `__syncthreads()`.

Net diff: 22 insertions / 32 deletions across both kernels (commit `bee78fb`).

## Correctness

Built and tested on a fresh clone (`/home/dayanb/research/SPMM-smemy-capped` on the remote GPU host, to
avoid colliding with other agents using the shared checkout). `pytest tests/torch_fast_spdmm/test_pbr.py
tests/torch_fast_spdmm/test_csr.py -q` → **3704 passed**, 1 pre-existing unrelated warning. Pure refactor,
same accumulated sum, no regressions.

## Profiling method

Identical methodology to `STALL_REASONS.md`: `ncu --profile-from-start off --metrics <same 15-metric
list as profile_after_20260714.sh> --csv --target-processes all python -m benchmark.profile_pbr
--matrix-profile <mat> --size <n> --features <f> --block 8 --min-nnz 2 --smem-size 2048 --kernel block`,
run via `jobs/profile_stall_reasons/profile_smemy_capped_20260714.sh` (only `kernel=block` reprofiled;
cuSPARSE is untouched by this change, so its baseline numbers remain valid). 5 matrices × 2 sizes × 2
feature counts = 20 runs, all succeeded, CSVs in `ncu_out/stall_reasons_smemy_capped/` (untracked).

**N=1024 caveat (same as `STALL_REASONS.md`):** too small/fast for reliable stall sampling — flagged
directional-only.

## Results — N=262144 (the real data)

| matrix/feat | time before→after (ms) | Δtime | SM% before→after | `mio_throttle` before→after | `barrier` before→after |
|---|---|---|---|---|---|
| `diag_b0.1`/128 | 2.013 → 2.004 | −0.4% | 54.83 → 53.68 | 10.77 → **13.94** (+3.2) | 19.87 → 19.42 (−0.5) |
| `diag_b0.1`/512 | 8.021 → 7.945 | −0.9% | 33.06 → 31.54 | 17.66 → **21.00** (+3.3) | 25.29 → **27.58** (+2.3) |
| `diag_b0.3`/128 | 3.336 → 3.352 | +0.5% | 44.25 → 42.24 | 17.84 → **22.35** (+4.5) | 14.10 → **16.67** (+2.6) |
| `diag_b0.3`/512 | 12.986 → 12.928 | −0.4% | 30.28 → 28.36 | 21.32 → **23.78** (+2.5) | 20.11 → **26.20** (+6.1) |
| `scattered_1x8`/128 | 0.129 → 0.131 | +1.4% | 49.66 → 49.92 | 0.02 → 0.03 (~0) | 25.39 → 22.83 (−2.6) |
| `scattered_1x8`/512 | 0.496 → 0.497 | ~0% | 26.56 → 26.70 | 0.29 → 0.57 (+0.3) | 36.58 → 33.43 (−3.2) |
| `scattered_2x2`/128 | 0.091 → 0.093 | +2.0% | 67.49 → 70.37 (+2.9) | 0.41 → 0.30 (−0.1) | 29.47 → 28.32 (−1.2) |
| `scattered_2x2`/512 | 0.354 → 0.353 | ~0% | 30.32 → 32.27 (+2.0) | 22.56 → 21.38 (−1.2) | 9.90 → 9.66 (~0) |
| `scattered_8x1`/128 | 0.288 → 0.288 | ~0% | 38.18 → 39.97 (+1.8) | 17.97 → 17.05 (−0.9) | 21.12 → 19.81 (−1.3) |
| `scattered_8x1`/512 | 1.090 → 1.091 | ~0% | 26.30 → 28.69 (+2.4) | 22.33 → 21.23 (−1.1) | 25.34 → 23.21 (−2.1) |

(`sm__throughput`, `mio_throttle`, `barrier` all in %; time in ms.)

## Appendix — N=1024 (directional only)

| matrix/feat | time before→after (µs) | `mio_throttle` before→after | `barrier` before→after |
|---|---|---|---|
| `diag_b0.1`/128 | 13.15 → 13.2 | 4.19 → 7.43 | 15.53 → 14.77 |
| `diag_b0.1`/512 | 40.29 → 41.4 | 14.00 → 17.91 | 24.09 → 23.01 |
| `diag_b0.3`/128 | 18.62 → 18.3 | 9.63 → 14.77 | 12.53 → 10.70 |
| `diag_b0.3`/512 | 60.48 → 60.4 | 17.38 → 21.10 | 19.32 → 21.31 |
| `scattered_1x8`/128 | 5.09 → 4.8 | 0.00 → 0.00 | 19.85 → 15.71 |
| `scattered_1x8`/512 | 8.64 → 8.0 | 0.00 → 0.00 | 28.23 → 21.25 |
| `scattered_2x2`/128 | 4.58 → 4.6 | 0.00 → 0.00 | 21.54 → 21.31 |
| `scattered_2x2`/512 | 5.89 → 5.9 | 0.00 → 0.00 | 4.03 → 3.75 |
| `scattered_8x1`/128 | 5.66 → 5.4 | 0.00 → 0.00 | 13.84 → 12.36 |
| `scattered_8x1`/512 | 12.10 → 10.3 | 0.00 → 0.00 | 32.30 → 30.59 |

Same order-of-magnitude as before, consistent with the N=1024 sampling-noise caveat — not treated as
evidence either way.

## Verdict: mixed, and specifically NOT a win on the matrices the fix targeted

This isolates cleanly from the previous attempt (iteration count/burst size held fixed — confirmed by
construction, not just assumption), and the result is a genuine split, not a win:

1. **On `diag_b0.1`/`diag_b0.3` (the dense, many-rows-per-block matrices the original H6 hypothesis and
   the previous attempt both targeted), `mio_throttle` got WORSE in all 4 configs** — by +2.5 to +4.5
   percentage points, the same direction as the previous (reverted) attempt, just smaller in magnitude.
   `barrier` also worsened in 3 of 4 diag configs (up to +6.1 points on `diag_b0.3`/512). Wall-clock time
   is flat either way (±1%, within noise). So even with the burst-size confound removed, merging Step
   B+C does not help — and by the `mio_throttle`/`barrier` metrics specifically, it hurts — on exactly
   the matrix shapes where H6's mechanism was supposed to apply.

2. **On the scattered matrices (`scattered_1x8`, `scattered_2x2`, `scattered_8x1`), the picture is mildly
   positive**: `sm__throughput` up in 4 of 6 configs (+1.8 to +2.9 points), `mio_throttle` flat-or-down in
   5 of 6, `barrier` down in 5 of 6. But wall-clock time is unchanged everywhere (all deltas within
   ±2%, i.e. noise floor) — so these stall-metric improvements don't translate into measurable speed.

3. **No configuration shows a real (>noise) time improvement.** The largest time deltas anywhere are
   ±2%, indistinguishable from run-to-run ncu/kernel-launch variance at these matrix sizes.

**Bottom line:** removing the `smem_Y` store/reload in isolation, with `p` held fixed, is not a win. It
reproduces a smaller version of the previous attempt's `mio_throttle`/`barrier` regression on the dense
diag matrices, is a wash on wall-clock time everywhere, and only shows stall-metric (not speed) benefits
on the scattered matrices. This closes out the "isolate the store/reload from the p-growth" question
raised in `STALL_REASONS.md`: the answer is that the store/reload removal was never the beneficial part
of that attempt — if anything it's mildly harmful on its own for the matrix shapes that motivated the
whole investigation. `mio_throttle`/`barrier` reduction on `block` remains an open problem; this
particular lever (removing the smem_Y round trip) does not solve it, with or without letting `p` grow.

## Disposition

Kernel change is committed on `feat/pbr-smemy-capped-p` for the record (all correctness tests pass, so
it is not incorrect — merely not a performance win). Not recommended for merge into a production kernel
path without further justification; no other production kernel (`rowgroup`/`colgroup`/`tile`) was
touched by this change.
