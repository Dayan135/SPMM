# Direct atomicAdd-removal ablation (2026-07-15, branch `feat/pbr-noatomic-ablation`)

## Motivation

`STALL_REASONS.md` (branch `feat/pbr-block-stall-fix`) used `ncu --query-metrics` to check which
physical pipe `atomicAdd` issues on, found it to be the LSU pipe (not MIO), and concluded from that
indirect evidence that `mio_throttle` is driven purely by `block`'s `smem_X`/`smem_Y` traffic, not by
the Y-flush `atomicAdd`. Four separate kernel-modification attempts that changed *how data reaches* the
atomicAdd (smem_Y merge, capped-p, adaptive bypass, warp-sync) all left the atomic instruction itself in
place and all showed the "whack-a-mole" pattern: the targeted stall reason drops, another absorbs the
freed cycles, wall-clock time doesn't move. None of those four is a direct test of the atomic
instruction's own contribution. This ablation is that direct test: swap `atomicAdd` for a plain
non-atomic store in `block`'s Step C Y-flush (both `pbr_spmm_zero_idle_kernel` and
`pbr_spmm_vec_kernel`), same instruction count, same memory addresses touched, only the RMW→store
difference. Output is numerically wrong whenever >1 PBR block writes the same `(row, feature)` — not a
candidate for merge, diagnostic only.

## Method

Same `ncu` metric set and N=262144 matrix/feature grid as `STALL_REASONS.md`, `kernel=block` only (no
cuSPARSE leg — baseline already exists). Job: `jobs/profile_noatomic_ablation/profile_20260715.sh`,
output `ncu_out/noatomic_ablation/` (untracked, local only). Parsing validated by reproducing the
`STALL_REASONS.md` baseline numbers exactly from the original `ncu_out/stall_reasons/*_block.csv` files
using the same script (filter to `pbr_spmm_*_kernel` rows, exclude the zero-init fill kernel).

## Results

| matrix/feat | time: atomic → noatomic | Δtime | `mio_throttle`: atomic → noatomic | `barrier` | `long_scoreboard` | SM%: atomic → noatomic |
|---|---|---|---|---|---|---|
| `diag_b0.1`/128 | 2.013 → 1.215 ms | **−39.6%** | 10.77 → **0.15** | 19.87→13.77 | 24.90→13.99 | 54.8 → 90.0 |
| `diag_b0.1`/512 | 8.021 → 3.211 ms | **−60.0%** | 17.66 → **4.34** | 25.29→20.44 | 19.26→16.98 | 33.1 → 81.4 |
| `diag_b0.3`/128 | 3.336 → 1.626 ms | **−51.3%** | 17.84 → **0.46** | 14.10→9.83  | 20.48→14.41 | 44.3 → 89.7 |
| `diag_b0.3`/512 | 12.986 → 5.069 ms | **−61.0%** | 21.32 → **9.90** | 20.11→16.07 | 17.57→17.40 | 30.3 → 76.3 |
| `scattered_1x8`/128 | 0.129 → 0.121 ms | −6.2% | 0.02 → 0.01 | 25.39→24.44 | 46.99→44.71 | 49.7 → 53.6 |
| `scattered_1x8`/512 | 0.496 → 0.447 ms | −9.9% | 0.29 → 0.03 | 36.58→33.83 | 49.91→53.11 | 26.6 → 29.5 |
| `scattered_2x2`/128 | 0.091 → 0.077 ms | −15.4% | 0.41 → 0.02 | 29.47→25.53 | 27.42→19.86 | 67.5 → 78.6 |
| `scattered_2x2`/512 | 0.354 → 0.205 ms | **−42.1%** | 22.56 → **3.97** | 9.90→6.95   | 22.58→55.67 | 30.3 → 52.3 |
| `scattered_8x1`/128 | 0.288 → 0.123 ms | **−57.3%** | 17.97 → **0.34** | 21.12→14.53 | 16.06→14.37 | 38.2 → 87.4 |
| `scattered_8x1`/512 | 1.090 → 0.374 ms | **−65.7%** | 22.33 → **5.14** | 25.34→28.25 | 11.65→18.07 | 26.3 → 75.3 |

## Conclusions

1. **Removing the atomic instruction collapses `mio_throttle` and cuts wall-clock time 40–66%** on every
   matrix/feature combination that had high baseline `mio_throttle` (`diag_b0.1`, `diag_b0.3`,
   `scattered_8x1`, `scattered_2x2`/f512 — the same set flagged in `STALL_REASONS.md`'s own matrix-shape
   correlation). SM throughput roughly doubles in several cases (e.g. `diag_b0.1`/512: 33%→81%,
   `scattered_8x1`/128: 38%→87%).

2. **On matrices where baseline `mio_throttle` was already ~0** (`scattered_1x8` both feature counts,
   `scattered_2x2`/f128), removing the atomic barely moves anything (−6% to −15%, vs. −40% to −66%
   elsewhere) — exactly the differential the correlation predicts, and a clean negative control showing
   this isn't just "atomics are always free to remove."

3. **This directly contradicts the `ncu --query-metrics` pipe-separation claim.** That check verified
   which functional unit *executes* the `atomicAdd` SASS instruction (LSU), which is a different
   question from whether removing it changes the `mio_throttle` *warp-issue-stall* counter. The
   ablation shows it clearly does. Most likely reconciliation: the atomic RMW's global round-trip
   creates downstream backpressure (L2/memory-subsystem queueing) that also blocks the MIO-classified
   `smem_X`/`smem_Y` instructions from issuing, even though the atomic instruction itself dispatches on
   a separate pipe from shared-memory ops. The instruction-level pipe assignment and the SM-level
   issue-stall accounting are not the same measurement, and the previous branch's correction only
   checked the former.

4. **This is different from the four prior negative attempts** (`feat/pbr-block-stall-fix`'s smem_Y
   merge, `feat/pbr-smemy-capped-p`, `feat/pbr-adaptive-smem-bypass`, `feat/pbr-warp-sync`) precisely
   because it is the first one to remove the atomic instruction itself rather than changing how data
   reaches it. Those four all left `atomicAdd` in place and saw freed cycles reallocated to a different
   stall reason with no net wall-clock win — consistent with atomics being the actual bottleneck their
   fixes never touched.

## Important caveat

**This is a ceiling, not a mergeable result.** The ablation is only correct in cases with ≤1 PBR block
per output row per launch (true for none of the matrices tested here — all have `blocks_per_stripe`
&gt;1 for `diag_*`, and `scattered_8x1`/`scattered_2x2` also alias). A production fix needs a
correctness-preserving way to cut atomic frequency or cost — e.g. one wider/vectorized atomic per row
per launch instead of one per `(row, feature)`, or restructuring the flush so only one thread per row
issues the RMW and combines multiple features per atomic. That is the concrete next lever this result
justifies investigating; it hasn't been attempted here.

## Disposition

Not merged — diagnostic only, `torch_fast_spdmm/kernels/pbr_kernel.cu`'s Step C on this branch produces
incorrect output by design. Kept for the record on `feat/pbr-noatomic-ablation` (forked from
`feat/pbr-block-stall-fix`).
