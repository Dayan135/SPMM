# Next kernel ideas: evaluation (CPU only, 2026-11-11)

Status: complete, no GPU timing. Four proposed ideas were evaluated. The evaluation also surfaced
two others (empty-row skipping and vectorized X staging) that rank above all four.

Labels: **computed** = exact from the real BOO structures plus the occupancy model, which matched
4,344 timed arms. **fit** = regression on measured ncu counters. **estimated** = per-path
instruction counts read from SASS and applied to computed work counts.

## Data and method

- `scripts/research_panel/next_ideas_blocks.py` rebuilt the real BOO structures on a CPU node (job
  24067936, snapshot `/home/dayanb/research/kc-p-20261110`, same labels as the k_c x P gate). All
  30 `boo_digest` values match the gate. It writes per-block n_r, n_c, nnz, active columns,
  nonempty rows and every row length (`blocks/`, 31 MB).
- `scripts/research_panel/next_ideas_eval.py` runs ideas 1-4 (`eval/*.csv`, printed in
  `eval_report.txt`). The instruction budget is `instruction_budget.py` -> `eval/instruction_budget.csv`.
- Inputs: fem2d_permuted, fem2d, roadnet, diag_dd0.05, diag_b1.0. Shapes: 32x32, 64x32, 128x32,
  128x64, 256x32, 256x64. F512.
- Validation: the computed X-staging traffic, sum(n_c) x F x 4 B / 32 B, matches the measured
  global-load sectors. fem2d_permuted 128x64: 86.1M computed vs 90.3M measured; 256x64: 63.1M vs
  about 68M; 128x32: 64.8M vs 69.8M. X staging is nearly all global-load traffic.

## Where the instructions go (fit + SASS)

A constrained fit (non-negative least squares) to the measured P8-P128 instruction counts
(`fit_instructions.py`, output `eval/instruction_fit.txt`; within 0.4% on all 10 points) gives, for
fem2d_permuted 128x64 F512 T256:

  inst = 977M + 16.8 x (warp loop trips x passes) + 17.8 x (warps per CTA x passes x blocks)

The P-independent 977M decomposes, from SASS of the rowseg T256/P32/FPT4 kernel:

| Part | Cost per unit (SASS) | Work (computed) | Instructions |
|---|---|---|---|
| Step A, X staging | 25 instr per staged element per warp (scalar 4-byte LDG + STS, 64-bit address math) | 21.5M warp-iterations | about 538M (32%) |
| Step B work-item setup + commit | about 44 instr per warp-item for an empty row; more for a nonempty one | 10.4M warp-items | about 360-450M |
| Step B inner loop | 16.8 per warp trip (fit) | 37.7M trips | about 630M (38%) |
| Per pass, per warp | 17.8 (fit) | 3.2M | about 57M |

An empty work item still pays about 10 instructions of setup and about 30 in the commit section:
load the row id, build the 64-bit Y address, and test 4 accumulators before skipping the atomics.

## Idea 1: active-column compaction (computed)

| Input | Shape | Active / staged columns | X sectors now -> compacted | max_active = max_nc? |
|---|---|---|---|---|
| fem2d_permuted | 128x64 / 256x64 | 0.414 / 0.478 | 86.1M -> 35.6M / 63.1M -> 30.1M | yes |
| fem2d | 128x64 / 256x64 | 0.414 / 0.480 | 86.1M -> 35.7M / 62.9M -> 30.2M | yes |
| roadnet | 128x64 / 256x64 | 0.399 / 0.483 | 347M -> 138M / 278M -> 134M | yes |
| diag_dd0.05 | 128x64 | 0.964 | 16.8M -> 16.2M | yes |
| diag_b1.0 | all | 1.000 | unchanged | yes |

- Traffic: X staging falls 52-60% on FEM and roadnet, and barely changes on the diagonals.
- Arena: no change. In every shape some block uses all of its columns, so max_active = max_nc and
  the largest P at any occupancy target is unchanged.
- Instructions (estimated): Step A shrinks with the staged elements, -12% to -22% of the total on
  FEM/roadnet and -1% on diag_dd0.05.
- Cost: a format change. `boo_matrix.cpp` writes a per-block active-column list and remaps nz_idx
  local columns, and the kernel loads the per-block list instead of the column group's ids. The
  16-bit index gets smaller, so no new limits. It needs the full test and sanitizer suite.
- Decisive test: fem2d_permuted and roadnet at 128x64 / 256x64, F512. Prediction: global-load
  sectors fall to about 0.48 / 0.40 x. Falsifier: sectors fall but time is unchanged (within 3%),
  which would mean the staging traffic is no longer on the critical path.

## Idea 2: size-class launches (computed): reject

Blocks are homogeneous in their arena requirement. Without compaction, no 2-class cut raises P for
any input or shape. With compaction, fem2d/fem2d_permuted 128x64/256x64 can move 24% of nonzeros
from P64 to P128, and roadnet 46-73%. But the k_c x P experiment measured P64 -> P128 at P >= 32
as about -1% instructions (1,638M -> 1,624M), so the gain is negligible, and each extra launch adds
overhead and a tail wave.

## Idea 3: row-length-aware mapping (computed lane efficiency; fit + estimate)

- Sorting rows by length raises lane efficiency only at low P: fem2d_permuted 128x64 0.42 -> 0.77
  at P8 and 0.63 -> 0.92 at P16, but 0.996 already at P32 (fem2d rows come in equal-length groups)
  and 1.0 at P >= 64. diag_dd0.05 is 0.77 -> 0.98 at P32.
- With the fitted model, predicted instructions are 0.75x at P8, 0.85x at P16 and 1.00x at
  P32-P128 (fem2d_permuted). Sorted P8 would still be about 2,020M, above P32's 1,667M. **It does
  not help at the P the winners use (P32-P64).**
- The useful part of this idea is not sorting but **skipping empty rows**. See the next section.

## Empty-row skipping (computed + estimated): new, rank 1

In tall blocks most rows have no nonzeros in the block's column group, yet Step B gives every row a
work item on every pass.

| Input | 128x64 | 256x64 | 32x32 |
|---|---|---|---|
| fem2d_permuted | 74% empty | 82% | 55% |
| roadnet | 78% | 86% | 70% |
| diag_dd0.05 | 52% | 76% | 19% |
| diag_b1.0 | 50% | 75% | 0% |

Warp-items per run, now -> nonempty rows only: fem2d_permuted 128x64 P32 10.4M -> 2.7M; 256x64
P64 14.8M -> 2.7M; roadnet 128x64 P32 42.4M -> 9.7M. At about 44 instructions per empty warp-item
(SASS), the estimated savings are:

| Winner | Measured instructions | Saved by skipping empty rows |
|---|---|---|
| fem2d_permuted 128x64 P32 | 1,667M | about 336M (-20%) |
| fem2d_permuted 256x64 P64 T512 | 1,733M | about 529M (-31%) |
| roadnet 128x64 P32 | 5,924M | about 1,441M (-24%) |
| diag_dd0.05 128x64 P64 | 442M | about 96M (-22%) |

It is a **kernel-only** change. After the row-start table, build a list of nonempty local rows in
shared memory (rows with start < end, via warp ballot and prefix sum; uint16, +2 x max_nr bytes of
arena), then map Step B work items over that list. The arena grows slightly: for 256x64 P64 T512 it
goes 31,764 -> 32,276 B, alloc 33,408 B, still 3 CTAs; that must be checked per shape. Atomics are
unchanged, since empty rows commit nothing. Tall blocks benefit most, which makes the current
winners better.

## Vectorized X staging (estimated): new, rank 2

Step A stages one float per iteration at 25 instructions. P is a multiple of 8 and X rows are
contiguous, so 16-byte loads and stores (4 floats per iteration) cut Step A instructions about 4x,
and put 4x the bytes in flight per load, which should help a latency-bound loop. It needs
F % 4 == 0 for alignment, with a scalar fallback otherwise.

Estimated savings: -24% (fem2d_permuted 128x64 P32), -17% (256x64 P64), -27% (roadnet 128x64),
-18% (diag_dd0.05). This is kernel-only and touches only the Step A loop.

## Idea 4: A in registers across passes (computed): reject

At the winners, the busiest thread in a block must hold 88-188 nonzeros (p99) on fem2d_permuted
and 13-23 on roadnet. With 8 / 16 / 32 extra registers, the share of nonzeros in blocks that fit is
< 1% / 2-4% / 11-26% on FEM, and 9-15% / 32-77% / 95-100% on roadnet. Extra registers cost CTAs
at T256: 6 -> 5 at +8, 6 -> 4 at +16; at T512, 3 -> 2. It only fits roadnet, at an occupancy cost,
and would need a fallback path.

## Ranking

| Rank | Idea | Expected effect at the winners | Cost | Confidence |
|---|---|---|---|---|
| 1 | Empty-row skipping | -20% to -31% instructions | Kernel only, small | High on the work count (computed), medium on cost per item |
| 2 | Vectorized X staging | -17% to -27% instructions, more memory-level parallelism | Kernel only, small | Medium-high |
| 3 | Active-column compaction (idea 1) | -52% to -60% X traffic on FEM/roadnet; -12% to -22% instructions (about -3% to -14% on top of rank 2) | Format + kernel | High on traffic (computed); time effect unknown |
| 4 | Row sorting (idea 3) | Only at P8-P16, not used | Kernel/format | High that it doesn't help |
| 5 | Size classes (idea 2) | About 0-1% | Host, multiple launches | High that it doesn't help |
| 6 | A in registers (idea 4) | Fits only roadnet, occupancy cost | Hard | High that it doesn't help |

**Against cuSPARSE:** ranks 1 + 2 together are estimated at about -48% instructions on
fem2d_permuted 256x64 P64 (1,733M -> about 860M), below cuSPARSE's 1,145M on the same reordering.
Rank 3 also halves X traffic. This is the first candidate set that plausibly closes the gap where
BOO was at parity. Instructions are not time, though: Step A is latency-bound, so the time gain will
be smaller than the instruction cut. That is what the GPU test must show.

## Recommended GPU experiment (frozen predictions)

Kernel-only A/B on the quick-decode harness (`experiments/2026-11-10_quick_decode/quick_decode_job.sh`
pattern: one GPU, A and B rebuilt together, full BOO suite plus memcheck, racecheck and synccheck,
three alternating rounds, float64 check on every arm, ncu on key arms). Variants:
- B1: empty-row skipping
- B2: vectorized X staging
- B3: both

Configurations: fem2d_permuted and roadnet F512 at 128x64 P32 T256, 256x64 P64 T512 and tuned 32x32
P64 T256, plus diag_dd0.05 128x64 P64 T256 and diag_b1.0 64x32 P128 T256.

Predictions:
1. Warp instructions, B3 vs A: fem2d_permuted 256x64 0.45-0.65x; 128x64 0.50-0.70x; roadnet
   128x64 0.45-0.65x. B1 and B2 each land within ±10 percentage points of the estimates above.
2. Global-load sectors are unchanged for B1, and unchanged in total for B2 (same bytes, fewer
   requests).
3. Time: B3 faster than A by > 5% on both FEM winners and on roadnet 128x64. The time gain is
   smaller than the instruction cut.
4. 32x32 gains less from B1 than the tall shapes (55-70% empty rows vs 74-86%), so the tall-block
   advantage over 32x32 grows.

Falsifier: instructions fall as predicted but time moves < 3%, meaning the kernel is not
instruction-bound at these points. Then traffic (rank 3) and latency, not instructions, are the
remaining levers.

## The P8 "more CTAs slower" oddity

It stays deprioritized. It appears only at P8, which no winner uses, and both ranked fixes reduce
its likely source (more warps issuing staging and empty-row work at once). If it is tested later,
the decisive check is P8 native (6 CTAs) against padded (2 CTAs) at matched instructions, with
ncu L1/LSU pipe utilization and long-scoreboard stalls by source line from a `-lineinfo` build.
