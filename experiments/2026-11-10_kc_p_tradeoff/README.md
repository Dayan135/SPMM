# k_c × P: staging, instructions and occupancy

Status (2026-11-10): complete. CPU gate 24057296, stage0 24057339 (23 tests passed, memcheck,
racecheck and synccheck clean), timing allocations 24057348 and 24057349, nsys audit 24057438 and ncu
profiles 24057439. The report script is `scripts/research_panel/kcp_report.py`; results are below.
The optional -lineinfo build was not started. Agent 2's session ended during the timing run, and the
audit, profiles and report were completed without changing the frozen manifest or analysis plan.

## Question

At a fixed BOO structure, does reducing P help through occupancy, or hurt through more passes and a different row-to-warp mapping? Can wider column groups reduce enough Y commits to offset their staging and arena costs?

## Fixed before timing

- Inputs: fem2d_permuted (HEM), fem2d (HEM, natural order), roadnet (METIS), diag_dd0.05 and diag_b1.0 (identity). Generated N=262144; roadnet native size. F128 and F512.
- Structures: 32x32 anchor; 64x32; 128x{32,64,128,256,512}; 256x{32,64,128,256}. The excluded 256x512 corner is explicitly marked as exceeding uint16 encoding. Build real BOO structures, hash labels and BOO contents. Reuse saved labels; generate absent k256/k512 labels only on CPU nodes with the frozen method/seed and the existing feasible-first clustering path. Rectangles use row labels from k_r and column labels from k_c, not a newly jointly optimized clustering.
- CPU table: P={8,16,32,64,128}, T={128,256,512}, fixed FPT2 on diag_dd0.05 and FPT4 elsewhere. Registers/static shared/local/stack come from `cuobjdump --dump-resource-usage` on the unchanged validated extension, sm_86. These are static binary resources, not measurements from a new GPU run.
- Timing: feasible arms at T256 and T512. FPT fixed across shapes/P within input. Each (input,F) is one interleaved cell with all feasible structures and P values, avoiding between-cell comparisons for the shape sweep. This is a mechanism study, not an exhaustive FPT retuning competition.
- Matched occupancy: for each (input,shape,T256), target the minimum predicted natural CTAs/SM over feasible P. Use existing `_ctaN` padding; compare on the direct-binding path with `_pad0` controls for every P. Where the natural arena already reaches the target and padding's safety margin will not fit, use `_pad0` itself. All padded targets must be checked against cudaOccupancy; mismatched targets are unsupported, never silently relabeled. Record exact padding. Fixed occupancy does not fix warp mapping, cache capacity or waves lost to scheduling tails.
- Controls: `_dup` at largest feasible P for each shape/T; `_pad0` at each P/T256. Complete call, launch_mode auto, slab512, float32, warmup5, repeats20, independent seeded shuffle per repeat, float64 checks, original artifact hashes. Inputs/X/references from nnzflat-20261014. Stage0 validates P8/kc512 and runs memcheck/racecheck/synccheck on padded cases.
- Replication: two allocations on disjoint eligible node sets, excluding ise-3090-01 and ise-3090-04; verify GPU UUIDs per comparison. At most two GPU jobs at once, including audits/profiles. Claims require both ratios >1.05; [0.95,1.05] is a descriptive band, not equivalence. Report min/median/max per-GPU ratios and both unfiltered and filtered results. Drop an entire allocation/cell in filtered reports if any `_dup` or `_pad0` ratio leaves [0.95,1.05]; retain all unfiltered data. Do not claim from <2 accepted distinct GPUs.
- Launch audit: Kineto plus nsys for all cells/F, with exact cell+arm keys. No alias acceptance based on an audit of another F.
- No timing-based pruning or retuning after freeze. An unsupported cell remains unsupported. Best P is descriptive screen selection; report all P rather than treating the selected minimum as independent confirmation.

## Model and analysis plan

Test the proposed model verbatim: `arena=6*max_nnz+8*max_nr+4*max_nc+4*P*max_nc+20`, `alloc=ceil((arena+1024)/128)*128`, `CTAs=min(102400//alloc,1536//T,65536//(regs*T),16)`. A cell is infeasible if packed local coordinates exceed 65535 or dynamic arena exceeds 101376 B. Static shared memory is expected zero and recorded; nonzero static memory is a model discrepancy. Requested shape and actual maxima are separate columns. One outlier sets every CTA's arena.

Q1: join every accepted native/padded arm to the real structure plus padding; exact integer comparisons of predicted arena and CTAs against launch_info; for the preselected ncu subset also allocated shared memory and individual resource limits. List all discrepancies rather than averaging. Include register-per-warp rounding as an explicitly separate diagnostic formula; it does not replace the original model after seeing results. ncu coverage is a subset, not every timing arm.

Q2: show P curves per fixed shape/T/FPT/F. Distinguish largest launchable P (cap only) from largest P preserving a specified CTA count. The existing dd0.05 measurements already disfavor the literal universal largest-launchable-P claim.

Q3: compare all P on direct binding at the common target CTA count, both between P and each padded arm versus its own pad0. Check local-memory allocation/spills. A comparison is invalid if occupancy targets differ or a control fails. Padding changes requested shared memory and may change L1/shared carveout; record ncu shared configuration/L1 counters rather than claiming an occupancy-only causal intervention without qualification.

Q4: at fixed k_r=128 compare all k_c with fixed P where feasible and with best measured P. The k_r=256 sweep tests row growth; 32x32 and 64x32 connect to earlier work. Relate changes to measured X staging, Y commits and remainder. Do not infer a universal optimum from five inputs.

Q5: preselect fem2d_permuted and diag_dd0.05, F512/T256: all feasible P for 128x64, native and occupancy-matched; plus k_c={32,64,128,256,512} at the largest P retaining >=3 CTAs, or largest fitting P if none retain three. This is model-selected P, not a timing-selected best. One ncu full capture/call per arm, durations diagnostic only. Collect threads/CTAs/limits, registers/local memory, smem allocation/configuration, waves, achieved occupancy, cache hits, L2/DRAM bytes, global-load and RED sectors, shared wavefronts/bank conflicts, warp instructions, issue active and barrier/scoreboard/MIO stalls. Use existing shape_profile and shape_profile_report. Optional separate lineinfo build requires separate binary hash and validation; no timings from it count.

## Predictions, frozen before new timing

1. Arena and allocated shared bytes will match exactly. CTA counts should match except possibly register-allocation granularity; no spills expected from the unchanged launch-bound specializations.
2. At matched occupancy, P8/P16 will usually lose to P32/P64: more barriers/pass control and altered row-to-warp grouping. Logical X bytes (x_staged*F*4), nonzero-FMA work and logical output commits stay fixed with P, but transaction counts/instructions can change. Row-start binary search is once per CTA, so its dynamic work should not multiply by F/P; repeated row-start loads in Step B can change.
3. Best P will not universally be the largest launchable P. Expect P32/P64 on sparse FEM/road and P128 on the dense narrow-column control, with regressions when larger P loses too much residency. This is a directional prediction, not a preselected winner.
4. Wider k_c should reduce Y fanout most where blocks share row support, but k_c256/512 will generally lose after P is forced down. Sparse dd0.05 is the best chance for a wider-column gain; dense diag_b1.0 will hit sparse-payload arena pressure early. FEM/road should favor 32/64 over 512.
5. If low-P slowdown persists at matched occupancy with similar DRAM bytes, expect more warp instructions/barriers or poorer shared/RED transaction efficiency, not an occupancy explanation. If low P is faster after matching occupancy, reject the monotone per-pass-cost expectation and examine imbalance/bank conflicts. If wide k_c wins, expect reduced RED sectors sufficient to offset staging/instruction costs.

## Results (2026-11-10)

Two allocations on disjoint node sets (alloc 1: cs-3090-01, cs-pheno-*; alloc 2: ise-pheno-*), 10
cells each. Every arm passed the float64 check and the nsys launch audit (2,172 of 2,172 per
allocation). The control filter keeps 18 of 20 (allocation, cell) groups. The two dropped are
alloc 1 diag_b1.0 F128 and diag_dd0.05 F128, where `_pad0` runs 5-11% faster than the native arm on
short (~1 ms) cells. That is probably the direct-binding launch path, not noise, and no conclusion
below changes with the filter. Outputs are in `analysis/report/`: `report.txt`, `q*_*.csv`,
`kc_p_grid.png`, `p_matched_occupancy.png` and `profile/`.

**Q1, the model is exact.** Predicted arena bytes and CTAs/SM match launch_info for all 4,344 timed
arms, native and padded. There are 0 discrepancies, 39-40 registers per thread, no local memory and
no static shared memory. On the 28 ncu captures, allocated shared memory and the occupancy limits
also agree.

**Q3, at matched occupancy, larger P is always faster.** With every P padded to the same CTAs/SM,
time falls monotonically with P in every group (`p_matched_occupancy.png`, orange). P8/P16 are more
than 5% slower than the best matched P in 210 of 216 comparisons, on both GPUs, and the best matched
P is P128 in 75 of 108 groups. The ncu panel (fem2d_permuted 128x64, F512) shows why:

| P | warp inst | shared ld wavefronts | barrier stall / issue | global ld sectors | RED sectors | DRAM |
|---|---|---|---|---|---|---|
| 8 | 2,693M | 554M | 15.5 | 88.9M | 175M | 4.8 GB |
| 32 | 1,667M | 311M | 3.4 | 90.3M | 175M | 4.6 GB |
| 128 | 1,624M | 288M | 0.7 | 90.3M | 175M | 4.2 GB |

Memory traffic and atomics are fixed by the structure. Instructions, shared loads and barrier
stalls grow as P shrinks, because each pass re-walks the block's nonzeros. Padding does not change
the instruction counts.

**Q2, native best P balances two opposite costs.** Raising P cuts per-pass work but grows the arena
(4·k_c bytes per unit of P) and drops CTAs/SM. The native optimum is the largest P that still keeps
about 4-6 CTAs/SM, usually P32 or P64. It is the largest launchable P in only 75 of 210 consistent
groups (best P counts: P8 7, P16 31, P32 61, P64 64, P128 47). At P128 most shapes fall to 1-2
CTAs/SM and lose 1.2-1.6x (`p_matched_occupancy.png`, blue). This confirms frozen prediction 3.

**Q4, wide k_c loses everywhere.** Speedup over the fastest 32x32 arm, best P and T, minimum of 2 GPUs:

| Input | best shape (F128 / F512) | k_c 128 | k_c 256 | k_c 512 |
|---|---|---|---|---|
| fem2d_permuted | 256x64 1.27 / 256x64 1.27 | 0.80-0.88 | 0.49-0.66 | 0.36-0.39 |
| fem2d | 256x64 1.13 / 1.15 | 0.76-0.85 | 0.52-0.67 | 0.34-0.44 |
| roadnet | 256x64 1.23 / 256x64 1.21 | 1.02-1.09 | 0.72-0.86 | 0.45-0.56 |
| diag_dd0.05 | 128x64 1.11 / 1.09 | 1.07-1.09 | 0.81-0.91 | 0.55-0.66 |
| diag_b1.0 | 128x32 1.04 / 1.01 (band) | 0.78-0.80 | 0.53-0.54 | 0.31-0.34 |

Wider column groups do cut Y commits (to 0.47-0.80x of 32x32), but they stage more X, not less
(1.0-2.1x at k_c 256-512), because a 128-row group touches only part of a 512-column group. They
also force P down or CTAs/SM to 1. In ncu, fem2d_permuted at k_r=128 behaves as follows as k_c goes
32 -> 512:
- global load sectors 70M -> 236M
- RED sectors 223M -> 105M
- DRAM 3.6 -> 8.8 GB
- L2 hit 69% -> 26%
- time 7.6 -> 43.8 ms

The best structures keep k_c at 32-64 and grow k_r (256x32, 256x64, 128x64), which matches the
block-shape confirmation. This confirms frozen prediction 4. The diag_dd0.05 exception is modest:
128x128 reaches 1.07-1.09, still at or below 128x64.

**Prediction 2** (low P loses at matched occupancy through more instructions and barriers, while
logical bytes stay fixed) is confirmed. **Prediction 5's** alternative, low P faster after matching,
did not occur.

**Unexpected: at low P, more CTAs can hurt.** On diag_dd0.05 F512, P8 runs at 2.57x the best time
with its native 6 CTAs/SM, but only 1.58x when padded down to 2 CTAs/SM. roadnet F512 and
fem2d_permuted F512 show the same at P8. ncu on diag_dd0.05 128x64 P8 shows the same 602M
instructions either way, but 6 CTAs/SM raise long-scoreboard stalls from 5.9 to 26.5 and barrier
stalls from 1.6 to 16.9 per issue. That points to memory-pipeline contention among co-resident CTAs
during staging. This is inferred, not isolated by an experiment.

## Reproduction and provenance

New adapter scripts reuse `shape_gate` functions, `bench_shape`, `shape_job` audit mode, `shape_launch_audit`, `shape_profile` and its report parser. `kcp_freeze.py` is CPU-only and write-once. Isolated remote source: `/home/dayanb/research/kc-p-20261110`; validated extension SHA256 `9d4318d5149ce732a5dc38d8c3e0bd9a507491561069b956563cb0d21214f1cd`. No commits. Uploaded scripts must match local SHA256 before dependent submissions. SSH uses a persistent control connection; submissions use unique names, persisted job IDs, queue/accounting/output checks before retry.
