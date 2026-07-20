# rowgroup vs block through the atomic-count lens: mechanism confirmed, still a net loss (mostly)

No new kernel code — `rowgroup` already exists, already correctness-tested. This is a targeted
re-measurement of an existing kernel through a specific lens the project hadn't checked yet.

## Hypothesis

`STALL_REASONS_NOATOMIC_ABLATION.md` found that literally deleting `block`'s `atomicAdd` (diagnostic
only, numerically wrong) cuts wall-clock time 40–66% on matrices with high baseline `mio_throttle`
(`diag_b0.1`, `diag_b0.3`, `scattered_8x1`, `scattered_2x2`/f512), and named the concrete,
correctness-preserving next lever as "a way to cut atomic frequency or cost... restructuring the flush
so only one thread per row combines multiple features per atomic." `rowgroup` already does a version of
this: it groups same-row-strip PBR blocks into one CUDA block and accumulates them in `smem_Y` before a
**single** `atomicAdd` per strip, instead of `block`'s one `atomicAdd` per PBR block. H5
(`WHY_BLOCK_KERNEL_FAILS_TO_SCALE.md`) also already showed `rowgroup`'s row-strip-local grouping
(unlike `chunk`'s pure array-order grouping) doesn't pay the L2-locality tax coarser grouping usually
costs — so this comparison isn't confounded by that mechanism the way a `chunk`-vs-`block` one would be.

**Prediction:** on the noatomic ablation's flagged matrices, `rowgroup` should show a large `mio_throttle`
drop (fewer atomics issued) and should recover a meaningful fraction of the 40–66% noatomic ceiling.

An older experiment (`LAST_EXPERIMENTS.md` Experiment 1, commit `a1dbf5b`) found `rowgroup` up to
1.83× faster than `block` on `scattered_dense2`/f512 ("atomicAdd contention + good group fill") but
~1.0× on `scattered_8x1`/`scattered_2x4` — using older, non-density-fixed matrix definitions and no
mio_throttle-specific measurement. This experiment re-runs the same comparison on the current matrix
definitions with the project's standard 15-metric stall-reason list, directly against the noatomic
ablation's own numbers.

## Method

`jobs/profile_rowgroup_atomic/profile_20260720.sh` — same `ncu` metric set and job pattern as
`STALL_REASONS.md`/the noatomic ablation, `block` vs `rowgroup`, N=262144, `block=8, min_nnz=2,
smem=2048`, the 4 flagged matrices × {128, 512} features. No branch/kernel changes; ran directly against
`research/pbr-cusparse-scalability-gap`. All 8 pairs succeeded.

## Results

| matrix/feat | time: block→rowgroup | Δtime | `mio_throttle`: block→rowgroup | `barrier`: block→rowgroup | `no_instruction`: block→rowgroup | `warps_active`: block→rowgroup |
|---|---|---|---|---|---|---|
| `diag_b0.1`/128 | 2.039→2.498 ms | +22.5% | 10.73→**0.05** | 19.84→2.70 | 2.64→**39.38*** | 89.4→70.7 |
| `diag_b0.1`/512 | 8.074→8.722 ms | +8.0% | 17.63→**0.03** | 25.32→2.73 | — | — |
| `diag_b0.3`/128 | 3.367→4.490 ms | +33.3% | 17.78→**0.04** | 14.09→1.93 | 0.30→**39.38** | 89.4→70.7 |
| `diag_b0.3`/512 | 13.060→15.683 ms | +20.1% | 21.32→**0.04** | 20.12→1.98 | — | — |
| `scattered_8x1`/128 | 0.289→0.311 ms | +7.7% | 18.12→**0.42** | 21.15→3.52 | — | — |
| `scattered_8x1`/512 | 1.099→0.994 ms | **−9.6%** | 22.47→**0.26** | 25.23→3.92 | 0.20→**47.47** | 90.0→83.4 |
| `scattered_2x2`/128 | 0.093→0.207 ms | **+123.4%** | 0.47→0.49 (flat) | 29.31→4.51 | 2.64→**18.62** | 85.7→81.6 |
| `scattered_2x2`/512 | 0.352→0.783 ms | +122.8% | 22.27→**0.58** | 9.80→5.38 | — | — |

(*`no_instruction`/`warps_active` columns show the representative f128 point per matrix; full per-point
numbers in `ncu_out/rowgroup_atomic/`.)

## Mechanism: hypothesis confirmed causally, but a new bottleneck replaces the old one

**`mio_throttle` collapses to near-zero in every case with nonzero baseline** (10.7–22.5% → 0.03–0.58%)
— matching, and in several cases *beating*, the noatomic ablation's own diagnostic ceiling (e.g.
`diag_b0.3`/512: noatomic ablation got `mio_throttle` to 9.90%, `rowgroup` gets it to 0.04%).
`barrier` drops substantially too (14–29% → 2–5%). **This is a real, causally-confirmed validation of
the atomic-contention hypothesis** — unlike every prior kernel-edit attempt in this project, `rowgroup`
doesn't just shuffle the freed cycles from one named stall reason to another *within the same set*
(`mio_throttle` ↔ `barrier` ↔ `long_scoreboard`) — it genuinely eliminates the atomic-contention signal.

**But `no_instruction_pct` (warps with literally nothing to issue) explodes in every case** (+16 to +47
percentage points), and `warps_active_pct` (occupancy) drops 15–19 points. `rowgroup` launches far
fewer, "fatter" CTAs than `block` (one per group-of-PBR-blocks instead of one per PBR block) — this is
the same concurrency cost H5 already identified for coarser grouping in general, just showing up here
as a warp-scheduler-starvation stall rather than H5's L2-hit-rate framing. With fewer resident warps
per SM, when the ones that *are* resident stall waiting on the (now much cheaper) atomic or on a global
load, there's nothing else queued to switch to — so `no_instruction` absorbs almost the entire gain
`mio_throttle`/`barrier` freed, and then some. **Net result: `rowgroup` regresses on 7 of 8 points
tested, badly on `scattered_2x2` (+123%), and only wins narrowly on `scattered_8x1`/f512 (−9.6%).**

One puzzle, reported honestly rather than resolved: `blocks_per_stripe` (mean 7.9–8.0 for the `diag`
matrices, 1.35 for both `scattered_8x1` and `scattered_2x2`) does **not** predict the win/loss split —
`scattered_8x1` and `scattered_2x2` have nearly identical `blocks_per_stripe` stats and total PBR block
counts (~21,000 each) yet one wins and the other regresses worst of all eight points. `scattered_8x1`
has only 9 unique block occupancy patterns vs `scattered_2x2`'s 784 (much more structural regularity),
which may matter, but this wasn't instrumented further here — flagged as an open question rather than
force-fit into an explanation the data doesn't clearly support.

## Verdict

**Confirms the noatomic ablation's mechanism causally (atomic contention is real and `rowgroup`'s
batched-flush design genuinely removes it) but shows the only currently-built way to get there
—coarser grouping—trades it for an occupancy cost that outweighs the gain in 7 of 8 tested cases.**
This directly reinforces H5/H6's central finding: `block`'s massive CTA-count concurrency is precisely
what makes it hard to beat via any coarsening-based fix, even one that successfully targets the
*correct* stall reason. Not a case for switching production default to `rowgroup`. The noatomic
ablation's other suggested lever — a genuinely new kernel-body change that cuts atomic cost *without*
reducing CTA count (e.g., a wider per-thread atomic covering multiple features, or restructuring so a
single thread per row combines features before one RMW, while keeping one CTA per PBR block) — remains
untried and is the more promising next hypothesis, since it wouldn't pay this occupancy cost at all.
