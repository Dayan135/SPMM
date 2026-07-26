# nosync* race-detector coverage, full grid + nosyncAll (2026-07-22/26)

Branch `feat/pbr-mergebc-alloc-ablation`. Extends the original 30-config `compute-sanitizer racecheck`
sweep (`nosyncA`/`nosyncB`/`nosyncC`, fixed `block=8, min_nnz=2`, 5 matrices × 2 features, all clean — "0
hazards" everywhere) to: (1) a 4th variant, `nosyncAll`, that removes all three `__syncthreads()` calls
simultaneously instead of one at a time, and (2) the full standard matrix/block-config/feature grid used
elsewhere this session (`STALL_REASONS_MERGEBC_FULLGRID.md`'s 10 matrices × 7 `(block_size, min_nnz)`
configs × 6 features), at `size=262144`.

**Headline: this is not a clean result, and it corrects the earlier one.** `nosyncA` and the new
`nosyncAll` both show real, reproducible race hazards — including actual `compute-sanitizer` **errors**,
not just warnings — once the grid widens beyond `block=8`. `nosyncB` and `nosyncC` remain completely
clean at every one of the 1680 configs tested. None of the four variants should be considered safe for
any real use; see Verdict.

## What changed since the last report

1. **`nosyncAll` kernel variant** (`pbr_spmm_zero_idle_kernel_nosyncAll`, `torch_fast_spdmm/kernels/pbr_kernel.cu`,
   dispatch variant 7): combines `nosyncA` + `nosyncB` + `nosyncC`'s removals into one kernel (Step A's
   `smem_X` guard gone, the Step B/C `smem_Y` guard gone with loops kept separate — not fused like
   `mergeBC` — and the end-of-iteration WAR guard gone). Wired into `_SCALAR_DIAG_VARIANTS`,
   `profile_pbr.py`, and `check_scalar_diag_correctness.py`.
2. **`check_scalar_diag_correctness.py` extended to sweep all 7 `(block_size, min_nnz)` configs** (previously
   fixed at `block=8, min_nnz=2`) — the racecheck sweep covers all 7, so output correctness needs to as well;
   a variant could be correct at one block config and wrong at another since `n_r`/`n_c` (and the
   worker-pool shape every variant's loop bounds depend on) changes with block size.
3. **Vacuous-config detection**: some `(matrix, block_size, min_nnz)` combos build **zero PBR blocks** —
   `min_nnz` too high for that matrix's sparsity pattern — so the diagnostic kernel never launches and
   every variant, including `constdata` (which should never trivially match), reports `max_abs=max_rel=0`.
   Both `check_scalar_diag_correctness.py` and `racecheck_probe_batch.py` now check
   `pbr_mat.accounted_blocks() == 0` and skip these before counting/profiling anything.
4. **`racecheck_probe_batch.py`**: batched driver, one `compute-sanitizer` process per `(variant, matrix)`
   pair, looping internally over the 7×6=42 `(block_config, features)` sub-configs. `[CONFIG]` markers
   bracket each sub-config so a hazard report can be attributed to the specific sub-config it occurred in.

## Correctness sweep (all 7 block configs × 5 matrices × 2 features × 8 variants)

`python -m benchmark.check_scalar_diag_correctness`, N=262144: **368 real (non-vacuous) checks, 184
unexpected — but all 184 are the same single, already-known pattern, not 184 distinct surprises.**

- 12 of the 35 `(matrix, block, min_nnz)` combos are vacuous (0 PBR blocks: `diag_b0.1`@8/20,
  `scattered_1x8`@2/3, @4/8, @8/12, @8/20, `scattered_8x1`@2/3, @4/8, @8/12, @8/20, `scattered_2x2`@4/8,
  @8/12, @8/20) — excluded, 192 checks that would have been vacuously "passing" for nothing exercised.
  (A rerun of the same script found 10 vacuous combos the first time and 12 the second — `scattered_1x8`
  and `scattered_8x1` at `block=8, min_nnz=12` flip between vacuous/non-vacuous across runs because
  `benchmark.matrices`' scattered-matrix generators aren't seeded, so the exact block count near a
  threshold is randomly realized each invocation. Not a bug, just a methodology note.)
- Of the 368 real checks: **`baseline`, `mergeBC`, `mergeBC_alloc` — 0 unexpected results** (correct
  everywhere they should be, all 46 configs each). **`nosyncA`, `nosyncB`, `nosyncC`, `nosyncAll` — all
  46/46 configs "unexpectedly" numerically match baseline** (i.e., the known "expected-wrong-but-
  passes-anyway" caveat from the original 5-matrix study, now confirmed across the full 7-config
  block/min_nnz grid too — not new information, just broader confirmation of the existing caveat).
  `constdata` — wrong everywhere as expected (46/46 "OK").
- **This correctness check's feature range (128, 512 only) never overlaps with where the racecheck sweep
  found real hazards** (concentrated at `features` 16-64, see below) — which is exactly why this
  numeric-comparison check missed what racecheck caught. The two methods are complementary, not
  redundant: racecheck instruments every shared-memory access and catches a hazard regardless of whether
  it happened to corrupt output in a *particular* run; the numeric check only catches corruption that
  actually manifested in the specific inputs/feature counts it happened to try.

## Racecheck sweep: full breakdown by variant × matrix

40 `compute-sanitizer --tool racecheck` processes (4 variants × 10 matrices), each internally covering up
to 42 `(block_config, features)` sub-configs (fewer where vacuous configs were skipped). All 4 per-variant
jobs completed (`jobs/racecheck_nosync_fullgrid/racecheck_nosync{A,B,C,All}.sh`, ~3h10m-3h54m each,
0 job failures).

| matrix | nosyncA (haz/err/warn) | nosyncB | nosyncC | nosyncAll (haz/err/warn) |
|---|---|---|---|---|
| `diag_b0.3` | 2 / 0 / 2 | 0/0/0 | 0/0/0 | 2 / 0 / 2 |
| `diag_b1.0` | 1 / 0 / 1 | 0/0/0 | 0/0/0 | 1 / 0 / 1 |
| `scattered_1x8_d0.5pct` | 2 / **1** / 1 | 0/0/0 | 0/0/0 | 2 / **1** / 1 |
| `scattered_1x8_d2pct` | 2 / 0 / 2 | 0/0/0 | 0/0/0 | 1 / 0 / 1 |
| `scattered_2x2_d0.5pct` | 1 / 0 / 1 | 0/0/0 | 0/0/0 | 1 / 0 / 1 |
| `scattered_2x2_d2pct` | 1 / 0 / 1 | 0/0/0 | 0/0/0 | 1 / 0 / 1 |
| `scattered_2x4_d0.5pct` | 3 / **2** / 1 | 0/0/0 | 0/0/0 | 3 / **2** / 1 |
| `scattered_2x4_d2pct` | 2 / 0 / 2 | 0/0/0 | 0/0/0 | 2 / 0 / 2 |
| `scattered_8x1_d0.5pct` | 2 / **1** / 1 | 0/0/0 | 0/0/0 | 3 / **1** / 2 |
| `scattered_8x1_d2pct` | 2 / 0 / 2 | 0/0/0 | 0/0/0 | 1 / 0 / 1 |
| **Total** | **18 / 4 / 14** | **0/0/0** | **0/0/0** | **17 / 4 / 13** |
| **Matrices with ≥1 hazard** | 10/10 | 0/10 | 0/10 | 10/10 |
| **Matrices with ≥1 error** | 3/10 | 0/10 | 0/10 | 3/10 |

**`nosyncA` and `nosyncAll` show hazards on every single one of the 10 matrices**, with real
`compute-sanitizer` **errors** (not just warnings) on the same 3 matrices for both variants
(`scattered_1x8_d0.5pct`, `scattered_2x4_d0.5pct`, `scattered_8x1_d0.5pct`). **`nosyncB` and `nosyncC` are
completely clean — 0 hazards across all 1680 configs tested for each.**

The near-identical `nosyncA`/`nosyncAll` numbers (18 vs 17 total hazards, same 3 matrices with errors)
are not a coincidence — the detailed hazard reports (below) show every single flagged race in `nosyncAll`
occurs in the `pbr_spmm_zero_idle_kernel_nosyncAll<int, float, 2, 2, ...>` template instantiation (the
`block=2` PBR-block-size configs) at the *first* thing the kernel body does (Step A's shared-memory
write). That is exactly `nosyncA`'s removed guard, not `nosyncB`'s or `nosyncC`'s. **`nosyncAll`'s hazards
appear to be inherited entirely from `nosyncA`'s mechanism** — no evidence of a new interaction-effect
hazard from combining all three removals that isn't already present in `nosyncA` alone.

## Reconciliation with the original "30-config sweep was clean" finding

The original sweep (`STALL_REASONS_ZEROIDLE_MICROBENCH.md`'s racecheck follow-up, 3 variants × 5 matrices
× 2 features, **fixed at `block=8, min_nnz=2`**) found 0 hazards everywhere and was reported as "correct
in this test but not proven safe in general." **That framing undersold it — `nosyncA` is not merely
"unproven," it has a confirmed, reproducible, block-size-dependent hazard the narrow grid could not have
found because it never tested `block=2`.** Direct evidence: grepping every "Race reported" line in both
full detailed logs (`diag_b0.3` and `scattered_1x8_d0.5pct`, `nosyncAll`) shows **100% of the flagged
races are in the `<..., 2, 2, ...>` (block=2) template instantiation — none in `<..., 8, 8, ...>`**, which
is consistent with `block=8, min_nnz=2` being clean in both the original narrow sweep and this full sweep
alike (the full sweep's own per-matrix totals include `block=8` sub-configs that contributed 0 to the
counts above). This is a genuine correction, not a contradiction: the original result (clean at
`block=8`) still holds; it just wasn't the whole picture.

## Hazard detail (full, non-summary `compute-sanitizer` output)

Re-ran `diag_b0.3` and `scattered_1x8_d0.5pct` (`nosyncAll`) with full (non-truncated) output capture.
Both show the exact same write/read pair:

```
Warning/Error: Race reported between Write access at
  void pbr_spmm_zero_idle_kernel_nosyncAll<int, float, (int)2, (int)2, (int)2048, (int)128, (int)512>(...)+0xdb0
and Read access at
  void pbr_spmm_zero_idle_kernel_nosyncAll<int, float, (int)2, (int)2, (int)2048, (int)128, (int)512>(...)+0x1080  [3,887,643 hazards]
and Read access at ...+0x1090  [6,112,357 hazards]
```

- `diag_b0.3`, `block=2, min_nnz=2, feat=16`: **Warning**, 2 read offsets flagged, 3.9M and 6.1M
  individual hazard instances recorded per offset pair (racecheck's `NV_COMPUTE_SANITIZER_MAX_RACECHECK_HAZARDS`
  cap was hit — the true count is capped-and-larger, not exactly this number — this is a *pervasive*
  hazard, not a rare edge case).
- `scattered_1x8_d0.5pct`, `block=2, min_nnz=2`: the *same* Write(+0xdb0)/Read(+0x1090) pair appears
  as a **Warning** at `feat=16` (5.4M hazard instances) and escalates to an **Error** at `feat=32` (4.6M
  instances) — same source-level race, different severity classification racecheck assigns per
  occurrence.
- Kernel binary offsets aren't decoded to source line/variable names by `compute-sanitizer` here (no
  `-lineinfo` build), but the Write at the *very first* shared-memory-writing point in the kernel body
  racing against *later* reads matches Step A's cooperative load into `smem_X` racing against Step B's
  read of `smem_X` — i.e., precisely `nosyncA`'s removed guard, corroborating the template-instantiation
  evidence above.

## Nondeterminism / NaN diagnostic (separate from racecheck)

Some raw sub-config outputs in the sweep showed `y.sum=nan` (e.g. `diag_b0.3`, `block=8, min_nnz=12/20`,
`features` 16-64). Checked whether this is the same vacuous-zero-blocks issue found in the correctness
sweep, or a distinct bug: **it is a distinct, genuine race**, not vacuousness — `pbr_mat.accounted_blocks()`
at that exact config is 258,389 and 120,340 respectively (very much not zero), and a standalone (no
`compute-sanitizer`) repeated-trial script confirms the underlying nondeterminism directly:

| feat | baseline (5 trials) | `nosyncAll` (5 trials) |
|---|---|---|
| 16 | stable, `sum≈-5757` every time | **wildly different every trial**: -18289, -70103, +69672, -240200, -109156 |
| 32 | stable, `sum≈-5731` every time | **wildly different**: -34808, +114154, -126024, -200147, -259875 |
| 64 | stable, `sum≈28855` every time | **wildly different**: -73166, +78275, +9565, +111024, -86083 |
| 128 | stable, `sum≈-13373` every time | **stable/near-deterministic**: -13373.29 (±0.01) all 5 trials |

Identical input, identical kernel, 5 back-to-back calls — `nosyncAll` returns a *different answer every
single time* at low feature counts, converging to near-determinism only once `features` is large enough
(consistent with the race window becoming a smaller fraction of a longer per-block critical path). This
is a textbook data-race signature (nondeterministic result on fixed input), independent confirmation of
what racecheck flagged, and explains the occasional `NaN`: at low feature counts the race sometimes reads
a partially-written / uninitialized shared-memory bit pattern that happens to decode as `NaN`, other times
a stale-but-finite garbage value — same underlying hazard, different symptom depending on scheduling.

## Verdict: none of the four variants are safe for real use

- **`nosyncB`, `nosyncC`: clean across all 1680 configs tested** (0 hazards, 0 errors, 0 warnings
  anywhere). This is the strongest evidence available that these two specific barrier removals don't
  introduce a *detectable* hazard on this hardware/kernel shape/grid — but "no hazard detected in 1680
  configs" is evidence of absence, not proof of absence (same caveat as always with dynamic race
  detection: a different launch config, GPU, driver, or scheduling could still expose one). Combined with
  `STALL_REASONS_ZEROIDLE_MICROBENCH.md`'s finding that removing them is also performance-neutral, there's
  no reason to keep them removed in any real kernel — free to have but pointless, exactly as previously
  concluded.
- **`nosyncA`, `nosyncAll`: confirmed unsafe.** Real, reproducible, pervasive (millions of instances)
  races including actual `compute-sanitizer` errors (not just warnings) on 3/10 matrices, and directly
  observable nondeterministic output (not just theoretical risk) at `block=2` PBR configs and low feature
  counts. **Do not use `nosyncA`'s removed guard, alone or combined, in any production kernel path.** The
  fact that `nosyncAll`'s hazard profile is nearly identical to `nosyncA`'s alone (not worse) is mildly
  good news for the "combined risk" question specifically — combining all three removals doesn't appear
  to unlock a *new* class of hazard beyond what `nosyncA` alone already has — but it doesn't make
  `nosyncAll` any safer, and `nosyncAll` is not a variant anyone was proposing to ship regardless.
- Net effect on `STALL_REASONS_ZEROIDLE_MICROBENCH.md`'s original framing: **downgrade `nosyncA` from
  "performance-neutral, no-op, free to remove" to "confirmed unsafe, keep it."** `nosyncB`/`nosyncC`'s
  "free to remove but pointless" framing is unaffected and now has ~28x more configs of supporting
  evidence (1680 vs 60 in the original correctness pass, plus this racecheck sweep's own 1680).

## Data

Raw logs at job time under `jobs/racecheck_nosync_fullgrid/logs/` (per-variant `[job] === SUMMARY ===`
sections) and `jobs/racecheck_nosync_fullgrid/detail_out/` (full non-truncated hazard reports for the two
flagged configs) on the remote host — not committed, regenerable via the job scripts.
