# Adaptive smem bypass for low-reuse PBR blocks — findings

Branch `feat/pbr-adaptive-smem-bypass` (off `origin/feat/pbr-nnzgroup`). One of three parallel
attacks on `block`'s `mio_throttle`/`barrier` stalls (see `STALL_REASONS.md`); the other two are
`feat/pbr-smemy-capped-p` (negative) and `feat/pbr-warp-sync` (wash).

## What was built

`STALL_REASONS.md` found cuSPARSE executes **zero** shared-memory instructions on every matrix
tested — it reads `X` straight from global memory (L1/L2 cache), never staging into shared memory.
`block`'s `smem_X` staging exists to let multiple rows in a PBR block reuse the same loaded columns —
but when a block has very few populated rows (`n_r` small, e.g. `scattered_1x8`'s pattern, `n_r=1`),
each `smem_X` column is read back by only 1 row: zero actual reuse, pure overhead.

Added a CTA-uniform runtime branch (`n_r` is identical for every thread in a CTA, so this isn't
per-thread divergent) in both `pbr_spmm_zero_idle_kernel`/`pbr_spmm_vec_kernel`: when `n_r <= 1`, skip
the cooperative `smem_X` load (Step A) and its `__syncthreads()` entirely, and read `X` directly via
`__ldg` in Step B's per-row column loop instead — the same access pattern cuSPARSE uses. The `n_r > 1`
path is untouched. Correctness: **3704/3704 tests pass.**

## Methodology note: a real measurement bug caught and fixed

The first profiling pass mixed `sm__sass_inst_executed_op_shared.sum` into the same `ncu` invocation
as the 15 stall-reason metrics. This changed `ncu`'s replay-pass structure and corrupted the
stall-reason numbers — confirmed because `diag_b0.3` **never triggers this bypass at all** (a direct
census found 0% of its PBR blocks have `n_r<=1`, mean `n_r=7.5`), yet its profile still differed
substantially from baseline, while `launch__registers_per_thread` (40) and
`launch__waves_per_multiprocessor` (290.63) matched baseline exactly. Rerun with the corrected,
validated 15-metric list alone (`profile_adaptive_bypass_fixed_20260714.sh`) — see that job's header
comment for the full diagnosis.

Separately, the orchestrating session found and fixed a *second*, independent bug while
cross-checking this result: the shared `/home/dayanb/research/SPMM` remote clone (used for the
"why doesn't cuSPARSE have this problem" comparison) was still checked out at a pre-revert commit
(the failed `smem_Y`-merge experiment) and had never been rebuilt after the revert — so an earlier
reported baseline shared-instruction count for `block` (28,038,536 on `diag_b0.3`) was measured
against the wrong binary. The corrected, verified-current baseline is **43,842,784** (confirmed by
`barrier`=20.11%/`mio_throttle`=21.31% matching `STALL_REASONS.md` exactly on that same rebuild).
This correction doesn't change any conclusion below (all comparisons here use this branch's own
freshly-built binary against the validated `STALL_REASONS.md` table), but is noted for the record.

## Results (N=262144, corrected metrics)

| matrix/feat | time before→after | SM% before→after | `mio_throttle` before→after | `barrier` before→after |
|---|---|---|---|---|
| `diag_b0.1`/128 | 2.013→2.039ms (+1.3%) | 54.8→66.3 | 10.8→12.9 | 19.9→**0.0** |
| `diag_b0.1`/512 | 8.021→7.795ms (−2.8%) | 33.1→42.4 | 17.7→25.3 | 25.3→**0.0** |
| `diag_b0.3`/128 | 3.336→3.344ms (+0.2%) | 44.3→48.0 | 17.8→20.4 | 14.1→**0.0** |
| `diag_b0.3`/512 | 12.99→12.34ms (−5.0%) | 30.3→34.1 | 21.3→28.9 | 20.1→**0.0** |
| `scattered_1x8`/128 | 0.129→0.129ms (−0.1%) | 49.7→72.3 | 0.02→0.13 | 25.4→**0.0** |
| `scattered_1x8`/512 | 0.496→0.498ms (+0.4%) | 26.6→45.8 | 0.29→1.49 | 36.6→**0.0** |
| `scattered_2x2`/128 | 0.091→0.102ms (+12.9%) | 67.5→85.2 | 0.41→0.13 | 29.5→**0.0** |
| `scattered_2x2`/512 | 0.354→0.353ms (−0.2%) | 30.3→31.9 | 22.6→25.6 | 9.9→**0.0** |
| `scattered_8x1`/128 | 0.288→0.286ms (−0.6%) | 38.2→41.9 | 18.0→22.8 | 21.1→**0.0** |
| `scattered_8x1`/512 | 1.090→1.181ms (+8.4%) | 26.3→26.5 | 22.3→28.8 | 25.3→**0.0** |

`barrier` collapses to **exactly 0.00%** in all 10 configs — including `diag_b0.3`, where the bypass
never even triggers, and `scattered_2x2`/`scattered_8x1` (n_r=2/8, also outside the `n_r<=1` bypass
condition). This is reproducible (checked twice, isolated single-metric query and the full validated
list) and is not a measurement artifact — but the mechanism isn't fully understood: the un-bypassed
code path is logically unchanged, so this looks like a compiler/SASS-level side effect of adding a
CTA-uniform branch around the existing `__syncthreads()`, not a result of the bypass logic itself
(most of these matrices never take the bypass branch). Flagged as unexplained rather than claimed.

**Net effect: SM% improves substantially everywhere (up to +30 points), `barrier` vanishes
everywhere, but `mio_throttle`/`long_scoreboard` absorb the freed cycles in most configs, so
wall-clock time is a wash** — ranges from −5.0% to +12.9%, no config shows a clear win, several show
a real regression. `scattered_1x8` (the matrix this change specifically targets) shows the starkest
disconnect: SM% nearly doubles (26.6→45.8 at f512) yet time is flat (+0.4%) — consistent with
`scattered_1x8` being fundamentally memory-latency-bound (`long_scoreboard`=49.9% at baseline, rising
further to 70.5% after), so improving issue-slot efficiency doesn't help when DRAM latency is already
the real critical path.

## Verdict: negative/wash, same pattern as the other two approaches

Like `feat/pbr-smemy-capped-p` and `feat/pbr-warp-sync`, this removes a named stall reason
(`barrier`, and for the targeted low-`n_r` matrices, genuinely reduces shared-memory instruction
count too) without translating into a reliable wall-clock improvement. Freed cycles get reallocated to
`mio_throttle`/`long_scoreboard` rather than shortening the critical path. Not recommended for merge.
The unexplained "barrier vanishes even where the bypass never triggers" result may be worth a closer
SASS-level look (`cuobjdump --dump-sass`) if this line of investigation is revisited, but that is out
of scope here.
