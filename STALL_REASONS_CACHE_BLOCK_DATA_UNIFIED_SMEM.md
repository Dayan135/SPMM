# block_data cache, redesigned: unified smem budget instead of additive allocation (2026-07-30)

Branch `feat/pbr-cache-block-data-unified-smem`, forked from `feat/pbr-cache-block-data`. Follow-up to
`STALL_REASONS_CACHE_BLOCK_DATA.md` (the original `s_block_data` cache: a separate, additive
`__shared__` array on top of `smem[TOTAL_SHARED]`, found to be a pure occupancy tax in every config —
`launch__occupancy_limit_shared_mem` regressed unconditionally since the extra allocation never traded
off against anything, netting "whack-a-mole": `long_scoreboard` genuinely dropped from removing
redundant `__ldg` reads, but `barrier`/occupancy loss absorbed the gain, net wall-clock flat-to-negative).

## The redesign

Instead of a separate array, `s_block_data` is now carved out of the **front of the same
`smem[TOTAL_SHARED]` budget** `smem_X`/`smem_Y` already share — `p` (the feature-tile width) is computed
over the *remaining* budget after `block_data`'s reservation, instead of the full budget. Static
shared-memory footprint is therefore **identical to baseline at every config** (verified directly, see
below); the cost of caching `block_data` is paid entirely by a smaller `p` (more `base_f` grid-stride
passes, more repeated `smem_X` staging) instead of by occupancy. Applied to both
`pbr_spmm_zero_idle_kernel` and `pbr_spmm_vec_kernel` (mirrors the original branch's scope); the vec
kernel additionally pads `total_nnz` up to the next multiple of `L` (`total_nnz_aligned`) before shifting
`smem_X`/`smem_Y`, since their vectorized `float4`/`double2` access requires 16-byte alignment and
`total_nnz` is an arbitrary sparsity-scan-dependent count. Full details and the documented (comfortably
unhit) overflow invariant are in the kernel comments, `torch_fast_spdmm/kernels/pbr_kernel.cu`.

## Correctness

`pytest tests/torch_fast_spdmm/test_pbr.py tests/torch_fast_spdmm/test_csr.py -v` (the real correctness
suite — `test_benchmark.py` is the 79k-item GPU benchmark sweep, not a correctness gate, per this
session's earlier finding): **3704 passed, 1 pre-existing unrelated warning, in 9.3s.** Covers both the
scalar and vectorized kernel paths across block sizes 2/4/8 and multiple feature counts.

## Method

Three-way comparison, same grid as the original single-shot occupancy/stall-reason study
(`jobs/profile_smem_block_data/`): matrices `{diag_b0.3, scattered_1x8, scattered_8x1}` × blocks
`{2,4,8}` × `smem_size` `{512,1024,2048}` × features `{129,255,517}` (non-multiple-of-4, so
`kernel="block"`'s `use_vec` auto-select forces `pbr_spmm_zero_idle_kernel`, not the vec kernel), size
262144 — but **N=5 repeated trials this time**, not single-shot. Two separate `ncu` passes per rep
(occupancy metrics, stall-reason metrics), never mixed, per this project's established methodology note.
81 configs × 2 passes × 5 reps = 810 `ncu` invocations per arm (baseline / additive / unified), 2430 total.

**Infrastructure note, worth recording because it nearly produced a wrong result:** running baseline,
additive, and unified as three concurrent jobs against the shared `fem` conda env hit a real,
non-obvious contamination bug. `pip install -e` registers into environment-global site-packages
regardless of source directory; `flock`-serializing the install step alone is *not* sufficient — the
registration is a global, mutable, shared resource for the rest of every job's runtime, so a sibling
job's *later* install silently overwrites it mid-run even after a job's own one-time post-install check
passed. Confirmed empirically (a live diagnostic mid-sweep showed `torch_fast_spdmm.core.__file__`
resolving to a different clone's `.so` well after the affected job's own install had completed). Fixed by
exporting `PYTHONPATH="$SLURM_SUBMIT_DIR/torch_fast_spdmm:$PYTHONPATH"` after install — durable per-process
isolation that survives sibling jobs' later installs, verified both before/after the full ~1hr profiling
loop. Separately, running the `unified` arm out of the shared default clone (`/home/dayanb/research/SPMM`,
used throughout this session for many other tasks) caused 422/810 mid-run failures when something else
checked out a different branch there concurrently; fixed by giving `unified` its own dedicated clone
(`SPMM-unified-smem-cmp`), matching the isolation already used for `baseline`/`additive`. Final data below
is from the fully isolated, 0-failure rerun of all three arms.

## Occupancy: the core mechanism claim, confirmed directly

At `diag_b0.3`, every `(block, smem_size)` combination:

| block | smem | baseline `smem_static`/`occ_limit_smem` | additive | unified |
|---|---|---|---|---|
| 2 | 512  | 2048 / 21 | 2064 / 20 | **2048 / 21** |
| 2 | 1024 | 4096 / 12 | 4112 / 12 | **4096 / 12** |
| 2 | 2048 | 8192 / 11 | 8208 / 10 | **8192 / 11** |
| 4 | 512  | 2048 / 21 | 2112 / 20 | **2048 / 21** |
| 4 | 1024 | 4096 / 12 | 4160 / 12 | **4096 / 12** |
| 4 | 2048 | 8192 / 11 | 8256 / 10 | **8192 / 11** |
| 8 | 512  | 2048 / 21 | 2304 / 19 | **2048 / 21** |
| 8 | 1024 | 4096 / 12 | 4352 / 12 | **4096 / 12** |
| 8 | 2048 | 8192 / 11 | 8448 / 10 | **8192 / 11** |

`additive`'s `smem_static` is exactly `baseline + BLOCK_ROWS×BLOCK_COLS×4` bytes at every config (16/64/256
bytes for block 2/4/8) — genuinely additive, as designed, and it costs 1-2 occupancy blocks/SM in 6/9
cases here. `unified`'s `smem_static` and `occ_limit_smem` are **bit-identical to baseline at every single
config** — the redesign's core claim holds exactly as intended; the occupancy tax is fully eliminated.

## Wall-clock: win/loss breakdown (±3% WIN/REGRESSION/WASH)

Positive % = right-hand side faster. 81 configs per comparison.

| comparison | geomean | median | WIN | REGRESSION | WASH |
|---|---|---|---|---|---|
| additive vs baseline | −1.90% | −0.7% | 13 (16.0%) | 29 (35.8%) | 39 (48.1%) |
| **unified vs baseline** | **−3.29%** | −2.9% | 5 (6.2%) | 38 (46.9%) | 38 (46.9%) |
| **unified vs additive** | **−1.41%** | −0.9% | 10 (12.3%) | 17 (21.0%) | 54 (66.7%) |

**Headline: no. Shrinking `p` is not a cheaper trade than the occupancy tax was — on net it's a bit more
expensive.** `unified` is slower than `baseline` more often and by more than `additive` was (46.9%
REGRESSION vs 35.8%, −3.29% geomean vs −1.90%), and slower than `additive` itself on net too (−1.41%
geomean, 21.0% REGRESSION vs 12.3% WIN). Removing the occupancy tax did not translate into a net win —
it just swapped which mechanism does the losing.

### But the `smem_size`/matrix-density breakdown matches the hypothesis precisely

By `smem_size` (`unified` vs `additive`):

| smem_size | geomean | WIN | REGRESSION |
|---|---|---|---|
| 512  | −3.60% | 0 (0.0%) | 9 (33.3%) |
| 1024 | −2.26% | 0 (0.0%) | 7 (25.9%) |
| **2048** | **+1.71%** | **10 (37.0%)** | **1 (3.7%)** |

By matrix (`unified` vs `additive`):

| matrix | geomean | WIN | REGRESSION |
|---|---|---|---|
| `diag_b0.3` (dense, up to 64 nnz/8×8 block) | **−4.73%** | 1 (3.7%) | 17 (63.0%) |
| `scattered_1x8` (sparse) | **+0.42%** | 6 (22.2%) | 0 (0.0%) |
| `scattered_8x1` (sparse) | **+0.17%** | 3 (11.1%) | 0 (0.0%) |

This is exactly the shape the redesign's own mechanism predicts: `unified`'s cost is `total_nnz`'s bite
out of the shared budget, shrinking `p`. At `smem_size=2048` there's enough slack that even a dense
block's `total_nnz` reservation barely dents `p` — `unified` wins clearly there (+1.71% geomean, 37% WIN,
essentially 0 regression risk). At `smem_size=512` there's much less slack, so the same reservation bites
harder — `unified` loses to `additive` there. Matrix density tells the same story more starkly:
`scattered_1x8`/`scattered_8x1` have tiny per-block `nnz` (little to reserve, `p` barely shrinks) and
`unified` is a small, clean, *zero-regression* win over `additive` on both (0/27 REGRESSION each). `diag_b0.3`
has up to 64 nnz per 8×8 block — the reservation is large enough that `p`-shrinkage costs more than the
occupancy recovery saves, and `unified` loses to `additive` badly (63% REGRESSION, worst single cell
−13.6% at `block=2, smem=512, feat=255`).

The coordinator's specific hypothesis ("especially at large `smem_size`... and small `block` size") is
**half-confirmed**: `smem_size` is the dominant, cleanly-monotonic driver (2048 flips the sign entirely).
`block` size alone is not — `unified` vs `additive` is slightly negative at block 2, 4, *and* 8 alike
(−1.5%, −1.5%, −1.2%) when averaged across all matrices/smem sizes; what actually matters is the
*resulting* `total_nnz` (a function of both block size and matrix density together, not block size in
isolation) relative to the available budget.

## Verdict

**Not a clear win to recommend over the original additive version, and neither variant beats leaving
`block_data` caching alone entirely** — both `additive` and `unified` are net-negative against plain
`baseline` in this measurement (−1.90% and −3.29% geomean respectively, both with REGRESSION cells
outnumbering WIN cells). The original `STALL_REASONS_CACHE_BLOCK_DATA.md` verdict ("whack-a-mole, not
recommended for merge") stands, and this redesign doesn't change the bottom line — it just relocates
where the "mole" pops up (occupancy → `p`-shrinkage) rather than eliminating whack-a-mole itself.

That said, the redesign is not a wash: it cleanly **wins over `additive`** specifically for
**sparse-block matrices at generous `smem_size`** (`scattered_1x8`/`scattered_8x1` at `smem=2048`: small,
consistent, zero-regression gains), and it succeeds completely at its stated mechanical goal (occupancy
parity with baseline, confirmed bit-for-bit). If `block_data` caching were ever revisited for a
production path, `unified`'s allocation strategy is strictly the more defensible design of the two
(never worse than baseline on the occupancy axis, unlike `additive`) — but the underlying idea of caching
`block_data` in shared memory at all remains a net loss on this grid regardless of which allocation
strategy carries it, and isn't worth carrying forward as-is either way.

## Data

Raw CSVs (2430 files: `{baseline,additive,unified}_{matrix}_b{block}_s{smem}_f{feat}_rep{0-4}.csv` × 2
passes) at job time under `ncu_out/cache_unified_smem/{occupancy,stall_reasons}/` on each of the three
remote clones (`/home/dayanb/research/SPMM-baseline-cache-cmp`, `SPMM-cache-block-data`,
`SPMM-unified-smem-cmp`) — not committed, regenerable via `jobs/profile_cache_unified_smem/profile_{baseline,additive,unified}.sh`.
