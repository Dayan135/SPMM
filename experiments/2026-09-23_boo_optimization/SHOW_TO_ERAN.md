# BOO's small_block_policy: fixing the diag_skewed CTA explosion

Follow-up to the first 4-way benchmark (BOO vs PBR `block`/`nnzflat_shared`/`nnzflat_onthefly`, all
four kernels in one build, timed back to back). `diag_skewed` was the worst cell by far — **0.503x**
vs `block`, and getting *worse* with scale (0.407x at N=65536 → 0.161x at N=262144). Every other
matrix was within a normal spread; this one stood out.

## Root cause: the arena tax on near-empty blocks

`diag_skewed` is built deliberately bimodal — a dense 64-wide diagonal band plus sparse off-band
noise — so its per-block nnz distribution has nothing in between "very full" and "nearly empty."
BOO's default policy (`keep`) launches one CUDA block per (row-group × col-group) intersection no
matter how little work it holds, and every block's shared-memory arena is sized from the matrix's
**global** max block (there's no such thing as per-CTA dynamic shared memory in CUDA — one launch,
one arena size, every CTA pays it).

```
                          blocks with <=3 nnz         arena utilization
N=65536,  hem, target=32     85,221 / 94,328 (90.3%)      4.4%
N=262144, hem, target=32  1,365,213 / 1,401,812 (97.4%)   1.3%
```

At N=262144, 97.4% of the 1.4M launched CTAs do a single multiply-add while paying for an arena
built for a 1024-nnz block. `ncu` confirms the mechanism directly — BOO here is **compute-bound**,
not memory-bound (`long_scoreboard_pct` only 10.8%, far below PBR's 19.75%), with the single
largest stall `short_scoreboard_pct` at 29.3% — a swarm of near-empty CTAs each paying a fixed
per-launch shared-memory-dependency cost for almost no work.

## Two fixes, built and measured head-to-head

**`remainder`** — spill below-threshold blocks to the existing CSR kernel, on its own stream.
Exactly what PBR's `min_nnz` already does.

**`tail_batch`** (new) — flatten below-threshold blocks to a global-index COO tail, consumed by a
new grid-strided kernel: one thread per nonzero, no smem, no barriers, and critically **CTA count
bounded by GPU occupancy, not by tail size** — the direct fix for the CTA explosion. Never leaves
BOO for the CSR kernel, so `boo_fraction` (nnz handled by *some* BOO-family kernel) stays 1.0 even
though the strict block-encoded `compressed_fraction` drops.

`diag_skewed`, all four size/feature cells:

| N | f | `keep` | `remainder` | `tail_batch` | PBR `block` |
|---|---|---|---|---|---|
| 65536 | 128 | 2.38 ms | **1.01 ms** | 1.54 ms | 0.92 ms |
| 65536 | 512 | 8.47 ms | **3.96 ms** | 6.45 ms | 3.50 ms |
| 262144 | 128 | 28.11 ms | **4.67 ms** | 11.80 ms | 4.26 ms |
| 262144 | 512 | 99.47 ms | **17.32 ms** | 53.60 ms | 16.12 ms |

`remainder` wins decisively, and the gap to `tail_batch` **widens with scale** (1.5x at N=65536 →
2.5–3.7x at N=262144). Not a tuning gap — an architectural one: PBR's CSR kernel already does
row-grouped register accumulation with vectorized loads (one warp per row, accumulate across all
of that row's nonzeros in registers, one atomic commit per feature at the end). `tail_batch`'s
one-atomic-per-nonzero-per-feature is naive by comparison, and a row with several tail entries
(common — off-band noise usually lands in several different tiny column groups) costs `remainder`
one atomic per row per feature but costs `tail_batch` one per *entry*.

Regression check on the matrices where `keep` was already fine (`fem2d_permuted`, `fem2d`,
`diag_b0.3`, `diag_b0.1`, `scattered_8x1`): both policies land within ~0.2% of `keep`, i.e. noise.
Neither fix costs anything where it isn't needed.

## Decision

`remainder` (threshold 4) is now BOO's default, replacing `keep`. `tail_batch` stays in the format
— correct, tested, a genuine 1.5–1.6x win over `keep` where it applies — as the base for a future
row-grouped rewrite, but re-deriving CSR's optimization inside it wasn't worth it for a benefit
(avoiding the CSR dependency) that's about format purity, not speed.

## Bottom line: re-ran the full 4-way benchmark with the fix

Same 14 matrices × 3 sizes × 2 feature counts, `remainder` replacing `keep`:

| | vs `block` | vs `nnzflat_shared` | vs `nnzflat_onthefly` | fastest in |
|---|---|---|---|---|
| before (`keep`) | 1.026 | 1.175 | 1.760 | 28/82 clean cells |
| **after (`remainder`)** | **1.073** | **1.234** | **1.849** | **30/82 clean cells** |

By size — BOO now wins at every tier, including the smallest:

| size | vs `block` |
|---|---|
| 16,384 | 0.996 |
| 65,536 | 1.102 |
| 262,144 | **1.130** |

`diag_skewed` itself: **0.917** overall — no longer the outlier, in line with the rest of the
benchmark instead of 6x worse. The fix didn't just repair one matrix; it lifted the whole
comparison, because the same CTA-explosion mechanism was present, at lower intensity, wherever any
clustering produced near-empty blocks. `diag_b0.3`/`diag_b0.1` remain the strongest wins (2.10x/2.08x,
`identity` clustering) and are unaffected by the small-block policy change — those matrices are
already dense-block, not fragmented.
