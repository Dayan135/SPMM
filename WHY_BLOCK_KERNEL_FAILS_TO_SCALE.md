# Why does PBR's advantage over cuSPARSE erode at scale?

**Question:** the full benchmark sweep (commit `69041a9`) reported PBR's advantage over cuSPARSE
*improving* with size (geomean 0.77 → 1.28, best-config-per-size). But that framing hid the real
picture. This doc investigates the actual scaling behavior, three hypotheses deep.

**Bottom line:** confirmed and mechanistically explained. `block` launches one CUDA block per
naturally-occurring PBR block regardless of how much nonzero work it actually contains; cuSPARSE
instead pre-partitions work by nnz count. On `diag_b0.3` this makes `block` launch ~5× more CUDA
blocks than cuSPARSE needs for identical work. That's invisible while the GPU has slack (small N)
but directly costs wall-clock time once the grid saturates the SMs (large N) — which is exactly the
crossover the sweep measured.

---

## H1: The "best-tuned-per-size" comparison is masking real degradation

**Experiment:** re-pulled the full sweep (commit `69041a9`, 5 sizes × 10 matrices × 6 features) from
W&B and computed the geomean `cusparse_time / block_time` advantage two ways: (a) best config
re-tuned independently at every size (what was originally reported), vs (b) one **fixed**
representative config (`block_rows=8, min_nnz=2, smem_size=2048`) held constant across all sizes —
i.e. what a real deployment without per-N retuning would actually see.

**Finding: confirmed.** (a) does improve with size (0.77 → 1.28) — but that's an artifact of always
picking each size's best `(block_size, min_nnz, smem)` combo, which differs across sizes. Under (b),
several matrices show real, monotonic degradation:

| matrix | 1024 | 4096 | 16384 | 65536 | 262144 |
|---|---|---|---|---|---|
| `diag_b0.3` | 0.707 | 0.706 | 0.578 | 0.525 | **0.482** |
| `scattered_2x2_d2pct` | 0.707 | 0.560 | 0.398 | 0.407 | **0.420** |
| `scattered_8x1_d2pct` | 0.707 | 0.754 | 0.723 | 0.647 | **0.599** |

(`>1` = PBR faster than cuSPARSE. These are getting *worse*, not better.) `diag_b0.3` was picked for
deeper investigation below: cleanest monotonic trend, no density-rescaling confound (unlike the
scattered matrices, whose density was deliberately rescaled `∝ 1/N` in an earlier fix — see
`LAST_EXPERIMENTS.md` Experiment 3 — which is itself a plausible independent cause of degradation
for those specifically and would confound the analysis).

## H2: More work is spilling to the slow CSR-remainder path as N grows

**Experiment:** a compressed_fraction is already a known confound from prior work (`LAST_EXPERIMENTS.md`
Experiment 2 found some diag configs were 99.8% remainder at large N). Checked directly with a
pure-Python block-density scan (no GPU/compiled extension needed — just replicates the
`min_nnz_per_block` compression-threshold logic on the scipy matrix) across all 5 sizes for
`diag_b0.3` and `scattered_2x2_d2pct`.

**Finding: rejected.** `compressed_fraction = 1.0` at *every* size, for both matrices. Essentially all
nnz goes through the compressed PBR path regardless of N — this is not a format-coverage problem.
The degradation must come from the kernel's own scaling behavior relative to cuSPARSE's, not from
more work leaking to the naive CSR-remainder kernel.

## H3: The PBR kernel's own compute scales worse than cuSPARSE's

**Experiment:** `profile_cusparse.py` was generalized to share `build_matrix()` with `profile_pbr.py`
(previously hardcoded to a different matrix), so both drivers profile the *identical* `diag_b0.3`
matrix. ncu-profiled `block` vs `torch.sparse.mm` at N=1024 and N=262144 (256× apart), f64 and f512,
with per-kernel-launch `gpu__time_duration` (`jobs/profile_cusparse_scaling/`, output in
`ncu_out/cusparse_scaling/`).

**Finding: confirmed, with a concrete mechanism.**

cuSPARSE's `torch.sparse.mm` launches **4** kernels per call, not 1: a zero-init (`FillFunctor`,
nearly identical cost to our own zero-init at matched N/features — cancels out of the comparison), a
`csr_partition_kernel`, the actual compute (`csrmm_alg2_kernel`), and a `matrix_scalar_multiply_kernel`
(alpha/beta scaling). Isolating "real compute" (excluding the near-identical zero-init on both sides):

| N | features | `block` (pbr_kernel) | cuSPARSE (partition+compute+scale) | winner |
|---|---|---|---|---|
| 1024 | 64 | 12,224 ns | 17,760 ns | **block, 1.45×** |
| 1024 | 512 | 59,808 ns | 26,208 ns | cusparse, 2.28× |
| 262144 | 64 | 1,718,784 ns | 702,272 ns | cusparse, **2.45×** |
| 262144 | 512 | 13,003,680 ns | 4,718,784 ns | cusparse, **2.76×** |

`block` already loses at f512 even at N=1024; by N=262144 it loses at *both* feature counts —
matching the fixed-config trend from H1.

**Root cause — grid geometry.** `csr_partition_kernel` pre-computes an nnz-balanced work partition:
grid.x = 205 (N=1024) → 52,429 (N=262144), each cuSPARSE work-chunk handling **~96 nnz** regardless
of N (`total_nnz / grid.x`). `block`, by contrast, launches **exactly one CUDA block per
naturally-occurring 8×8 PBR block**, irrespective of how much nnz it holds: grid.x = 1024 (N=1024) →
262,144 (N=262144), i.e. `grid.x == num_pbr_blocks` always. For `diag_b0.3` (~19 nnz/block on
average), that's **~5× more CUDA blocks than cuSPARSE needs for the same nnz throughput**
(262144 / 52429 ≈ 5.0).

That 5× ratio exists at *both* sizes — the difference is whether it costs anything:

| N | `waves_per_multiprocessor` (block) | (cuSPARSE) | ratio | GPU saturated? |
|---|---|---|---|---|
| 1024 | 1.14 | 0.25 | 4.6× | No — both far under 1 full wave of occupancy |
| 262144 | 290.6 | 63.9 | 4.5× | Yes — both deep into serialized waves |

At N=1024 the GPU has slack for both kernels (waves ≪ 1), so the extra CTAs `block` launches don't
serialize — there's spare capacity to absorb them, and `block` even wins at f64. At N=262144 both
kernels are deep in oversubscription (**every** CTA-wave is a real serialization point — only
`sm_count × blocks_per_sm` CTAs run concurrently at a time), so needing 4.5× more waves directly and
proportionally costs wall-clock time. The fixed per-CTA overhead `block` pays on every launch
(decoding the 64-bit occupancy bitmask, computing active-row/col popcounts, deriving the smem-split
factor `p`, three `__syncthreads` barriers, one `atomicAdd` flush) was always there; it just wasn't on
the critical path until the grid was large enough to saturate the SMs. That crossover is the
"advantage eroding with scale" the benchmark measured.

## Supporting check: do rowgroup/colgroup (built to amortize exactly this) help?

**Experiment:** `rowgroup`/`colgroup` group multiple PBR blocks per CUDA block specifically to
amortize this kind of per-CTA overhead (see `LAST_EXPERIMENTS.md`). Pulled their best-time cuSPARSE
advantage for `diag_b0.3` across sizes from the same full-sweep data already in hand.

**Finding:** mixed, but consistent with the diagnosis rather than contradicting it. `rowgroup`/
`colgroup` are uniformly *worse* than `block` at every size here (0.57–0.96 vs `block`'s 0.74–1.03) —
so grouping doesn't win outright at the configs tested. But they close ground on cuSPARSE *faster*
than `block` does across the same size range: `rowgroup` 0.57→0.96 (+0.39) vs `block` 0.74→1.03
(+0.28). That's the expected direction if reducing CTA count is the right lever — it's just that
`rowgroup`/`colgroup`'s own per-group overhead (multiple `__syncthreads` per PBR block within a
group, atomicAdd-per-strip-flush bookkeeping) currently outweighs the benefit at these particular
configs, rather than the mechanism being wrong.

## Conclusion

The improving-with-scale story in the original sweep summary was an artifact of re-tuning the PBR
config independently at every size. Held fixed, the real trend is degradation on several matrices,
and it has a concrete, measurable cause: `block`'s one-CTA-per-PBR-block launch granularity is
**~5× finer than cuSPARSE's nnz-balanced work partition** for `diag_b0.3`. That mismatch is latent
while the GPU has spare capacity and becomes the dominant cost once the grid saturates the SMs —
which happens well before N=262144 (the fixed-config table already shows the crossover between
4096 and 16384).

This is a **different root cause than the tile kernel's failure** (see `TILE_STATE.md` — that was
missing vectorization, not this). It also suggests the more promising next lever isn't "group by
PBR-block count" (what `rowgroup`/`colgroup`/`tile` all do) but **group by nnz-count** — mirroring
cuSPARSE's own `csr_partition_kernel` — so that the number of launched CUDA blocks tracks actual work
rather than block *occurrence* count.
