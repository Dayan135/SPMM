# Why does PBR's advantage over cuSPARSE erode at scale?

**Question:** the full benchmark sweep (commit `69041a9`) reported PBR's advantage over cuSPARSE
*improving* with size (geomean 0.77 → 1.28, best-config-per-size). But that framing hid the real
picture. This doc investigates the actual scaling behavior, three hypotheses deep.

**Bottom line:** H1–H3 confirm and mechanistically explain the correlation — `block` launches ~5×
more CUDA blocks than cuSPARSE needs on `diag_b0.3`, invisible while the GPU has slack (small N) but
directly on the critical path once the grid saturates the SMs (large N). **H4 then built a
purpose-designed kernel to test whether that mechanism is *sufficient* on its own — and it isn't.**
Reducing CTA count in isolation (no other change) doesn't recover cuSPARSE's speed and gets *worse*
past a small chunk size, with DRAM traffic exploding up to 7.4× for identical total work. The real
gap is at least two mechanisms, not one: grid-size mismatch (confirmed) **and** cuSPARSE's genuinely
nnz-*balanced* (not just block-*count*-balanced) work distribution, which our naive chunking doesn't
replicate. See H4 for the full, honestly-reported (partially negative) result.

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

## Raw evidence: full ncu breakdown, every kernel launch

Complete per-kernel-launch data behind H3, `diag_b0.3`, all 4 (N, features) points, both engines.
Source: `ncu_out/cusparse_scaling/diag_b0.3_{N}_f{features}_{block,cusparse}.csv` (raw ncu `--csv`
output, `jobs/profile_cusparse_scaling/profile_20260712.sh`). `waves/SM` = `launch__waves_per_multiprocessor`
(CTA-waves needed to run the whole grid given this GPU's SM count and this kernel's occupancy);
`warps%`/`SM%`/`DRAM%` are `sm__warps_active`/`sm__throughput`/`dram__throughput`
`.avg.pct_of_peak_sustained_{active,elapsed}`; hit rates are `l1tex__t_sector_hit_rate.pct` /
`lts__t_sector_hit_rate.pct`.

### N=1024, features=64

| engine | kernel | grid | block | time (ms) | SM% | DRAM% | DRAM rd (MB) | DRAM wr (MB) | warps% | waves/SM | regs | smem (B) | L1 hit% | L2 hit% |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| block    | zero_init           | (64,1,1)    | (128,1,1) | 0.0023 |  0.77 |  0.21 |   0.00 |   0.00 |  8.21 |  0.07 | 16 |    0 |  0.00 | 98.93 |
| block    | **pbr_kernel**      | (1024,1,1)  | (128,1,1) | 0.0122 | 37.85 |  6.55 |   0.64 |   0.09 | 62.64 |  1.14 | 40 | 8192 |  8.61 | 94.63 |
| cusparse | zero_init           | (64,1,1)    | (128,1,1) | 0.0024 |  0.80 |  0.20 |   0.00 |   0.00 |  8.21 |  0.07 | 16 |    0 |  0.00 | 99.48 |
| cusparse | scalar_mul          | (1,256,1)   | (32,4,1)  | 0.0027 |  7.08 |  0.20 |   0.00 |   0.00 | 23.06 |  0.26 | 31 |    0 |  0.00 | 98.82 |
| cusparse | **csr_partition**   | (4,1,1)     | (256,1,1) | 0.0052 |  0.15 |  0.15 |   0.01 |   0.00 | 13.72 |  0.01 | 16 |    0 | 70.99 | 152.86 |
| cusparse | **csrmm_alg2**      | (205,1,1)   | (128,1,1) | 0.0098 | 23.64 |  7.72 |   0.66 |   0.02 | 18.92 |  0.25 | 44 |    0 | 43.61 | 81.12 |

### N=1024, features=512

| engine | kernel | grid | block | time (ms) | SM% | DRAM% | DRAM rd (MB) | DRAM wr (MB) | warps% | waves/SM | regs | smem (B) | L1 hit% | L2 hit% |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| block    | zero_init           | (512,1,1)   | (128,1,1) | 0.0035 |  4.17 | 30.26 |   0.00 |   0.94 | 30.50 |  0.52 | 16 |    0 |  0.00 | 101.74 |
| block    | **pbr_kernel**      | (1024,1,1)  | (128,1,1) | 0.0598 | 25.29 | 10.40 |   4.31 |   1.36 | 70.80 |  1.14 | 40 | 8192 |  3.38 | 94.64 |
| cusparse | zero_init           | (512,1,1)   | (128,1,1) | 0.0036 |  4.16 | 30.32 |   0.00 |   0.95 | 30.41 |  0.52 | 16 |    0 |  0.00 | 99.88 |
| cusparse | scalar_mul          | (4,256,1)   | (32,4,1)  | 0.0037 | 19.66 | 36.97 |   0.00 |   1.22 | 52.99 |  1.04 | 31 |    0 |  0.00 | 99.81 |
| cusparse | **csr_partition**   | (4,1,1)     | (256,1,1) | 0.0052 |  0.14 |  0.15 |   0.01 |   0.00 | 13.61 |  0.01 | 16 |    0 | 70.24 | 90.32 |
| cusparse | **csrmm_alg2**      | (205,4,1)   | (128,1,1) | 0.0173 | 55.69 | 34.21 |   4.02 |   1.36 | 68.00 |  1.00 | 44 |    0 | 40.05 | 85.53 |

### N=262144, features=64

| engine | kernel | grid | block | time (ms) | SM% | DRAM% | DRAM rd (MB) | DRAM wr (MB) | warps% | waves/SM | regs | smem (B) | L1 hit% | L2 hit% |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| block    | zero_init           | (16384,1,1) | (128,1,1) | 0.0772 |  6.03 | 93.89 |   0.01 |  65.94 | 89.84 |  16.65 | 16 |    0 |  0.00 | 99.99 |
| block    | **pbr_kernel**      | (262144,1,1)| (128,1,1) | **1.7188** | 67.31 | 14.44 | 159.63 |  66.54 | 88.06 | **290.63** | 40 | 8192 |  8.29 | 93.80 |
| cusparse | zero_init           | (16384,1,1) | (128,1,1) | 0.0771 |  6.01 | 93.94 |   0.01 |  65.93 | 89.90 |  16.65 | 16 |    0 |  0.00 | 99.99 |
| cusparse | scalar_mul          | (1,65535,1) | (32,4,1)  | 0.0772 | 55.66 | 93.77 |   0.01 |  65.94 | 73.01 |  66.60 | 31 |    0 |  0.00 | 99.99 |
| cusparse | **csr_partition**   | (820,1,1)   | (256,1,1) | 0.0138 | 24.06 | 11.06 |   1.05 |   0.33 | 80.51 |   1.67 | 16 |    0 | 75.69 | 57.29 |
| cusparse | **csrmm_alg2**      | (52429,1,1) | (128,1,1) | **0.6112** | 95.38 | 41.59 | 165.01 |  66.65 | 77.87 | **63.94** | 44 |    0 | 28.91 | 84.41 |

### N=262144, features=512

| engine | kernel | grid | block | time (ms) | SM% | DRAM% | DRAM rd (MB) | DRAM wr (MB) | warps% | waves/SM | regs | smem (B) | L1 hit% | L2 hit% |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| block    | zero_init           | (131072,1,1)| (128,1,1) | 0.6040 |  6.08 | 97.39 |   0.02 | 536.00 | 92.61 | 133.20 | 16 |    0 |  0.00 | 100.00 |
| block    | **pbr_kernel**      | (262144,1,1)| (128,1,1) | **13.0037** | 30.28 | 13.80 | 1099.41 | 536.24 | 91.00 | **290.63** | 40 | 8192 |  3.30 | 94.59 |
| cusparse | zero_init           | (131072,1,1)| (128,1,1) | 0.6055 |  6.08 | 97.10 |   0.04 | 535.72 | 92.52 | 133.20 | 16 |    0 |  0.00 | 100.02 |
| cusparse | scalar_mul          | (4,65535,1) | (32,4,1)  | 0.6047 | 27.77 | 97.29 |   0.02 | 535.99 | 79.53 | 266.40 | 31 |    0 |  0.00 | 100.00 |
| cusparse | **csr_partition**   | (820,1,1)   | (256,1,1) | 0.0137 | 23.63 |  8.56 |   1.05 |   0.01 | 80.54 |   1.67 | 16 |    0 | 75.69 | 56.91 |
| cusparse | **csrmm_alg2**      | (52429,4,1) | (128,1,1) | **4.1004** | 58.40 | 48.79 | 1202.64 | 620.48 | 76.02 | **255.75** | 44 |    0 | 22.48 | 86.52 |

**Reading the evidence:**
- **Grid.x is the smoking gun.** `pbr_kernel`'s grid never varies with features (1024 or 262144 — always exactly `num_pbr_blocks`), while `csrmm_alg2`'s grid.x is fixed by nnz-partitioning (205 or 52429) and only its grid.y grows with `feat_tiles` (1→4). At N=262144, `pbr_kernel` needs **290.63 waves/SM** vs `csrmm_alg2`'s **63.94–255.75** — 1.1–4.5× more, and it's the wave count, not raw grid size, that gates wall time once both are oversubscribed.
- **DRAM traffic is comparable or better for us, so it's not a bandwidth story.** At N=262144/f512, `pbr_kernel` reads *less* (1099 MB vs cuSPARSE's 1203 MB) and both write about the same (~536–620 MB) — yet `pbr_kernel` takes 13.00ms vs `csrmm_alg2`'s 4.10ms. The bottleneck is occupancy/launch geometry, not memory volume.
- **`sm__throughput` tells the same story from the compute side.** At N=262144/f64, `csrmm_alg2` hits 95.38% SM throughput (essentially compute-saturated, doing real work every cycle) vs `pbr_kernel`'s 67.31% — cuSPARSE is extracting far more useful work per cycle from the SMs it occupies.
- **The zero-init kernel is identical between engines to within noise** (0.0771–0.0772ms and 0.6040–0.6055ms at matched N/features) — confirms it's not a differentiator, consistent with H2's rejection of the remainder/overhead-asymmetry story.
- **At N=1024, `waves/SM < 1.2` for every kernel** — there's no oversubscription yet, so the grid-size mismatch is free; `pbr_kernel` even wins head-to-head at f64 (0.0122ms vs `csr_partition`+`csrmm_alg2`'s combined 0.0150ms). The crossover only appears once grids grow enough to force multiple serialized waves per SM.

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

## H4: Direct verification via a purpose-built diagnostic kernel

**Hypothesis:** if H3's mechanism (CTA count / wave count) is the dominant cause, then a kernel that
changes *only* CTA count — nothing else — should close most of the gap to cuSPARSE as chunk size
grows, with DRAM traffic roughly unchanged (same total work, just batched differently).

**Experiment:** built `pbr_spmm_chunk_kernel` — a new, minimal kernel whose per-iteration body is
byte-for-byte identical to `pbr_spmm_vec_kernel`'s (own smem_X/smem_Y carve-out, own load, own
`atomicAdd` flush, zero reuse across iterations), wrapped in a `#pragma unroll` loop over
`chunk_size` PBR blocks per CTA (`grid.x = ceil(num_pbr_blocks / chunk_size)`). Deliberately changes
*nothing else* `rowgroup`/`colgroup`/`tile` also change (no persistent smem_Y, no extra sync
patterns, no occupancy-tuned grouping) — an ablation that isolates the one variable H3 blamed.
Correctness: 480+96 new parametrized cases (`TestPBRChunkKernel`, chunk_size ∈ {2,4,8,16,32,64}) all
pass against `block` as ground truth. ncu-profiled `block` vs `chunk` (all 6 sizes) vs cuSPARSE on
the identical `diag_b0.3` matrix, N=1024 and N=262144, f64 and f512
(`jobs/profile_chunk_scaling/`, output in `ncu_out/chunk_scaling/`).

**Finding: the simple version of the hypothesis does not hold — and the failure mode is itself
informative.**

At N=1024 (GPU has slack, `waves/SM < 1.2` for `block`): time gets **monotonically worse** as
chunk_size grows — expected, since there's no oversubscription to relieve, so batching PBR blocks
sequentially per CTA is pure added serialization with no compensating benefit:

| chunk_size | f64 time (ms) | f512 time (ms) | waves/SM |
|---|---|---|---|
| `block` (=1) | 0.0124 | 0.0595 | 1.14 |
| 2 | 0.0127 | 0.0644 | 0.62 |
| 4 | 0.0150 | 0.0677 | 0.31 |
| 8 | 0.0207 | 0.0832 | 0.17 |
| 16 | 0.0457 | 0.1324 | 0.09 |
| 32 | 0.1012 | 0.2482 | 0.04 |
| 64 | 0.1304 | 0.4666 | 0.02 |
| cuSPARSE | 0.0173 | 0.0261 | 0.25 |

At N=262144 (the regime H3 predicts should benefit) — **it doesn't.** `waves/SM` drops exactly as
designed (290.63 → 159.84 → 79.92 → 44.40 → 22.20 → 11.10 → 4.54, essentially halving each step,
confirming the grid-shrink mechanism works exactly as modeled), but wall-clock time stays flat for
small chunks and then gets **worse**, never approaching cuSPARSE at any chunk size:

| chunk_size | f64 time (ms) | f512 time (ms) | waves/SM | regs/thread | f512 DRAM rd (MB) |
|---|---|---|---|---|---|
| `block` (=1) | 1.7204 | 12.9913 | 290.63 | 40 | 1099.2 |
| 2 | 1.6947 | 12.9072 | 159.84 | 48 | 1270.3 |
| 4 | 1.6978 | 12.9497 | 79.92 | 48 | 3118.6 |
| 8 | 1.7173 | 14.8244 | 44.40 | 56 | 5449.6 |
| 16 | 1.7895 | 15.6322 | 22.20 | 56 | 6803.6 |
| 32 | 1.9128 | 16.4131 | 11.10 | 56 | 7776.6 |
| 64 | 1.7948 | 16.7506 | 4.54 | 40 | 8141.3 |
| cuSPARSE (`csrmm_alg2`) | 0.6114 | 4.1068 | 63.94 / 255.75 | 44 | 165.0 / 1203.2 |

Even `chunk=64` — **64× fewer waves than `block`** — is still ~13× slower than cuSPARSE at f512, and
markedly *slower* than `chunk=2`. Two things stand out as the actual cause:
- **DRAM traffic explodes with chunk_size for supposedly identical total work** — f512 reads climb
  1099 → 1270 → 3119 → 5450 → 6804 → 7777 → **8141 MB**, a 7.4× increase from `block` to `chunk=64`,
  despite every chunk size processing the exact same matrix/features. Register count rises modestly
  (40→48→56) and `sm__warps_active%` *drops* (91%→~74-83%) for most chunk sizes before an anomalous
  partial recovery at chunk=64 — consistent with the `#pragma unroll`'d loop increasing per-thread
  live state/register pressure and degrading cache locality (each CTA now touches a wider, more
  scattered working set sequentially, with no cross-iteration reuse designed in) rather than a clean
  occupancy win.
- **cuSPARSE's `csr_partition_kernel` balances by nnz, our chunking only balances by block *count*.**
  `diag_b0.3` blocks don't all carry identical nnz (density=0.3 gives real variance per 8×8 block).
  Grouping a *fixed count* of blocks per CTA does not guarantee a *fixed amount of work* per CTA — as
  chunk_size grows, per-CTA nnz variance grows too, so the slowest CTA (the actual determinant of
  kernel end time) becomes a worse outlier. This directly tracks the observed `sm__throughput`
  decline (30.26% → 23.54% from `block` to `chunk=32` at f512) and gives a mechanistic explanation
  for why bigger chunks measured *worse*, not better.

**Conclusion of H4:** CTA-count/wave-count (H3) is a real, confirmed contributor, but it is **not
sufficient on its own** — a kernel that changes only that variable, with no other change, fails to
close the gap and actively regresses beyond a small chunk size. The remaining gap is most plausibly
cuSPARSE's genuinely *nnz-balanced* work partitioning (equal work per launched unit, not just equal
block count) combined with a more register/cache-efficient per-nonzero compute path — neither of
which a naive `#pragma unroll` chunking loop replicates. This is exactly the kind of result a good
ablation should produce: it rules out an oversimplified version of the hypothesis and sharpens what
to build next.

## Conclusion

The improving-with-scale story in the original sweep summary was an artifact of re-tuning the PBR
config independently at every size. Held fixed, the real trend is degradation on several matrices.
H1–H3 identified a real, measurable contributor: `block`'s one-CTA-per-PBR-block launch granularity
is **~5× finer than cuSPARSE's nnz-balanced work partition** for `diag_b0.3`, latent while the GPU
has spare capacity and on the critical path once the grid saturates the SMs (well before N=262144).

**H4 then falsified the simple version of that story.** A kernel built specifically to change only
CTA count — nothing else — does not recover cuSPARSE's speed, and actively gets worse past a small
chunk size, with DRAM traffic exploding for identical total work. The honest picture: CTA count is a
*real but insufficient* contributor. The rest of the gap is most plausibly cuSPARSE's genuinely
nnz-*balanced* (not just block-*count*-balanced) partitioning — its `csr_partition_kernel` equalizes
actual work per launched unit, which a naive count-based grouping does not — combined with a more
register/cache-efficient per-nonzero compute path.

This is a **different root cause than the tile kernel's failure** (see `TILE_STATE.md` — that was
missing vectorization, not this). It also sharpens the next lever worth trying: not "group by
PBR-block count" (what `rowgroup`/`colgroup`/`tile`/`chunk` all do) but **group by nnz-count**,
mirroring cuSPARSE's own `csr_partition_kernel`, so the number of launched CUDA blocks — and the
work assigned to each — tracks actual nnz rather than block *occurrence* count. Any future attempt
at this should be benchmarked against the `chunk` kernel's numbers above as the honest baseline, not
against `block` alone, since H4 shows naive grouping is not automatically an improvement.
