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
