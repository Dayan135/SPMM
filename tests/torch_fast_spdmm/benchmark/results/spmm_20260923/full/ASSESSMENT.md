# BOO comparison: 2026-09-23

BOO is competitive with cuSPARSE CSR on this synthetic workload collection,
particularly at the largest tested size. PBR has the strongest overall results
in the measured parameter grids. BOO's advantages depend on matrix structure and
cluster size; the measurements do not establish a universal replacement.

[W&B run](https://wandb.ai/private-dayan/torch-fast-spdmm/runs/4nk9coe9)
contains the raw samples, summaries, and result artifacts. See
[comparison.md](comparison.md) for every workload and [summary.csv](summary.csv)
for every configuration. [samples.csv](samples.csv) retains all individual times.

## Measurement scope

- RTX 3090, PyTorch 2.10.0+cu128, float32 values and int32 CSR indices.
- 14 profiles × 3 requested sizes × 2 feature widths = 84 workloads.
- Eight PBR block/min_nnz configurations, four BOO HEM sizes (8/16/32/64),
  and one PyTorch `torch.sparse.mm` CSR baseline: 1,092 configurations total.
- Five warmup calls followed by 20 measured calls per configuration:
  21,840 samples, all recorded in milliseconds.
- Identical seeded matrix and dense input within every workload. All numerical
  checks passed against CPU float64 references. Maximum absolute error across
  reported outputs was approximately 1.17e-5.
- Complete matmul calls with resident inputs are timed. Preprocessing and staging
  are excluded from those times and recorded separately in milliseconds.
- Family comparisons below choose the smallest observed median within each
  family's measured grid, separately per workload. This is retrospective tuning.
  Counts include close medians and are not statistical significance claims.

## Results

| Family | Fastest observed median, out of 84 | Geometric-mean speedup over CSR |
|---|---:|---:|
| cuSPARSE CSR | 20 | 1.000× |
| PBR, best measured configuration | 52 | 1.110× |
| BOO, best measured k | 12 | 0.982× |

BOO beats the CSR baseline in 45 of 84 workloads, although PBR is faster in many
of those. Larger sizes favor BOO relative to both baselines:

| Requested N | BOO speedup over CSR | BOO speedup over best measured PBR |
|---:|---:|---:|
| 16,384 | 0.825× | 0.802× |
| 65,536 | 1.018× | 0.900× |
| 262,144 | 1.128× | 0.959× |

These are geometric means of per-workload ratios; greater than one means BOO
is faster. No workloads were filtered by compressed fraction.

Examples at requested N=262,144 and 512 features (median milliseconds):

| Matrix | cuSPARSE CSR | PBR, best measured | BOO, best measured |
|---|---:|---:|---:|
| Permuted FEM | 13.07902 | 14.04306 | **11.42059** |
| Diagonal, density 0.1 | 3.98986 | 2.95024 | **2.83534** |
| Corner-cluster pattern | 1.85958 | 1.25798 | **1.15251** |
| Dense diagonal blocks | 13.90541 | **11.13749** | 12.63104 |
| Unpermuted FEM | **6.57035** | 8.96496 | 10.97933 |
| Skewed diagonal | **12.65341** | 15.68398 | 17.39418 |

The FEM pair supports investigating clustering as a way to recover useful local
structure after permutation. This is an interpretation of these measurements;
profiling would be needed to isolate the performance mechanism.

## Choice of k and preprocessing

Best measured BOO k: 32 in 51 workloads, 64 in 21, 8 in 8, and 16 in 4.
For permuted FEM at N=262,144 / features=512, k=32 takes 11.42059 ms,
whereas k=64 takes 14.76557 ms.

For the same k=32 example, BOO construction took 15,507.26 ms and staging took
16.54 ms. The CSR baseline's construction/staging took 7.48/8.40 ms. BOO saves
1.65843 ms per multiplication, giving an estimated break-even of about 9,350
multiplications with the same sparse matrix. This estimate excludes searching
over k and assumes the measured costs remain representative. The BOO win is
therefore most useful for repeated reuse of a fixed matrix.

## Validation and provenance

The isolated remote build passed 1,152 tests, with 13 expected skips (12
rectangular RCM cases and one multi-GPU case). The end-to-end smoke run verified
78 summaries and 1,560 samples. The full result download was checked for all
1,092 summaries, exactly 20 samples each, matching medians, matching input/source
hashes within each workload, and passed numerical checks throughout.

Build/test/smoke job: `21618321`; full sweep job: `21618650`.
Remote directory: `/home/dayanb/research/spmm-compare-20260923-140004`.
The full result artifact was retrieved through W&B after SSH connectivity was
lost; W&B reports the run finished and metadata reports all 84 workloads complete.
`metadata.json` identifies the exact source snapshot and compiled extension.
