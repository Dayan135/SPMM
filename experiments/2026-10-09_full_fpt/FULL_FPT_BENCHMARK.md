# Full FPT benchmark — 2026-10-09

Status: completed successfully on 2026-10-09 at 22:01 Israel time. All 104 CPU
and 104 GPU shards exited 0:0. Results retrieved locally on 2026-10-12.

Measured 1,836 BOO complete-call combinations; the remaining 36 requested
combinations reuse identical HEM/Graclus k32/k64 diagonal-stripe labelings and
were explicitly deduplicated. Across all scopes/baselines, 14,734 output rows
passed numerical validation. Some optional baselines were skipped for resource
limits; the aggregate records the number of direct algorithms measured.

[Completed aggregate report](../../tests/torch_fast_spdmm/benchmark/results/full_fpt_20261009/aggregate/report.md)
and [all configuration timings](../../tests/torch_fast_spdmm/benchmark/results/full_fpt_20261009/aggregate/summary.csv).

## Requested scope

- FPT: **1, 2, 4**; maximum row/column group size k: **32, 64**.
- Clustering: **identity, HEM, Graclus, METIS**, seed 0, feasible-first grouping.
- Dense feature widths: **32, 128, 512**; float32 values / int32 canonical CSR.
- Generated matrix sizes: **262,144 and 1,048,576**.
- Native SuiteSparse dimensions are preserved, including the rectangular input.

SuiteSparse inputs: Boeing/pwtk, Schmid/thermal2, AMD/G2_circuit,
Williams/webbase-1M, SNAP/roadNet-CA, LPnetlib/lp_cre_b.

Generated inputs at both sizes: fem2d, fem2d_permuted, diag_dd0.017,
diag_dd0.033, diag_dd0.05, diag_b0.1, diag_b0.3, diag_b1.0, diag_skewed,
pattern_diag_stripe. FEM here is the repository's generated FEM model.

This is **26 inputs × 4 methods × 2 k values × 3 feature widths × 3 FPT values
= 1,872 requested BOO complete-call combinations**. C++ measurements and
baselines are additional. Duplicate labelings and unsupported configurations
remain explicit in the results; requested count is not a promise that all fit.

## Measurement and validation

One shard per (input, clustering method): 104 shards. CPU jobs save labels,
and the corresponding GPU jobs hash-check them before timing. Clustering,
format construction, staging and plan preparation are outside timing.
Each cell interleaves FPT1/2/4, PyTorch CSR and all four direct cuSPARSE
algorithms, subject to recorded memory/support limits. Reordered CSR and
per-call permutation baselines are retained. Five warmups and 20 repeats.

The full sweep uses the real-panel harness for generated matrices too, through
[full_sweep.py](../../scripts/research_panel/full_sweep.py). This gives both input
families the same float64 CPU reference, row-wise numerical error bounds,
transient GPU memory budget and output schema. The adapter registers generated
inputs only within its process; the external agent's files are unchanged.

Preflight job **22418047 completed 0:0**: **151 tests passed**, followed by
end-to-end generated FEM N1024/F35 smoke cells for all four methods, both k
values and FPT1/2/4. Earlier preflights exposed a corrected adapter import and
missing PyMETIS. PyMETIS 2025.2.2 and typing-extensions 4.16.0 were installed
only in this snapshot's `deps/`; the shared environment is unchanged.
The extension is the previously validated FPT binary; no new kernel changes.

## Execution and results

Isolated remote root: `/home/dayanb/research/boo-full-fpt-20261009`.

| Job | Role | Final state |
|---|---|---|
| 22418047 | Preflight | Completed successfully |
| 22418048 | CPU array 0–103, maximum 4 concurrent, 8 CPUs/128 GiB per task | All 104 tasks completed 0:0 |
| 22418049 | RTX 3090 array 0–103, maximum 2 concurrent, 8 CPUs/128 GiB per task | All 104 tasks completed 0:0 |
| 22418066 | Aggregate report after all GPU tasks terminate | Report retained in `aggregate/` |

CPU tasks had a 24-hour limit, GPU tasks 6 hours. Each GPU task depended on its
matching CPU task via Slurm `aftercorr`. The aggregate report records measured
and skipped cells. Per-task files are preserved in the separate archive; see
[RESULTS_ARCHIVE.md](../../RESULTS_ARCHIVE.md).

- `sweep.json`: exact indexed task manifest.
- `benchmark_source.json`: parent, extension and sweep source hashes.
- `labels/`: hash-validated labels; copied real identity/HEM labels are reused.
- `logs/{cluster,matmul}-<job>_<task>.log`: per-task progress and errors.
- `results/task_NNN/{summary.csv,samples.csv,metadata.json,report.md}` (archive).
- `aggregate/{summary.csv,status.json,report.md}`: combined output after completion.

Generate an intermediate aggregate safely at any time:

```bash
python3 scripts/research_panel/full_sweep_report.py /home/dayanb/research/boo-full-fpt-20261009
```

The aggregate reports best BOO and a **same-cell original-order direct** baseline,
including how many direct algorithms were measured. Raw CSV preserves all
parameter combinations. The inherited per-shard FPT table has an inverted
heading for its final direct ratio; use raw timings or the aggregate's explicitly
labeled `Direct / BOO` ratio. Independent allocation replication remains a
separate experiment; 20 timing repeats are not independent allocations.
