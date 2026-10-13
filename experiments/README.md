# Experiments

Each dated folder has a `README.md` with its question, report, figures, and
paths to code and retained result tables. Dates identify the measured run or,
for the two early PBR figure sets, the date the figures were added. Raw captures
are described in [the archive manifest](../RESULTS_ARCHIVE.md).

For new runs, use `YYYY-MM-DD_short_name/`. Record the question, setup,
measurements, conclusion, and limits in its `README.md` or a linked report;
put selected plots in `plots/`, and link to the driver and summary tables.
Plots are optional when a run has no useful figure.

| Date | Experiment | Record |
|---|---|---|
| 2026-06-23 | [PBR baseline figures](2026-06-23_pbr_baseline/README.md) | Figure set |
| 2026-06-23 | [Shared-memory and column sorting figures](2026-06-23_smem_colsort/README.md) | Figure set |
| 2026-09-23 | [PBR / BOO / CSR baseline](2026-09-23_spmm_baseline/README.md) | Completed |
| 2026-09-23 | [BOO optimization](2026-09-23_boo_optimization/README.md) | Completed |
| 2026-09-28 | [Reordering versus BOO](2026-09-28_reorder_vs_boo/README.md) | Completed |
| 2026-09-29 | [Direct cuSPARSE comparison](2026-09-29_direct_cusparse/README.md) | Completed |
| 2026-09-29 | [Matched-scope cuSPARSE comparison](2026-09-29_direct_cusparse_symmetric/README.md) | Completed |
| 2026-10-06 | [Clustering backends](2026-10-06_clustering/README.md) | Completed |
| 2026-10-08 | [Profiling panel](2026-10-08_profiling_panel/README.md) | Completed |
| 2026-10-09 | [Full FPT sweep](2026-10-09_full_fpt/README.md) | Completed |
| 2026-10-12 | [Tile and slab optimization](2026-10-12_tile_optimization/README.md) | Completed and replicated |

The [next experiment](../NEXT_EXPERIMENTS.md) tests the output-write path.
