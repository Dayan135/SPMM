# Reorder versus BOO report tables

These six CSVs support [the experiment report](../../../../../../experiments/2026-09-28_reorder_vs_boo/REORDER_VS_BOO.md).
`REORDER_VS_BOO_E1.csv` is the historical E1 configuration table;
`REORDER_VS_BOO_E1_STRUCTURE.csv` adds deterministic structure counters.
`REORDER_VS_BOO_REPLICATES.csv` contains the three-job configuration summaries.
The BEST, RATIOS, and JOB_WINNERS tables are derived from those summaries.

The three root-level per-call sample CSVs are in the
[separate archive](../../../../../../RESULTS_ARCHIVE.md). Regenerate the
three-job tables with `report_reorder_vs_boo.py` after restoring the full raw
run; generated sample CSVs are ignored by Git.
