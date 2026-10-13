# Symmetric direct cuSPARSE comparison tables

These tables support [the matched-scope comparison](../../../../../../experiments/2026-09-29_direct_cusparse_symmetric/CUSPARSE_DIRECT_SYMMETRIC.md).
`all_configs.csv` contains every measured configuration. The API-level,
C++-level, and diagnostic workload tables are kept separate, as are their
geometric means. The three `profiles_*.csv` files summarize GPU-busy traces;
`metadata.json` records the three runs.

Regenerate the report and tables with
`scripts/cusparse_direct/report_symmetric.py` after restoring the original
run. Its default output directory is this folder.
