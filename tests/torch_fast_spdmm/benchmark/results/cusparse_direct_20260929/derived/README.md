# Direct cuSPARSE comparison tables

These tables support [the direct comparison](../../../../../../experiments/2026-09-29_direct_cusparse/CUSPARSE_DIRECT.md).
`all_configs.csv` contains every measured configuration; `workloads.csv` and
`geomeans.csv` summarize the selected configurations. The `torch_*` tables
decompose one-call profiler traces. The JSON files record run and profiler
metadata. Individual CUDA-event samples are in the
[result archive](../../../../../../RESULTS_ARCHIVE.md).

Regenerate the report and tables with `scripts/cusparse_direct/report.py` after
restoring the original run. Its default output directory is this folder.
