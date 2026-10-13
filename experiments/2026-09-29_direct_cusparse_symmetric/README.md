# Matched-scope direct cuSPARSE comparison — 2026-09-29

The [report](CUSPARSE_DIRECT_SYMMETRIC.md) measures API-level and C++-level
timings separately for direct cuSPARSE, BOO, and PBR.

- Driver: [`bench_spmm.py`](../../tests/torch_fast_spdmm/benchmark/bench_spmm.py)
- Report generator: [`report_symmetric.py`](../../scripts/cusparse_direct/report_symmetric.py)
- Tables: [derived data](../../tests/torch_fast_spdmm/benchmark/results/cusparse_direct_symmetric_20260929/derived/README.md)
