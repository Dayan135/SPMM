# Direct cuSPARSE comparison — 2026-09-29

The [report](CUSPARSE_DIRECT.md) compares direct cuSPARSE with BOO, PBR, and
PyTorch CSR, and records the timing scope and profiler decomposition.

- Driver: [`bench_spmm.py`](../../tests/torch_fast_spdmm/benchmark/bench_spmm.py)
- Report generator: [`report.py`](../../scripts/cusparse_direct/report.py)
- Tables: [derived data](../../tests/torch_fast_spdmm/benchmark/results/cusparse_direct_20260929/derived/README.md)
- Individual samples: [archive](../../RESULTS_ARCHIVE.md)
