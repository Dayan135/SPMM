Work in `/Users/dayan/local/research/SPMM-boo`.

Read `CLAUDE.md` and `experiments/2026-10-08_profiling_panel/RESEARCH_CONTINUATION.md`. Execute the first real-matrix
benchmarking milestone for BOO SpMM. The workload is a fixed sparsity pattern
reused across many SpMM calls.

Another agent owns the synthetic profiling panel and its remote jobs. Your
scope is the real-matrix loader, benchmark harness, focused tests, and results.
Inspect existing partial work first. Prefer these files:

- `tests/torch_fast_spdmm/benchmark/real_matrices.py`
- `tests/torch_fast_spdmm/benchmark/bench_real.py`
- `tests/torch_fast_spdmm/benchmark/test_real_matrices.py`

Requirements:

1. Support SuiteSparse's Boeing/pwtk, Schmid/thermal2, AMD/G2_circuit,
   Williams/webbase-1M, SNAP/roadNet-CA, and LPnetlib/lp_cre_b.
2. Cache downloads outside tracked results. Record sources, hashes, dimensions,
   nnz and preprocessing. Handle Matrix Market symmetry, duplicates and
   explicit zeros consistently; preserve directed and rectangular matrices.
3. Separate download, CPU clustering and GPU timing phases. Save labels with
   matrix hashes and validate them when reusing them.
4. Start with identity/HEM, group targets 32/64, F=32/128/512, FP32,
   feasible-first clustering, row-major BOO, auto launch and remainder threshold 4.
5. Compare original CSR, direct cuSPARSE, BOO, and cuSPARSE after the exact same
   BOO label permutation. Report steady-state reordered execution separately
   from calls that gather X and restore Y. Reuse existing measurement helpers
   where appropriate, but keep timing scopes matched.
6. Validate numerics independently before timing. Interleave samples, report
   unsupported configurations, and record preprocessing outside steady-state
   timings. Do not benchmark intentionally incorrect raw-store diagnostics.
7. Add meaningful loader, label-validation and permutation tests. All tests and
   CUDA builds run remotely per CLAUDE.md.
8. Start with lp_cre_b as the smoke case, then run the six-matrix panel after
   validation. Use a distinct isolated remote snapshot and in-place extension
   build; avoid reinstalling the shared environment's package.

Do not edit kernels, `benchmark/research_panel.py`, `benchmark/panel_structure.py`,
`benchmark/test_research_panel.py`, or `scripts/research_panel/`. Coordinate any
necessary edits to shared `bench_spmm.py`: the profiling track added the optional
`diagnostics=False` argument to `measure_cell`/`_symmetric_variants`, preserving
existing defaults. That track's remote snapshot is
`/home/dayanb/research/boo-profile-20261008-a`; use a different path.

Do not commit or push. Deliver code, reproduction commands, test results, and
a brief report distinguishing completed runs from pending work.
