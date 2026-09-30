## Nsight Compute: occupancy and instruction-fetch stalls

One warmed call per mode/profile at N=262,144, F=512. Row-exclusive uses its Phase A best C++ configuration. Current BOO uses the same HEM target and hence the same grouping; its feature tile stays at p=64. This isolates execution-mode resource differences. All launched kernels, including output initialization and CSR remainder work, are retained in the [kernel CSV](BOO_ROWEXCL_ncu_kernels.csv). The table compares the row-exclusive kernel with the current BOO row-segmented kernel; if a mode has multiple such launches, percentages are duration-weighted descriptive averages. These profiled durations are diagnostic, not the unprofiled benchmark timings. Caches and clocks are left uncontrolled; raw logs retain profiler warnings.

| Profile | Mode | k / feature slab | Achieved occupancy % | no_instruction % | Main-kernel ms |
|---|---|---|---:|---:|---:|
| diag_b1.0 | boo | 64 / 64 | 33.03 | 0.61 | 11.02970 |
| diag_b1.0 | rowexcl | 64 / 32 | 66.29 | 1.73 | 12.99942 |
| fem2d | boo | 32 / 64 | 99.63 | 1.27 | 9.69018 |
| fem2d_permuted | boo | 32 / 64 | 99.69 | 1.24 | 11.07046 |
| fem2d_permuted | rowexcl | 32 / 32 | 66.35 | 1.50 | 13.79078 |
| fem2d | rowexcl | 32 / 32 | 66.27 | 1.61 | 11.67750 |

The `no_instruction` counter describes warps waiting for instruction fetch or an instruction-cache miss; it is not by itself proof of too few runnable CTAs. Compare it with achieved occupancy, the recorded resident-CTA resource bound, CTA nnz distribution, and unprofiled timings. [NVIDIA profiling guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/).

[Per-capture provenance and plan statistics](BOO_ROWEXCL_ncu_metadata.json); [summary CSV](BOO_ROWEXCL_ncu_summary.csv).
