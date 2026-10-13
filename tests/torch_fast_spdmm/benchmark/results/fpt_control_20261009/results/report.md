# BOO profiling panel

Single-session results. Best configurations are post-hoc selections; speedups use matching timing scopes. Profiler timings are diagnostic.

| Profile | N | F | Scope | BOO config | BOO ms | Direct config | Direct ms | Direct / BOO |
|---|---:|---:|---|---|---:|---|---:|---:|
| diag_b1.0 | 262144 | 128 | full_call | boo_hem_k32_fpt4 | 1.5508 | direct_DEFAULT_api | 3.1524 | 2.033 |
| diag_b1.0 | 262144 | 128 | cpp_only | boo_hem_k32_fpt4_cpp | 1.5242 | direct_DEFAULT | 3.1539 | 2.069 |
| diag_b1.0 | 262144 | 512 | full_call | boo_hem_k32_fpt4 | 5.7395 | direct_CSR_ALG2_api | 13.5025 | 2.353 |
| diag_b1.0 | 262144 | 512 | cpp_only | boo_hem_k32_fpt4_cpp | 5.7795 | direct_CSR_ALG2 | 12.2783 | 2.124 |
| fem2d | 262144 | 128 | full_call | boo_hem_k32_fpt4 | 1.8575 | direct_CSR_ALG2_api | 1.4510 | 0.781 |
| fem2d | 262144 | 128 | cpp_only | boo_hem_k32_fpt4_cpp | 1.8140 | direct_DEFAULT | 1.4618 | 0.806 |
| fem2d | 262144 | 512 | full_call | boo_hem_k32_fpt4 | 7.0236 | direct_DEFAULT_api | 6.1937 | 0.882 |
| fem2d | 262144 | 512 | cpp_only | boo_hem_k32_fpt4_cpp | 7.4665 | direct_CSR_ALG2 | 5.4830 | 0.734 |
| fem2d_permuted | 262144 | 128 | full_call | boo_hem_k32_fpt4 | 2.1524 | direct_CSR_ALG3_api | 2.0874 | 0.970 |
| fem2d_permuted | 262144 | 128 | cpp_only | boo_hem_k32_fpt4_cpp | 2.1115 | direct_CSR_ALG3 | 2.0905 | 0.990 |
| fem2d_permuted | 262144 | 512 | full_call | boo_hem_k32_fpt4 | 7.9724 | direct_CSR_ALG2_api | 12.5046 | 1.568 |
| fem2d_permuted | 262144 | 512 | cpp_only | boo_hem_k32_fpt4_cpp | 8.0568 | direct_CSR_ALG2 | 12.4483 | 1.545 |
| pattern_row_skewed | 262144 | 128 | full_call | boo_hem_k32_fpt2 | 0.4782 | direct_DEFAULT_api | 0.4536 | 0.949 |
| pattern_row_skewed | 262144 | 128 | cpp_only | boo_hem_k32_fpt2_cpp | 0.4439 | direct_CSR_ALG2 | 0.4506 | 1.015 |
| pattern_row_skewed | 262144 | 512 | full_call | boo_hem_k32_fpt2 | 1.6303 | direct_CSR_ALG3_api | 1.6809 | 1.031 |
| pattern_row_skewed | 262144 | 512 | cpp_only | boo_hem_k32_fpt2_cpp | 1.5954 | direct_CSR_ALG3 | 1.6773 | 1.051 |

## Paired features-per-thread controls

| Profile | F | Scope | Variant | FPT1 ms | Variant ms | FPT1 / variant |
|---|---:|---|---|---:|---:|---:|
| fem2d | 128 | full_call | boo_hem_k32_fpt2 | 2.8119 | 2.0628 | 1.363 |
| fem2d | 128 | full_call | boo_hem_k32_fpt4 | 2.8119 | 1.8575 | 1.514 |
| fem2d | 128 | cpp_only | boo_hem_k32_fpt2_cpp | 2.7766 | 2.0444 | 1.358 |
| fem2d | 128 | cpp_only | boo_hem_k32_fpt4_cpp | 2.7766 | 1.8140 | 1.531 |
| fem2d | 512 | full_call | boo_hem_k32_fpt2 | 11.3505 | 7.6376 | 1.486 |
| fem2d | 512 | full_call | boo_hem_k32_fpt4 | 11.3505 | 7.0236 | 1.616 |
| fem2d | 512 | cpp_only | boo_hem_k32_fpt2_cpp | 11.8134 | 7.5822 | 1.558 |
| fem2d | 512 | cpp_only | boo_hem_k32_fpt4_cpp | 11.8134 | 7.4665 | 1.582 |
| fem2d_permuted | 128 | full_call | boo_hem_k32_fpt2 | 3.0305 | 2.3244 | 1.304 |
| fem2d_permuted | 128 | full_call | boo_hem_k32_fpt4 | 3.0305 | 2.1524 | 1.408 |
| fem2d_permuted | 128 | cpp_only | boo_hem_k32_fpt2_cpp | 2.9731 | 2.2989 | 1.293 |
| fem2d_permuted | 128 | cpp_only | boo_hem_k32_fpt4_cpp | 2.9731 | 2.1115 | 1.408 |
| fem2d_permuted | 512 | full_call | boo_hem_k32_fpt2 | 11.5804 | 8.3046 | 1.394 |
| fem2d_permuted | 512 | full_call | boo_hem_k32_fpt4 | 11.5804 | 7.9724 | 1.453 |
| fem2d_permuted | 512 | cpp_only | boo_hem_k32_fpt2_cpp | 11.7340 | 8.1608 | 1.438 |
| fem2d_permuted | 512 | cpp_only | boo_hem_k32_fpt4_cpp | 11.7340 | 8.0568 | 1.456 |
| diag_b1.0 | 128 | full_call | boo_hem_k32_fpt2 | 3.4770 | 1.9799 | 1.756 |
| diag_b1.0 | 128 | full_call | boo_hem_k32_fpt4 | 3.4770 | 1.5508 | 2.242 |
| diag_b1.0 | 128 | cpp_only | boo_hem_k32_fpt2_cpp | 3.2472 | 1.9456 | 1.669 |
| diag_b1.0 | 128 | cpp_only | boo_hem_k32_fpt4_cpp | 3.2472 | 1.5242 | 2.130 |
| diag_b1.0 | 512 | full_call | boo_hem_k32_fpt2 | 13.6893 | 7.2914 | 1.877 |
| diag_b1.0 | 512 | full_call | boo_hem_k32_fpt4 | 13.6893 | 5.7395 | 2.385 |
| diag_b1.0 | 512 | cpp_only | boo_hem_k32_fpt2_cpp | 13.9991 | 7.1265 | 1.964 |
| diag_b1.0 | 512 | cpp_only | boo_hem_k32_fpt4_cpp | 13.9991 | 5.7795 | 2.422 |
| pattern_row_skewed | 128 | full_call | boo_hem_k32_fpt2 | 0.4925 | 0.4782 | 1.030 |
| pattern_row_skewed | 128 | full_call | boo_hem_k32_fpt4 | 0.4925 | 0.5059 | 0.974 |
| pattern_row_skewed | 128 | cpp_only | boo_hem_k32_fpt2_cpp | 0.4547 | 0.4439 | 1.024 |
| pattern_row_skewed | 128 | cpp_only | boo_hem_k32_fpt4_cpp | 0.4547 | 0.4731 | 0.961 |
| pattern_row_skewed | 512 | full_call | boo_hem_k32_fpt2 | 1.6507 | 1.6303 | 1.013 |
| pattern_row_skewed | 512 | full_call | boo_hem_k32_fpt4 | 1.6507 | 1.7536 | 0.941 |
| pattern_row_skewed | 512 | cpp_only | boo_hem_k32_fpt2_cpp | 1.6092 | 1.5954 | 1.009 |
| pattern_row_skewed | 512 | cpp_only | boo_hem_k32_fpt4_cpp | 1.6092 | 1.7193 | 0.936 |

API BOO uses auto launch. The existing preallocated C++ BOO harness uses the legacy stream fork/join; both include required output initialization and remainder work.

## Block structure

| Matrix | Config | Blocks | Staged X rows | Active X rows | Unused X % | Active output rows across blocks |
|---|---|---:|---:|---:|---:|---:|
| diag_b1.0_n262144_structure | boo_hem_k32 | 35784 | 772608 | 772608 | 0.00 | 772608 |
| fem2d_n262144_structure | boo_hem_k32 | 67922 | 1916265 | 873524 | 54.42 | 873524 |
| fem2d_permuted_n262144_structure | boo_hem_k32 | 67961 | 1916328 | 872120 | 54.49 | 872120 |
| pattern_row_skewed_n262144_structure | boo_hem_k32 | 11749 | 160142 | 128093 | 20.01 | 125789 |

Extracted 0 Nsight Systems kernel records and 0 Nsight Compute metric records. Raw traces and per-profile provenance remain beside this report.

Active support is structural: cancellation can reduce actual output atomics. Logical X counts do not imply equivalent DRAM traffic.
