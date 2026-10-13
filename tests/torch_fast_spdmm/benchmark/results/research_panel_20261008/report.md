# BOO profiling panel

Single-session results. Best configurations are post-hoc selections; speedups use matching timing scopes. Profiler timings are diagnostic.

| Profile | N | F | Scope | BOO config | BOO ms | Direct config | Direct ms | Direct / BOO |
|---|---:|---:|---|---|---:|---|---:|---:|
| diag_b0.3 | 262144 | 128 | full_call | boo_identity_k64 | 1.2609 | direct_DEFAULT_api | 1.0076 | 0.799 |
| diag_b0.3 | 262144 | 128 | cpp_only | boo_identity_k64_cpp | 1.2149 | direct_CSR_ALG2 | 1.0020 | 0.825 |
| diag_b0.3 | 262144 | 512 | full_call | boo_hem_k64 | 4.7186 | direct_DEFAULT_api | 3.9142 | 0.830 |
| diag_b0.3 | 262144 | 512 | cpp_only | boo_hem_k64_cpp | 4.6669 | direct_CSR_ALG2 | 3.9342 | 0.843 |
| diag_b1.0 | 262144 | 128 | full_call | boo_identity_k32 | 3.0976 | direct_CSR_ALG3_api | 3.1483 | 1.016 |
| diag_b1.0 | 262144 | 128 | cpp_only | boo_identity_k32_cpp | 3.0585 | direct_DEFAULT | 3.1532 | 1.031 |
| diag_b1.0 | 262144 | 512 | full_call | boo_identity_k64 | 12.1076 | direct_CSR_ALG2_api | 12.5039 | 1.033 |
| diag_b1.0 | 262144 | 512 | cpp_only | boo_identity_k64_cpp | 12.0974 | direct_CSR_ALG2 | 12.8287 | 1.060 |
| diag_skewed | 262144 | 128 | full_call | boo_identity_k32 | 4.2316 | direct_DEFAULT_api | 3.0198 | 0.714 |
| diag_skewed | 262144 | 128 | cpp_only | boo_identity_k32_cpp | 4.2081 | direct_DEFAULT | 3.0192 | 0.717 |
| diag_skewed | 262144 | 512 | full_call | boo_identity_k64 | 16.3884 | direct_CSR_ALG2_api | 11.6952 | 0.714 |
| diag_skewed | 262144 | 512 | cpp_only | boo_identity_k64_cpp | 16.5826 | direct_CSR_ALG2 | 11.9818 | 0.723 |
| fem2d | 262144 | 128 | full_call | boo_hem_k32 | 2.7984 | direct_DEFAULT_api | 1.4433 | 0.516 |
| fem2d | 262144 | 128 | cpp_only | boo_hem_k32_cpp | 2.7821 | direct_DEFAULT | 1.4372 | 0.517 |
| fem2d | 262144 | 512 | full_call | boo_hem_k32 | 10.5000 | direct_CSR_ALG2_api | 5.6902 | 0.542 |
| fem2d | 262144 | 512 | cpp_only | boo_hem_k32_cpp | 10.5046 | direct_CSR_ALG2 | 5.8542 | 0.557 |
| fem2d_permuted | 262144 | 128 | full_call | boo_hem_k32 | 3.0770 | direct_CSR_ALG3_api | 2.1409 | 0.696 |
| fem2d_permuted | 262144 | 128 | cpp_only | boo_hem_k32_cpp | 3.0663 | direct_CSR_ALG3 | 2.1955 | 0.716 |
| fem2d_permuted | 262144 | 512 | full_call | boo_hem_k32 | 11.5220 | direct_DEFAULT_api | 12.4298 | 1.079 |
| fem2d_permuted | 262144 | 512 | cpp_only | boo_hem_k32_cpp | 11.4677 | direct_CSR_ALG3 | 12.3418 | 1.076 |
| pattern_row_skewed | 262144 | 128 | full_call | boo_hem_k32 | 0.4966 | direct_DEFAULT_api | 0.4526 | 0.911 |
| pattern_row_skewed | 262144 | 128 | cpp_only | boo_hem_k32_cpp | 0.4547 | direct_DEFAULT | 0.4484 | 0.986 |
| pattern_row_skewed | 262144 | 512 | full_call | boo_hem_k64 | 1.6384 | direct_CSR_ALG3_api | 1.6947 | 1.034 |
| pattern_row_skewed | 262144 | 512 | cpp_only | boo_hem_k64_cpp | 1.5929 | direct_CSR_ALG3 | 1.6865 | 1.059 |

API BOO uses auto launch. The existing preallocated C++ BOO harness uses the legacy stream fork/join; both include required output initialization and remainder work.

## Block structure

| Matrix | Config | Blocks | Staged X rows | Active X rows | Unused X % | Active output rows across blocks |
|---|---|---:|---:|---:|---:|---:|
| diag_b0.3_n262144_structure | boo_identity_k32 | 16384 | 524288 | 524284 | 0.00 | 524283 |
| diag_b0.3_n262144_structure | boo_identity_k64 | 4096 | 262144 | 262144 | 0.00 | 262144 |
| diag_b0.3_n262144_structure | boo_hem_k32 | 16386 | 524288 | 524273 | 0.00 | 524312 |
| diag_b0.3_n262144_structure | boo_hem_k64 | 4096 | 262144 | 262144 | 0.00 | 262144 |
| diag_b1.0_n262144_structure | boo_identity_k32 | 16384 | 524288 | 524288 | 0.00 | 524288 |
| diag_b1.0_n262144_structure | boo_identity_k64 | 4096 | 262144 | 262144 | 0.00 | 262144 |
| diag_b1.0_n262144_structure | boo_hem_k32 | 35784 | 772608 | 772608 | 0.00 | 772608 |
| diag_b1.0_n262144_structure | boo_hem_k64 | 4096 | 262144 | 262144 | 0.00 | 262144 |
| diag_skewed_n262144_structure | boo_identity_k32 | 16384 | 524288 | 524288 | 0.00 | 524288 |
| diag_skewed_n262144_structure | boo_identity_k64 | 4119 | 263616 | 262231 | 0.53 | 262235 |
| diag_skewed_n262144_structure | boo_hem_k32 | 36407 | 892881 | 784090 | 12.18 | 782394 |
| diag_skewed_n262144_structure | boo_hem_k64 | 7542 | 410781 | 356843 | 13.13 | 356007 |
| fem2d_n262144_structure | boo_identity_k32 | 72004 | 2304128 | 974352 | 57.71 | 974352 |
| fem2d_n262144_structure | boo_identity_k64 | 35236 | 2255104 | 876304 | 61.14 | 876304 |
| fem2d_n262144_structure | boo_hem_k32 | 67922 | 1916265 | 873524 | 54.42 | 873524 |
| fem2d_n262144_structure | boo_hem_k64 | 34930 | 1872250 | 684576 | 63.44 | 684576 |
| fem2d_permuted_n262144_structure | boo_identity_k32 | 527490 | 16879680 | 2345928 | 86.10 | 2345928 |
| fem2d_permuted_n262144_structure | boo_identity_k64 | 517386 | 33112704 | 2344644 | 92.92 | 2344644 |
| fem2d_permuted_n262144_structure | boo_hem_k32 | 67961 | 1916328 | 872120 | 54.49 | 872120 |
| fem2d_permuted_n262144_structure | boo_hem_k64 | 34789 | 1869569 | 683472 | 63.44 | 683472 |
| pattern_row_skewed_n262144_structure | boo_identity_k32 | 20967 | 670944 | 167760 | 75.00 | 167760 |
| pattern_row_skewed_n262144_structure | boo_identity_k64 | 20956 | 1341184 | 167760 | 87.49 | 167752 |
| pattern_row_skewed_n262144_structure | boo_hem_k32 | 11749 | 160142 | 128093 | 20.01 | 125789 |
| pattern_row_skewed_n262144_structure | boo_hem_k64 | 10236 | 135168 | 124563 | 7.85 | 124132 |

Extracted 23 Nsight Systems kernel records and 1081 Nsight Compute metric records. Raw traces and per-profile provenance remain beside this report.

Active support is structural: cancellation can reduce actual output atomics. Logical X counts do not imply equivalent DRAM traffic.
