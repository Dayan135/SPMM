# Full FPT sweep

Completed tasks: 104/104. Validated BOO full-call rows: 1836/1872 requested combinations.

Requested duplicates, unsupported variants and incomplete tasks are retained in raw data/status. Each ratio below uses original-order direct cuSPARSE in the same task, labeling and feature cell. Winners are post-hoc; this is one allocation per task, not independent replication.

| Matrix | F | Clustering | k | FPT | BOO ms | Best measured direct ms | Direct / BOO | Direct algorithms measured |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| AMD/G2_circuit | 32 | identity | 64 | 2 | 0.2749 | 0.1029 | 0.374 | 4 / 4 |
| AMD/G2_circuit | 128 | identity | 32 | 2 | 0.7542 | 0.3707 | 0.492 | 4 / 4 |
| AMD/G2_circuit | 512 | identity | 32 | 2 | 2.5626 | 1.4449 | 0.564 | 4 / 4 |
| Boeing/pwtk | 32 | identity | 64 | 4 | 0.6943 | 0.6318 | 0.910 | 4 / 4 |
| Boeing/pwtk | 128 | hem | 32 | 4 | 1.8560 | 1.9333 | 1.042 | 4 / 4 |
| Boeing/pwtk | 512 | hem | 32 | 4 | 6.4701 | 7.9114 | 1.223 | 4 / 4 |
| LPnetlib/lp_cre_b | 32 | identity | 64 | 4 | 0.1597 | 0.0553 | 0.346 | 4 / 4 |
| LPnetlib/lp_cre_b | 128 | identity | 32 | 4 | 0.3743 | 0.1546 | 0.413 | 4 / 4 |
| LPnetlib/lp_cre_b | 512 | identity | 32 | 4 | 1.0875 | 0.6267 | 0.576 | 4 / 4 |
| SNAP/roadNet-CA | 32 | metis | 64 | 4 | 2.2943 | 1.1356 | 0.495 | 4 / 4 |
| SNAP/roadNet-CA | 128 | metis | 64 | 4 | 7.0636 | 4.4093 | 0.624 | 4 / 4 |
| SNAP/roadNet-CA | 512 | metis | 64 | 2 | 26.7392 | 22.0083 | 0.823 | 1 / 4 |
| Schmid/thermal2 | 32 | metis | 64 | 4 | 2.2682 | 1.0685 | 0.471 | 4 / 4 |
| Schmid/thermal2 | 128 | graclus | 32 | 4 | 7.2381 | 4.4595 | 0.616 | 4 / 4 |
| Schmid/thermal2 | 512 | hem | 32 | 2 | 27.5968 | 18.9880 | 0.688 | 2 / 4 |
| Williams/webbase-1M | 32 | hem | 32 | 2 | 1.3384 | 0.5489 | 0.410 | 4 / 4 |
| Williams/webbase-1M | 128 | identity | 32 | 2 | 4.1815 | 2.1084 | 0.504 | 4 / 4 |
| Williams/webbase-1M | 512 | identity | 32 | 2 | 16.2191 | 8.2053 | 0.506 | 4 / 4 |
| generated/diag_b0.1_n1048576 | 32 | identity | 64 | 2 | 0.7255 | 0.8806 | 1.214 | 4 / 4 |
| generated/diag_b0.1_n1048576 | 128 | identity | 64 | 2 | 2.7197 | 3.3659 | 1.238 | 4 / 4 |
| generated/diag_b0.1_n1048576 | 512 | identity | 64 | 2 | 10.4684 | 13.4584 | 1.286 | 4 / 4 |
| generated/diag_b0.1_n262144 | 32 | identity | 64 | 2 | 0.2212 | 0.2355 | 1.065 | 4 / 4 |
| generated/diag_b0.1_n262144 | 128 | identity | 64 | 2 | 0.7260 | 0.8509 | 1.172 | 4 / 4 |
| generated/diag_b0.1_n262144 | 512 | identity | 64 | 2 | 2.6829 | 3.3741 | 1.258 | 4 / 4 |
| generated/diag_b0.3_n1048576 | 32 | hem | 64 | 4 | 0.8576 | 2.0966 | 2.445 | 4 / 4 |
| generated/diag_b0.3_n1048576 | 128 | graclus | 64 | 4 | 3.2041 | 3.9030 | 1.218 | 4 / 4 |
| generated/diag_b0.3_n1048576 | 512 | graclus | 64 | 4 | 12.0074 | 15.5223 | 1.293 | 4 / 4 |
| generated/diag_b0.3_n262144 | 32 | hem | 64 | 4 | 0.2560 | 0.5319 | 2.078 | 4 / 4 |
| generated/diag_b0.3_n262144 | 128 | identity | 64 | 4 | 0.8366 | 0.9917 | 1.185 | 4 / 4 |
| generated/diag_b0.3_n262144 | 512 | identity | 64 | 4 | 3.0781 | 3.8856 | 1.262 | 4 / 4 |
| generated/diag_b1.0_n1048576 | 32 | metis | 32 | 4 | 1.8739 | 3.6782 | 1.963 | 4 / 4 |
| generated/diag_b1.0_n1048576 | 128 | metis | 32 | 4 | 5.2152 | 12.9101 | 2.475 | 4 / 4 |
| generated/diag_b1.0_n1048576 | 512 | metis | 32 | 4 | 20.4104 | 55.1316 | 2.701 | 3 / 4 |
| generated/diag_b1.0_n262144 | 32 | identity | 32 | 4 | 0.5043 | 0.8509 | 1.687 | 4 / 4 |
| generated/diag_b1.0_n262144 | 128 | identity | 32 | 4 | 1.3266 | 3.1534 | 2.377 | 4 / 4 |
| generated/diag_b1.0_n262144 | 512 | identity | 32 | 4 | 4.9295 | 12.8840 | 2.614 | 4 / 4 |
| generated/diag_dd0.017_n1048576 | 32 | identity | 64 | 2 | 0.8335 | 0.4925 | 0.591 | 4 / 4 |
| generated/diag_dd0.017_n1048576 | 128 | graclus | 32 | 2 | 2.5558 | 1.9292 | 0.755 | 4 / 4 |
| generated/diag_dd0.017_n1048576 | 512 | graclus | 32 | 2 | 9.6210 | 7.7725 | 0.808 | 3 / 4 |
| generated/diag_dd0.017_n262144 | 32 | identity | 64 | 4 | 0.2376 | 0.1362 | 0.573 | 4 / 4 |
| generated/diag_dd0.017_n262144 | 128 | graclus | 32 | 2 | 0.5806 | 0.4628 | 0.797 | 4 / 4 |
| generated/diag_dd0.017_n262144 | 512 | graclus | 64 | 2 | 2.0787 | 1.6225 | 0.781 | 4 / 4 |
| generated/diag_dd0.033_n1048576 | 32 | identity | 64 | 2 | 0.9088 | 0.5852 | 0.644 | 4 / 4 |
| generated/diag_dd0.033_n1048576 | 128 | identity | 64 | 2 | 2.8646 | 2.3695 | 0.827 | 4 / 4 |
| generated/diag_dd0.033_n1048576 | 512 | identity | 64 | 2 | 11.2435 | 10.2159 | 0.909 | 4 / 4 |
| generated/diag_dd0.033_n262144 | 32 | identity | 64 | 2 | 0.2601 | 0.1654 | 0.636 | 4 / 4 |
| generated/diag_dd0.033_n262144 | 128 | identity | 64 | 2 | 0.7014 | 0.5852 | 0.834 | 4 / 4 |
| generated/diag_dd0.033_n262144 | 512 | identity | 64 | 2 | 2.5421 | 2.0972 | 0.825 | 4 / 4 |
| generated/diag_dd0.05_n1048576 | 32 | identity | 64 | 2 | 0.9441 | 0.6446 | 0.683 | 4 / 4 |
| generated/diag_dd0.05_n1048576 | 128 | identity | 64 | 2 | 2.9747 | 2.7996 | 0.941 | 4 / 4 |
| generated/diag_dd0.05_n1048576 | 512 | identity | 64 | 2 | 11.6623 | 11.9752 | 1.027 | 4 / 4 |
| generated/diag_dd0.05_n262144 | 32 | identity | 64 | 2 | 0.2708 | 0.1812 | 0.669 | 4 / 4 |
| generated/diag_dd0.05_n262144 | 128 | identity | 64 | 2 | 0.7352 | 0.6835 | 0.930 | 4 / 4 |
| generated/diag_dd0.05_n262144 | 512 | identity | 64 | 2 | 2.6445 | 2.5651 | 0.970 | 4 / 4 |
| generated/diag_skewed_n1048576 | 32 | metis | 32 | 4 | 5.6980 | 8.7378 | 1.533 | 4 / 4 |
| generated/diag_skewed_n1048576 | 128 | identity | 32 | 4 | 19.9029 | 18.4996 | 0.929 | 4 / 4 |
| generated/diag_skewed_n1048576 | 512 | identity | 32 | 4 | 78.6591 | 81.1576 | 1.032 | 4 / 4 |
| generated/diag_skewed_n262144 | 32 | identity | 32 | 4 | 0.7880 | 1.0865 | 1.379 | 4 / 4 |
| generated/diag_skewed_n262144 | 128 | identity | 32 | 4 | 2.4376 | 3.0295 | 1.243 | 4 / 4 |
| generated/diag_skewed_n262144 | 512 | identity | 32 | 4 | 9.3768 | 11.9890 | 1.279 | 4 / 4 |
| generated/fem2d_n1048576 | 32 | graclus | 64 | 4 | 2.3127 | 2.2692 | 0.981 | 4 / 4 |
| generated/fem2d_n1048576 | 128 | hem | 32 | 4 | 7.8965 | 6.0928 | 0.772 | 4 / 4 |
| generated/fem2d_n1048576 | 512 | graclus | 32 | 4 | 29.3545 | 25.5959 | 0.872 | 3 / 4 |
| generated/fem2d_n262144 | 32 | graclus | 64 | 4 | 0.5985 | 0.5222 | 0.873 | 4 / 4 |
| generated/fem2d_n262144 | 128 | hem | 32 | 4 | 1.8985 | 1.4720 | 0.775 | 4 / 4 |
| generated/fem2d_n262144 | 512 | hem | 32 | 4 | 7.2432 | 5.9421 | 0.820 | 4 / 4 |
| generated/fem2d_permuted_n1048576 | 32 | metis | 64 | 4 | 2.7106 | 2.3639 | 0.872 | 4 / 4 |
| generated/fem2d_permuted_n1048576 | 128 | hem | 32 | 4 | 8.4557 | 8.6758 | 1.026 | 4 / 4 |
| generated/fem2d_permuted_n1048576 | 512 | hem | 32 | 4 | 31.7466 | 51.4396 | 1.620 | 3 / 4 |
| generated/fem2d_permuted_n262144 | 32 | metis | 64 | 4 | 0.6947 | 0.5468 | 0.787 | 4 / 4 |
| generated/fem2d_permuted_n262144 | 128 | hem | 32 | 4 | 2.1796 | 2.2380 | 1.027 | 4 / 4 |
| generated/fem2d_permuted_n262144 | 512 | hem | 32 | 4 | 8.0000 | 12.4370 | 1.555 | 4 / 4 |
| generated/pattern_diag_stripe_n1048576 | 32 | hem | 32 | 2 | 0.7224 | 0.3859 | 0.534 | 4 / 4 |
| generated/pattern_diag_stripe_n1048576 | 128 | graclus | 32 | 2 | 1.8104 | 1.3906 | 0.768 | 4 / 4 |
| generated/pattern_diag_stripe_n1048576 | 512 | hem | 32 | 2 | 6.5823 | 5.4938 | 0.835 | 3 / 4 |
| generated/pattern_diag_stripe_n262144 | 32 | graclus | 32 | 2 | 0.2206 | 0.1060 | 0.480 | 4 / 4 |
| generated/pattern_diag_stripe_n262144 | 128 | graclus | 32 | 2 | 0.4961 | 0.3660 | 0.738 | 4 / 4 |
| generated/pattern_diag_stripe_n262144 | 512 | graclus | 32 | 2 | 1.6858 | 1.3368 | 0.793 | 4 / 4 |

See summary.csv for every configuration, matching C++ scopes, reordered baselines, permutation costs, validation and skip reasons; status.json identifies missing/failed tasks.
