| matrix | F | labels | torch | direct api (best) | BOO | perm steady api (best) | perm gather api (best) | direct cpp | perm cpp | BOO cpp | BOO/direct api | BOO/perm gather |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| AMD/G2_circuit | 32 | hem_k32 | 0.1618 | 0.0993 | 0.4024 | 0.1060 | 0.3461 | 0.0993 | 0.1034 | 0.3461 | 0.247 | 0.860 |
| AMD/G2_circuit | 32 | hem_k64 | 0.1618 | 0.0998 | 0.3625 | 0.1055 | 0.3441 | 0.0993 | 0.1024 | 0.3082 | 0.275 | 0.949 |
| AMD/G2_circuit | 32 | identity_k32 | 0.1715 | 0.1029 | 0.3348 | - | - | 0.0978 | - | 0.2806 | 0.307 | - |
| AMD/G2_circuit | 32 | identity_k64 | 0.1618 | 0.1004 | 0.2811 | - | - | 0.0947 | - | 0.2263 | 0.357 | - |
| AMD/G2_circuit | 128 | hem_k32 | 0.6216 | 0.3604 | 0.8950 | 0.3702 | 0.7521 | 0.3610 | 0.3656 | 0.8131 | 0.403 | 0.840 |
| AMD/G2_circuit | 128 | hem_k64 | 0.6211 | 0.3604 | 0.9528 | 0.3635 | 0.7444 | 0.3579 | 0.3599 | 0.8796 | 0.378 | 0.781 |
| AMD/G2_circuit | 128 | identity_k32 | 0.6349 | 0.3635 | 0.7880 | - | - | 0.3574 | - | 0.7286 | 0.461 | - |
| AMD/G2_circuit | 128 | identity_k64 | 0.6339 | 0.3692 | 0.8115 | - | - | 0.3645 | - | 0.7598 | 0.455 | - |
| AMD/G2_circuit | 512 | hem_k32 | 2.3880 | 1.4382 | 3.1319 | 1.4003 | 2.8769 | 1.4367 | 1.3978 | 3.0141 | 0.459 | 0.919 |
| AMD/G2_circuit | 512 | hem_k64 | 2.4018 | 1.4397 | 3.7048 | 1.3880 | 2.8682 | 1.4413 | 1.3860 | 3.5702 | 0.389 | 0.774 |
| AMD/G2_circuit | 512 | identity_k32 | 2.4069 | 1.4428 | 2.7438 | - | - | 1.4372 | - | 2.6762 | 0.526 | - |
| AMD/G2_circuit | 512 | identity_k64 | 2.4095 | 1.4474 | 2.8396 | - | - | 1.4408 | - | 2.7725 | 0.510 | - |
| Boeing/pwtk | 32 | hem_k32 | 1.1090 | 0.6072 | 1.0291 | 0.6042 | 0.9242 | 0.6052 | 0.5980 | 0.9646 | 0.590 | 0.898 |
| Boeing/pwtk | 32 | hem_k64 | 1.1085 | 0.6016 | 1.0097 | 0.6011 | 0.9216 | 0.6001 | 0.5955 | 0.9400 | 0.596 | 0.913 |
| Boeing/pwtk | 32 | identity_k32 | 1.1484 | 0.6164 | 0.9902 | - | - | 0.6113 | - | 0.9196 | 0.623 | - |
| Boeing/pwtk | 32 | identity_k64 | 1.1372 | 0.6124 | 0.9841 | - | - | 0.6077 | - | 0.9216 | 0.622 | - |
| Boeing/pwtk | 128 | hem_k32 | 2.1233 | 1.9594 | 3.0628 | 2.0424 | 2.5882 | 1.9599 | 2.0265 | 3.0285 | 0.640 | 0.845 |
| Boeing/pwtk | 128 | hem_k64 | 2.1064 | 1.9517 | 3.4212 | 2.0419 | 2.5923 | 1.9574 | 2.0311 | 3.3674 | 0.570 | 0.758 |
| Boeing/pwtk | 128 | identity_k32 | 2.1299 | 1.9983 | 3.0141 | - | - | 1.9763 | - | 2.9322 | 0.663 | - |
| Boeing/pwtk | 128 | identity_k64 | 2.1320 | 1.9645 | 3.4028 | - | - | 1.9594 | - | 3.3121 | 0.577 | - |
| Boeing/pwtk | 512 | hem_k32 | 8.8120 | 8.0261 | 12.1636 | 8.1715 | 10.7274 | 7.7056 | 8.1372 | 11.0689 | 0.660 | 0.882 |
| Boeing/pwtk | 512 | hem_k64 | 8.7332 | 7.9124 | 14.0621 | 8.1923 | 10.6542 | 7.7491 | 8.1459 | 12.8497 | 0.563 | 0.758 |
| Boeing/pwtk | 512 | identity_k32 | 9.1080 | 7.9227 | 11.2082 | - | - | 8.0476 | - | 10.9245 | 0.707 | - |
| Boeing/pwtk | 512 | identity_k64 | 8.6717 | 7.8700 | 12.7744 | - | - | 7.8889 | - | 12.4964 | 0.616 | - |
| LPnetlib/lp_cre_b | 32 | hem_k32 | 0.0829 | 0.0512 | 0.2509 | 0.0512 | 0.1260 | 0.0522 | 0.0497 | 0.1956 | 0.204 | 0.502 |
| LPnetlib/lp_cre_b | 32 | hem_k64 | 0.0819 | 0.0512 | 0.2519 | 0.0522 | 0.1260 | 0.0522 | 0.0502 | 0.1961 | 0.203 | 0.500 |
| LPnetlib/lp_cre_b | 32 | identity_k32 | 0.0942 | 0.0532 | 0.1869 | - | - | 0.0492 | - | 0.1331 | 0.285 | - |
| LPnetlib/lp_cre_b | 32 | identity_k64 | 0.0932 | 0.0522 | 0.1618 | - | - | 0.0481 | - | 0.1116 | 0.323 | - |
| LPnetlib/lp_cre_b | 128 | hem_k32 | 0.1997 | 0.1526 | 0.5509 | 0.1516 | 0.2693 | 0.1546 | 0.1490 | 0.4895 | 0.277 | 0.489 |
| LPnetlib/lp_cre_b | 128 | hem_k64 | 0.2002 | 0.1516 | 0.7526 | 0.1505 | 0.2683 | 0.1546 | 0.1475 | 0.6958 | 0.201 | 0.356 |
| LPnetlib/lp_cre_b | 128 | identity_k32 | 0.2130 | 0.1546 | 0.3922 | - | - | 0.1495 | - | 0.3369 | 0.394 | - |
| LPnetlib/lp_cre_b | 128 | identity_k64 | 0.2145 | 0.1546 | 0.4168 | - | - | 0.1495 | - | 0.3620 | 0.371 | - |
| LPnetlib/lp_cre_b | 512 | hem_k32 | 0.6861 | 0.6260 | 1.8335 | 0.6083 | 1.0419 | 0.6277 | 0.6047 | 1.7556 | 0.341 | 0.568 |
| LPnetlib/lp_cre_b | 512 | hem_k64 | 0.6851 | 0.6257 | 2.7028 | 0.6011 | 1.0353 | 0.6267 | 0.5970 | 2.6184 | 0.231 | 0.383 |
| LPnetlib/lp_cre_b | 512 | identity_k32 | 0.6999 | 0.6292 | 1.2170 | - | - | 0.6246 | - | 1.1535 | 0.517 | - |
| LPnetlib/lp_cre_b | 512 | identity_k64 | 0.7066 | 0.6292 | 1.3875 | - | - | 0.6236 | - | 1.3256 | 0.454 | - |
| SNAP/roadNet-CA | 32 | hem_k32 | 1.4285 | 1.1295 | 3.2732 | 1.1059 | 3.9332 | 1.1315 | 1.1018 | 3.2015 | 0.345 | 1.202 |
| SNAP/roadNet-CA | 32 | hem_k64 | 1.4300 | 1.1325 | 2.8175 | 1.1090 | 3.9588 | 1.1341 | 1.1075 | 2.7500 | 0.402 | 1.405 |
| SNAP/roadNet-CA | 32 | identity_k32 | 1.4413 | 1.1392 | 3.5594 | - | - | 1.1382 | - | 3.4970 | 0.320 | - |
| SNAP/roadNet-CA | 32 | identity_k64 | 1.4444 | 1.1433 | 3.6337 | - | - | 1.1397 | - | 3.5794 | 0.315 | - |
| SNAP/roadNet-CA | 128 | hem_k32 | 6.6273 | 4.4201 | 7.9401 | 4.3889 | 9.2580 | 4.4237 | 4.3750 | 7.9171 | 0.557 | 1.166 |
| SNAP/roadNet-CA | 128 | hem_k64 | 6.6330 | 4.4175 | 8.6625 | 4.3950 | 9.2641 | 4.4211 | 4.3709 | 8.5965 | 0.510 | 1.069 |
| SNAP/roadNet-CA | 128 | identity_k32 | 6.6442 | 4.4626 | 9.5662 | - | - | 4.4442 | - | 9.5032 | 0.466 | - |
| SNAP/roadNet-CA | 128 | identity_k64 | 6.6478 | 4.4641 | 11.9301 | - | - | 4.4462 | - | 11.8308 | 0.374 | - |
| SNAP/roadNet-CA | 512 | hem_k32 | 26.5370 | 21.9904 | 28.7590 | 21.6330 | 40.8781 | 21.9930 | 21.6233 | - | 0.765 | 1.421 |
| SNAP/roadNet-CA | 512 | hem_k64 | 26.5375 | 22.0134 | 32.6461 | 21.6274 | 40.8970 | 22.0109 | 21.6320 | - | 0.674 | 1.253 |
| SNAP/roadNet-CA | 512 | identity_k32 | 26.5728 | 22.0114 | 35.4284 | - | - | 22.0083 | - | 34.8063 | 0.621 | - |
| SNAP/roadNet-CA | 512 | identity_k64 | 26.5866 | 21.9756 | 45.7784 | - | - | 21.9858 | - | 45.0678 | 0.480 | - |
| Schmid/thermal2 | 32 | hem_k32 | 1.2877 | 1.0486 | 3.1508 | 0.9953 | 2.7786 | 1.0501 | 0.9928 | 3.0817 | 0.333 | 0.882 |
| Schmid/thermal2 | 32 | hem_k64 | 1.2918 | 1.0470 | 2.5492 | 0.9964 | 2.7945 | 1.0496 | 0.9923 | 2.4893 | 0.411 | 1.096 |
| Schmid/thermal2 | 32 | identity_k32 | 1.2969 | 1.0537 | 4.5430 | - | - | 1.0476 | - | 4.4774 | 0.232 | - |
| Schmid/thermal2 | 32 | identity_k64 | 1.3164 | 1.0522 | 5.1876 | - | - | 1.0476 | - | 5.1220 | 0.203 | - |
| Schmid/thermal2 | 128 | hem_k32 | 5.5020 | 4.4820 | 7.9227 | 4.0064 | 7.0477 | 4.4861 | 3.9997 | 7.8561 | 0.566 | 0.890 |
| Schmid/thermal2 | 128 | hem_k64 | 5.5101 | 4.4708 | 9.6220 | 4.0259 | 7.0661 | 4.4728 | 4.0177 | 9.4976 | 0.465 | 0.734 |
| Schmid/thermal2 | 128 | identity_k32 | 5.5250 | 4.4718 | 12.6259 | - | - | 4.4611 | - | 12.6249 | 0.354 | - |
| Schmid/thermal2 | 128 | identity_k64 | 5.5460 | 4.4790 | 18.7904 | - | - | 4.4728 | - | 18.7796 | 0.238 | - |
| Schmid/thermal2 | 512 | hem_k32 | 21.8875 | 18.9906 | 29.5255 | 17.0194 | 29.0263 | 18.9916 | 17.0189 | 30.5587 | 0.643 | 0.983 |
| Schmid/thermal2 | 512 | hem_k64 | 21.8803 | 18.9804 | 36.4365 | 16.8637 | 28.8681 | 18.9804 | 16.8545 | 36.9864 | 0.521 | 0.792 |
| Schmid/thermal2 | 512 | identity_k32 | 21.8808 | 17.3199 | 49.4479 | - | - | 17.3128 | - | 48.7910 | 0.350 | - |
| Schmid/thermal2 | 512 | identity_k64 | 21.8844 | 17.3404 | 75.0198 | - | - | 17.3286 | - | 74.7633 | 0.231 | - |
| Williams/webbase-1M | 32 | hem_k32 | 0.7598 | 0.5458 | 1.3716 | 0.5489 | 1.9876 | 0.5463 | 0.5448 | 1.2984 | 0.398 | 1.449 |
| Williams/webbase-1M | 32 | hem_k64 | 0.7567 | 0.5453 | 1.6932 | 0.5468 | 1.9784 | 0.5437 | 0.5437 | 1.6128 | 0.322 | 1.168 |
| Williams/webbase-1M | 32 | identity_k32 | 0.7818 | 0.5673 | 1.7280 | - | - | 0.5612 | - | 1.6676 | 0.328 | - |
| Williams/webbase-1M | 32 | identity_k64 | 0.7782 | 0.5540 | 2.1120 | - | - | 0.5473 | - | 2.0567 | 0.262 | - |
| Williams/webbase-1M | 128 | hem_k32 | 2.8221 | 2.0716 | 4.2547 | 2.0721 | 4.5722 | 2.0741 | 2.0669 | 4.1815 | 0.487 | 1.075 |
| Williams/webbase-1M | 128 | hem_k64 | 2.8201 | 2.0526 | 6.5172 | 2.0649 | 4.5548 | 2.0526 | 2.0552 | 6.3882 | 0.315 | 0.699 |
| Williams/webbase-1M | 128 | identity_k32 | 2.8314 | 2.1053 | 4.3540 | - | - | 2.1053 | - | 4.3034 | 0.484 | - |
| Williams/webbase-1M | 128 | identity_k64 | 2.8488 | 2.0685 | 7.5187 | - | - | 2.0608 | - | 7.4368 | 0.275 | - |
| Williams/webbase-1M | 512 | hem_k32 | 11.2384 | 8.0343 | 16.6303 | 8.7997 | 18.5457 | 8.0302 | 8.7813 | 16.5924 | 0.483 | 1.115 |
| Williams/webbase-1M | 512 | hem_k64 | 11.2123 | 8.0118 | 25.0798 | 8.7460 | 18.5093 | 8.0082 | 8.7383 | 24.8248 | 0.319 | 0.738 |
| Williams/webbase-1M | 512 | identity_k32 | 11.2599 | 8.2591 | 16.7962 | - | - | 8.1408 | - | 16.6712 | 0.492 | - |
| Williams/webbase-1M | 512 | identity_k64 | 11.2486 | 8.0635 | 29.0734 | - | - | 8.0195 | - | 29.0232 | 0.277 | - |

Features-per-thread variants, paired with FPT=1 in the same cell (ratio = FPT1 ms / variant ms; >1 is faster):
| matrix | F | labels | variant | ms | FPT1 ms | ratio | variant / best direct (same scope) |
|---|---:|---|---|---:|---:|---:|---:|
| AMD/G2_circuit | 32 | hem_k32 | boo_fpt2 | 0.3850 | 0.4024 | 1.045 | 0.258 |
| AMD/G2_circuit | 32 | hem_k32 | boo_fpt2_cpp | 0.3308 | 0.3461 | 1.046 | 0.300 |
| AMD/G2_circuit | 32 | hem_k32 | boo_fpt4 | 0.3773 | 0.4024 | 1.066 | 0.263 |
| AMD/G2_circuit | 32 | hem_k32 | boo_fpt4_cpp | 0.3256 | 0.3461 | 1.063 | 0.305 |
| AMD/G2_circuit | 32 | hem_k64 | boo_fpt2 | 0.3487 | 0.3625 | 1.040 | 0.286 |
| AMD/G2_circuit | 32 | hem_k64 | boo_fpt2_cpp | 0.2949 | 0.3082 | 1.045 | 0.337 |
| AMD/G2_circuit | 32 | hem_k64 | boo_fpt4 | 0.3369 | 0.3625 | 1.076 | 0.296 |
| AMD/G2_circuit | 32 | hem_k64 | boo_fpt4_cpp | 0.2877 | 0.3082 | 1.071 | 0.345 |
| AMD/G2_circuit | 32 | identity_k32 | boo_fpt2 | 0.3149 | 0.3348 | 1.063 | 0.327 |
| AMD/G2_circuit | 32 | identity_k32 | boo_fpt2_cpp | 0.2652 | 0.2806 | 1.058 | 0.369 |
| AMD/G2_circuit | 32 | identity_k32 | boo_fpt4 | 0.3251 | 0.3348 | 1.030 | 0.317 |
| AMD/G2_circuit | 32 | identity_k32 | boo_fpt4_cpp | 0.2775 | 0.2806 | 1.011 | 0.352 |
| AMD/G2_circuit | 32 | identity_k64 | boo_fpt2 | 0.2616 | 0.2811 | 1.074 | 0.384 |
| AMD/G2_circuit | 32 | identity_k64 | boo_fpt2_cpp | 0.2150 | 0.2263 | 1.052 | 0.440 |
| AMD/G2_circuit | 32 | identity_k64 | boo_fpt4 | 0.2734 | 0.2811 | 1.028 | 0.367 |
| AMD/G2_circuit | 32 | identity_k64 | boo_fpt4_cpp | 0.2273 | 0.2263 | 0.995 | 0.417 |
| AMD/G2_circuit | 128 | hem_k32 | boo_fpt2 | 0.8361 | 0.8950 | 1.070 | 0.431 |
| AMD/G2_circuit | 128 | hem_k32 | boo_fpt2_cpp | 0.7629 | 0.8131 | 1.066 | 0.473 |
| AMD/G2_circuit | 128 | hem_k32 | boo_fpt4 | 0.8202 | 0.8950 | 1.091 | 0.439 |
| AMD/G2_circuit | 128 | hem_k32 | boo_fpt4_cpp | 0.7613 | 0.8131 | 1.068 | 0.474 |
| AMD/G2_circuit | 128 | hem_k64 | boo_fpt2 | 0.8888 | 0.9528 | 1.072 | 0.406 |
| AMD/G2_circuit | 128 | hem_k64 | boo_fpt2_cpp | 0.8269 | 0.8796 | 1.064 | 0.433 |
| AMD/G2_circuit | 128 | hem_k64 | boo_fpt4 | 0.8607 | 0.9528 | 1.107 | 0.419 |
| AMD/G2_circuit | 128 | hem_k64 | boo_fpt4_cpp | 0.7982 | 0.8796 | 1.102 | 0.448 |
| AMD/G2_circuit | 128 | identity_k32 | boo_fpt2 | 0.7429 | 0.7880 | 1.061 | 0.489 |
| AMD/G2_circuit | 128 | identity_k32 | boo_fpt2_cpp | 0.6886 | 0.7286 | 1.058 | 0.519 |
| AMD/G2_circuit | 128 | identity_k32 | boo_fpt4 | 0.8397 | 0.7880 | 0.938 | 0.433 |
| AMD/G2_circuit | 128 | identity_k32 | boo_fpt4_cpp | 0.7875 | 0.7286 | 0.925 | 0.454 |
| AMD/G2_circuit | 128 | identity_k64 | boo_fpt2 | 0.7660 | 0.8115 | 1.059 | 0.482 |
| AMD/G2_circuit | 128 | identity_k64 | boo_fpt2_cpp | 0.7188 | 0.7598 | 1.057 | 0.507 |
| AMD/G2_circuit | 128 | identity_k64 | boo_fpt4 | 0.8627 | 0.8115 | 0.941 | 0.428 |
| AMD/G2_circuit | 128 | identity_k64 | boo_fpt4_cpp | 0.8187 | 0.7598 | 0.928 | 0.445 |
| AMD/G2_circuit | 512 | hem_k32 | boo_fpt2 | 2.9225 | 3.1319 | 1.072 | 0.492 |
| AMD/G2_circuit | 512 | hem_k32 | boo_fpt2_cpp | 2.8339 | 3.0141 | 1.064 | 0.507 |
| AMD/G2_circuit | 512 | hem_k32 | boo_fpt4 | 2.9563 | 3.1319 | 1.059 | 0.486 |
| AMD/G2_circuit | 512 | hem_k32 | boo_fpt4_cpp | 2.8605 | 3.0141 | 1.054 | 0.502 |
| AMD/G2_circuit | 512 | hem_k64 | boo_fpt2 | 3.4273 | 3.7048 | 1.081 | 0.420 |
| AMD/G2_circuit | 512 | hem_k64 | boo_fpt2_cpp | 3.3398 | 3.5702 | 1.069 | 0.432 |
| AMD/G2_circuit | 512 | hem_k64 | boo_fpt4 | 3.3516 | 3.7048 | 1.105 | 0.430 |
| AMD/G2_circuit | 512 | hem_k64 | boo_fpt4_cpp | 3.2722 | 3.5702 | 1.091 | 0.440 |
| AMD/G2_circuit | 512 | identity_k32 | boo_fpt2 | 2.5672 | 2.7438 | 1.069 | 0.562 |
| AMD/G2_circuit | 512 | identity_k32 | boo_fpt2_cpp | 2.5098 | 2.6762 | 1.066 | 0.573 |
| AMD/G2_circuit | 512 | identity_k32 | boo_fpt4 | 3.0290 | 2.7438 | 0.906 | 0.476 |
| AMD/G2_circuit | 512 | identity_k32 | boo_fpt4_cpp | 2.9809 | 2.6762 | 0.898 | 0.482 |
| AMD/G2_circuit | 512 | identity_k64 | boo_fpt2 | 2.6803 | 2.8396 | 1.059 | 0.540 |
| AMD/G2_circuit | 512 | identity_k64 | boo_fpt2_cpp | 2.6179 | 2.7725 | 1.059 | 0.550 |
| AMD/G2_circuit | 512 | identity_k64 | boo_fpt4 | 3.0746 | 2.8396 | 0.924 | 0.471 |
| AMD/G2_circuit | 512 | identity_k64 | boo_fpt4_cpp | 3.0275 | 2.7725 | 0.916 | 0.476 |
| Boeing/pwtk | 32 | hem_k32 | boo_fpt2 | 0.8192 | 1.0291 | 1.256 | 0.741 |
| Boeing/pwtk | 32 | hem_k32 | boo_fpt2_cpp | 0.7496 | 0.9646 | 1.287 | 0.807 |
| Boeing/pwtk | 32 | hem_k32 | boo_fpt4 | 0.7398 | 1.0291 | 1.391 | 0.821 |
| Boeing/pwtk | 32 | hem_k32 | boo_fpt4_cpp | 0.6820 | 0.9646 | 1.414 | 0.887 |
| Boeing/pwtk | 32 | hem_k64 | boo_fpt2 | 0.7951 | 1.0097 | 1.270 | 0.757 |
| Boeing/pwtk | 32 | hem_k64 | boo_fpt2_cpp | 0.7214 | 0.9400 | 1.303 | 0.832 |
| Boeing/pwtk | 32 | hem_k64 | boo_fpt4 | 0.7020 | 1.0097 | 1.438 | 0.857 |
| Boeing/pwtk | 32 | hem_k64 | boo_fpt4_cpp | 0.6395 | 0.9400 | 1.470 | 0.938 |
| Boeing/pwtk | 32 | identity_k32 | boo_fpt2 | 0.7864 | 0.9902 | 1.259 | 0.784 |
| Boeing/pwtk | 32 | identity_k32 | boo_fpt2_cpp | 0.7301 | 0.9196 | 1.259 | 0.837 |
| Boeing/pwtk | 32 | identity_k32 | boo_fpt4 | 0.7004 | 0.9902 | 1.414 | 0.880 |
| Boeing/pwtk | 32 | identity_k32 | boo_fpt4_cpp | 0.6513 | 0.9196 | 1.412 | 0.939 |
| Boeing/pwtk | 32 | identity_k64 | boo_fpt2 | 0.7803 | 0.9841 | 1.261 | 0.785 |
| Boeing/pwtk | 32 | identity_k64 | boo_fpt2_cpp | 0.7276 | 0.9216 | 1.267 | 0.835 |
| Boeing/pwtk | 32 | identity_k64 | boo_fpt4 | 0.6943 | 0.9841 | 1.417 | 0.882 |
| Boeing/pwtk | 32 | identity_k64 | boo_fpt4_cpp | 0.6420 | 0.9216 | 1.435 | 0.947 |
| Boeing/pwtk | 128 | hem_k32 | boo_fpt2 | 2.1934 | 3.0628 | 1.396 | 0.893 |
| Boeing/pwtk | 128 | hem_k32 | boo_fpt2_cpp | 2.0285 | 3.0285 | 1.493 | 0.966 |
| Boeing/pwtk | 128 | hem_k32 | boo_fpt4 | 1.8908 | 3.0628 | 1.620 | 1.036 |
| Boeing/pwtk | 128 | hem_k32 | boo_fpt4_cpp | 1.7997 | 3.0285 | 1.683 | 1.089 |
| Boeing/pwtk | 128 | hem_k64 | boo_fpt2 | 2.5421 | 3.4212 | 1.346 | 0.768 |
| Boeing/pwtk | 128 | hem_k64 | boo_fpt2_cpp | 2.3793 | 3.3674 | 1.415 | 0.823 |
| Boeing/pwtk | 128 | hem_k64 | boo_fpt4 | 2.1268 | 3.4212 | 1.609 | 0.918 |
| Boeing/pwtk | 128 | hem_k64 | boo_fpt4_cpp | 2.0204 | 3.3674 | 1.667 | 0.969 |
| Boeing/pwtk | 128 | identity_k32 | boo_fpt2 | 2.0378 | 3.0141 | 1.479 | 0.981 |
| Boeing/pwtk | 128 | identity_k32 | boo_fpt2_cpp | 1.9620 | 2.9322 | 1.495 | 1.007 |
| Boeing/pwtk | 128 | identity_k32 | boo_fpt4 | 1.8929 | 3.0141 | 1.592 | 1.056 |
| Boeing/pwtk | 128 | identity_k32 | boo_fpt4_cpp | 1.8406 | 2.9322 | 1.593 | 1.074 |
| Boeing/pwtk | 128 | identity_k64 | boo_fpt2 | 2.4776 | 3.4028 | 1.373 | 0.793 |
| Boeing/pwtk | 128 | identity_k64 | boo_fpt2_cpp | 2.4125 | 3.3121 | 1.373 | 0.812 |
| Boeing/pwtk | 128 | identity_k64 | boo_fpt4 | 2.1683 | 3.4028 | 1.569 | 0.906 |
| Boeing/pwtk | 128 | identity_k64 | boo_fpt4_cpp | 2.1187 | 3.3121 | 1.563 | 0.925 |
| Boeing/pwtk | 512 | hem_k32 | boo_fpt2 | 7.5156 | 12.1636 | 1.618 | 1.068 |
| Boeing/pwtk | 512 | hem_k32 | boo_fpt2_cpp | 7.9698 | 11.0689 | 1.389 | 0.967 |
| Boeing/pwtk | 512 | hem_k32 | boo_fpt4 | 6.5935 | 12.1636 | 1.845 | 1.217 |
| Boeing/pwtk | 512 | hem_k32 | boo_fpt4_cpp | 6.7697 | 11.0689 | 1.635 | 1.138 |
| Boeing/pwtk | 512 | hem_k64 | boo_fpt2 | 9.2375 | 14.0621 | 1.522 | 0.857 |
| Boeing/pwtk | 512 | hem_k64 | boo_fpt2_cpp | 9.9656 | 12.8497 | 1.289 | 0.778 |
| Boeing/pwtk | 512 | hem_k64 | boo_fpt4 | 7.8259 | 14.0621 | 1.797 | 1.011 |
| Boeing/pwtk | 512 | hem_k64 | boo_fpt4_cpp | 8.2483 | 12.8497 | 1.558 | 0.939 |
| Boeing/pwtk | 512 | identity_k32 | boo_fpt2 | 7.3196 | 11.2082 | 1.531 | 1.082 |
| Boeing/pwtk | 512 | identity_k32 | boo_fpt2_cpp | 7.2182 | 10.9245 | 1.513 | 1.115 |
| Boeing/pwtk | 512 | identity_k32 | boo_fpt4 | 6.7031 | 11.2082 | 1.672 | 1.182 |
| Boeing/pwtk | 512 | identity_k32 | boo_fpt4_cpp | 6.7855 | 10.9245 | 1.610 | 1.186 |
| Boeing/pwtk | 512 | identity_k64 | boo_fpt2 | 8.8929 | 12.7744 | 1.436 | 0.885 |
| Boeing/pwtk | 512 | identity_k64 | boo_fpt2_cpp | 8.7982 | 12.4964 | 1.420 | 0.897 |
| Boeing/pwtk | 512 | identity_k64 | boo_fpt4 | 7.6349 | 12.7744 | 1.673 | 1.031 |
| Boeing/pwtk | 512 | identity_k64 | boo_fpt4_cpp | 7.6913 | 12.4964 | 1.625 | 1.026 |
| LPnetlib/lp_cre_b | 32 | hem_k32 | boo_fpt2 | 0.2458 | 0.2509 | 1.021 | 0.208 |
| LPnetlib/lp_cre_b | 32 | hem_k32 | boo_fpt2_cpp | 0.1884 | 0.1956 | 1.038 | 0.277 |
| LPnetlib/lp_cre_b | 32 | hem_k32 | boo_fpt4 | 0.2396 | 0.2509 | 1.047 | 0.214 |
| LPnetlib/lp_cre_b | 32 | hem_k32 | boo_fpt4_cpp | 0.1874 | 0.1956 | 1.044 | 0.279 |
| LPnetlib/lp_cre_b | 32 | hem_k64 | boo_fpt2 | 0.2493 | 0.2519 | 1.010 | 0.205 |
| LPnetlib/lp_cre_b | 32 | hem_k64 | boo_fpt2_cpp | 0.1889 | 0.1961 | 1.038 | 0.276 |
| LPnetlib/lp_cre_b | 32 | hem_k64 | boo_fpt4 | 0.2376 | 0.2519 | 1.060 | 0.216 |
| LPnetlib/lp_cre_b | 32 | hem_k64 | boo_fpt4_cpp | 0.1843 | 0.1961 | 1.064 | 0.283 |
| LPnetlib/lp_cre_b | 32 | identity_k32 | boo_fpt2 | 0.1782 | 0.1869 | 1.049 | 0.299 |
| LPnetlib/lp_cre_b | 32 | identity_k32 | boo_fpt2_cpp | 0.1270 | 0.1331 | 1.048 | 0.387 |
| LPnetlib/lp_cre_b | 32 | identity_k32 | boo_fpt4 | 0.1766 | 0.1869 | 1.058 | 0.301 |
| LPnetlib/lp_cre_b | 32 | identity_k32 | boo_fpt4_cpp | 0.1260 | 0.1331 | 1.057 | 0.390 |
| LPnetlib/lp_cre_b | 32 | identity_k64 | boo_fpt2 | 0.1551 | 0.1618 | 1.043 | 0.337 |
| LPnetlib/lp_cre_b | 32 | identity_k64 | boo_fpt2_cpp | 0.1044 | 0.1116 | 1.069 | 0.461 |
| LPnetlib/lp_cre_b | 32 | identity_k64 | boo_fpt4 | 0.1464 | 0.1618 | 1.105 | 0.357 |
| LPnetlib/lp_cre_b | 32 | identity_k64 | boo_fpt4_cpp | 0.0983 | 0.1116 | 1.135 | 0.490 |
| LPnetlib/lp_cre_b | 128 | hem_k32 | boo_fpt2 | 0.5320 | 0.5509 | 1.036 | 0.287 |
| LPnetlib/lp_cre_b | 128 | hem_k32 | boo_fpt2_cpp | 0.4705 | 0.4895 | 1.040 | 0.329 |
| LPnetlib/lp_cre_b | 128 | hem_k32 | boo_fpt4 | 0.5135 | 0.5509 | 1.073 | 0.297 |
| LPnetlib/lp_cre_b | 128 | hem_k32 | boo_fpt4_cpp | 0.4598 | 0.4895 | 1.065 | 0.336 |
| LPnetlib/lp_cre_b | 128 | hem_k64 | boo_fpt2 | 0.7270 | 0.7526 | 1.035 | 0.208 |
| LPnetlib/lp_cre_b | 128 | hem_k64 | boo_fpt2_cpp | 0.6687 | 0.6958 | 1.041 | 0.231 |
| LPnetlib/lp_cre_b | 128 | hem_k64 | boo_fpt4 | 0.6968 | 0.7526 | 1.080 | 0.217 |
| LPnetlib/lp_cre_b | 128 | hem_k64 | boo_fpt4_cpp | 0.6436 | 0.6958 | 1.081 | 0.240 |
| LPnetlib/lp_cre_b | 128 | identity_k32 | boo_fpt2 | 0.3656 | 0.3922 | 1.073 | 0.423 |
| LPnetlib/lp_cre_b | 128 | identity_k32 | boo_fpt2_cpp | 0.3144 | 0.3369 | 1.072 | 0.476 |
| LPnetlib/lp_cre_b | 128 | identity_k32 | boo_fpt4 | 0.3594 | 0.3922 | 1.091 | 0.430 |
| LPnetlib/lp_cre_b | 128 | identity_k32 | boo_fpt4_cpp | 0.3144 | 0.3369 | 1.072 | 0.476 |
| LPnetlib/lp_cre_b | 128 | identity_k64 | boo_fpt2 | 0.3876 | 0.4168 | 1.075 | 0.399 |
| LPnetlib/lp_cre_b | 128 | identity_k64 | boo_fpt2_cpp | 0.3364 | 0.3620 | 1.076 | 0.444 |
| LPnetlib/lp_cre_b | 128 | identity_k64 | boo_fpt4 | 0.3697 | 0.4168 | 1.127 | 0.418 |
| LPnetlib/lp_cre_b | 128 | identity_k64 | boo_fpt4_cpp | 0.3226 | 0.3620 | 1.122 | 0.463 |
| LPnetlib/lp_cre_b | 512 | hem_k32 | boo_fpt2 | 1.7777 | 1.8335 | 1.031 | 0.352 |
| LPnetlib/lp_cre_b | 512 | hem_k32 | boo_fpt2_cpp | 1.6998 | 1.7556 | 1.033 | 0.369 |
| LPnetlib/lp_cre_b | 512 | hem_k32 | boo_fpt4 | 1.7121 | 1.8335 | 1.071 | 0.366 |
| LPnetlib/lp_cre_b | 512 | hem_k32 | boo_fpt4_cpp | 1.6512 | 1.7556 | 1.063 | 0.380 |
| LPnetlib/lp_cre_b | 512 | hem_k64 | boo_fpt2 | 2.6076 | 2.7028 | 1.037 | 0.240 |
| LPnetlib/lp_cre_b | 512 | hem_k64 | boo_fpt2_cpp | 2.5364 | 2.6184 | 1.032 | 0.247 |
| LPnetlib/lp_cre_b | 512 | hem_k64 | boo_fpt4 | 2.5103 | 2.7028 | 1.077 | 0.249 |
| LPnetlib/lp_cre_b | 512 | hem_k64 | boo_fpt4_cpp | 2.4407 | 2.6184 | 1.073 | 0.257 |
| LPnetlib/lp_cre_b | 512 | identity_k32 | boo_fpt2 | 1.1264 | 1.2170 | 1.080 | 0.559 |
| LPnetlib/lp_cre_b | 512 | identity_k32 | boo_fpt2_cpp | 1.0737 | 1.1535 | 1.074 | 0.582 |
| LPnetlib/lp_cre_b | 512 | identity_k32 | boo_fpt4 | 1.0870 | 1.2170 | 1.120 | 0.579 |
| LPnetlib/lp_cre_b | 512 | identity_k32 | boo_fpt4_cpp | 1.0435 | 1.1535 | 1.105 | 0.599 |
| LPnetlib/lp_cre_b | 512 | identity_k64 | boo_fpt2 | 1.2867 | 1.3875 | 1.078 | 0.489 |
| LPnetlib/lp_cre_b | 512 | identity_k64 | boo_fpt2_cpp | 1.2314 | 1.3256 | 1.077 | 0.506 |
| LPnetlib/lp_cre_b | 512 | identity_k64 | boo_fpt4 | 1.2160 | 1.3875 | 1.141 | 0.517 |
| LPnetlib/lp_cre_b | 512 | identity_k64 | boo_fpt4_cpp | 1.1699 | 1.3256 | 1.133 | 0.533 |
| SNAP/roadNet-CA | 32 | hem_k32 | boo_fpt2 | 3.1078 | 3.2732 | 1.053 | 0.363 |
| SNAP/roadNet-CA | 32 | hem_k32 | boo_fpt2_cpp | 3.0920 | 3.2015 | 1.035 | 0.366 |
| SNAP/roadNet-CA | 32 | hem_k32 | boo_fpt4 | 3.0828 | 3.2732 | 1.062 | 0.366 |
| SNAP/roadNet-CA | 32 | hem_k32 | boo_fpt4_cpp | 3.0080 | 3.2015 | 1.064 | 0.376 |
| SNAP/roadNet-CA | 32 | hem_k64 | boo_fpt2 | 2.6895 | 2.8175 | 1.048 | 0.421 |
| SNAP/roadNet-CA | 32 | hem_k64 | boo_fpt2_cpp | 2.6537 | 2.7500 | 1.036 | 0.427 |
| SNAP/roadNet-CA | 32 | hem_k64 | boo_fpt4 | 2.6445 | 2.8175 | 1.065 | 0.428 |
| SNAP/roadNet-CA | 32 | hem_k64 | boo_fpt4_cpp | 2.5795 | 2.7500 | 1.066 | 0.440 |
| SNAP/roadNet-CA | 32 | identity_k32 | boo_fpt2 | 3.3920 | 3.5594 | 1.049 | 0.336 |
| SNAP/roadNet-CA | 32 | identity_k32 | boo_fpt2_cpp | 3.3464 | 3.4970 | 1.045 | 0.340 |
| SNAP/roadNet-CA | 32 | identity_k32 | boo_fpt4 | 3.3311 | 3.5594 | 1.069 | 0.342 |
| SNAP/roadNet-CA | 32 | identity_k32 | boo_fpt4_cpp | 3.2911 | 3.4970 | 1.063 | 0.346 |
| SNAP/roadNet-CA | 32 | identity_k64 | boo_fpt2 | 3.4872 | 3.6337 | 1.042 | 0.328 |
| SNAP/roadNet-CA | 32 | identity_k64 | boo_fpt2_cpp | 3.4381 | 3.5794 | 1.041 | 0.331 |
| SNAP/roadNet-CA | 32 | identity_k64 | boo_fpt4 | 3.3649 | 3.6337 | 1.080 | 0.340 |
| SNAP/roadNet-CA | 32 | identity_k64 | boo_fpt4_cpp | 3.3234 | 3.5794 | 1.077 | 0.343 |
| SNAP/roadNet-CA | 128 | hem_k32 | boo_fpt2 | 7.6221 | 7.9401 | 1.042 | 0.580 |
| SNAP/roadNet-CA | 128 | hem_k32 | boo_fpt2_cpp | 7.4429 | 7.9171 | 1.064 | 0.594 |
| SNAP/roadNet-CA | 128 | hem_k32 | boo_fpt4 | 7.5832 | 7.9401 | 1.047 | 0.583 |
| SNAP/roadNet-CA | 128 | hem_k32 | boo_fpt4_cpp | 7.4486 | 7.9171 | 1.063 | 0.594 |
| SNAP/roadNet-CA | 128 | hem_k64 | boo_fpt2 | 8.2970 | 8.6625 | 1.044 | 0.532 |
| SNAP/roadNet-CA | 128 | hem_k64 | boo_fpt2_cpp | 8.0911 | 8.5965 | 1.062 | 0.546 |
| SNAP/roadNet-CA | 128 | hem_k64 | boo_fpt4 | 8.0886 | 8.6625 | 1.071 | 0.546 |
| SNAP/roadNet-CA | 128 | hem_k64 | boo_fpt4_cpp | 7.9380 | 8.5965 | 1.083 | 0.557 |
| SNAP/roadNet-CA | 128 | identity_k32 | boo_fpt2 | 9.0609 | 9.5662 | 1.056 | 0.493 |
| SNAP/roadNet-CA | 128 | identity_k32 | boo_fpt2_cpp | 9.0312 | 9.5032 | 1.052 | 0.492 |
| SNAP/roadNet-CA | 128 | identity_k32 | boo_fpt4 | 9.0373 | 9.5662 | 1.059 | 0.494 |
| SNAP/roadNet-CA | 128 | identity_k32 | boo_fpt4_cpp | 9.0568 | 9.5032 | 1.049 | 0.491 |
| SNAP/roadNet-CA | 128 | identity_k64 | boo_fpt2 | 11.1811 | 11.9301 | 1.067 | 0.399 |
| SNAP/roadNet-CA | 128 | identity_k64 | boo_fpt2_cpp | 11.1826 | 11.8308 | 1.058 | 0.398 |
| SNAP/roadNet-CA | 128 | identity_k64 | boo_fpt4 | 10.8052 | 11.9301 | 1.104 | 0.413 |
| SNAP/roadNet-CA | 128 | identity_k64 | boo_fpt4_cpp | 10.8662 | 11.8308 | 1.089 | 0.409 |
| SNAP/roadNet-CA | 512 | hem_k32 | boo_fpt2 | 27.7284 | 28.7590 | 1.037 | 0.793 |
| SNAP/roadNet-CA | 512 | hem_k32 | boo_fpt4 | 28.5343 | 28.7590 | 1.008 | 0.771 |
| SNAP/roadNet-CA | 512 | hem_k64 | boo_fpt2 | 31.2837 | 32.6461 | 1.044 | 0.704 |
| SNAP/roadNet-CA | 512 | hem_k64 | boo_fpt4 | 31.5924 | 32.6461 | 1.033 | 0.697 |
| SNAP/roadNet-CA | 512 | identity_k32 | boo_fpt2 | 34.0352 | 35.4284 | 1.041 | 0.647 |
| SNAP/roadNet-CA | 512 | identity_k32 | boo_fpt2_cpp | 34.2395 | 34.8063 | 1.017 | 0.643 |
| SNAP/roadNet-CA | 512 | identity_k32 | boo_fpt4 | 34.1043 | 35.4284 | 1.039 | 0.645 |
| SNAP/roadNet-CA | 512 | identity_k32 | boo_fpt4_cpp | 34.0797 | 34.8063 | 1.021 | 0.646 |
| SNAP/roadNet-CA | 512 | identity_k64 | boo_fpt2 | 43.5180 | 45.7784 | 1.052 | 0.505 |
| SNAP/roadNet-CA | 512 | identity_k64 | boo_fpt2_cpp | 43.8958 | 45.0678 | 1.027 | 0.501 |
| SNAP/roadNet-CA | 512 | identity_k64 | boo_fpt4 | 41.9789 | 45.7784 | 1.091 | 0.523 |
| SNAP/roadNet-CA | 512 | identity_k64 | boo_fpt4_cpp | 41.9753 | 45.0678 | 1.074 | 0.524 |
| Schmid/thermal2 | 32 | hem_k32 | boo_fpt2 | 2.9763 | 3.1508 | 1.059 | 0.352 |
| Schmid/thermal2 | 32 | hem_k32 | boo_fpt2_cpp | 2.9251 | 3.0817 | 1.054 | 0.359 |
| Schmid/thermal2 | 32 | hem_k32 | boo_fpt4 | 2.9158 | 3.1508 | 1.081 | 0.360 |
| Schmid/thermal2 | 32 | hem_k32 | boo_fpt4_cpp | 2.8380 | 3.0817 | 1.086 | 0.370 |
| Schmid/thermal2 | 32 | hem_k64 | boo_fpt2 | 2.4172 | 2.5492 | 1.055 | 0.433 |
| Schmid/thermal2 | 32 | hem_k64 | boo_fpt2_cpp | 2.3665 | 2.4893 | 1.052 | 0.444 |
| Schmid/thermal2 | 32 | hem_k64 | boo_fpt4 | 2.3388 | 2.5492 | 1.090 | 0.448 |
| Schmid/thermal2 | 32 | hem_k64 | boo_fpt4_cpp | 2.2799 | 2.4893 | 1.092 | 0.460 |
| Schmid/thermal2 | 32 | identity_k32 | boo_fpt2 | 4.3295 | 4.5430 | 1.049 | 0.243 |
| Schmid/thermal2 | 32 | identity_k32 | boo_fpt2_cpp | 4.2772 | 4.4774 | 1.047 | 0.245 |
| Schmid/thermal2 | 32 | identity_k32 | boo_fpt4 | 4.2061 | 4.5430 | 1.080 | 0.251 |
| Schmid/thermal2 | 32 | identity_k32 | boo_fpt4_cpp | 4.1595 | 4.4774 | 1.076 | 0.252 |
| Schmid/thermal2 | 32 | identity_k64 | boo_fpt2 | 4.9925 | 5.1876 | 1.039 | 0.211 |
| Schmid/thermal2 | 32 | identity_k64 | boo_fpt2_cpp | 4.9377 | 5.1220 | 1.037 | 0.212 |
| Schmid/thermal2 | 32 | identity_k64 | boo_fpt4 | 4.7160 | 5.1876 | 1.100 | 0.223 |
| Schmid/thermal2 | 32 | identity_k64 | boo_fpt4_cpp | 4.6725 | 5.1220 | 1.096 | 0.224 |
| Schmid/thermal2 | 128 | hem_k32 | boo_fpt2 | 7.3620 | 7.9227 | 1.076 | 0.609 |
| Schmid/thermal2 | 128 | hem_k32 | boo_fpt2_cpp | 7.2084 | 7.8561 | 1.090 | 0.622 |
| Schmid/thermal2 | 128 | hem_k32 | boo_fpt4 | 7.2668 | 7.9227 | 1.090 | 0.617 |
| Schmid/thermal2 | 128 | hem_k32 | boo_fpt4_cpp | 7.1583 | 7.8561 | 1.097 | 0.627 |
| Schmid/thermal2 | 128 | hem_k64 | boo_fpt2 | 8.9083 | 9.6220 | 1.080 | 0.502 |
| Schmid/thermal2 | 128 | hem_k64 | boo_fpt2_cpp | 8.7316 | 9.4976 | 1.088 | 0.512 |
| Schmid/thermal2 | 128 | hem_k64 | boo_fpt4 | 8.5274 | 9.6220 | 1.128 | 0.524 |
| Schmid/thermal2 | 128 | hem_k64 | boo_fpt4_cpp | 8.4178 | 9.4976 | 1.128 | 0.531 |
| Schmid/thermal2 | 128 | identity_k32 | boo_fpt2 | 11.8149 | 12.6259 | 1.069 | 0.378 |
| Schmid/thermal2 | 128 | identity_k32 | boo_fpt2_cpp | 11.7985 | 12.6249 | 1.070 | 0.378 |
| Schmid/thermal2 | 128 | identity_k32 | boo_fpt4 | 11.4637 | 12.6259 | 1.101 | 0.390 |
| Schmid/thermal2 | 128 | identity_k32 | boo_fpt4_cpp | 11.5922 | 12.6249 | 1.089 | 0.385 |
| Schmid/thermal2 | 128 | identity_k64 | boo_fpt2 | 17.5478 | 18.7904 | 1.071 | 0.255 |
| Schmid/thermal2 | 128 | identity_k64 | boo_fpt2_cpp | 17.7265 | 18.7796 | 1.059 | 0.252 |
| Schmid/thermal2 | 128 | identity_k64 | boo_fpt4 | 16.6308 | 18.7904 | 1.130 | 0.269 |
| Schmid/thermal2 | 128 | identity_k64 | boo_fpt4_cpp | 16.8187 | 18.7796 | 1.117 | 0.266 |
| Schmid/thermal2 | 512 | hem_k32 | boo_fpt2 | 27.8410 | 29.5255 | 1.061 | 0.682 |
| Schmid/thermal2 | 512 | hem_k32 | boo_fpt2_cpp | 27.9941 | 30.5587 | 1.092 | 0.678 |
| Schmid/thermal2 | 512 | hem_k32 | boo_fpt4 | 28.0468 | 29.5255 | 1.053 | 0.677 |
| Schmid/thermal2 | 512 | hem_k32 | boo_fpt4_cpp | 27.7632 | 30.5587 | 1.101 | 0.684 |
| Schmid/thermal2 | 512 | hem_k64 | boo_fpt2 | 33.8570 | 36.4365 | 1.076 | 0.561 |
| Schmid/thermal2 | 512 | hem_k64 | boo_fpt2_cpp | 33.9789 | 36.9864 | 1.089 | 0.559 |
| Schmid/thermal2 | 512 | hem_k64 | boo_fpt4 | 33.3594 | 36.4365 | 1.092 | 0.569 |
| Schmid/thermal2 | 512 | hem_k64 | boo_fpt4_cpp | 33.0445 | 36.9864 | 1.119 | 0.574 |
| Schmid/thermal2 | 512 | identity_k32 | boo_fpt2 | 46.2556 | 49.4479 | 1.069 | 0.374 |
| Schmid/thermal2 | 512 | identity_k32 | boo_fpt2_cpp | 47.4465 | 48.7910 | 1.028 | 0.365 |
| Schmid/thermal2 | 512 | identity_k32 | boo_fpt4 | 46.2510 | 49.4479 | 1.069 | 0.374 |
| Schmid/thermal2 | 512 | identity_k32 | boo_fpt4_cpp | 46.2152 | 48.7910 | 1.056 | 0.375 |
| Schmid/thermal2 | 512 | identity_k64 | boo_fpt2 | 70.9990 | 75.0198 | 1.057 | 0.244 |
| Schmid/thermal2 | 512 | identity_k64 | boo_fpt2_cpp | 71.7286 | 74.7633 | 1.042 | 0.242 |
| Schmid/thermal2 | 512 | identity_k64 | boo_fpt4 | 68.1580 | 75.0198 | 1.101 | 0.254 |
| Schmid/thermal2 | 512 | identity_k64 | boo_fpt4_cpp | 68.2291 | 74.7633 | 1.096 | 0.254 |
| Williams/webbase-1M | 32 | hem_k32 | boo_fpt2 | 1.3471 | 1.3716 | 1.018 | 0.405 |
| Williams/webbase-1M | 32 | hem_k32 | boo_fpt2_cpp | 1.2739 | 1.2984 | 1.019 | 0.429 |
| Williams/webbase-1M | 32 | hem_k32 | boo_fpt4 | 1.4198 | 1.3716 | 0.966 | 0.384 |
| Williams/webbase-1M | 32 | hem_k32 | boo_fpt4_cpp | 1.3471 | 1.2984 | 0.964 | 0.406 |
| Williams/webbase-1M | 32 | hem_k64 | boo_fpt2 | 1.6148 | 1.6932 | 1.049 | 0.338 |
| Williams/webbase-1M | 32 | hem_k64 | boo_fpt2_cpp | 1.5411 | 1.6128 | 1.047 | 0.353 |
| Williams/webbase-1M | 32 | hem_k64 | boo_fpt4 | 1.6097 | 1.6932 | 1.052 | 0.339 |
| Williams/webbase-1M | 32 | hem_k64 | boo_fpt4_cpp | 1.5493 | 1.6128 | 1.041 | 0.351 |
| Williams/webbase-1M | 32 | identity_k32 | boo_fpt2 | 1.6297 | 1.7280 | 1.060 | 0.348 |
| Williams/webbase-1M | 32 | identity_k32 | boo_fpt2_cpp | 1.5724 | 1.6676 | 1.061 | 0.357 |
| Williams/webbase-1M | 32 | identity_k32 | boo_fpt4 | 1.7838 | 1.7280 | 0.969 | 0.318 |
| Williams/webbase-1M | 32 | identity_k32 | boo_fpt4_cpp | 1.7357 | 1.6676 | 0.961 | 0.323 |
| Williams/webbase-1M | 32 | identity_k64 | boo_fpt2 | 1.9732 | 2.1120 | 1.070 | 0.281 |
| Williams/webbase-1M | 32 | identity_k64 | boo_fpt2_cpp | 1.9261 | 2.0567 | 1.068 | 0.284 |
| Williams/webbase-1M | 32 | identity_k64 | boo_fpt4 | 2.0209 | 2.1120 | 1.045 | 0.274 |
| Williams/webbase-1M | 32 | identity_k64 | boo_fpt4_cpp | 1.9748 | 2.0567 | 1.041 | 0.277 |
| Williams/webbase-1M | 128 | hem_k32 | boo_fpt2 | 4.1774 | 4.2547 | 1.019 | 0.496 |
| Williams/webbase-1M | 128 | hem_k32 | boo_fpt2_cpp | 4.1318 | 4.1815 | 1.012 | 0.502 |
| Williams/webbase-1M | 128 | hem_k32 | boo_fpt4 | 4.4595 | 4.2547 | 0.954 | 0.465 |
| Williams/webbase-1M | 128 | hem_k32 | boo_fpt4_cpp | 4.4247 | 4.1815 | 0.945 | 0.469 |
| Williams/webbase-1M | 128 | hem_k64 | boo_fpt2 | 6.0744 | 6.5172 | 1.073 | 0.338 |
| Williams/webbase-1M | 128 | hem_k64 | boo_fpt2_cpp | 6.0237 | 6.3882 | 1.061 | 0.341 |
| Williams/webbase-1M | 128 | hem_k64 | boo_fpt4 | 6.0017 | 6.5172 | 1.086 | 0.342 |
| Williams/webbase-1M | 128 | hem_k64 | boo_fpt4_cpp | 5.9090 | 6.3882 | 1.081 | 0.347 |
| Williams/webbase-1M | 128 | identity_k32 | boo_fpt2 | 4.1651 | 4.3540 | 1.045 | 0.505 |
| Williams/webbase-1M | 128 | identity_k32 | boo_fpt2_cpp | 4.1073 | 4.3034 | 1.048 | 0.513 |
| Williams/webbase-1M | 128 | identity_k32 | boo_fpt4 | 4.8963 | 4.3540 | 0.889 | 0.430 |
| Williams/webbase-1M | 128 | identity_k32 | boo_fpt4_cpp | 4.8435 | 4.3034 | 0.888 | 0.435 |
| Williams/webbase-1M | 128 | identity_k64 | boo_fpt2 | 6.9130 | 7.5187 | 1.088 | 0.299 |
| Williams/webbase-1M | 128 | identity_k64 | boo_fpt2_cpp | 6.8547 | 7.4368 | 1.085 | 0.301 |
| Williams/webbase-1M | 128 | identity_k64 | boo_fpt4 | 6.8961 | 7.5187 | 1.090 | 0.300 |
| Williams/webbase-1M | 128 | identity_k64 | boo_fpt4_cpp | 6.8480 | 7.4368 | 1.086 | 0.301 |
| Williams/webbase-1M | 512 | hem_k32 | boo_fpt2 | 16.3727 | 16.6303 | 1.016 | 0.491 |
| Williams/webbase-1M | 512 | hem_k32 | boo_fpt2_cpp | 16.2826 | 16.5924 | 1.019 | 0.493 |
| Williams/webbase-1M | 512 | hem_k32 | boo_fpt4 | 17.5171 | 16.6303 | 0.949 | 0.459 |
| Williams/webbase-1M | 512 | hem_k32 | boo_fpt4_cpp | 17.3819 | 16.5924 | 0.955 | 0.462 |
| Williams/webbase-1M | 512 | hem_k64 | boo_fpt2 | 23.7891 | 25.0798 | 1.054 | 0.337 |
| Williams/webbase-1M | 512 | hem_k64 | boo_fpt2_cpp | 23.4742 | 24.8248 | 1.058 | 0.341 |
| Williams/webbase-1M | 512 | hem_k64 | boo_fpt4 | 23.1895 | 25.0798 | 1.082 | 0.345 |
| Williams/webbase-1M | 512 | hem_k64 | boo_fpt4_cpp | 23.0410 | 24.8248 | 1.077 | 0.348 |
| Williams/webbase-1M | 512 | identity_k32 | boo_fpt2 | 16.2785 | 16.7962 | 1.032 | 0.507 |
| Williams/webbase-1M | 512 | identity_k32 | boo_fpt2_cpp | 16.3615 | 16.6712 | 1.019 | 0.498 |
| Williams/webbase-1M | 512 | identity_k32 | boo_fpt4 | 19.0244 | 16.7962 | 0.883 | 0.434 |
| Williams/webbase-1M | 512 | identity_k32 | boo_fpt4_cpp | 19.1565 | 16.6712 | 0.870 | 0.425 |
| Williams/webbase-1M | 512 | identity_k64 | boo_fpt2 | 26.8611 | 29.0734 | 1.082 | 0.300 |
| Williams/webbase-1M | 512 | identity_k64 | boo_fpt2_cpp | 26.9696 | 29.0232 | 1.076 | 0.297 |
| Williams/webbase-1M | 512 | identity_k64 | boo_fpt4 | 26.8360 | 29.0734 | 1.083 | 0.300 |
| Williams/webbase-1M | 512 | identity_k64 | boo_fpt4_cpp | 26.8959 | 29.0232 | 1.079 | 0.298 |

Unsupported / skipped:
- Williams/webbase-1M F=512 hem_k32 direct_perm_CSR_ALG3: skipped: GPU memory budget (needs one call of transient headroom)
- Williams/webbase-1M F=512 hem_k32 direct_perm_CSR_ALG3_steady_api: skipped: GPU memory budget (needs one call of transient headroom)
- Williams/webbase-1M F=512 hem_k32 direct_perm_CSR_ALG3_gather_api: skipped: GPU memory budget (needs one call of transient headroom)
- Williams/webbase-1M F=512 hem_k32 perm_restore_y: skipped: GPU memory budget
- Williams/webbase-1M F=512 hem_k64 direct_perm_CSR_ALG3: skipped: GPU memory budget (needs one call of transient headroom)
- Williams/webbase-1M F=512 hem_k64 direct_perm_CSR_ALG3_steady_api: skipped: GPU memory budget (needs one call of transient headroom)
- Williams/webbase-1M F=512 hem_k64 direct_perm_CSR_ALG3_gather_api: skipped: GPU memory budget (needs one call of transient headroom)
- Williams/webbase-1M F=512 hem_k64 perm_restore_y: skipped: GPU memory budget
- Schmid/thermal2 F=512 hem_k32 direct_CSR_ALG1: skipped: GPU memory budget (needs one call of transient headroom)
- Schmid/thermal2 F=512 hem_k32 direct_CSR_ALG1_api: skipped: GPU memory budget (needs one call of transient headroom)
- Schmid/thermal2 F=512 hem_k32 direct_perm_CSR_ALG1: skipped: GPU memory budget (needs one call of transient headroom)
- Schmid/thermal2 F=512 hem_k32 direct_perm_CSR_ALG1_steady_api: skipped: GPU memory budget (needs one call of transient headroom)
- Schmid/thermal2 F=512 hem_k32 direct_perm_CSR_ALG1_gather_api: skipped: GPU memory budget (needs one call of transient headroom)
- Schmid/thermal2 F=512 hem_k32 direct_CSR_ALG3: skipped: GPU memory budget (needs one call of transient headroom)
- Schmid/thermal2 F=512 hem_k32 direct_CSR_ALG3_api: skipped: GPU memory budget (needs one call of transient headroom)
- Schmid/thermal2 F=512 hem_k32 direct_perm_CSR_ALG3: skipped: GPU memory budget (needs one call of transient headroom)
- Schmid/thermal2 F=512 hem_k32 direct_perm_CSR_ALG3_steady_api: skipped: GPU memory budget (needs one call of transient headroom)
- Schmid/thermal2 F=512 hem_k32 direct_perm_CSR_ALG3_gather_api: skipped: GPU memory budget (needs one call of transient headroom)
- Schmid/thermal2 F=512 hem_k32 perm_restore_y: skipped: GPU memory budget
- Schmid/thermal2 F=512 hem_k64 direct_CSR_ALG1: skipped: GPU memory budget (needs one call of transient headroom)
- Schmid/thermal2 F=512 hem_k64 direct_CSR_ALG1_api: skipped: GPU memory budget (needs one call of transient headroom)
- Schmid/thermal2 F=512 hem_k64 direct_perm_CSR_ALG1: skipped: GPU memory budget (needs one call of transient headroom)
- Schmid/thermal2 F=512 hem_k64 direct_perm_CSR_ALG1_steady_api: skipped: GPU memory budget (needs one call of transient headroom)
- Schmid/thermal2 F=512 hem_k64 direct_perm_CSR_ALG1_gather_api: skipped: GPU memory budget (needs one call of transient headroom)
- Schmid/thermal2 F=512 hem_k64 direct_CSR_ALG3: skipped: GPU memory budget (needs one call of transient headroom)
- Schmid/thermal2 F=512 hem_k64 direct_CSR_ALG3_api: skipped: GPU memory budget (needs one call of transient headroom)
- Schmid/thermal2 F=512 hem_k64 direct_perm_CSR_ALG3: skipped: GPU memory budget (needs one call of transient headroom)
- Schmid/thermal2 F=512 hem_k64 direct_perm_CSR_ALG3_steady_api: skipped: GPU memory budget (needs one call of transient headroom)
- Schmid/thermal2 F=512 hem_k64 direct_perm_CSR_ALG3_gather_api: skipped: GPU memory budget (needs one call of transient headroom)
- Schmid/thermal2 F=512 hem_k64 perm_restore_y: skipped: GPU memory budget
- SNAP/roadNet-CA F=512 identity_k32 direct_CSR_ALG1: skipped: GPU memory budget (needs one call of transient headroom)
- SNAP/roadNet-CA F=512 identity_k32 direct_CSR_ALG1_api: skipped: GPU memory budget (needs one call of transient headroom)
- SNAP/roadNet-CA F=512 identity_k32 direct_CSR_ALG3: skipped: GPU memory budget (needs one call of transient headroom)
- SNAP/roadNet-CA F=512 identity_k32 direct_CSR_ALG3_api: skipped: GPU memory budget (needs one call of transient headroom)
- SNAP/roadNet-CA F=512 identity_k64 direct_CSR_ALG1: skipped: GPU memory budget (needs one call of transient headroom)
- SNAP/roadNet-CA F=512 identity_k64 direct_CSR_ALG1_api: skipped: GPU memory budget (needs one call of transient headroom)
- SNAP/roadNet-CA F=512 identity_k64 direct_CSR_ALG3: skipped: GPU memory budget (needs one call of transient headroom)
- SNAP/roadNet-CA F=512 identity_k64 direct_CSR_ALG3_api: skipped: GPU memory budget (needs one call of transient headroom)
- SNAP/roadNet-CA F=512 hem_k32 boo_cpp: skipped: GPU memory budget (needs one call of transient headroom)
- SNAP/roadNet-CA F=512 hem_k32 direct_CSR_ALG2: skipped: GPU memory budget (needs one call of transient headroom)
- SNAP/roadNet-CA F=512 hem_k32 direct_CSR_ALG2_api: skipped: GPU memory budget (needs one call of transient headroom)
- SNAP/roadNet-CA F=512 hem_k32 direct_perm_CSR_ALG2: skipped: GPU memory budget (needs one call of transient headroom)
- SNAP/roadNet-CA F=512 hem_k32 direct_perm_CSR_ALG2_steady_api: skipped: GPU memory budget (needs one call of transient headroom)
- SNAP/roadNet-CA F=512 hem_k32 direct_perm_CSR_ALG2_gather_api: skipped: GPU memory budget (needs one call of transient headroom)
- SNAP/roadNet-CA F=512 hem_k32 direct_CSR_ALG1: skipped: GPU memory budget (needs one call of transient headroom)
- SNAP/roadNet-CA F=512 hem_k32 direct_CSR_ALG1_api: skipped: GPU memory budget (needs one call of transient headroom)
- SNAP/roadNet-CA F=512 hem_k32 direct_perm_CSR_ALG1: skipped: GPU memory budget (needs one call of transient headroom)
- SNAP/roadNet-CA F=512 hem_k32 direct_perm_CSR_ALG1_steady_api: skipped: GPU memory budget (needs one call of transient headroom)
- SNAP/roadNet-CA F=512 hem_k32 direct_perm_CSR_ALG1_gather_api: skipped: GPU memory budget (needs one call of transient headroom)
- SNAP/roadNet-CA F=512 hem_k32 direct_CSR_ALG3: skipped: GPU memory budget (needs one call of transient headroom)
- SNAP/roadNet-CA F=512 hem_k32 direct_CSR_ALG3_api: skipped: GPU memory budget (needs one call of transient headroom)
- SNAP/roadNet-CA F=512 hem_k32 direct_perm_CSR_ALG3: skipped: GPU memory budget (needs one call of transient headroom)
- SNAP/roadNet-CA F=512 hem_k32 direct_perm_CSR_ALG3_steady_api: skipped: GPU memory budget (needs one call of transient headroom)
- SNAP/roadNet-CA F=512 hem_k32 direct_perm_CSR_ALG3_gather_api: skipped: GPU memory budget (needs one call of transient headroom)
- SNAP/roadNet-CA F=512 hem_k32 perm_restore_y: skipped: GPU memory budget
- SNAP/roadNet-CA F=512 hem_k64 boo_cpp: skipped: GPU memory budget (needs one call of transient headroom)
- SNAP/roadNet-CA F=512 hem_k64 direct_CSR_ALG2: skipped: GPU memory budget (needs one call of transient headroom)
- SNAP/roadNet-CA F=512 hem_k64 direct_CSR_ALG2_api: skipped: GPU memory budget (needs one call of transient headroom)
- SNAP/roadNet-CA F=512 hem_k64 direct_perm_CSR_ALG2: skipped: GPU memory budget (needs one call of transient headroom)
- SNAP/roadNet-CA F=512 hem_k64 direct_perm_CSR_ALG2_steady_api: skipped: GPU memory budget (needs one call of transient headroom)
- SNAP/roadNet-CA F=512 hem_k64 direct_perm_CSR_ALG2_gather_api: skipped: GPU memory budget (needs one call of transient headroom)
- SNAP/roadNet-CA F=512 hem_k64 direct_CSR_ALG1: skipped: GPU memory budget (needs one call of transient headroom)
- SNAP/roadNet-CA F=512 hem_k64 direct_CSR_ALG1_api: skipped: GPU memory budget (needs one call of transient headroom)
- SNAP/roadNet-CA F=512 hem_k64 direct_perm_CSR_ALG1: skipped: GPU memory budget (needs one call of transient headroom)
- SNAP/roadNet-CA F=512 hem_k64 direct_perm_CSR_ALG1_steady_api: skipped: GPU memory budget (needs one call of transient headroom)
- SNAP/roadNet-CA F=512 hem_k64 direct_perm_CSR_ALG1_gather_api: skipped: GPU memory budget (needs one call of transient headroom)
- SNAP/roadNet-CA F=512 hem_k64 direct_CSR_ALG3: skipped: GPU memory budget (needs one call of transient headroom)
- SNAP/roadNet-CA F=512 hem_k64 direct_CSR_ALG3_api: skipped: GPU memory budget (needs one call of transient headroom)
- SNAP/roadNet-CA F=512 hem_k64 direct_perm_CSR_ALG3: skipped: GPU memory budget (needs one call of transient headroom)
- SNAP/roadNet-CA F=512 hem_k64 direct_perm_CSR_ALG3_steady_api: skipped: GPU memory budget (needs one call of transient headroom)
- SNAP/roadNet-CA F=512 hem_k64 direct_perm_CSR_ALG3_gather_api: skipped: GPU memory budget (needs one call of transient headroom)
- SNAP/roadNet-CA F=512 hem_k64 perm_restore_y: skipped: GPU memory budget
