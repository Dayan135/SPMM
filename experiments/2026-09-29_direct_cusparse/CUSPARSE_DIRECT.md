# Direct cuSPARSE comparison

The supporting CSV/JSON tables are in the [dated result directory](../../tests/torch_fast_spdmm/benchmark/results/cusparse_direct_20260929/derived/README.md). Individual CUDA-event samples are in the [result archive](../../RESULTS_ARCHIVE.md).

Measured source commit: `af0bfbc97f99a0e613fd5b4589a35b732a63e8b6`. Source SHA256: `18dabc207f3d714fe2d1a9fa59f2bec7bc4f3f7a1bfa8911da7a995bb6212e6b`. Extension SHA256: `6e0b47800d75ca19d96ad8219cac50b0cfd70b836db11ce557c837a8a886ba15`.

GPU: NVIDIA GeForce RTX 3090; PyTorch 2.10.0+cu128; PyTorch CUDA 12.8. Runtime cuSPARSE version(s): 12508.

cuSPARSE build-header version(s): 12510. Runtime and header versions are recorded separately; the extension was built with the cluster conda CUDA 12.9 toolkit.

Validation: `735 passed, 12 skipped, 1 warning in 22.07s`. Profiler: NVIDIA Nsight Systems version 2025.2.1.0.

| Replicate | Slurm job | GPU node |
|---|---|---|
| 1 | 21732456 | cs-3090-03.auth.ad.bgu.ac.il |
| 2 | 21732457 | cs-3090-04.auth.ad.bgu.ac.il |
| 3 | 21732458 | cs-3090-05.auth.ad.bgu.ac.il |

All 84 workloads use FP32 CSR values, int32 indices, seeded identical A and X across methods and jobs: 14 profiles × N={16,384; 65,536; 262,144} × F={128; 512}. Matrix and X hashes are checked by the report generator. Every supported result passed an independent SciPy FP64 reference at rtol=atol=1e-4.

Five warmups, then 20 individual CUDA-event timings in seeded, interleaved configuration order. Each job reports its median; tables give the median and [minimum, maximum] of the three job medians. Conversion, staging, validation, bufferSize and preprocess are outside timing.

Direct: DEFAULT, CSR_ALG1, CSR_ALG2, CSR_ALG3; non-transposed A/X, row-major X/Y, alpha=1, beta=0. Each algorithm owns a reusable sparse descriptor, dense descriptors, output and workspace. C++ CUDA events enclose only cusparseSpMM. bufferSize and optional preprocess are each called once per plan; statuses and workspace bytes are preserved in the all-config CSV. Internal initialization performed by cusparseSpMM is included. [NVIDIA SpMM API](https://docs.nvidia.com/cuda/archive/12.9.1/cusparse/index.html#cusparsespmm).

PBR: kernel=block, csr_mode=0, smem=2048, (block size:min_nnz)={2:2, 2:3, 4:2, 4:4, 4:8, 8:2, 8:8, 8:12}. BOO: HEM target={8,16,32,64}, threads=256, p=64, rowseg, row-major nonzeros, small blocks below 4 sent to remainder, hem_feasible_first=False, launch_mode=legacy. These retain the earlier 84-workload sweep. Both use complete API calls, including allocating/zeroing Y and joining remainder work; torch.sparse.mm is also a complete call. Direct cuSPARSE reuses Y, so it has a deliberately narrower timing scope.

For each workload and method, select the configuration with the lowest median of three job medians, then retain that same configuration in all three jobs. Selection is retrospective, not a production dispatcher. All four cuSPARSE algorithms and all PBR/BOO configurations remain in the raw CSV. Speedup is direct/method, paired within each job; >1 means faster than direct. The geometric means first combine the 28 workloads per size within each job, then report median and range across jobs. Do not read small differences (roughly below 1.3×) as robust wins.

## Geometric means versus direct cuSPARSE

| N | Method | Speedup, median [range] |
|---:|---|---:|
| 16384 | torch | 0.621 [0.615, 0.660]× |
| 16384 | pbr | 0.662 [0.646, 0.696]× |
| 16384 | boo | 0.504 [0.492, 0.553]× |
| 65536 | torch | 0.734 [0.733, 0.741]× |
| 65536 | pbr | 0.833 [0.829, 0.845]× |
| 65536 | boo | 0.735 [0.735, 0.766]× |
| 262144 | torch | 0.732 [0.732, 0.735]× |
| 262144 | pbr | 0.864 [0.862, 0.871]× |
| 262144 | boo | 0.820 [0.820, 0.837]× |

Plain reading: the direct baseline removes PyTorch wrapper/setup costs from the baseline comparison. The following counts require at least a 1.3× margin in **every** job; the remaining workloads have smaller or inconsistent margins. These are measurements on three allocated GPUs, not confidence intervals or universal rankings.

| Method | Faster than direct by ≥1.3× in every job | Slower by ≥1.3× in every job | Remaining |
|---|---:|---:|---:|
| torch | 0 | 51 | 33 |
| pbr | 0 | 37 | 47 |
| boo | 0 | 44 | 40 |

## Per workload

Times in ms, median [range]. Full selected configuration, fractions, paths and speedup ranges are in [workload CSV](../../tests/torch_fast_spdmm/benchmark/results/cusparse_direct_20260929/derived/CUSPARSE_DIRECT_workloads.csv). `block+csr` means block kernel plus remainder; `csr_only` means no compressed blocks.

| Profile | N | F | Direct algorithm | Direct ms | Torch ms | PBR ms (path; fraction) | BOO ms (fraction) | Speedup torch / PBR / BOO |
|---|---:|---:|---|---|---|---|---|---|
| diag_b0.1 | 16384 | 128 | CSR_ALG3 | 0.0625 [0.0614, 0.0635] | 0.0973 [0.0860, 0.0993] | 0.0819 [0.0737, 0.0850] (block+csr; 0.000) | 0.1229 [0.1039, 0.1260] (1.000) | 0.653 / 0.775 / 0.517 |
| diag_b0.3 | 16384 | 128 | DEFAULT | 0.0778 [0.0758, 0.0788] | 0.1106 [0.0983, 0.1116] | 0.1270 [0.1193, 0.1290] (block+csr; 0.132) | 0.1802 [0.1597, 0.1848] (1.000) | 0.706 / 0.613 / 0.432 |
| diag_b1.0 | 16384 | 128 | DEFAULT | 0.2253 [0.2227, 0.2268] | 0.2647 [0.2453, 0.2724] | 0.2309 [0.2237, 0.2355] (block; 1.000) | 0.2980 [0.2826, 0.3087] (1.000) | 0.857 / 0.982 / 0.761 |
| diag_skewed | 16384 | 128 | DEFAULT | 0.2130 [0.2094, 0.2309] | 0.2611 [0.2319, 0.2678] | 0.2268 [0.2140, 0.2427] (block+csr; 0.995) | 0.2964 [0.2693, 0.3087] (0.995) | 0.862 / 0.951 / 0.748 |
| fem2d | 16384 | 128 | DEFAULT | 0.1065 [0.1055, 0.1065] | 0.1393 [0.1270, 0.1418] | 0.1782 [0.1720, 0.1807] (block; 1.000) | 0.2601 [0.2468, 0.2662] (1.000) | 0.765 / 0.598 / 0.409 |
| fem2d_permuted | 16384 | 128 | DEFAULT | 0.1260 [0.1249, 0.1270] | 0.1628 [0.1485, 0.1633] | 0.2493 [0.2437, 0.2529] (block; 1.000) | 0.2621 [0.2468, 0.2673] (1.000) | 0.777 / 0.509 / 0.484 |
| pattern_checkerboard | 16384 | 128 | CSR_ALG3 | 0.0338 [0.0328, 0.0338] | 0.0737 [0.0625, 0.0758] | 0.0645 [0.0573, 0.0666] (block; 1.000) | 0.0963 [0.0783, 0.1004] (1.000) | 0.458 / 0.524 / 0.351 |
| pattern_corner_cluster | 16384 | 128 | CSR_ALG3 | 0.0297 [0.0272, 0.0297] | 0.0686 [0.0553, 0.0717] | 0.0604 [0.0512, 0.0645] (block; 1.000) | 0.0916 [0.0742, 0.0973] (1.000) | 0.433 / 0.492 / 0.324 |
| pattern_diag_stripe | 16384 | 128 | CSR_ALG3 | 0.0317 [0.0307, 0.0317] | 0.0778 [0.0655, 0.0788] | 0.0635 [0.0532, 0.0666] (csr_only; 0.000) | 0.1004 [0.0768, 0.1024] (0.408) | 0.408 / 0.500 / 0.316 |
| pattern_row_skewed | 16384 | 128 | CSR_ALG3 | 0.0328 [0.0317, 0.0328] | 0.0737 [0.0635, 0.0768] | 0.0635 [0.0573, 0.0666] (block; 1.000) | 0.0978 [0.0778, 0.1034] (0.985) | 0.444 / 0.516 / 0.335 |
| scattered_1x8 | 16384 | 128 | CSR_ALG3 | 0.0287 [0.0256, 0.0287] | 0.0727 [0.0594, 0.0737] | 0.0604 [0.0502, 0.0635] (block; 1.000) | 0.0922 [0.0742, 0.0963] (1.000) | 0.394 / 0.475 / 0.311 |
| scattered_2x2 | 16384 | 128 | CSR_ALG3 | 0.0287 [0.0276, 0.0287] | 0.0768 [0.0666, 0.0788] | 0.0604 [0.0492, 0.0635] (csr_only; 0.000) | 0.0942 [0.0737, 0.0983] (1.000) | 0.373 / 0.475 / 0.304 |
| scattered_2x4 | 16384 | 128 | CSR_ALG3 | 0.0287 [0.0276, 0.0287] | 0.0707 [0.0604, 0.0727] | 0.0604 [0.0502, 0.0645] (csr_only; 0.000) | 0.0922 [0.0737, 0.0973] (1.000) | 0.406 / 0.475 / 0.311 |
| scattered_8x1 | 16384 | 128 | CSR_ALG3 | 0.0317 [0.0307, 0.0328] | 0.0707 [0.0604, 0.0717] | 0.0604 [0.0512, 0.0645] (csr_only; 0.000) | 0.0911 [0.0737, 0.0963] (1.000) | 0.457 / 0.525 / 0.348 |
| diag_b0.1 | 16384 | 512 | DEFAULT | 0.2253 [0.2237, 0.2253] | 0.2744 [0.2703, 0.2765] | 0.2202 [0.2150, 0.2232] (block+csr; 0.000) | 0.2703 [0.2560, 0.2785] (1.000) | 0.821 / 1.023 / 0.833 |
| diag_b0.3 | 16384 | 512 | DEFAULT | 0.2621 [0.2596, 0.2683] | 0.3098 [0.3067, 0.3185] | 0.3958 [0.3932, 0.4178] (block+csr; 0.132) | 0.4659 [0.4511, 0.4838] (1.000) | 0.846 / 0.660 / 0.563 |
| diag_b1.0 | 16384 | 512 | DEFAULT | 0.7864 [0.7834, 0.7972] | 0.8494 [0.8340, 0.8530] | 0.7178 [0.7132, 0.7270] (block; 1.000) | 0.9042 [0.8888, 0.9185] (1.000) | 0.935 / 1.096 / 0.868 |
| diag_skewed | 16384 | 512 | DEFAULT | 0.7798 [0.7777, 0.7818] | 0.8340 [0.8212, 0.8453] | 0.7583 [0.7465, 0.7603] (block+csr; 0.995) | 0.9221 [0.8980, 0.9370] (0.995) | 0.932 / 1.031 / 0.843 |
| fem2d | 16384 | 512 | CSR_ALG2 | 0.3645 [0.3476, 0.3758] | 0.4209 [0.4050, 0.4219] | 0.6124 [0.5683, 0.6154] (block; 1.000) | 0.7685 [0.7409, 0.7772] (1.000) | 0.864 / 0.612 / 0.474 |
| fem2d_permuted | 16384 | 512 | DEFAULT | 0.4250 [0.4204, 0.4250] | 0.4787 [0.4710, 0.4818] | 0.8709 [0.8658, 0.8719] (block; 1.000) | 0.7660 [0.7644, 0.7690] (1.000) | 0.882 / 0.487 / 0.555 |
| pattern_checkerboard | 16384 | 512 | CSR_ALG3 | 0.0952 [0.0942, 0.0963] | 0.1731 [0.1679, 0.1731] | 0.1444 [0.1382, 0.1464] (csr_only; 0.000) | 0.1720 [0.1597, 0.1746] (1.000) | 0.556 / 0.667 / 0.560 |
| pattern_corner_cluster | 16384 | 512 | CSR_ALG3 | 0.0804 [0.0799, 0.0809] | 0.1382 [0.1341, 0.1393] | 0.1116 [0.1075, 0.1137] (block; 1.000) | 0.1362 [0.1260, 0.1393] (1.000) | 0.581 / 0.720 / 0.590 |
| pattern_diag_stripe | 16384 | 512 | CSR_ALG3 | 0.0922 [0.0901, 0.0922] | 0.1500 [0.1464, 0.1505] | 0.1388 [0.1341, 0.1413] (csr_only; 0.000) | 0.1628 [0.1485, 0.1628] (0.408) | 0.614 / 0.664 / 0.566 |
| pattern_row_skewed | 16384 | 512 | CSR_ALG3 | 0.1085 [0.1085, 0.1091] | 0.1700 [0.1659, 0.1710] | 0.1485 [0.1444, 0.1516] (block; 1.000) | 0.1705 [0.1567, 0.1741] (0.824) | 0.642 / 0.734 / 0.640 |
| scattered_1x8 | 16384 | 512 | DEFAULT | 0.0840 [0.0819, 0.0840] | 0.1311 [0.1280, 0.1321] | 0.0993 [0.0952, 0.1014] (block; 1.000) | 0.1300 [0.1183, 0.1331] (0.997) | 0.640 / 0.845 / 0.646 |
| scattered_2x2 | 16384 | 512 | CSR_ALG3 | 0.0748 [0.0737, 0.0748] | 0.1249 [0.1208, 0.1249] | 0.0916 [0.0870, 0.0942] (block+csr; 0.179) | 0.1208 [0.1101, 0.1239] (1.000) | 0.598 / 0.816 / 0.619 |
| scattered_2x4 | 16384 | 512 | DEFAULT | 0.0799 [0.0788, 0.0799] | 0.1280 [0.1244, 0.1290] | 0.0963 [0.0921, 0.0983] (csr_only; 0.000) | 0.1311 [0.1167, 0.1326] (0.999) | 0.624 / 0.830 / 0.609 |
| scattered_8x1 | 16384 | 512 | DEFAULT | 0.0829 [0.0819, 0.0840] | 0.1300 [0.1270, 0.1321] | 0.1249 [0.1208, 0.1270] (csr_only; 0.000) | 0.1403 [0.1306, 0.1434] (1.000) | 0.638 / 0.664 / 0.591 |
| diag_b0.1 | 65536 | 128 | DEFAULT | 0.2243 [0.2232, 0.2253] | 0.2744 [0.2693, 0.2765] | 0.2079 [0.2007, 0.2079] (block+csr; 0.000) | 0.2637 [0.2437, 0.2714] (1.000) | 0.817 / 1.084 / 0.850 |
| diag_b0.3 | 65536 | 128 | DEFAULT | 0.2550 [0.2540, 0.2555] | 0.3072 [0.3000, 0.3174] | 0.3707 [0.3661, 0.3738] (block+csr; 0.131) | 0.3896 [0.3645, 0.3912] (1.000) | 0.830 / 0.688 / 0.654 |
| diag_b1.0 | 65536 | 128 | DEFAULT | 0.7926 [0.7921, 0.8008] | 0.8509 [0.8402, 0.8658] | 0.7501 [0.7414, 0.7521] (block; 1.000) | 0.8878 [0.8704, 0.8996] (1.000) | 0.931 / 1.065 / 0.893 |
| diag_skewed | 65536 | 128 | DEFAULT | 0.7916 [0.7910, 0.7982] | 0.8484 [0.8381, 0.8571] | 0.8540 [0.8484, 0.8576] (block+csr; 0.980) | 0.9692 [0.9523, 0.9692] (0.980) | 0.941 / 0.931 / 0.824 |
| fem2d | 65536 | 128 | DEFAULT | 0.3564 [0.3553, 0.3635] | 0.4096 [0.4014, 0.4285] | 0.5734 [0.5673, 0.5837] (block; 1.000) | 0.7613 [0.7480, 0.7736] (1.000) | 0.870 / 0.623 / 0.470 |
| fem2d_permuted | 65536 | 128 | CSR_ALG3 | 0.5345 [0.5330, 0.5437] | 0.7398 [0.7275, 0.7465] | 0.8627 [0.8556, 0.8842] (block; 1.000) | 0.8141 [0.7921, 0.8264] (1.000) | 0.728 / 0.620 / 0.658 |
| pattern_checkerboard | 65536 | 128 | CSR_ALG3 | 0.1126 [0.1116, 0.1137] | 0.1761 [0.1731, 0.1772] | 0.1403 [0.1362, 0.1434] (csr_only; 0.000) | 0.1766 [0.1669, 0.1792] (1.000) | 0.645 / 0.810 / 0.643 |
| pattern_corner_cluster | 65536 | 128 | CSR_ALG2 | 0.0942 [0.0937, 0.0947] | 0.1413 [0.1377, 0.1413] | 0.1096 [0.1055, 0.1116] (block; 1.000) | 0.1372 [0.1260, 0.1393] (1.000) | 0.670 / 0.860 / 0.687 |
| pattern_diag_stripe | 65536 | 128 | DEFAULT | 0.1014 [0.1004, 0.1024] | 0.1495 [0.1464, 0.1516] | 0.1352 [0.1331, 0.1393] (csr_only; 0.000) | 0.1638 [0.1556, 0.1700] (0.404) | 0.678 / 0.750 / 0.619 |
| pattern_row_skewed | 65536 | 128 | DEFAULT | 0.1239 [0.1219, 0.1239] | 0.1710 [0.1674, 0.1720] | 0.1413 [0.1362, 0.1434] (csr_only; 0.000) | 0.1700 [0.1556, 0.1710] (0.990) | 0.725 / 0.877 / 0.729 |
| scattered_1x8 | 65536 | 128 | DEFAULT | 0.0860 [0.0840, 0.0860] | 0.1341 [0.1300, 0.1341] | 0.1014 [0.0963, 0.1024] (block; 1.000) | 0.1423 [0.1290, 0.1428] (1.000) | 0.641 / 0.848 / 0.604 |
| scattered_2x2 | 65536 | 128 | DEFAULT | 0.0768 [0.0758, 0.0778] | 0.1260 [0.1208, 0.1260] | 0.0911 [0.0860, 0.0922] (csr_only; 0.000) | 0.1331 [0.1208, 0.1341] (1.000) | 0.618 / 0.844 / 0.580 |
| scattered_2x4 | 65536 | 128 | DEFAULT | 0.0819 [0.0809, 0.0819] | 0.1300 [0.1249, 0.1300] | 0.0942 [0.0891, 0.0963] (csr_only; 0.000) | 0.1352 [0.1229, 0.1357] (1.000) | 0.630 / 0.870 / 0.606 |
| scattered_8x1 | 65536 | 128 | DEFAULT | 0.0850 [0.0829, 0.0850] | 0.1321 [0.1285, 0.1341] | 0.1239 [0.1203, 0.1270] (csr_only; 0.000) | 0.1495 [0.1382, 0.1516] (1.000) | 0.643 / 0.686 / 0.568 |
| diag_b0.1 | 65536 | 512 | CSR_ALG2 | 0.8581 [0.8561, 0.8607] | 1.0199 [1.0143, 1.0214] | 0.7649 [0.7578, 0.7654] (block+csr; 0.000) | 0.8033 [0.7905, 0.8187] (1.000) | 0.843 / 1.124 / 1.068 |
| diag_b0.3 | 65536 | 512 | DEFAULT | 0.9759 [0.9738, 0.9856] | 1.1372 [1.1341, 1.1520] | 1.4740 [1.4689, 1.4915] (block+csr; 0.131) | 1.2800 [1.2524, 1.3123] (1.000) | 0.856 / 0.661 / 0.761 |
| diag_b1.0 | 65536 | 512 | CSR_ALG2 | 3.1575 [3.1483, 3.1647] | 3.3787 [3.3469, 3.3823] | 2.7914 [2.7653, 2.8042] (block; 1.000) | 3.2748 [3.2313, 3.2942] (1.000) | 0.936 / 1.131 / 0.964 |
| diag_skewed | 65536 | 512 | CSR_ALG2 | 3.0413 [3.0336, 3.0566] | 3.2614 [3.2563, 3.2814] | 3.2256 [3.2230, 3.2379] (block+csr; 0.980) | 3.6050 [3.5814, 3.6163] (0.980) | 0.932 / 0.943 / 0.847 |
| fem2d | 65536 | 512 | CSR_ALG2 | 1.4111 [1.4018, 1.4162] | 1.5749 [1.5596, 1.5800] | 2.2154 [2.2011, 2.2262] (block; 1.000) | 2.7162 [2.6772, 2.7274] (1.000) | 0.896 / 0.637 / 0.520 |
| fem2d_permuted | 65536 | 512 | DEFAULT | 2.7156 [2.7146, 2.7177] | 2.8764 [2.8672, 2.8795] | 3.4350 [3.4330, 3.4703] (block; 1.000) | 2.8047 [2.7848, 2.8257] (1.000) | 0.944 / 0.790 / 0.968 |
| pattern_checkerboard | 65536 | 512 | CSR_ALG3 | 0.3604 [0.3594, 0.3615] | 0.6287 [0.6267, 0.6303] | 0.4608 [0.4567, 0.4659] (csr_only; 0.000) | 0.4680 [0.4567, 0.4726] (1.000) | 0.574 / 0.782 / 0.770 |
| pattern_corner_cluster | 65536 | 512 | CSR_ALG3 | 0.3052 [0.3036, 0.3077] | 0.4792 [0.4731, 0.4818] | 0.3441 [0.3374, 0.3446] (block; 1.000) | 0.3482 [0.3338, 0.3517] (1.000) | 0.639 / 0.894 / 0.876 |
| pattern_diag_stripe | 65536 | 512 | CSR_ALG3 | 0.3471 [0.3405, 0.3492] | 0.5197 [0.5181, 0.5238] | 0.4531 [0.4526, 0.4593] (csr_only; 0.000) | 0.4639 [0.4582, 0.4731] (0.404) | 0.667 / 0.760 / 0.738 |
| pattern_row_skewed | 65536 | 512 | CSR_ALG3 | 0.4239 [0.4219, 0.4275] | 0.5996 [0.5949, 0.6011] | 0.4890 [0.4823, 0.4905] (block; 1.000) | 0.4659 [0.4526, 0.4680] (0.990) | 0.711 / 0.872 / 0.914 |
| scattered_1x8 | 65536 | 512 | DEFAULT | 0.2908 [0.2888, 0.2908] | 0.4516 [0.4469, 0.4547] | 0.3072 [0.3000, 0.3082] (block; 1.000) | 0.3482 [0.3318, 0.3502] (1.000) | 0.644 / 0.947 / 0.835 |
| scattered_2x2 | 65536 | 512 | DEFAULT | 0.2437 [0.2427, 0.2447] | 0.4076 [0.4035, 0.4081] | 0.2678 [0.2632, 0.2698] (csr_only; 0.000) | 0.3031 [0.2903, 0.3041] (1.000) | 0.600 / 0.910 / 0.805 |
| scattered_2x4 | 65536 | 512 | DEFAULT | 0.2898 [0.2888, 0.2898] | 0.4506 [0.4480, 0.4526] | 0.2867 [0.2806, 0.2877] (csr_only; 0.000) | 0.3277 [0.3123, 0.3308] (1.000) | 0.641 / 1.011 / 0.881 |
| scattered_8x1 | 65536 | 512 | DEFAULT | 0.2816 [0.2806, 0.2821] | 0.4419 [0.4372, 0.4434] | 0.3999 [0.3958, 0.4014] (csr_only; 0.000) | 0.3860 [0.3707, 0.3871] (1.000) | 0.637 / 0.706 / 0.731 |
| diag_b0.1 | 262144 | 128 | DEFAULT | 0.8509 [0.8489, 0.8515] | 1.0148 [1.0076, 1.0158] | 0.7188 [0.7107, 0.7199] (block+csr; 0.000) | 0.7839 [0.7613, 0.7885] (1.000) | 0.839 / 1.184 / 1.086 |
| diag_b0.3 | 262144 | 128 | DEFAULT | 0.9728 [0.9708, 0.9795] | 1.1372 [1.1320, 1.1500] | 1.3768 [1.3619, 1.3783] (block+csr; 0.131) | 1.2616 [1.2237, 1.2692] (1.000) | 0.854 / 0.711 / 0.772 |
| diag_b1.0 | 262144 | 128 | CSR_ALG2 | 3.1370 [3.1232, 3.1462] | 3.4017 [3.3521, 3.4125] | 2.8785 [2.8559, 2.8856] (block; 1.000) | 3.2973 [3.2568, 3.3183] (1.000) | 0.922 / 1.090 / 0.954 |
| diag_skewed | 262144 | 128 | CSR_ALG2 | 2.9389 [2.9332, 2.9660] | 3.2256 [3.1744, 3.2399] | 3.9757 [3.9685, 3.9951] (block+csr; 0.924) | 4.3750 [4.3494, 4.3991] (0.924) | 0.915 / 0.739 / 0.674 |
| fem2d | 262144 | 128 | CSR_ALG2 | 1.4372 [1.4280, 1.4408] | 1.5969 [1.5944, 1.6051] | 2.2057 [2.1980, 2.2170] (block; 1.000) | 2.8477 [2.8191, 2.8549] (1.000) | 0.898 / 0.650 / 0.505 |
| fem2d_permuted | 262144 | 128 | CSR_ALG3 | 2.1345 [2.0936, 2.1622] | 3.2210 [3.2128, 3.2236] | 3.3408 [3.3372, 3.3777] (block; 1.000) | 3.0971 [3.0899, 3.1288] (1.000) | 0.664 / 0.639 / 0.691 |
| pattern_checkerboard | 262144 | 128 | CSR_ALG3 | 0.4209 [0.4116, 0.4219] | 0.6569 [0.6492, 0.6574] | 0.4649 [0.4577, 0.4659] (csr_only; 0.000) | 0.5238 [0.5233, 0.5396] (1.000) | 0.642 / 0.905 / 0.786 |
| pattern_corner_cluster | 262144 | 128 | DEFAULT | 0.3354 [0.3348, 0.3389] | 0.5059 [0.5059, 0.5094] | 0.3389 [0.3359, 0.3430] (block; 1.000) | 0.3732 [0.3615, 0.3779] (1.000) | 0.663 / 0.989 / 0.898 |
| pattern_diag_stripe | 262144 | 128 | DEFAULT | 0.3645 [0.3635, 0.3645] | 0.5279 [0.5212, 0.5279] | 0.4598 [0.4526, 0.4628] (csr_only; 0.000) | 0.5079 [0.4936, 0.5100] (0.409) | 0.691 / 0.793 / 0.718 |
| pattern_row_skewed | 262144 | 128 | DEFAULT | 0.4526 [0.4495, 0.4547] | 0.6129 [0.6062, 0.6185] | 0.4649 [0.4577, 0.4710] (csr_only; 0.000) | 0.5079 [0.4920, 0.5222] (0.989) | 0.739 / 0.974 / 0.891 |
| scattered_1x8 | 262144 | 128 | DEFAULT | 0.3005 [0.2980, 0.3011] | 0.4623 [0.4588, 0.4634] | 0.3098 [0.3031, 0.3113] (block; 1.000) | 0.3932 [0.3779, 0.3942] (1.000) | 0.650 / 0.972 / 0.766 |
| scattered_2x2 | 262144 | 128 | DEFAULT | 0.2509 [0.2488, 0.2519] | 0.4127 [0.4080, 0.4152] | 0.2688 [0.2621, 0.2693] (csr_only; 0.000) | 0.3610 [0.3441, 0.3651] (1.000) | 0.608 / 0.937 / 0.695 |
| scattered_2x4 | 262144 | 128 | DEFAULT | 0.2949 [0.2944, 0.2954] | 0.4547 [0.4495, 0.4577] | 0.2867 [0.2791, 0.2872] (csr_only; 0.000) | 0.3676 [0.3523, 0.3681] (1.000) | 0.649 / 1.030 / 0.803 |
| scattered_8x1 | 262144 | 128 | DEFAULT | 0.2836 [0.2821, 0.2847] | 0.4449 [0.4393, 0.4465] | 0.4035 [0.3948, 0.4035] (csr_only; 0.000) | 0.4342 [0.4106, 0.4342] (1.000) | 0.640 / 0.706 / 0.656 |
| diag_b0.1 | 262144 | 512 | DEFAULT | 3.3746 [3.3741, 3.3792] | 3.9905 [3.9813, 3.9916] | 2.9517 [2.9440, 2.9548] (block+csr; 0.000) | 2.8570 [2.8416, 2.8815] (1.000) | 0.847 / 1.144 / 1.181 |
| diag_b0.3 | 262144 | 512 | DEFAULT | 3.8943 [3.8856, 3.9091] | 4.5251 [4.5087, 4.5389] | 5.7846 [5.7810, 5.7994] (block+csr; 0.131) | 4.7012 [4.6361, 4.7058] (1.000) | 0.861 / 0.673 / 0.831 |
| diag_b1.0 | 262144 | 512 | CSR_ALG2 | 12.8707 [12.7355, 12.9695] | 13.8020 [13.7723, 13.9443] | 11.3295 [11.3044, 11.3649] (block; 1.000) | 12.7519 [12.7099, 12.8722] (1.000) | 0.930 / 1.132 / 1.008 |
| diag_skewed | 262144 | 512 | CSR_ALG2 | 11.6485 [11.5912, 11.7786] | 12.6428 [12.4621, 12.6976] | 15.6744 [15.5996, 16.0133] (block+csr; 0.924) | 17.2196 [17.1018, 17.3747] (0.924) | 0.928 / 0.743 / 0.678 |
| fem2d | 262144 | 512 | CSR_ALG2 | 5.7892 [5.7605, 5.7953] | 6.6237 [6.5935, 6.6427] | 9.0250 [9.0220, 9.1244] (block; 1.000) | 10.7095 [10.7023, 10.8124] (1.000) | 0.874 / 0.638 / 0.538 |
| fem2d_permuted | 262144 | 512 | CSR_ALG3 | 12.0161 [11.8717, 12.0351] | 13.1077 [13.1011, 13.1108] | 13.5782 [13.5639, 13.6428] (block; 1.000) | 11.1815 [11.1775, 11.2666] (1.000) | 0.917 / 0.882 / 1.068 |
| pattern_checkerboard | 262144 | 512 | CSR_ALG3 | 1.3768 [1.3722, 1.3804] | 2.4934 [2.4842, 2.4945] | 1.7285 [1.7162, 1.7295] (csr_only; 0.000) | 1.6389 [1.6210, 1.6415] (1.000) | 0.552 / 0.796 / 0.839 |
| pattern_corner_cluster | 262144 | 512 | CSR_ALG3 | 1.1576 [1.1515, 1.1668] | 1.8586 [1.8544, 1.8642] | 1.2549 [1.2508, 1.2580] (block; 1.000) | 1.1648 [1.1535, 1.1663] (1.000) | 0.623 / 0.922 / 0.994 |
| pattern_diag_stripe | 262144 | 512 | CSR_ALG3 | 1.3353 [1.3322, 1.3373] | 2.0029 [1.9948, 2.0070] | 1.7213 [1.7121, 1.7311] (csr_only; 0.000) | 1.6998 [1.6794, 1.7172] (0.409) | 0.666 / 0.774 / 0.784 |
| pattern_row_skewed | 262144 | 512 | CSR_ALG3 | 1.6804 [1.6773, 1.6840] | 2.3542 [2.3480, 2.3562] | 1.8412 [1.8340, 1.8437] (block; 1.000) | 1.6650 [1.6466, 1.6650] (0.989) | 0.713 / 0.913 / 1.009 |
| scattered_1x8 | 262144 | 512 | DEFAULT | 1.1018 [1.0998, 1.1049] | 1.7157 [1.7070, 1.7213] | 1.1325 [1.1233, 1.1418] (block; 1.000) | 1.1489 [1.1346, 1.1730] (1.000) | 0.642 / 0.973 / 0.959 |
| scattered_2x2 | 262144 | 512 | DEFAULT | 0.9196 [0.9165, 0.9236] | 1.5334 [1.5258, 1.5386] | 0.9708 [0.9636, 0.9743] (csr_only; 0.000) | 1.0168 [0.9969, 1.0445] (1.000) | 0.600 / 0.948 / 0.904 |
| scattered_2x4 | 262144 | 512 | DEFAULT | 1.1167 [1.1141, 1.1197] | 1.7347 [1.7300, 1.7362] | 1.0440 [1.0373, 1.0470] (csr_only; 0.000) | 1.0813 [1.0650, 1.0819] (1.000) | 0.644 / 1.073 / 1.035 |
| scattered_8x1 | 262144 | 512 | DEFAULT | 1.0609 [1.0593, 1.0609] | 1.6783 [1.6691, 1.6788] | 1.4802 [1.4751, 1.4838] (csr_only; 0.000) | 1.3583 [1.3251, 1.3629] (1.000) | 0.632 / 0.717 / 0.781 |

## One-call torch.sparse.mm traces

Nsight Systems captures exactly one warmed call per workload in a separate process, after the timed sweep in replicate 1. Capture uses cudaProfilerApi; no kernel filter. [Every launched kernel and its duration](../../tests/torch_fast_spdmm/benchmark/results/cusparse_direct_20260929/derived/CUSPARSE_DIRECT_torch_kernels.csv), [memory activities](../../tests/torch_fast_spdmm/benchmark/results/cusparse_direct_20260929/derived/CUSPARSE_DIRECT_torch_memory.csv), and [per-call decomposition](../../tests/torch_fast_spdmm/benchmark/results/cusparse_direct_20260929/derived/CUSPARSE_DIRECT_torch_profiles.csv) are retained. Kernel names containing `spmm` or `csrmm` are classified as SpMM; all names remain auditable.

The table below uses the same profiled call for CUDA-event time and trace durations. GPU activity is the union of kernel and memory-operation intervals; event minus that union includes launch gaps and instrumentation overhead, not a pure CPU cost. Kernel durations can overlap, so their sum need not equal GPU busy time. Profiling perturbs execution: these traces are diagnostics, not the timing used for speedups. No subtraction is made between a profiled kernel time and an unprofiled benchmark time.

| Profile | N | F | Kernels | Profiled event ms | SpMM kernels ms | Other kernels ms | Memory ms | Event minus GPU busy ms | Outside SpMM |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| diag_b0.1 | 16384 | 128 | 4 | 0.32506 | 0.04259 | 0.02662 | 0.00000 | 0.25584 | 86.9% |
| diag_b0.3 | 16384 | 128 | 4 | 0.33939 | 0.05443 | 0.02640 | 0.00000 | 0.25856 | 84.0% |
| diag_b1.0 | 16384 | 128 | 4 | 0.48064 | 0.20250 | 0.02656 | 0.00000 | 0.25158 | 57.9% |
| diag_skewed | 16384 | 128 | 4 | 0.55146 | 0.20294 | 0.02669 | 0.00000 | 0.32182 | 63.2% |
| fem2d | 16384 | 128 | 4 | 0.44512 | 0.08387 | 0.02682 | 0.00000 | 0.33443 | 81.2% |
| fem2d_permuted | 16384 | 128 | 4 | 0.49110 | 0.10534 | 0.02653 | 0.00000 | 0.35923 | 78.5% |
| pattern_checkerboard | 16384 | 128 | 4 | 0.36742 | 0.01914 | 0.02678 | 0.00000 | 0.32150 | 94.8% |
| pattern_corner_cluster | 16384 | 128 | 4 | 0.33059 | 0.01459 | 0.02669 | 0.00000 | 0.28931 | 95.6% |
| pattern_diag_stripe | 16384 | 128 | 4 | 0.42611 | 0.02134 | 0.02691 | 0.00000 | 0.37786 | 95.0% |
| pattern_row_skewed | 16384 | 128 | 4 | 0.33760 | 0.01974 | 0.02666 | 0.00000 | 0.29120 | 94.2% |
| scattered_1x8 | 16384 | 128 | 4 | 0.34966 | 0.01738 | 0.02755 | 0.00000 | 0.30474 | 95.0% |
| scattered_2x2 | 16384 | 128 | 4 | 0.33894 | 0.02250 | 0.02701 | 0.00000 | 0.28944 | 93.4% |
| scattered_2x4 | 16384 | 128 | 4 | 0.32013 | 0.01568 | 0.02688 | 0.00000 | 0.27757 | 95.1% |
| scattered_8x1 | 16384 | 128 | 4 | 0.32186 | 0.01568 | 0.02717 | 0.00000 | 0.27901 | 95.1% |
| diag_b0.1 | 16384 | 512 | 4 | 0.47670 | 0.17293 | 0.08307 | 0.00000 | 0.22070 | 63.7% |
| diag_b0.3 | 16384 | 512 | 4 | 0.50957 | 0.21578 | 0.08310 | 0.00000 | 0.21069 | 57.7% |
| diag_b1.0 | 16384 | 512 | 4 | 1.10694 | 0.81821 | 0.08333 | 0.00000 | 0.20540 | 26.1% |
| diag_skewed | 16384 | 512 | 4 | 1.11696 | 0.80551 | 0.08333 | 0.00000 | 0.22813 | 27.9% |
| fem2d | 16384 | 512 | 4 | 0.61997 | 0.32656 | 0.08330 | 0.00000 | 0.21011 | 47.3% |
| fem2d_permuted | 16384 | 512 | 4 | 0.69539 | 0.40080 | 0.08310 | 0.00000 | 0.21149 | 42.4% |
| pattern_checkerboard | 16384 | 512 | 4 | 0.35629 | 0.07264 | 0.08333 | 0.00000 | 0.20032 | 79.6% |
| pattern_corner_cluster | 16384 | 512 | 4 | 0.32275 | 0.03802 | 0.08342 | 0.00000 | 0.20131 | 88.2% |
| pattern_diag_stripe | 16384 | 512 | 4 | 0.33421 | 0.04886 | 0.08352 | 0.00000 | 0.20182 | 85.4% |
| pattern_row_skewed | 16384 | 512 | 4 | 0.35949 | 0.07053 | 0.08451 | 0.00000 | 0.20445 | 80.4% |
| scattered_1x8 | 16384 | 512 | 4 | 0.33235 | 0.03197 | 0.08355 | 0.00000 | 0.21683 | 90.4% |
| scattered_2x2 | 16384 | 512 | 4 | 0.33920 | 0.02480 | 0.08358 | 0.00000 | 0.23082 | 92.7% |
| scattered_2x4 | 16384 | 512 | 4 | 0.33994 | 0.02896 | 0.08339 | 0.00000 | 0.22758 | 91.5% |
| scattered_8x1 | 16384 | 512 | 4 | 0.33427 | 0.03037 | 0.08384 | 0.00000 | 0.22006 | 90.9% |
| diag_b0.1 | 65536 | 128 | 4 | 0.53155 | 0.17158 | 0.08410 | 0.00000 | 0.27587 | 67.7% |
| diag_b0.3 | 65536 | 128 | 4 | 0.52195 | 0.21574 | 0.08400 | 0.00000 | 0.22221 | 58.7% |
| diag_b1.0 | 65536 | 128 | 4 | 1.16541 | 0.81402 | 0.08794 | 0.00000 | 0.26345 | 30.2% |
| diag_skewed | 65536 | 128 | 4 | 1.10682 | 0.81261 | 0.08813 | 0.00000 | 0.20608 | 26.6% |
| fem2d | 65536 | 128 | 4 | 0.63814 | 0.32963 | 0.08451 | 0.00000 | 0.22400 | 48.3% |
| fem2d_permuted | 65536 | 128 | 4 | 0.91370 | 0.63018 | 0.08403 | 0.00000 | 0.19949 | 31.0% |
| pattern_checkerboard | 65536 | 128 | 4 | 0.37328 | 0.08035 | 0.08384 | 0.00000 | 0.20909 | 78.5% |
| pattern_corner_cluster | 65536 | 128 | 4 | 0.33379 | 0.04246 | 0.08432 | 0.00000 | 0.20701 | 87.3% |
| pattern_diag_stripe | 65536 | 128 | 4 | 0.33088 | 0.04934 | 0.08416 | 0.00000 | 0.19737 | 85.1% |
| pattern_row_skewed | 65536 | 128 | 4 | 0.35718 | 0.07082 | 0.08374 | 0.00000 | 0.20262 | 80.2% |
| scattered_1x8 | 65536 | 128 | 4 | 0.32982 | 0.03322 | 0.08410 | 0.00000 | 0.21251 | 89.9% |
| scattered_2x2 | 65536 | 128 | 4 | 0.32755 | 0.02426 | 0.08419 | 0.00000 | 0.21910 | 92.6% |
| scattered_2x4 | 65536 | 128 | 4 | 0.44346 | 0.03030 | 0.08410 | 0.00000 | 0.32906 | 93.2% |
| scattered_8x1 | 65536 | 128 | 4 | 0.33370 | 0.03235 | 0.08422 | 0.00000 | 0.21712 | 90.3% |
| diag_b0.1 | 65536 | 512 | 4 | 1.16483 | 0.69258 | 0.30999 | 0.00000 | 0.16227 | 40.5% |
| diag_b0.3 | 65536 | 512 | 4 | 1.41376 | 0.86541 | 0.30979 | 0.00000 | 0.23856 | 38.8% |
| diag_b1.0 | 65536 | 512 | 4 | 3.78227 | 3.28526 | 0.31373 | 0.00000 | 0.18328 | 13.1% |
| diag_skewed | 65536 | 512 | 4 | 3.66838 | 3.15483 | 0.31421 | 0.00000 | 0.19935 | 14.0% |
| fem2d | 65536 | 512 | 4 | 1.82762 | 1.33956 | 0.31002 | 0.00000 | 0.17804 | 26.7% |
| fem2d_permuted | 65536 | 512 | 4 | 3.03405 | 2.56583 | 0.30963 | 0.00000 | 0.15858 | 15.4% |
| pattern_checkerboard | 65536 | 512 | 4 | 0.84294 | 0.30205 | 0.30999 | 0.00000 | 0.23091 | 64.2% |
| pattern_corner_cluster | 65536 | 512 | 4 | 0.63613 | 0.15382 | 0.30960 | 0.00000 | 0.17270 | 75.8% |
| pattern_diag_stripe | 65536 | 512 | 4 | 0.67194 | 0.19347 | 0.30979 | 0.00000 | 0.16867 | 71.2% |
| pattern_row_skewed | 65536 | 512 | 4 | 0.74157 | 0.27187 | 0.30960 | 0.00000 | 0.16009 | 63.3% |
| scattered_1x8 | 65536 | 512 | 4 | 0.59290 | 0.12691 | 0.31005 | 0.00000 | 0.15593 | 78.6% |
| scattered_2x2 | 65536 | 512 | 4 | 0.55453 | 0.08214 | 0.31027 | 0.00000 | 0.16211 | 85.2% |
| scattered_2x4 | 65536 | 512 | 4 | 0.59245 | 0.12496 | 0.30995 | 0.00000 | 0.15753 | 78.9% |
| scattered_8x1 | 65536 | 512 | 4 | 0.59238 | 0.11728 | 0.31027 | 0.00000 | 0.16483 | 80.2% |
| diag_b0.1 | 262144 | 128 | 4 | 1.22790 | 0.68243 | 0.31117 | 0.00000 | 0.23430 | 44.4% |
| diag_b0.3 | 262144 | 128 | 4 | 1.37254 | 0.86519 | 0.31581 | 0.00000 | 0.19155 | 37.0% |
| diag_b1.0 | 262144 | 128 | 4 | 3.75958 | 3.24398 | 0.33245 | 0.00000 | 0.18315 | 13.7% |
| diag_skewed | 262144 | 128 | 4 | 3.53731 | 3.03559 | 0.33283 | 0.00000 | 0.16889 | 14.2% |
| fem2d | 262144 | 128 | 4 | 1.84003 | 1.35242 | 0.32224 | 0.00000 | 0.16537 | 26.5% |
| fem2d_permuted | 262144 | 128 | 4 | 3.38752 | 2.88776 | 0.32176 | 0.00000 | 0.17800 | 14.8% |
| pattern_checkerboard | 262144 | 128 | 4 | 0.80541 | 0.33450 | 0.31104 | 0.00000 | 0.15987 | 58.5% |
| pattern_corner_cluster | 262144 | 128 | 4 | 0.63760 | 0.16787 | 0.31123 | 0.00000 | 0.15849 | 73.7% |
| pattern_diag_stripe | 262144 | 128 | 4 | 0.66298 | 0.20253 | 0.31091 | 0.00000 | 0.14953 | 69.5% |
| pattern_row_skewed | 262144 | 128 | 4 | 0.75043 | 0.28304 | 0.31107 | 0.00000 | 0.15632 | 62.3% |
| scattered_1x8 | 262144 | 128 | 4 | 0.66342 | 0.13600 | 0.31120 | 0.00000 | 0.21622 | 79.5% |
| scattered_2x2 | 262144 | 128 | 4 | 0.57216 | 0.08288 | 0.31091 | 0.00000 | 0.17837 | 85.5% |
| scattered_2x4 | 262144 | 128 | 4 | 0.59712 | 0.13290 | 0.31120 | 0.00000 | 0.15302 | 77.7% |
| scattered_8x1 | 262144 | 128 | 4 | 0.62870 | 0.11408 | 0.31133 | 0.00000 | 0.20329 | 81.9% |
| diag_b0.1 | 262144 | 512 | 4 | 4.12640 | 2.75512 | 1.21415 | 0.00000 | 0.15713 | 33.2% |
| diag_b0.3 | 262144 | 512 | 4 | 4.85254 | 3.45880 | 1.21831 | 0.00000 | 0.17544 | 28.7% |
| diag_b1.0 | 262144 | 512 | 4 | 14.52589 | 13.12104 | 1.23527 | 0.00000 | 0.16958 | 9.7% |
| diag_skewed | 262144 | 512 | 4 | 13.39206 | 11.99127 | 1.23703 | 0.00000 | 0.16377 | 10.5% |
| fem2d | 262144 | 512 | 4 | 6.89248 | 5.43199 | 1.22468 | 0.00000 | 0.23581 | 21.2% |
| fem2d_permuted | 262144 | 512 | 4 | 13.16630 | 11.76705 | 1.22240 | 0.00000 | 0.17685 | 10.6% |
| pattern_checkerboard | 262144 | 512 | 4 | 2.62909 | 1.25280 | 1.21341 | 0.00000 | 0.16287 | 52.3% |
| pattern_corner_cluster | 262144 | 512 | 4 | 1.99194 | 0.62586 | 1.21354 | 0.00000 | 0.15254 | 68.6% |
| pattern_diag_stripe | 262144 | 512 | 4 | 2.15568 | 0.76458 | 1.21476 | 0.00000 | 0.17634 | 64.5% |
| pattern_row_skewed | 262144 | 512 | 4 | 2.49510 | 1.11428 | 1.21376 | 0.00000 | 0.16706 | 55.3% |
| scattered_1x8 | 262144 | 512 | 4 | 1.93645 | 0.48355 | 1.22672 | 0.00000 | 0.22617 | 75.0% |
| scattered_2x2 | 262144 | 512 | 4 | 1.71981 | 0.31136 | 1.21376 | 0.00000 | 0.19468 | 81.9% |
| scattered_2x4 | 262144 | 512 | 4 | 1.87354 | 0.50167 | 1.21386 | 0.00000 | 0.15801 | 73.2% |
| scattered_8x1 | 262144 | 512 | 4 | 1.82502 | 0.44423 | 1.21428 | 0.00000 | 0.16652 | 75.7% |

## Artifacts and reproduction

- [All configurations, all three jobs](../../tests/torch_fast_spdmm/benchmark/results/cusparse_direct_20260929/derived/CUSPARSE_DIRECT_all_configs.csv), including compressed_fraction for every PBR/BOO result and direct preprocess status.
- All individual CUDA-event samples are in the [result archive](../../RESULTS_ARCHIVE.md).
- [Job metadata](../../tests/torch_fast_spdmm/benchmark/results/cusparse_direct_20260929/derived/CUSPARSE_DIRECT_metadata.json) and [geometric means](../../tests/torch_fast_spdmm/benchmark/results/cusparse_direct_20260929/derived/CUSPARSE_DIRECT_geomeans.csv).
- [Per-workload profiler metadata](../../tests/torch_fast_spdmm/benchmark/results/cusparse_direct_20260929/derived/CUSPARSE_DIRECT_profiles_metadata.json), including source, matrix and X hashes.
- Submission recipe: `sbatch scripts/cusparse_direct/job.sh validate`, then separate `sbatch scripts/cusparse_direct/job.sh 1`, `2`, `3` submissions. Set the snapshot path in the script for another checkout. Profiling runs after replicate 1.
- Offline report: `python scripts/cusparse_direct/report.py <results-root>`. Raw Nsight SQLite/report files remain in `/home/dayanb/research/cusparse-direct-20260929/results/profiles`.
