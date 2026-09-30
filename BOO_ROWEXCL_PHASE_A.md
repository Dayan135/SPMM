# Row-exclusive BOO: Phase A

Measured source: `f65e4b2257e2bf1a97923923251576810a3be4bd`; source SHA256 `e59607e51f2d68f78cbd84f64adf6cf89833be15d0e4744d49ddeb93c0d0a710`; extension SHA256 `b5e6cda34a2281339cb8a960e0218a0e908dc28f6b3ecdc84120927441a8bd55`.

NVIDIA GeForce RTX 3090; PyTorch 2.10.0+cu128; PyTorch CUDA 12.8; FP32 values, int32 indices, seed 0.

| Replicate | Slurm job | Node | Timed / validated workloads |
|---|---|---|---|
| 1 | 21734701 | cs-3090-03.auth.ad.bgu.ac.il | 84 / 210 |
| 2 | 21734702 | cs-3090-04.auth.ad.bgu.ac.il | 84 / 84 |
| 3 | 21734703 | cs-3090-04.auth.ad.bgu.ac.il | 84 / 84 |

The BOO format and HEM implementation are unchanged. HEM targets {8,16,32,64}, feasible_first=False, rowmajor blocks, small_block_policy=remainder, threshold=4 match the existing comparison. Each row-exclusive plan references the **same BOO object** as its corresponding current-BOO arm; values and packed coordinates are not copied or reencoded. Auxiliary tables index block rows, row slices and column-group chunks.

A CTA owns up to 32 rows from one HEM group and one disjoint feature slab (16/32/64; API default 32), using 128 threads. It keeps sums in registers across all the group’s blocks, then processes the owned rows’ CSR remainders in the same CTA and writes Y with plain stores. Empty rows are scheduled and overwritten with zero. There is no Y memset, atomic update, temporary output, or separate remainder kernel.

Balance: greedily split groups **by rows only**, with requested 2,048 stored nonzeros/CTA and at most 32 rows/CTA. The effective cap rises to the maximum single-row degree if necessary; no implementation can enforce a smaller hard cap on an indivisible row without splitting that row. Both requested/effective caps and exact per-CTA histograms are recorded. The shared-memory budget is 8 KiB per CTA, raised when needed to hold one entire column group (up to the default 48 KiB limit). The union of a group’s column groups is staged at once when it fits this budget; otherwise whole column groups are packed into successive chunks. Chunking frequency is structural, exact, and recorded outside timing.

Timing: five warmups, then 20 individual CUDA-event samples in a seeded rotating order, with resident A and X. Preparation, transfers, independent SciPy FP64 validation (rtol=atol=1e-4), and plan construction are untimed. C++ row-exclusive timing encloses exactly its kernel launch(es) with Y preallocated. Direct cuSPARSE reuses descriptors/workspace/output and times only cusparseSpMM, alpha=1/beta=0, after bufferSize and optional preprocess. The full-API comparison calls both methods through Python: row-exclusive allocates uninitialized Y; current BOO allocates/zeros Y and joins its block/remainder streams. **No C++/API cross-scope speedup is reported.**

Each workload selects the lowest median of three job medians per family, retaining the same configuration across jobs. All 32 configurations/workload and all samples remain in the CSVs. Times and speedups below are median [min,max] across jobs; speedups are paired within each job. >1 means row-exclusive is faster. Geometric means combine 28 workloads per size within each job, then report median/range of the three means. Best-vs-best API selection can choose different HEM targets; the matched-k results isolate execution on the identical BOO grouping.

Validation covers the 84 requested workloads in all three jobs, plus all 126 additional workload cells at F={100,200,520} in replicate 1, across every HEM target and slab. GPU edge-case tests additionally poison Y to catch unwritten elements, force chunking and row splitting, and cover zero matrices, empty rows, rectangular inputs and pure CSR remainders.

## Geometric means

| N | Comparison | Speedup median [range] |
|---:|---|---:|
| 16384 | cpp_speedup_vs_direct | 0.388 [0.387, 0.401]× |
| 16384 | api_speedup_vs_best_boo | 0.797 [0.783, 0.832]× |
| 16384 | api_speedup_vs_same_k_boo | 0.840 [0.830, 0.900]× |
| 65536 | cpp_speedup_vs_direct | 0.324 [0.319, 0.325]× |
| 65536 | api_speedup_vs_best_boo | 0.448 [0.435, 0.455]× |
| 65536 | api_speedup_vs_same_k_boo | 0.464 [0.450, 0.472]× |
| 262144 | cpp_speedup_vs_direct | 0.306 [0.305, 0.307]× |
| 262144 | api_speedup_vs_best_boo | 0.370 [0.367, 0.371]× |
| 262144 | api_speedup_vs_same_k_boo | 0.378 [0.373, 0.379]× |

## Per workload

| Profile | N | F | Direct C++ ms | Row-exclusive C++ ms (config) | BOO API ms | Row-exclusive API ms (config) | C++ / API speedup |
|---|---:|---:|---|---|---|---|---|
| diag_b0.1 | 16384 | 128 | 0.0604 [0.0604, 0.0625] | 0.0625 [0.0625, 0.0625] (rowexcl_cpp_k64_s16) | 0.1239 [0.1213, 0.1244] | 0.0732 [0.0727, 0.0753] (rowexcl_api_k64_s32) | 0.967 / 1.657 |
| diag_b0.3 | 16384 | 128 | 0.0773 [0.0727, 0.0778] | 0.0932 [0.0860, 0.0942] (rowexcl_cpp_k64_s32) | 0.1756 [0.1741, 0.1761] | 0.1044 [0.1004, 0.1055] (rowexcl_api_k64_s32) | 0.830 / 1.681 |
| diag_b1.0 | 16384 | 128 | 0.2181 [0.2104, 0.2248] | 0.2084 [0.2002, 0.2191] (rowexcl_cpp_k64_s16) | 0.2924 [0.2883, 0.3128] | 0.2202 [0.2156, 0.2304] (rowexcl_api_k64_s16) | 1.047 / 1.337 |
| diag_skewed | 16384 | 128 | 0.2253 [0.2145, 0.2319] | 0.2437 [0.2345, 0.2509] (rowexcl_cpp_k64_s32) | 0.3087 [0.2934, 0.3113] | 0.2570 [0.2452, 0.2632] (rowexcl_api_k64_s32) | 0.924 / 1.196 |
| fem2d | 16384 | 128 | 0.1065 [0.1050, 0.1065] | 0.2099 [0.2028, 0.2099] (rowexcl_cpp_k32_s16) | 0.2642 [0.2637, 0.2703] | 0.2207 [0.2130, 0.2212] (rowexcl_api_k32_s16) | 0.507 / 1.195 |
| fem2d_permuted | 16384 | 128 | 0.1249 [0.1188, 0.1280] | 0.2017 [0.1951, 0.2099] (rowexcl_cpp_k32_s16) | 0.2662 [0.2478, 0.2719] | 0.2125 [0.2058, 0.2202] (rowexcl_api_k32_s16) | 0.610 / 1.209 |
| pattern_checkerboard | 16384 | 128 | 0.0297 [0.0297, 0.0317] | 0.1044 [0.1034, 0.1044] (rowexcl_cpp_k16_s32) | 0.1004 [0.0952, 0.1034] | 0.1126 [0.1126, 0.1137] (rowexcl_api_k16_s32) | 0.287 / 0.891 |
| pattern_corner_cluster | 16384 | 128 | 0.0256 [0.0246, 0.0287] | 0.1116 [0.1106, 0.1126] (rowexcl_cpp_k8_s32) | 0.1024 [0.0963, 0.1065] | 0.1208 [0.1208, 0.1224] (rowexcl_api_k16_s32) | 0.227 / 0.847 |
| pattern_diag_stripe | 16384 | 128 | 0.0276 [0.0276, 0.0292] | 0.1167 [0.1167, 0.1188] (rowexcl_cpp_k32_s32) | 0.1034 [0.1004, 0.1096] | 0.1270 [0.1270, 0.1290] (rowexcl_api_k16_s32) | 0.237 / 0.815 |
| pattern_row_skewed | 16384 | 128 | 0.0302 [0.0297, 0.0307] | 0.0922 [0.0911, 0.0922] (rowexcl_cpp_k64_s32) | 0.1034 [0.1024, 0.1096] | 0.1019 [0.1014, 0.1024] (rowexcl_api_k64_s32) | 0.328 / 1.010 |
| scattered_1x8 | 16384 | 128 | 0.0246 [0.0246, 0.0256] | 0.1331 [0.1321, 0.1331] (rowexcl_cpp_k32_s32) | 0.1009 [0.0963, 0.1034] | 0.1434 [0.1434, 0.1444] (rowexcl_api_k32_s32) | 0.185 / 0.699 |
| scattered_2x2 | 16384 | 128 | 0.0246 [0.0236, 0.0256] | 0.1219 [0.1219, 0.1219] (rowexcl_cpp_k8_s32) | 0.0983 [0.0952, 0.1060] | 0.1331 [0.1321, 0.1331] (rowexcl_api_k16_s32) | 0.202 / 0.738 |
| scattered_2x4 | 16384 | 128 | 0.0246 [0.0246, 0.0261] | 0.1198 [0.1198, 0.1208] (rowexcl_cpp_k32_s32) | 0.1024 [0.0993, 0.1050] | 0.1311 [0.1306, 0.1311] (rowexcl_api_k16_s32) | 0.205 / 0.781 |
| scattered_8x1 | 16384 | 128 | 0.0266 [0.0266, 0.0266] | 0.0896 [0.0891, 0.0901] (rowexcl_cpp_k32_s32) | 0.1004 [0.0963, 0.1014] | 0.1004 [0.0993, 0.1014] (rowexcl_api_k16_s32) | 0.297 / 0.990 |
| diag_b0.1 | 16384 | 512 | 0.2253 [0.2243, 0.2263] | 0.1812 [0.1802, 0.1823] (rowexcl_cpp_k64_s32) | 0.2586 [0.2570, 0.2642] | 0.1915 [0.1915, 0.1925] (rowexcl_api_k64_s32) | 1.243 / 1.343 |
| diag_b0.3 | 16384 | 512 | 0.2463 [0.2447, 0.2499] | 0.2852 [0.2847, 0.2852] (rowexcl_cpp_k64_s32) | 0.4424 [0.4413, 0.4572] | 0.2970 [0.2959, 0.2990] (rowexcl_api_k64_s32) | 0.864 / 1.491 |
| diag_b1.0 | 16384 | 512 | 0.8023 [0.8013, 0.8084] | 0.7537 [0.7485, 0.7588] (rowexcl_cpp_k64_s32) | 0.9272 [0.9252, 0.9431] | 0.7700 [0.7578, 0.7736] (rowexcl_api_k64_s32) | 1.065 / 1.204 |
| diag_skewed | 16384 | 512 | 0.8023 [0.7982, 0.8023] | 0.8381 [0.8351, 0.8397] (rowexcl_cpp_k64_s32) | 0.9564 [0.9452, 0.9661] | 0.8468 [0.8438, 0.8520] (rowexcl_api_k64_s32) | 0.956 / 1.123 |
| fem2d | 16384 | 512 | 0.3512 [0.3502, 0.3512] | 0.7434 [0.7393, 0.7460] (rowexcl_cpp_k32_s16) | 0.7485 [0.7429, 0.7608] | 0.7547 [0.7516, 0.7567] (rowexcl_api_k32_s16) | 0.472 / 0.992 |
| fem2d_permuted | 16384 | 512 | 0.4244 [0.4234, 0.4270] | 0.7373 [0.7342, 0.7429] (rowexcl_cpp_k32_s32) | 0.7644 [0.7624, 0.7788] | 0.7506 [0.7485, 0.7547] (rowexcl_api_k32_s32) | 0.575 / 1.018 |
| pattern_checkerboard | 16384 | 512 | 0.0911 [0.0891, 0.0932] | 0.3625 [0.3604, 0.3640] (rowexcl_cpp_k16_s32) | 0.1736 [0.1731, 0.1823] | 0.3717 [0.3697, 0.3727] (rowexcl_api_k16_s32) | 0.250 / 0.468 |
| pattern_corner_cluster | 16384 | 512 | 0.0748 [0.0737, 0.0783] | 0.3860 [0.3835, 0.3917] (rowexcl_cpp_k16_s32) | 0.1403 [0.1372, 0.1638] | 0.3963 [0.3932, 0.4014] (rowexcl_api_k16_s32) | 0.191 / 0.354 |
| pattern_diag_stripe | 16384 | 512 | 0.0870 [0.0860, 0.0896] | 0.4178 [0.4178, 0.4188] (rowexcl_cpp_k32_s32) | 0.1690 [0.1623, 0.1731] | 0.4285 [0.4280, 0.4285] (rowexcl_api_k32_s32) | 0.208 / 0.394 |
| pattern_row_skewed | 16384 | 512 | 0.1044 [0.1034, 0.1075] | 0.3174 [0.3164, 0.3174] (rowexcl_cpp_k64_s32) | 0.1741 [0.1715, 0.1853] | 0.3267 [0.3267, 0.3277] (rowexcl_api_k64_s32) | 0.329 / 0.531 |
| scattered_1x8 | 16384 | 512 | 0.0829 [0.0809, 0.0850] | 0.4700 [0.4690, 0.4710] (rowexcl_cpp_k32_s32) | 0.1362 [0.1352, 0.1428] | 0.4823 [0.4813, 0.4823] (rowexcl_api_k32_s32) | 0.177 / 0.283 |
| scattered_2x2 | 16384 | 512 | 0.0691 [0.0686, 0.0717] | 0.4306 [0.4306, 0.4311] (rowexcl_cpp_k32_s32) | 0.1244 [0.1229, 0.1306] | 0.4413 [0.4413, 0.4413] (rowexcl_api_k16_s32) | 0.160 / 0.282 |
| scattered_2x4 | 16384 | 512 | 0.0763 [0.0748, 0.0783] | 0.4260 [0.4260, 0.4270] (rowexcl_cpp_k16_s32) | 0.1393 [0.1377, 0.1444] | 0.4362 [0.4362, 0.4367] (rowexcl_api_k16_s32) | 0.179 / 0.319 |
| scattered_8x1 | 16384 | 512 | 0.0778 [0.0768, 0.0799] | 0.3082 [0.3082, 0.3103] (rowexcl_cpp_k32_s32) | 0.1444 [0.1439, 0.1505] | 0.3195 [0.3185, 0.3205] (rowexcl_api_k16_s32) | 0.252 / 0.450 |
| diag_b0.1 | 65536 | 128 | 0.2263 [0.2253, 0.2268] | 0.2181 [0.2161, 0.2202] (rowexcl_cpp_k64_s32) | 0.2657 [0.2555, 0.2688] | 0.2294 [0.2273, 0.2330] (rowexcl_api_k64_s32) | 1.040 / 1.141 |
| diag_b0.3 | 65536 | 128 | 0.2560 [0.2468, 0.2580] | 0.3180 [0.3133, 0.3215] (rowexcl_cpp_k64_s32) | 0.3999 [0.3732, 0.4009] | 0.3374 [0.3256, 0.3384] (rowexcl_api_k64_s32) | 0.803 / 1.182 |
| diag_b1.0 | 65536 | 128 | 0.8125 [0.8064, 0.8151] | 0.7660 [0.7624, 0.7680] (rowexcl_cpp_k64_s32) | 0.9093 [0.9047, 0.9160] | 0.7823 [0.7736, 0.7844] (rowexcl_api_k64_s32) | 1.061 / 1.168 |
| diag_skewed | 65536 | 128 | 0.8084 [0.7997, 0.8120] | 0.8678 [0.8632, 0.8755] (rowexcl_cpp_k64_s32) | 0.9943 [0.9861, 1.0004] | 0.8863 [0.8796, 0.8878] (rowexcl_api_k64_s32) | 0.927 / 1.121 |
| fem2d | 65536 | 128 | 0.3599 [0.3579, 0.3615] | 0.7793 [0.7741, 0.7803] (rowexcl_cpp_k32_s16) | 0.7690 [0.7680, 0.7721] | 0.7905 [0.7839, 0.7921] (rowexcl_api_k32_s32) | 0.462 / 0.975 |
| fem2d_permuted | 65536 | 128 | 0.5304 [0.5192, 0.5340] | 0.8238 [0.8207, 0.8284] (rowexcl_cpp_k32_s32) | 0.8084 [0.8044, 0.8131] | 0.8340 [0.8320, 0.8397] (rowexcl_api_k32_s32) | 0.640 / 0.968 |
| pattern_checkerboard | 65536 | 128 | 0.1085 [0.1080, 0.1096] | 0.4925 [0.4895, 0.4972] (rowexcl_cpp_k32_s32) | 0.1853 [0.1802, 0.1925] | 0.5048 [0.5038, 0.5110] (rowexcl_api_k64_s32) | 0.220 / 0.367 |
| pattern_corner_cluster | 65536 | 128 | 0.0896 [0.0891, 0.0916] | 0.5335 [0.5330, 0.5458] (rowexcl_cpp_k32_s32) | 0.1393 [0.1372, 0.1444] | 0.5437 [0.5437, 0.5576] (rowexcl_api_k32_s32) | 0.167 / 0.252 |
| pattern_diag_stripe | 65536 | 128 | 0.1044 [0.1019, 0.1055] | 0.5796 [0.5786, 0.5827] (rowexcl_cpp_k8_s32) | 0.1766 [0.1751, 0.1818] | 0.5919 [0.5919, 0.5949] (rowexcl_api_k32_s32) | 0.179 / 0.297 |
| pattern_row_skewed | 65536 | 128 | 0.1229 [0.1198, 0.1239] | 0.4311 [0.4291, 0.4337] (rowexcl_cpp_k64_s32) | 0.1828 [0.1731, 0.1843] | 0.4460 [0.4393, 0.4470] (rowexcl_api_k64_s32) | 0.285 / 0.409 |
| scattered_1x8 | 65536 | 128 | 0.0901 [0.0876, 0.0911] | 0.6554 [0.6548, 0.6564] (rowexcl_cpp_k16_s32) | 0.1556 [0.1485, 0.1562] | 0.6661 [0.6646, 0.6687] (rowexcl_api_k16_s32) | 0.137 / 0.234 |
| scattered_2x2 | 65536 | 128 | 0.0809 [0.0768, 0.0840] | 0.6083 [0.6072, 0.6093] (rowexcl_cpp_k16_s32) | 0.1516 [0.1388, 0.1567] | 0.6252 [0.6185, 0.6257] (rowexcl_api_k32_s32) | 0.133 / 0.242 |
| scattered_2x4 | 65536 | 128 | 0.0886 [0.0829, 0.0911] | 0.6001 [0.5965, 0.6006] (rowexcl_cpp_k16_s32) | 0.1551 [0.1403, 0.1562] | 0.6159 [0.6062, 0.6170] (rowexcl_api_k16_s32) | 0.148 / 0.252 |
| scattered_8x1 | 65536 | 128 | 0.0881 [0.0840, 0.0891] | 0.4362 [0.4352, 0.4372] (rowexcl_cpp_k16_s32) | 0.1638 [0.1556, 0.1659] | 0.4500 [0.4449, 0.4572] (rowexcl_api_k64_s32) | 0.202 / 0.358 |
| diag_b0.1 | 65536 | 512 | 0.8673 [0.8607, 0.8699] | 0.8356 [0.8253, 0.8402] (rowexcl_cpp_k64_s32) | 0.8381 [0.8064, 0.8438] | 0.8576 [0.8361, 0.8602] (rowexcl_api_k64_s32) | 1.038 / 0.974 |
| diag_b0.3 | 65536 | 512 | 0.9646 [0.9503, 0.9697] | 1.2175 [1.2022, 1.2191] (rowexcl_cpp_k64_s32) | 1.2979 [1.2534, 1.3107] | 1.2513 [1.2196, 1.2544] (rowexcl_api_k64_s32) | 0.792 / 1.037 |
| diag_b1.0 | 65536 | 512 | 3.2983 [3.2788, 3.3275] | 2.9983 [2.9522, 3.0198] (rowexcl_cpp_k64_s16) | 3.2568 [3.2384, 3.2809] | 2.9978 [2.9645, 3.0152] (rowexcl_api_k64_s16) | 1.102 / 1.088 |
| diag_skewed | 65536 | 512 | 3.1923 [3.1329, 3.2210] | 3.4253 [3.3987, 3.4509] (rowexcl_cpp_k32_s32) | 3.6649 [3.6183, 3.6792] | 3.4258 [3.3940, 3.4478] (rowexcl_api_k64_s32) | 0.932 / 1.067 |
| fem2d | 65536 | 512 | 1.4418 [1.4408, 1.4454] | 3.0198 [2.9978, 3.0346] (rowexcl_cpp_k32_s32) | 2.7474 [2.7325, 2.7617] | 3.0546 [3.0213, 3.0597] (rowexcl_api_k32_s32) | 0.479 / 0.904 |
| fem2d_permuted | 65536 | 512 | 2.7208 [2.7187, 2.7254] | 3.2246 [3.1903, 3.2394] (rowexcl_cpp_k32_s32) | 2.8344 [2.7981, 2.8442] | 3.2563 [3.2179, 3.2568] (rowexcl_api_k32_s32) | 0.844 / 0.870 |
| pattern_checkerboard | 65536 | 512 | 0.3476 [0.3461, 0.3482] | 1.9446 [1.9405, 1.9579] (rowexcl_cpp_k32_s32) | 0.4900 [0.4736, 0.4925] | 1.9651 [1.9517, 1.9809] (rowexcl_api_k32_s32) | 0.178 / 0.247 |
| pattern_corner_cluster | 65536 | 512 | 0.2898 [0.2888, 0.2908] | 2.1432 [2.1376, 2.1668] (rowexcl_cpp_k32_s32) | 0.3512 [0.3456, 0.3523] | 2.1545 [2.1504, 2.1770] (rowexcl_api_k32_s32) | 0.135 / 0.163 |
| pattern_diag_stripe | 65536 | 512 | 0.3389 [0.3369, 0.3410] | 2.3173 [2.3132, 2.3322] (rowexcl_cpp_k8_s32) | 0.5089 [0.4833, 0.5197] | 2.3465 [2.3388, 2.3639] (rowexcl_api_k32_s32) | 0.146 / 0.215 |
| pattern_row_skewed | 65536 | 512 | 0.4209 [0.4178, 0.4219] | 1.7050 [1.7014, 1.7152] (rowexcl_cpp_k64_s32) | 0.5033 [0.4716, 0.5094] | 1.7249 [1.7162, 1.7408] (rowexcl_api_k64_s32) | 0.246 / 0.289 |
| scattered_1x8 | 65536 | 512 | 0.3062 [0.2929, 0.3082] | 2.6061 [2.6025, 2.6194] (rowexcl_cpp_k64_s32) | 0.3912 [0.3594, 0.3994] | 2.6240 [2.6194, 2.6394] (rowexcl_api_k16_s32) | 0.118 / 0.148 |
| scattered_2x2 | 65536 | 512 | 0.2591 [0.2468, 0.2601] | 2.4294 [2.4207, 2.4392] (rowexcl_cpp_k64_s32) | 0.3502 [0.3128, 0.3625] | 2.4489 [2.4402, 2.4612] (rowexcl_api_k16_s32) | 0.106 / 0.142 |
| scattered_2x4 | 65536 | 512 | 0.2918 [0.2918, 0.2949] | 2.3777 [2.3726, 2.3921] (rowexcl_cpp_k16_s32) | 0.3722 [0.3364, 0.3809] | 2.4023 [2.3900, 2.4207] (rowexcl_api_k16_s32) | 0.123 / 0.154 |
| scattered_8x1 | 65536 | 512 | 0.2918 [0.2898, 0.2949] | 1.7454 [1.7418, 1.7592] (rowexcl_cpp_k32_s64) | 0.4332 [0.3963, 0.4408] | 1.7736 [1.7695, 1.7838] (rowexcl_api_k16_s32) | 0.167 / 0.243 |
| diag_b0.1 | 262144 | 128 | 0.8540 [0.8509, 0.8540] | 0.8074 [0.8059, 0.8110] (rowexcl_cpp_k64_s32) | 0.7695 [0.7670, 0.7716] | 0.8182 [0.8161, 0.8212] (rowexcl_api_k64_s32) | 1.058 / 0.941 |
| diag_b0.3 | 262144 | 128 | 0.9487 [0.9416, 0.9539] | 1.1909 [1.1853, 1.1950] (rowexcl_cpp_k64_s32) | 1.2155 [1.2124, 1.2237] | 1.2037 [1.1981, 1.2073] (rowexcl_api_k64_s32) | 0.797 / 1.014 |
| diag_b1.0 | 262144 | 128 | 3.2768 [3.2579, 3.3111] | 2.9778 [2.9655, 2.9957] (rowexcl_cpp_k64_s16) | 3.2840 [3.2261, 3.2896] | 2.9916 [2.9788, 3.0167] (rowexcl_api_k64_s16) | 1.100 / 1.090 |
| diag_skewed | 262144 | 128 | 3.1135 [3.0618, 3.1293] | 3.8113 [3.7878, 3.8579] (rowexcl_cpp_k32_s32) | 4.4611 [4.4073, 4.5082] | 3.8195 [3.7955, 3.8630] (rowexcl_api_k64_s32) | 0.811 / 1.167 |
| fem2d | 262144 | 128 | 1.4561 [1.4556, 1.4597] | 3.0182 [3.0075, 3.0285] (rowexcl_cpp_k32_s32) | 2.8652 [2.8605, 2.8698] | 3.0433 [3.0305, 3.0587] (rowexcl_api_k32_s32) | 0.482 / 0.940 |
| fem2d_permuted | 262144 | 128 | 2.1069 [2.0895, 2.1417] | 3.2614 [3.2466, 3.2927] (rowexcl_cpp_k32_s32) | 3.1012 [3.0833, 3.1155] | 3.2855 [3.2691, 3.3080] (rowexcl_api_k32_s32) | 0.646 / 0.943 |
| pattern_checkerboard | 262144 | 128 | 0.4029 [0.4014, 0.4055] | 1.9220 [1.9164, 1.9446] (rowexcl_cpp_k32_s32) | 0.5207 [0.5192, 0.5217] | 1.9333 [1.9261, 1.9558] (rowexcl_api_k32_s32) | 0.209 / 0.269 |
| pattern_corner_cluster | 262144 | 128 | 0.3389 [0.3338, 0.3410] | 2.1033 [2.1002, 2.1228] (rowexcl_cpp_k32_s32) | 0.4045 [0.3748, 0.4137] | 2.1187 [2.1166, 2.1402] (rowexcl_api_k32_s32) | 0.160 / 0.189 |
| pattern_diag_stripe | 262144 | 128 | 0.3666 [0.3640, 0.3692] | 2.3030 [2.2917, 2.3173] (rowexcl_cpp_k16_s32) | 0.5084 [0.5007, 0.5166] | 2.3122 [2.3014, 2.3255] (rowexcl_api_k16_s32) | 0.159 / 0.220 |
| pattern_row_skewed | 262144 | 128 | 0.4521 [0.4506, 0.4536] | 1.7009 [1.6870, 1.7060] (rowexcl_cpp_k64_s32) | 0.5028 [0.4992, 0.5048] | 1.7111 [1.6968, 1.7152] (rowexcl_api_k64_s32) | 0.267 / 0.295 |
| scattered_1x8 | 262144 | 128 | 0.3016 [0.3011, 0.3021] | 2.5938 [2.5702, 2.5999] (rowexcl_cpp_k16_s32) | 0.3860 [0.3850, 0.3937] | 2.6030 [2.5836, 2.6132] (rowexcl_api_k16_s32) | 0.116 / 0.149 |
| scattered_2x2 | 262144 | 128 | 0.2499 [0.2488, 0.2509] | 2.3972 [2.3828, 2.4136] (rowexcl_cpp_k16_s32) | 0.3564 [0.3528, 0.3610] | 2.4064 [2.3951, 2.4243] (rowexcl_api_k16_s32) | 0.105 / 0.149 |
| scattered_2x4 | 262144 | 128 | 0.2959 [0.2939, 0.2985] | 2.3736 [2.3578, 2.3818] (rowexcl_cpp_k32_s32) | 0.3671 [0.3625, 0.3676] | 2.3828 [2.3685, 2.3926] (rowexcl_api_k32_s32) | 0.126 / 0.154 |
| scattered_8x1 | 262144 | 128 | 0.2836 [0.2826, 0.2862] | 1.7009 [1.6886, 1.7101] (rowexcl_cpp_k32_s32) | 0.4280 [0.4224, 0.4301] | 1.7111 [1.7009, 1.7208] (rowexcl_api_k32_s32) | 0.167 / 0.250 |
| diag_b0.1 | 262144 | 512 | 3.3756 [3.3751, 3.3777] | 3.2343 [3.2297, 3.2389] (rowexcl_cpp_k64_s32) | 2.8303 [2.8288, 2.8314] | 3.2778 [3.2768, 3.2896] (rowexcl_api_k64_s32) | 1.044 / 0.863 |
| diag_b0.3 | 262144 | 512 | 3.7345 [3.7130, 3.7361] | 4.7375 [4.7252, 4.7498] (rowexcl_cpp_k64_s32) | 4.6085 [4.6075, 4.6218] | 4.8246 [4.8041, 4.8445] (rowexcl_api_k64_s32) | 0.787 / 0.955 |
| diag_b1.0 | 262144 | 512 | 13.4083 [13.1794, 13.5485] | 12.2691 [11.9869, 12.3720] (rowexcl_cpp_k64_s32) | 12.7263 [12.5256, 12.8538] | 12.1482 [11.9291, 12.3233] (rowexcl_api_k64_s16) | 1.095 / 1.048 |
| diag_skewed | 262144 | 512 | 12.1364 [12.0131, 12.3192] | 15.4962 [15.3989, 15.6426] (rowexcl_cpp_k32_s32) | 17.6148 [17.4925, 17.8703] | 15.4747 [15.4092, 15.6380] (rowexcl_api_k32_s32) | 0.783 / 1.138 |
| fem2d | 262144 | 512 | 5.8102 [5.7329, 5.8460] | 12.3494 [12.1876, 12.3602] (rowexcl_cpp_k32_s32) | 10.7356 [10.7121, 10.7587] | 12.3515 [12.1795, 12.3996] (rowexcl_api_k32_s32) | 0.470 / 0.871 |
| fem2d_permuted | 262144 | 512 | 12.3796 [12.2916, 12.5834] | 13.4554 [13.3407, 13.4922] (rowexcl_cpp_k32_s32) | 11.1652 [11.0843, 11.2543] | 13.3729 [13.2500, 13.4385] (rowexcl_api_k32_s32) | 0.921 / 0.837 |
| pattern_checkerboard | 262144 | 512 | 1.3599 [1.3537, 1.3660] | 7.7609 [7.7215, 7.8408] (rowexcl_cpp_k32_s32) | 1.6497 [1.6379, 1.6497] | 7.7665 [7.7266, 7.8500] (rowexcl_api_k32_s32) | 0.175 / 0.211 |
| pattern_corner_cluster | 262144 | 512 | 1.1438 [1.1423, 1.1479] | 8.5903 [8.5514, 8.6461] (rowexcl_cpp_k32_s32) | 1.2308 [1.1940, 1.2349] | 8.5985 [8.5796, 8.6825] (rowexcl_api_k32_s32) | 0.133 / 0.142 |
| pattern_diag_stripe | 262144 | 512 | 1.3292 [1.3235, 1.3312] | 9.2723 [9.2150, 9.3655] (rowexcl_cpp_k16_s32) | 1.7290 [1.7270, 1.7326] | 9.2836 [9.2293, 9.3778] (rowexcl_api_k16_s32) | 0.143 / 0.187 |
| pattern_row_skewed | 262144 | 512 | 1.6666 [1.6625, 1.6712] | 6.8598 [6.8383, 6.9028] (rowexcl_cpp_k64_s32) | 1.6758 [1.6753, 1.6809] | 6.8690 [6.8541, 6.9135] (rowexcl_api_k64_s32) | 0.243 / 0.244 |
| scattered_1x8 | 262144 | 512 | 1.1049 [1.1028, 1.1049] | 10.4084 [10.3685, 10.4934] (rowexcl_cpp_k16_s32) | 1.1648 [1.1628, 1.1791] | 10.4212 [10.3624, 10.5052] (rowexcl_api_k16_s32) | 0.106 / 0.112 |
| scattered_2x2 | 262144 | 512 | 0.9247 [0.9231, 0.9272] | 9.6650 [9.6451, 9.7930] (rowexcl_cpp_k64_s32) | 1.0342 [1.0301, 1.0419] | 9.6758 [9.6543, 9.8043] (rowexcl_api_k64_s32) | 0.096 / 0.107 |
| scattered_2x4 | 262144 | 512 | 1.1259 [1.1203, 1.1300] | 9.5498 [9.5027, 9.6256] (rowexcl_cpp_k32_s32) | 1.1116 [1.1075, 1.1146] | 9.5631 [9.5135, 9.6358] (rowexcl_api_k32_s32) | 0.117 / 0.116 |
| scattered_8x1 | 262144 | 512 | 1.0680 [1.0655, 1.0696] | 6.9253 [6.8915, 7.0180] (rowexcl_cpp_k32_s64) | 1.3619 [1.3609, 1.3716] | 6.9489 [6.9130, 7.0390] (rowexcl_api_k32_s64) | 0.154 / 0.196 |

## Robust-margin counts

These descriptive counts require at least a 1.3× margin in every job; they are not confidence intervals.

| Comparison | ≥1.3× faster in all jobs | ≥1.3× slower in all jobs | Remaining |
|---|---:|---:|---:|
| cpp_speedup_vs_direct | 0 | 58 | 26 |
| api_speedup_vs_best_boo | 5 | 41 | 38 |
| api_speedup_vs_same_k_boo | 8 | 41 | 35 |

## Artifacts

- [Every configuration](BOO_ROWEXCL_phase_a_all_configs.csv), [selected workloads](BOO_ROWEXCL_phase_a_workloads.csv), [same-grouping API comparisons](BOO_ROWEXCL_phase_a_matched_k.csv), [geometric means](BOO_ROWEXCL_phase_a_geomeans.csv).
- [Individual samples](BOO_ROWEXCL_phase_a_samples.csv.gz), [all correctness checks](BOO_ROWEXCL_phase_a_validation.csv.gz), [exact CTA nnz/chunk histograms](BOO_ROWEXCL_phase_a_cta_distribution.csv.gz), [metadata](BOO_ROWEXCL_phase_a_metadata.json).
- Kernel registers, resident CTA bound, shared memory, chunking frequency, row splits and nnz quantiles appear on every row-exclusive result in the all-config CSV. CTA histograms include the feature-slab multiplicity.
- Reproduce with the isolated `scripts/boo_rowexcl/job.sh`: validate, smoke, then three separate jobs with arguments 1, 2 and 3. No pip installation is used.
