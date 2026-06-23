## 2026-06-23

**What I tried:** ncu profiling of `scattered_8x1` at sizes 16384 and 262144 (features=64, block=2×2) to explain why PBR speedup over cusparse collapses from 3.91× to 1.28×.

**What worked:** The ncu data clearly isolates the cause. At size=16384: 0.18 waves/SM, 5% compute SOL, 3% DRAM — GPU is essentially idle, both kernels are dominated by fixed launch overhead. PBR's 3.91× is an overhead-ratio artifact, not throughput. At size=262144: 47.6 waves/SM, 84% occupancy, 47% compute — GPU saturated; 1.28× is the real asymptotic speedup. Confirmed that the "best" config at 262144 (2×2 min_nnz=3) actually runs `csr_spmm_kernel` throughout — min_nnz=3 pushes all 8×1 blocks into the COO remainder, so "best PBR" on that matrix bypasses the block kernel entirely.

**What didn't:** The 2×2 block kernel is a poor fit for 8×1 block structure. At scale it runs at 47% compute vs the CSR remainder at 63% DRAM — the block kernel is 1.7× slower than plain CSR on this family. min_nnz tuning reveals the block kernel as the bottleneck, not the solution.

**Next steps:** (1) Profile cusparse at both sizes to confirm its overhead dominates at 16384. (2) Try 8×1-native blocking (`block_rows=8, block_cols=1`) on the `scattered_8x1` family — the 2×2 blocks don't align with the natural 8×1 sparsity pattern.

**Open questions:** How much of the ~100µs gap between ncu kernel time (72µs) and benchmark wall time (172µs) is output-tensor allocation vs Python dispatch vs CUDA sync? Is the block kernel genuinely worse than CSR on 8×1 patterns due to wasted padding, or is it the min_nnz threshold leaving too few blocks behind?
