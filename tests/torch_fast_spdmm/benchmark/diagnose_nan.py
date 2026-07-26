"""Diagnose the NaN outputs seen for nosyncAll at diag_b0.3, block=8,
min_nnz=12/20, low feature counts. Checks: accounted_blocks() (vacuous?),
whether baseline also produces NaN (data issue vs nosyncAll-specific race),
and repeats each config a few times to see if NaN is reproducible/deterministic
or intermittent (race signature)."""
import torch
from benchmark.profile_pbr import build_matrix
from torch_fast_spdmm import csr_to_pbr, pbr_batched_matmul_cuda, pbr_batched_matmul_scalar_diag_cuda

SIZE = 262144
CONFIGS = [
    ("diag_b0.3", 8, 12, [16, 32, 64, 128]),
    ("diag_b0.3", 8, 20, [16, 32, 64, 128]),
]

for matrix, block, min_nnz, feats in CONFIGS:
    mat = build_matrix(SIZE, matrix)
    pbr_mat = csr_to_pbr(mat, block_rows=block, block_cols=block, min_nnz_per_block=min_nnz)
    n_blocks = pbr_mat.accounted_blocks()
    print(f"\n=== {matrix} block={block} min_nnz={min_nnz}: accounted_blocks={n_blocks} ===")
    pbr_mat = pbr_mat.to("cuda")

    for feat in feats:
        torch.manual_seed(0)
        x = torch.randn((1, mat.shape[1], feat), dtype=torch.float32, device="cuda")

        y_base = pbr_batched_matmul_cuda(pbr_mat, x)
        base_nan = torch.isnan(y_base).any().item()
        base_sum = y_base.sum().item()

        nan_count = 0
        sums = []
        for trial in range(5):
            y = pbr_batched_matmul_scalar_diag_cuda(pbr_mat, x, variant="nosyncAll")
            torch.cuda.synchronize()
            is_nan = torch.isnan(y).any().item()
            n_nan_elems = torch.isnan(y).sum().item()
            if is_nan:
                nan_count += 1
            sums.append((is_nan, n_nan_elems, y.sum().item() if not is_nan else float('nan')))

        print(f"  feat={feat:4d}  baseline_nan={base_nan} baseline_sum={base_sum:.4f}  "
              f"nosyncAll: {nan_count}/5 trials NaN  detail={sums}")
