#!/bin/bash
#SBATCH --job-name=build_occ
#SBATCH --partition=rtx3090
#SBATCH --account=erant
#SBATCH --qos=erant
#SBATCH --gres=gpu:rtx_3090:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=jobs/logs/%j.out
#SBATCH --error=jobs/logs/%j.err

# Build gate: rebuild after adding get_pbr_rowgroup_blocks_per_sm occupancy query
# and switching _build_group_row_starts to use it instead of the hardcoded
# _BLOCKS_PER_SM_EST × _PIPELINE_DEPTH = 8 estimate.
#
# Expected: all 528 rowgroup tests pass.  The only behaviour change is
# target_groups = sm_count × real_bps (likely 12 for RTX 3090) vs the previous
# sm_count × 8.  The grouper still produces a valid partition — just more groups.

echo "[job] started $(date)  job_id=$SLURM_JOB_ID  node=$SLURMD_NODENAME"

cd "$SLURM_SUBMIT_DIR"

source ~/.bashrc
conda activate fem

set -euo pipefail

export CUDA_HOME="$CONDA_PREFIX"
export CC=$(which x86_64-conda-linux-gnu-cc)
export CXX=$(which x86_64-conda-linux-gnu-c++)
export CUDAHOSTCXX=$CXX
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

export SPDMM_COMMIT=$(git rev-parse --short HEAD)
echo "[job] commit=$SPDMM_COMMIT"

echo "[job] building..."
pip install -e torch_fast_spdmm --no-build-isolation -q

echo "[job] smoke-check: query bps on RTX 3090"
python - <<'EOF'
import torch  # loads libtorch/libc10 so the extension can dlopen them
import torch_fast_spdmm_core as core
for br in [2, 4, 8]:
    for sm in [512, 1024, 2048]:
        bps = core.pbr_rowgroup_blocks_per_sm(br, sm)
        print(f"  block_rows={br}  smem={sm}  blocks_per_sm={bps}")
EOF

echo "[job] running tests..."
pytest tests/torch_fast_spdmm/ -v -q

echo "[job] done $(date)"
