#!/bin/bash
#SBATCH --job-name=boo-rowexcl
#SBATCH --partition=rtx3090
#SBATCH --account=erant
#SBATCH --qos=erant
#SBATCH --gres=gpu:rtx_3090:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/home/dayanb/research/boo-rowexcl-20260929/ncu-smoke-%j.log
#SBATCH --error=/home/dayanb/research/boo-rowexcl-20260929/ncu-smoke-%j.err
source ~/.bashrc
conda activate fem
set -euo pipefail
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=2 PYTHONUNBUFFERED=1
cd /home/dayanb/research/boo-rowexcl-20260929
export PYTHONPATH="$PWD/torch_fast_spdmm:$PWD/tests/torch_fast_spdmm:${PYTHONPATH:-}"
python - <<'CHECK'
import torch_fast_spdmm as m, torch_fast_spdmm_core as core
for module in (m,core):
    print(module.__file__)
    assert module.__file__.startswith('/home/dayanb/research/boo-rowexcl-20260929/')
CHECK
mkdir -p results/ncu_smoke
cd tests/torch_fast_spdmm
ncu --profile-from-start off --metrics gpu__time_duration.sum,sm__warps_active.avg.pct_of_peak_sustained_active,smsp__warp_issue_stalled_no_instruction_per_warp_active.pct --cache-control none --clock-control none --csv --log-file ../../results/ncu_smoke/counters.csv python -m benchmark.profile_boo_rowexcl --size 1024 --features 100 --profile fem2d_permuted --mode rowexcl --out ../../results/ncu_smoke
