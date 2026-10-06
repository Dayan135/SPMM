#!/bin/bash
# GPU phase: loads offline labels (never clusters), builds BOO, times matmuls.
# Usage: sbatch --dependency=afterok:<cluster array> --time=... matmul.sh <N>
#SBATCH --job-name=spmm-matmul
#SBATCH --partition=rtx3090
#SBATCH --account=erant
#SBATCH --qos=erant
#SBATCH --gres=gpu:rtx_3090:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --output=/home/dayanb/research/clustering-20261006/xl/logs/matmul-%j.log
source ~/.bashrc
conda activate fem
set -euo pipefail
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
ROOT=/home/dayanb/research/clustering-20261006
export PYTHONPATH="$ROOT/deps:$ROOT/torch_fast_spdmm:$ROOT/tests/torch_fast_spdmm"
N=$1
nvidia-smi --query-gpu=name,memory.total --format=csv
cd "$ROOT/tests/torch_fast_spdmm"
python -m benchmark.bench_clustering --phase matmul --labels "$ROOT/xl/labels" \
    --out "$ROOT/xl/n$N" --sizes "$N" --features 128 512 \
    --profiles fem2d fem2d_permuted diag_b0.3 diag_b1.0 --groups 8 16 32 64 128 256
