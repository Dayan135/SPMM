#!/bin/bash
# Offline clustering: CPU-only nodes, no GPU allocated. One task per (profile, method).
# Usage: sbatch --array=0-15 cluster-array.sh <N>
#SBATCH --job-name=spmm-cluster
#SBATCH --partition=cpu
#SBATCH --account=erant
#SBATCH --qos=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=/home/dayanb/research/clustering-20261006/xl/logs/cluster-%A_%a.log
source ~/.bashrc
conda activate fem
set -euo pipefail
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export CUDA_VISIBLE_DEVICES=
ROOT=/home/dayanb/research/clustering-20261006
export PYTHONPATH="$ROOT/deps:$ROOT/torch_fast_spdmm:$ROOT/tests/torch_fast_spdmm"
N=$1
PROFILES=(fem2d fem2d_permuted diag_b0.3 diag_b1.0)
METHODS=(identity hem metis graclus)
P=${PROFILES[$((SLURM_ARRAY_TASK_ID / 4))]}
M=${METHODS[$((SLURM_ARRAY_TASK_ID % 4))]}
cd "$ROOT/tests/torch_fast_spdmm"
python -m benchmark.bench_clustering --phase cluster --labels "$ROOT/xl/labels" \
    --sizes "$N" --profiles "$P" --methods "$M" --groups 8 16 32 64 128 256
