#!/bin/bash
#SBATCH --job-name=bench_mergebc_full_262144
#SBATCH --partition=rtx3090
#SBATCH --account=erant
#SBATCH --qos=erant
#SBATCH --gres=gpu:rtx_3090:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=jobs/bench_mergebc_fullgrid/logs/%j_262144.out
#SBATCH --error=jobs/bench_mergebc_fullgrid/logs/%j_262144.err

# Full-grid baseline vs mergeBC wall-clock sweep, scalar_diag path, ONE SIZE
# per job (this project has hit 24h timeouts on monolithic full-config-matrix
# jobs before -- see LAST_EXPERIMENTS.md's tileblock section -- so each size
# is a separate sbatch submission rather than one job looping all sizes).
#
# Grid: 10 matrices (test_benchmark.py's MATRIX_FAMILIES) x 6 features
# ([16,32,64,128,256,512]) x 7 (block_size,min_nnz) configs (test_benchmark.py's
# own _SCALAR_DIAG_BLOCK_CONFIGS) x 2 variants (baseline, mergeBC) x 5 repeats
# (+3 warmup, untimed) = this size's slice of the standard full benchmark grid.
# Pure wall-clock (torch.cuda.Event), no ncu -- much cheaper per point than the
# profiling jobs, hence feasible to run the full grid instead of the narrow
# 5-matrix/2-feature diagnostic set.

echo "[job] started $(date)  job_id=$SLURM_JOB_ID  node=$SLURMD_NODENAME  size=262144"

cd "$SLURM_SUBMIT_DIR"

source ~/.bashrc
conda activate fem

set -euo pipefail

export CUDA_HOME="$CONDA_PREFIX"
export CC=$(which x86_64-conda-linux-gnu-cc)
export CXX=$(which x86_64-conda-linux-gnu-c++)
export CUDAHOSTCXX=$CXX
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

echo "[job] building"
pip install -e torch_fast_spdmm --no-build-isolation

cd tests/torch_fast_spdmm
OUT="$SLURM_SUBMIT_DIR/ncu_out/mergebc_fullgrid"
mkdir -p "$OUT"

echo "[job] running full-grid benchmark for size=262144"
python -m benchmark.benchmark_mergebc_fullgrid --size 262144 \
    --out "$OUT/mergebc_fullgrid_262144.csv"

echo "[job] done $(date)"
