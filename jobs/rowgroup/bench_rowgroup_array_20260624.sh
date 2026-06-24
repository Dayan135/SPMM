#!/bin/bash
#SBATCH --job-name=bench_rowgroup
#SBATCH --partition=rtx3090
#SBATCH --account=erant
#SBATCH --qos=erant
#SBATCH --gres=gpu:rtx_3090:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --array=0-2
#SBATCH --output=jobs/rowgroup/logs/%A_%a.out
#SBATCH --error=jobs/rowgroup/logs/%A_%a.err

# Full benchmark sweep comparing the row-group fusion kernel against the
# original block kernel, cusparse, and csr_naive. The PBR axis now sweeps both
# kernels (block + rowgroup), so the PBR portion is ~2x the previous sweep —
# hence the 12h limit. One array task per system size keeps wall-clock to the
# longest single size. The .so is assumed already built by the build gate job
# (same branch, no .cu/.cpp changes since); array tasks only run pytest.

# SYS_SIZES must match SIZES in tests/torch_fast_spdmm/test_benchmark.py.
# Note: do NOT name this SIZE — `conda activate` exports SIZE as the binutils
# `size` tool (like CC/CXX/AR), which would clobber our value.
SYS_SIZES=(16384 65536 262144)
SYS_SIZE=${SYS_SIZES[$SLURM_ARRAY_TASK_ID]}

echo "[job] started $(date)  job_id=$SLURM_JOB_ID  task=$SLURM_ARRAY_TASK_ID  size=$SYS_SIZE  node=$SLURMD_NODENAME"

cd "$SLURM_SUBMIT_DIR"

source ~/.bashrc
conda activate fem

set -euo pipefail

# Runtime import needs the conda libstdc++ (CXXABI) on PATH.
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
# W&B offline so the batch run never blocks on auth; sync afterward.
export WANDB_MODE=offline
# Stamp every run with the kernel version so batches are orderable in W&B.
export SPDMM_COMMIT=$(git rev-parse --short HEAD)
echo "[job] benchmarking commit=$SPDMM_COMMIT"

echo "[job] benchmarking size=$SYS_SIZE"
# Test ids embed the size as `<family>_<size>`, so -k filters this task's slice.
pytest tests/torch_fast_spdmm/test_benchmark.py -v --benchmark-only -k "_${SYS_SIZE}"

echo "[job] done $(date)"
