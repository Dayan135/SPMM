#!/bin/bash
#SBATCH --job-name=ncu_cusparse
#SBATCH --partition=rtx3090
#SBATCH --account=erant
#SBATCH --qos=erant
#SBATCH --gres=gpu:rtx_3090:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=0:20:00
#SBATCH --output=jobs/profile_scattered8x1/logs/%j.out
#SBATCH --error=jobs/profile_scattered8x1/logs/%j.err

# Profile cusparse SpMM at sizes 16384 and 262144 (scattered_8x1, features=64)
# to confirm that fixed launch overhead dominates at small size and explains
# why PBR's 3.91x speedup there is an artifact.

echo "[job] started $(date)  job_id=$SLURM_JOB_ID  node=$SLURMD_NODENAME"

cd "$SLURM_SUBMIT_DIR"

source ~/.bashrc
conda activate fem

set -euo pipefail

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

cd tests/torch_fast_spdmm
OUT="$SLURM_SUBMIT_DIR/ncu_out"
mkdir -p "$OUT"

METRICS="gpu__time_duration.sum,\
sm__throughput.avg.pct_of_peak_sustained_elapsed,\
dram__throughput.avg.pct_of_peak_sustained_elapsed,\
sm__warps_active.avg.pct_of_peak_sustained_active,\
launch__grid_size,launch__block_size,launch__waves_per_multiprocessor,\
l1tex__t_sector_hit_rate.pct,lts__t_sector_hit_rate.pct"

profile() {  # tag size
    echo "[job] profiling $1 (size=$2)"
    ncu --profile-from-start off --metrics "$METRICS" --csv \
        --target-processes all \
        python -m benchmark.profile_cusparse --size "$2" --features 64 \
        > "$OUT/$1.csv" 2> "$OUT/$1.log" || { echo "  FAILED — see $OUT/$1.log"; tail -5 "$OUT/$1.log"; }
}

profile cusparse_16384  16384
profile cusparse_262144 262144

echo "[job] done $(date)"
