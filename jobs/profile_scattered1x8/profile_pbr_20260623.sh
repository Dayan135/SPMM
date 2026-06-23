#!/bin/bash
#SBATCH --job-name=ncu_pbr_1x8
#SBATCH --partition=rtx3090
#SBATCH --account=erant
#SBATCH --qos=erant
#SBATCH --gres=gpu:rtx_3090:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=0:45:00
#SBATCH --output=jobs/profile_scattered1x8/logs/%j.out
#SBATCH --error=jobs/profile_scattered1x8/logs/%j.err

# Profile PBR on scattered_1x8 at sizes 16384 and 262144 (features=128).
# Goal: see whether col-origin sort improved L1/L2 hit rates for X reads,
# and whether smaller smem_size (512 vs 2048) frees enough L1 to matter.
# Best block config for this profile is 8x8 (matches the 1x8 scatter pattern).

echo "[job] started $(date)  job_id=$SLURM_JOB_ID  node=$SLURMD_NODENAME"

cd "$SLURM_SUBMIT_DIR"

source ~/.bashrc
conda activate fem

set -euo pipefail

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

cd tests/torch_fast_spdmm
OUT="$SLURM_SUBMIT_DIR/ncu_out/profile_scattered1x8"
mkdir -p "$OUT"

METRICS="gpu__time_duration.sum,\
sm__throughput.avg.pct_of_peak_sustained_elapsed,\
dram__throughput.avg.pct_of_peak_sustained_elapsed,\
sm__warps_active.avg.pct_of_peak_sustained_active,\
launch__grid_size,launch__block_size,launch__waves_per_multiprocessor,\
l1tex__t_sector_hit_rate.pct,lts__t_sector_hit_rate.pct"

profile() {  # tag size block min_nnz smem_size
    echo "[job] profiling $1 (size=$2 block=$3 min_nnz=$4 smem=$5)"
    ncu --profile-from-start off --metrics "$METRICS" --csv \
        --target-processes all \
        python -m benchmark.profile_pbr \
            --size "$2" --features 128 \
            --block "$3" --min-nnz "$4" \
            --smem-size "$5" \
            --matrix-profile scattered_1x8 \
        > "$OUT/$1.csv" 2> "$OUT/$1.log" \
        || { echo "  FAILED — see $OUT/$1.log"; tail -5 "$OUT/$1.log"; }
}

# size=16384: smem sweep to see L1 hit rate vs tile-size tradeoff
profile small_16384_8x8min2_sm2048  16384  8 2 2048
profile small_16384_8x8min2_sm512   16384  8 2  512

# size=262144: same configs at scale
profile big_262144_8x8min2_sm2048   262144 8 2 2048
profile big_262144_8x8min2_sm512    262144 8 2  512

echo "[job] done $(date)"
