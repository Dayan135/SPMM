#!/bin/bash
#SBATCH --job-name=ncu_pbr
#SBATCH --partition=rtx3090
#SBATCH --account=erant
#SBATCH --qos=erant
#SBATCH --gres=gpu:rtx_3090:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=0:30:00
#SBATCH --output=jobs/profile_scattered8x1/logs/%j.out
#SBATCH --error=jobs/profile_scattered8x1/logs/%j.err

# Nsight Compute profiling of the PBR SpMM kernel on scattered_8x1, to explain
# why the cusparse speedup collapses from 3.91x (16384) to 1.28x (262144).
# Three points: best@small, best@big, and best@small's config @big (control to
# separate the size effect from the config/min_nnz effect).

echo "[job] started $(date)  job_id=$SLURM_JOB_ID  node=$SLURMD_NODENAME"

cd "$SLURM_SUBMIT_DIR"

source ~/.bashrc
conda activate fem

set -euo pipefail

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

cd tests/torch_fast_spdmm
OUT="$SLURM_SUBMIT_DIR/ncu_out"
mkdir -p "$OUT"

# Speed-of-light, occupancy, launch geometry, and cache hit rates — enough to
# tell latency/occupancy-bound (small) from memory-bound (large).
METRICS="gpu__time_duration.sum,\
sm__throughput.avg.pct_of_peak_sustained_elapsed,\
dram__throughput.avg.pct_of_peak_sustained_elapsed,\
sm__warps_active.avg.pct_of_peak_sustained_active,\
launch__grid_size,launch__block_size,launch__waves_per_multiprocessor,\
l1tex__t_sector_hit_rate.pct,lts__t_sector_hit_rate.pct"

profile() {  # tag size block min_nnz
    echo "[job] profiling $1 (size=$2 block=$3 min_nnz=$4)"
    ncu --profile-from-start off --metrics "$METRICS" --csv \
        --target-processes all \
        python -m benchmark.profile_pbr --size "$2" --features 64 \
            --block "$3" --min-nnz "$4" \
        > "$OUT/$1.csv" 2> "$OUT/$1.log" || { echo "  FAILED — see $OUT/$1.log"; tail -5 "$OUT/$1.log"; }
}

profile small_16384_2x2min2 16384  2 2
profile big_262144_2x2min3 262144 2 3
profile big_262144_2x2min2 262144 2 2   # control: small's config at big size

echo "[job] done $(date)"
