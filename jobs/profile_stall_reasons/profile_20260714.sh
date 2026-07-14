#!/bin/bash
#SBATCH --job-name=ncu_stall_reasons
#SBATCH --partition=rtx3090
#SBATCH --account=erant
#SBATCH --qos=erant
#SBATCH --gres=gpu:rtx_3090:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=jobs/profile_stall_reasons/logs/%j.out
#SBATCH --error=jobs/profile_stall_reasons/logs/%j.err

# Extends H6's warp-stall-reason verification (WHY_BLOCK_KERNEL_FAILS_TO_SCALE.md,
# STALL_REASONS.md) beyond the single diag_b0.3/262144/f512 data point, across the
# matrix/size/feature grid the user asked for:
#   matrices: diag_b0.1, diag_b0.3, scattered_1x8, scattered_8x1, scattered_2x2
#   sizes:    1024, 262144
#   features: 128, 512
#   kernels:  block (PBR) vs cusparse (torch.sparse.mm / csrmm_alg2)
#
# Same metric set as the original ad hoc stall-reason run (block vs cuSPARSE,
# diag_b0.3, 262144/f512) that produced the H6 correction:
#   - smsp__warp_issue_stalled_*_per_warp_active.pct (all reasons ncu reports)
#   - sm__pipe_{alu,fma}_cycles_active.avg.pct_of_peak_sustained_elapsed
#   - sm__warps_active.avg.pct_of_peak_sustained_active
# Plus gpu__time_duration.sum and sm__throughput for context in the writeup.

echo "[job] started $(date)  job_id=$SLURM_JOB_ID  node=$SLURMD_NODENAME"

cd "$SLURM_SUBMIT_DIR"

source ~/.bashrc
conda activate fem

set -euo pipefail

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

cd tests/torch_fast_spdmm
OUT="$SLURM_SUBMIT_DIR/ncu_out/stall_reasons"
mkdir -p "$OUT"

METRICS="\
gpu__time_duration.sum,\
sm__throughput.avg.pct_of_peak_sustained_elapsed,\
sm__pipe_alu_cycles_active.avg.pct_of_peak_sustained_elapsed,\
sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_elapsed,\
sm__warps_active.avg.pct_of_peak_sustained_active,\
smsp__warp_issue_stalled_barrier_per_warp_active.pct,\
smsp__warp_issue_stalled_drain_per_warp_active.pct,\
smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct,\
smsp__warp_issue_stalled_math_pipe_throttle_per_warp_active.pct,\
smsp__warp_issue_stalled_mio_throttle_per_warp_active.pct,\
smsp__warp_issue_stalled_no_instruction_per_warp_active.pct,\
smsp__warp_issue_stalled_not_selected_per_warp_active.pct,\
smsp__warp_issue_stalled_selected_per_warp_active.pct,\
smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct,\
smsp__warp_issue_stalled_wait_per_warp_active.pct"

profile_block() {
    local tag=$1 matrix=$2 size=$3 features=$4
    echo "[job] profiling $tag (kernel=block)"
    ncu --profile-from-start off --metrics "$METRICS" --csv --target-processes all \
        python -m benchmark.profile_pbr \
            --size "$size" --features "$features" \
            --block 8 --min-nnz 2 --smem-size 2048 \
            --matrix-profile "$matrix" --kernel block \
        > "$OUT/${tag}.csv" 2> "$OUT/${tag}.log" \
        || { echo "  FAILED — see $OUT/${tag}.log"; tail -5 "$OUT/${tag}.log"; }
    echo "[job] done $tag"
}

profile_cusparse() {
    local tag=$1 matrix=$2 size=$3 features=$4
    echo "[job] profiling $tag (kernel=cusparse)"
    ncu --profile-from-start off --metrics "$METRICS" --csv --target-processes all \
        python -m benchmark.profile_cusparse \
            --size "$size" --features "$features" \
            --matrix-profile "$matrix" \
        > "$OUT/${tag}.csv" 2> "$OUT/${tag}.log" \
        || { echo "  FAILED — see $OUT/${tag}.log"; tail -5 "$OUT/${tag}.log"; }
    echo "[job] done $tag"
}

MATRICES="diag_b0.1 diag_b0.3 scattered_1x8 scattered_8x1 scattered_2x2"
SIZES="1024 262144"
FEATURES="128 512"

for matrix in $MATRICES; do
    for size in $SIZES; do
        for features in $FEATURES; do
            tag="${matrix}_${size}_f${features}"
            profile_block "${tag}_block" "$matrix" "$size" "$features"
            profile_cusparse "${tag}_cusparse" "$matrix" "$size" "$features"
        done
    done
done

echo "[job] all profiles done $(date)"
