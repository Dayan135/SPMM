#!/bin/bash
#SBATCH --job-name=ncu_noatomic_ablation
#SBATCH --partition=rtx3090
#SBATCH --account=erant
#SBATCH --qos=erant
#SBATCH --gres=gpu:rtx_3090:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --output=jobs/profile_noatomic_ablation/logs/%j.out
#SBATCH --error=jobs/profile_noatomic_ablation/logs/%j.err

# Diagnostic-only ablation (branch feat/pbr-noatomic-ablation, forked from
# feat/pbr-block-stall-fix): pbr_kernel.cu's block-kernel Y-flush (Step C, both
# scalar and vec variants) has atomicAdd swapped for a plain non-atomic store.
# Output is numerically WRONG whenever >1 PBR block writes the same (row,
# feature) -- this build is never meant to be merged, only profiled.
#
# Question: STALL_REASONS.md (feat/pbr-block-stall-fix) established via
# ncu --query-metrics that atomicAdd issues on the LSU pipe, not MIO, so
# mio_throttle should be ~unaffected by this ablation. Does removing the
# atomic instruction entirely move mio_throttle, long_scoreboard, or wall-clock
# time at all? Same METRICS/matrix grid as profile_stall_reasons, restricted to
# N=262144 (the only size STALL_REASONS.md trusts for stall sampling) and the
# `block` kernel only (no cusparse leg -- baseline numbers already exist in
# ncu_out/stall_reasons/ from the unmodified kernel).

echo "[job] started $(date)  job_id=$SLURM_JOB_ID  node=$SLURMD_NODENAME"

cd "$SLURM_SUBMIT_DIR"

source ~/.bashrc
conda activate fem

set -euo pipefail

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

cd tests/torch_fast_spdmm
OUT="$SLURM_SUBMIT_DIR/ncu_out/noatomic_ablation"
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
    echo "[job] profiling $tag (kernel=block, noatomic)"
    ncu --profile-from-start off --metrics "$METRICS" --csv --target-processes all \
        python -m benchmark.profile_pbr \
            --size "$size" --features "$features" \
            --block 8 --min-nnz 2 --smem-size 2048 \
            --matrix-profile "$matrix" --kernel block \
        > "$OUT/${tag}.csv" 2> "$OUT/${tag}.log" \
        || { echo "  FAILED — see $OUT/${tag}.log"; tail -5 "$OUT/${tag}.log"; }
    echo "[job] done $tag"
}

MATRICES="diag_b0.1 diag_b0.3 scattered_1x8 scattered_8x1 scattered_2x2"
SIZE=262144
FEATURES="128 512"

for matrix in $MATRICES; do
    for features in $FEATURES; do
        tag="${matrix}_${SIZE}_f${features}_block_noatomic"
        profile_block "$tag" "$matrix" "$SIZE" "$features"
    done
done

echo "[job] all profiles done $(date)"
