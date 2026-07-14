#!/bin/bash
#SBATCH --job-name=ncu_adaptive_bypass_fixed
#SBATCH --partition=rtx3090
#SBATCH --account=erant
#SBATCH --qos=erant
#SBATCH --gres=gpu:rtx_3090:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=jobs/profile_stall_reasons/logs/%j.out
#SBATCH --error=jobs/profile_stall_reasons/logs/%j.err

# Corrected rerun of profile_adaptive_bypass_20260714.sh: that job mixed
# sm__sass_inst_executed_op_shared.sum into the same ncu invocation as the
# stall-reason metrics, which changed ncu's replay-pass structure and
# corrupted the stall-reason numbers (verified: diag_b0.3 never triggers the
# n_r<=1 bypass at all -- 0% of its PBR blocks qualify -- yet its profile
# still differed from baseline, while registers/occupancy/total-instruction-
# count matched baseline almost exactly). This job uses the EXACT validated
# 15-metric list from profile_after_20260714.sh with nothing added, so the
# stall-reason numbers are directly comparable to STALL_REASONS.md's baseline.
# Shared-memory instruction counts are collected separately (see the
# standalone verification in the findings doc).

echo "[job] started $(date)  job_id=$SLURM_JOB_ID  node=$SLURMD_NODENAME"

cd "$SLURM_SUBMIT_DIR"

source ~/.bashrc
conda activate fem

set -euo pipefail

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

cd tests/torch_fast_spdmm
OUT="$SLURM_SUBMIT_DIR/ncu_out/stall_reasons_adaptive_bypass_fixed"
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

MATRICES="diag_b0.1 diag_b0.3 scattered_1x8 scattered_8x1 scattered_2x2"
SIZES="1024 262144"
FEATURES="128 512"

for matrix in $MATRICES; do
    for size in $SIZES; do
        for features in $FEATURES; do
            tag="${matrix}_${size}_f${features}_block"
            profile_block "$tag" "$matrix" "$size" "$features"
        done
    done
done

echo "[job] all profiles done $(date)"
