#!/bin/bash
#SBATCH --job-name=ncu_rowgroup_atomic
#SBATCH --partition=rtx3090
#SBATCH --account=erant
#SBATCH --qos=erant
#SBATCH --gres=gpu:rtx_3090:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=jobs/profile_rowgroup_atomic/logs/%j.out
#SBATCH --error=jobs/profile_rowgroup_atomic/logs/%j.err

# Next loop iteration: does `rowgroup` (already-implemented, correctness-tested --
# no new kernel code) capture a meaningful fraction of the atomicAdd-removal
# ceiling that STALL_REASONS_NOATOMIC_ABLATION.md found (40-66% time cut on
# block's high-mio_throttle matrices, by deleting the atomic instruction --
# diagnostic-only, numerically wrong, not mergeable)?
#
# `block` issues one atomicAdd per PBR block's Y-flush; `blocks_per_stripe > 1`
# matrices (diag_b0.1, diag_b0.3, scattered_8x1, scattered_2x2 -- the exact set
# the noatomic ablation flagged as high-mio_throttle) have MULTIPLE PBR blocks,
# likely concurrent CTAs on different SMs, atomicAdd-ing into the same output
# row band. `rowgroup` groups same-row-strip PBR blocks into ONE CTA and
# accumulates them in smem_Y before a SINGLE atomicAdd per strip -- directly
# cutting that cross-CTA collision count, exactly the "concrete next lever" the
# ablation's own caveat named (this is the older, already-built version of that
# lever, not the newer wider-atomic-per-row idea, which still needs new code).
# H5 (WHY_BLOCK_KERNEL_FAILS_TO_SCALE.md) already showed rowgroup's row-strip-
# local grouping (unlike chunk's pure array-order grouping) doesn't pay the
# L2-locality tax coarser grouping usually costs -- so this isn't confounded
# by that mechanism the way a chunk-vs-block comparison would be.
#
# An older, coarser experiment (LAST_EXPERIMENTS.md Experiment 1) found
# rowgroup 1.83x FASTER than block on scattered_dense2/f512 ("atomicAdd
# contention + good group fill") but ~1.0x on scattered_8x1/scattered_2x4 --
# using different (older, non-density-fixed) matrix definitions and without
# the mio_throttle-specific lens. This job re-runs the comparison on the
# CURRENT matrix definitions, with the same 15-metric stall-reason list used
# throughout this project, directly against the noatomic ablation's own
# baseline numbers for an apples-to-apples ceiling comparison.

echo "[job] started $(date)  job_id=$SLURM_JOB_ID  node=$SLURMD_NODENAME"

cd "$SLURM_SUBMIT_DIR"

source ~/.bashrc
conda activate fem

set -euo pipefail

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

cd tests/torch_fast_spdmm
OUT="$SLURM_SUBMIT_DIR/ncu_out/rowgroup_atomic"
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

profile() {
    local tag=$1 kernel=$2 matrix=$3 features=$4
    echo "[job] profiling $tag (kernel=$kernel)"
    ncu --profile-from-start off --metrics "$METRICS" --csv --target-processes all \
        python -m benchmark.profile_pbr \
            --size 262144 --features "$features" \
            --block 8 --min-nnz 2 --smem-size 2048 \
            --matrix-profile "$matrix" --kernel "$kernel" \
        > "$OUT/${tag}.csv" 2> "$OUT/${tag}.log" \
        || { echo "  FAILED -- see $OUT/${tag}.log"; tail -5 "$OUT/${tag}.log"; }
    echo "[job] done $tag"
}

MATRICES="diag_b0.1 diag_b0.3 scattered_8x1 scattered_2x2"
FEATURES="128 512"

for matrix in $MATRICES; do
    for features in $FEATURES; do
        tag="${matrix}_262144_f${features}"
        profile "${tag}_block" block "$matrix" "$features"
        profile "${tag}_rowgroup" rowgroup "$matrix" "$features"
    done
done

echo "[job] all profiles done $(date)"
