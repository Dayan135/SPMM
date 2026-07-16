#!/bin/bash
#SBATCH --job-name=ncu_zeroidle_microbench
#SBATCH --partition=rtx3090
#SBATCH --account=erant
#SBATCH --qos=erant
#SBATCH --gres=gpu:rtx_3090:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --output=jobs/profile_zeroidle_microbench/logs/%j.out
#SBATCH --error=jobs/profile_zeroidle_microbench/logs/%j.err

# Diagnostic-only zero_idle-kernel microbench (branch feat/pbr-zeroidle-microbench,
# forked from feat/pbr-block-stall-fix): profiles 6 forced-scalar-dispatch
# variants of pbr_spmm_zero_idle_kernel -- baseline (first-ever profile of
# this kernel at f128/f512, never dispatched by the normal use_vec path),
# constdata (block_data global read -> constant), nosyncA/nosyncB/nosyncC
# (each __syncthreads() deleted individually), and mergeBC (Step B/C fused,
# atomicAdd directly from register, no smem_Y, p held fixed -- isolates the
# merge/sync-removal effect from the p-growth confound that sank both prior,
# never-actually-dispatched attempts on feat/pbr-block-stall-fix and
# feat/pbr-smemy-capped-p). Several variants produce numerically WRONG output
# by design -- see check_scalar_diag_correctness.py for the correctness pass,
# run separately (no ncu overhead).
#
# Same METRICS list and N=262144/matrix grid as jobs/profile_stall_reasons/
# profile_20260714.sh, crossed with 6 --scalar-variant values instead of
# block vs cusparse.

echo "[job] started $(date)  job_id=$SLURM_JOB_ID  node=$SLURMD_NODENAME"

cd "$SLURM_SUBMIT_DIR"

source ~/.bashrc
conda activate fem

set -euo pipefail

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

cd tests/torch_fast_spdmm
OUT="$SLURM_SUBMIT_DIR/ncu_out/zeroidle_microbench"
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

profile_variant() {
    local tag=$1 matrix=$2 size=$3 features=$4 variant=$5
    echo "[job] profiling $tag (kernel=scalar_diag, variant=$variant)"
    ncu --profile-from-start off --metrics "$METRICS" --csv --target-processes all \
        python -m benchmark.profile_pbr \
            --size "$size" --features "$features" \
            --block 8 --min-nnz 2 --smem-size 2048 \
            --matrix-profile "$matrix" --kernel scalar_diag --scalar-variant "$variant" \
        > "$OUT/${tag}.csv" 2> "$OUT/${tag}.log" \
        || { echo "  FAILED — see $OUT/${tag}.log"; tail -5 "$OUT/${tag}.log"; }
    echo "[job] done $tag"
}

MATRICES="diag_b0.1 diag_b0.3 scattered_1x8 scattered_8x1 scattered_2x2"
SIZE=262144
FEATURES="128 512"
VARIANTS="baseline constdata nosyncA nosyncB nosyncC mergeBC"

for matrix in $MATRICES; do
    for features in $FEATURES; do
        for variant in $VARIANTS; do
            tag="${matrix}_${SIZE}_f${features}_${variant}"
            profile_variant "$tag" "$matrix" "$SIZE" "$features" "$variant"
        done
    done
done

echo "[job] all profiles done $(date)"
