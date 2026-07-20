#!/bin/bash
#SBATCH --job-name=ncu_colmajor
#SBATCH --partition=rtx3090
#SBATCH --account=erant
#SBATCH --qos=erant
#SBATCH --gres=gpu:rtx_3090:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --output=jobs/profile_colmajor/logs/%j.out
#SBATCH --error=jobs/profile_colmajor/logs/%j.err

# Follow-up to the col-major wall-clock sweep (commits ca59466/8e63dd9 on this
# branch): that sweep only measured time, never explained WHY col-major sort
# ((col_origin, row_origin) instead of natural (row_origin, col_origin))
# sometimes wins big and sometimes regresses badly on the SAME unmodified
# kernel -- only the PBR block launch order changes. H5
# (WHY_BLOCK_KERNEL_FAILS_TO_SCALE.md) already established that block's core
# advantage over grouped kernels comes from L2 hit rate driven by which CTAs
# happen to be concurrently in flight, not from per-CTA work volume. This job
# tests whether the same L2-locality mechanism explains col-major's mixed
# results: reordering trades X-read locality against Y-write (atomicAdd
# destination) locality between concurrently-scheduled CTAs, and the net
# depends on the matrix's row/col spread.
#
# Three representative size=262144 points pulled from the W&B sweep:
#   diag_b1.0            f128 block=2 min=2 sm=2048  -- best win,       -28.4%
#   scattered_1x8_d2pct   f512 block=8 min=2 sm=2048  -- best scattered win, -36.6%
#   scattered_8x1_d2pct   f256 block=8 min=2 sm=512   -- worst regression,  +54.5%
# Each profiled baseline (row-major) vs --col-major, with BOTH the standard
# 15-metric stall-reason list AND an L2/DRAM locality list (H5's dram_growth
# job), as SEPARATE ncu invocations -- mixing extra counters into the
# stall-reason list previously changed ncu's replay-pass structure and
# corrupted results (see STALL_REASONS.md), so never combine them.

echo "[job] started $(date)  job_id=$SLURM_JOB_ID  node=$SLURMD_NODENAME"

cd "$SLURM_SUBMIT_DIR"

source ~/.bashrc
conda activate fem

set -euo pipefail

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

cd tests/torch_fast_spdmm
OUT="$SLURM_SUBMIT_DIR/ncu_out/colmajor"
mkdir -p "$OUT"

STALL_METRICS="\
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

LOCALITY_METRICS="\
gpu__time_duration.sum,\
sm__throughput.avg.pct_of_peak_sustained_elapsed,\
dram__throughput.avg.pct_of_peak_sustained_elapsed,\
dram__bytes_read.sum,\
dram__bytes_write.sum,\
sm__warps_active.avg.pct_of_peak_sustained_active,\
launch__grid_size,launch__block_size,launch__waves_per_multiprocessor,\
l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum,\
l1tex__t_bytes_pipe_lsu_mem_global_op_st.sum,\
l1tex__t_sector_hit_rate.pct,\
lts__t_sector_hit_rate.pct"

profile() {
    local tag=$1 metrics_name=$2 metrics=$3; shift 3
    echo "[job] profiling $tag ($metrics_name)"
    ncu --profile-from-start off --metrics "$metrics" --csv --target-processes all \
        python -m benchmark.profile_pbr "$@" \
        > "$OUT/${tag}_${metrics_name}.csv" 2> "$OUT/${tag}_${metrics_name}.log" \
        || { echo "  FAILED -- see $OUT/${tag}_${metrics_name}.log"; tail -5 "$OUT/${tag}_${metrics_name}.log"; }
    echo "[job] done $tag ($metrics_name)"
}

run_point() {
    local tag=$1 matrix=$2 features=$3 block=$4 min_nnz=$5 smem=$6 density=$7
    local common_args=(--size 262144 --features "$features" --block "$block" \
        --min-nnz "$min_nnz" --smem-size "$smem" --matrix-profile "$matrix" \
        --kernel block --density-pct "$density")

    profile "${tag}_rowmajor" stall "$STALL_METRICS" "${common_args[@]}"
    profile "${tag}_colmajor" stall "$STALL_METRICS" "${common_args[@]}" --col-major

    profile "${tag}_rowmajor" locality "$LOCALITY_METRICS" "${common_args[@]}"
    profile "${tag}_colmajor" locality "$LOCALITY_METRICS" "${common_args[@]}" --col-major
}

# diag_b1.0 has no density param (fully dense diag band) -- density-pct is
# ignored for "diag" profile kind, harmless to pass a placeholder.
run_point diag_b1.0_f128            diag_b1.0        128 2 2 2048 0.5
run_point scattered_1x8_d2pct_f512  scattered_1x8    512 8 2 2048 2.0
run_point scattered_8x1_d2pct_f256  scattered_8x1    256 8 2 512  2.0

echo "[job] all profiles done $(date)"
