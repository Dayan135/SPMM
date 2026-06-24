#!/bin/bash
#SBATCH --job-name=ncu_diag
#SBATCH --partition=rtx3090
#SBATCH --account=erant
#SBATCH --qos=erant
#SBATCH --gres=gpu:rtx_3090:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=2:00:00
#SBATCH --output=jobs/profile_diag/logs/%j.out
#SBATCH --error=jobs/profile_diag/logs/%j.err

# Profile diag_dd0.001 at 262144 with block and rowgroup kernels.
#
# Key question: why is rowgroup neutral (1.0×) on diagonal matrices even though
# it eliminates atomicAdd?  Hypothesis: diagonal sparsity → few blocks per
# row-strip → low Y-row contention in the block kernel → plain-store doesn't help.
#
# Diagnostic metrics added vs prior profile jobs:
#   smsp__warp_issue_stall_mio_throttle_per_warp_active.pct
#       — MIO (memory-I/O) stalls; captures atomicAdd throttle in block kernel.
#         If this is LOW on diag/block → atomicAdd is not the bottleneck → explains
#         why rowgroup doesn't help.
#   smsp__warp_issue_stall_long_scoreboard_per_warp_active.pct
#       — L2/DRAM latency stalls; high value means warps wait on cache misses.
#
# Feature counts:
#   f64  — single feat_tile (p=2048); measures the kernel overhead regime.
#   f256 — mid point.
#   f512 — four feat_tiles; rowgroup should benefit most here if it ever does.

echo "[job] started $(date)  job_id=$SLURM_JOB_ID  node=$SLURMD_NODENAME"

cd "$SLURM_SUBMIT_DIR"

source ~/.bashrc
conda activate fem

set -euo pipefail

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
export SPDMM_COMMIT=$(git rev-parse --short HEAD)
echo "[job] commit=$SPDMM_COMMIT"

cd tests/torch_fast_spdmm
OUT="$SLURM_SUBMIT_DIR/ncu_out/profile_diag"
mkdir -p "$OUT"

METRICS="\
gpu__time_duration.sum,\
sm__throughput.avg.pct_of_peak_sustained_elapsed,\
dram__throughput.avg.pct_of_peak_sustained_elapsed,\
sm__warps_active.avg.pct_of_peak_sustained_active,\
launch__grid_size,\
launch__block_size,\
launch__waves_per_multiprocessor,\
l1tex__t_sector_hit_rate.pct,\
lts__t_sector_hit_rate.pct,\
dram__bytes_read.sum,\
dram__bytes_write.sum,\
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,\
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum,\
sm__sass_thread_inst_executed_op_ffma_pred_on.sum,\
smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct,\
smsp__warp_issue_stall_mio_throttle_per_warp_active.pct,\
smsp__warp_issue_stall_long_scoreboard_per_warp_active.pct"

profile() {  # tag  size  features  block  min_nnz  smem  matrix_profile  kernel
    local tag=$1 size=$2 features=$3 block=$4 min_nnz=$5 smem=$6 mprofile=$7 kernel=$8
    echo ""
    echo "[job] === profiling: $tag ==="
    ncu --profile-from-start off \
        --metrics "$METRICS" \
        --csv \
        --target-processes all \
        python -m benchmark.profile_pbr \
            --size      "$size" \
            --features  "$features" \
            --block     "$block" \
            --min-nnz   "$min_nnz" \
            --smem-size "$smem" \
            --matrix-profile "$mprofile" \
            --kernel    "$kernel" \
        > "$OUT/${tag}.csv" 2> "$OUT/${tag}.log" \
        || { echo "  FAILED — see $OUT/${tag}.log"; tail -10 "$OUT/${tag}.log"; }

    grep -A 40 '\[pbr_info\]' "$OUT/${tag}.log" || true
}

# diag_dd0.001 — block kernel at f64, f256, f512
profile diag_262144_f64_block    262144  64 8 2 2048 diag_dd0.001 block
profile diag_262144_f256_block   262144 256 8 2 2048 diag_dd0.001 block
profile diag_262144_f512_block   262144 512 8 2 2048 diag_dd0.001 block

# diag_dd0.001 — rowgroup kernel at f64, f256, f512
profile diag_262144_f64_rowgroup    262144  64 8 2 2048 diag_dd0.001 rowgroup
profile diag_262144_f256_rowgroup   262144 256 8 2 2048 diag_dd0.001 rowgroup
profile diag_262144_f512_rowgroup   262144 512 8 2 2048 diag_dd0.001 rowgroup

echo ""
echo "[job] done $(date)"
