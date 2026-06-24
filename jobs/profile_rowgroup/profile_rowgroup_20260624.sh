#!/bin/bash
#SBATCH --job-name=ncu_rowgroup
#SBATCH --partition=rtx3090
#SBATCH --account=erant
#SBATCH --qos=erant
#SBATCH --gres=gpu:rtx_3090:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=1:30:00
#SBATCH --output=jobs/profile_rowgroup/logs/%j.out
#SBATCH --error=jobs/profile_rowgroup/logs/%j.err

# Head-to-head ncu profile: block kernel vs rowgroup kernel.
#
# Primary focus: scattered_dense2 (2x2 blocks, size=262144, feat=512) —
# the case where the benchmark showed rowgroup +71% over block.
# Secondary: scattered_8x1 (8x8 blocks, size=262144, feat=512) —
# a typical large-matrix case where rowgroup was ~+21%.
#
# Metrics collected (all per-kernel):
#   gpu__time_duration.sum              — kernel wall time
#   sm__throughput.avg.pct_of_peak_sustained_elapsed   — SM compute %
#   dram__throughput.avg.pct_of_peak_sustained_elapsed — DRAM BW %
#   sm__warps_active.avg.pct_of_peak_sustained_active  — warp occupancy %
#   launch__grid_size                   — total CUDA blocks launched
#   launch__block_size                  — threads per block
#   launch__waves_per_multiprocessor    — how many waves of blocks per SM
#   l1tex__t_sector_hit_rate.pct        — L1 hit rate (X column reads)
#   lts__t_sector_hit_rate.pct          — L2 hit rate
#   dram__bytes_read.sum                — total DRAM bytes read
#   l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum  — smem read bank conflicts
#   l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum  — smem write bank conflicts
#   sm__sass_thread_inst_executed_op_ffma_pred_on.sum — FMA instruction count
#   smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct — global load efficiency

echo "[job] started $(date)  job_id=$SLURM_JOB_ID  node=$SLURMD_NODENAME"

cd "$SLURM_SUBMIT_DIR"

source ~/.bashrc
conda activate fem

set -euo pipefail

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
export SPDMM_COMMIT=$(git rev-parse --short HEAD)
echo "[job] commit=$SPDMM_COMMIT"

cd tests/torch_fast_spdmm
OUT="$SLURM_SUBMIT_DIR/ncu_out/profile_rowgroup"
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
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,\
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum,\
sm__sass_thread_inst_executed_op_ffma_pred_on.sum,\
smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct"

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

    # Echo pbr_info (goes to stdout of the python process, captured in .log by ncu)
    grep -A 30 '\[pbr_info\]' "$OUT/${tag}.log" || true
}

# --- scattered_dense2: 2x2 blocks, 262144, feat=512 (rowgroup +71% in benchmark) ---
profile dense2_262144_f512_block    262144 512 2 2 2048 scattered_dense2 block
profile dense2_262144_f512_rowgroup 262144 512 2 2 2048 scattered_dense2 rowgroup

# --- scattered_dense2: same but feat=128 (moderate feature count) ---
profile dense2_262144_f128_block    262144 128 2 2 2048 scattered_dense2 block
profile dense2_262144_f128_rowgroup 262144 128 2 2 2048 scattered_dense2 rowgroup

# --- scattered_8x1: 8x8 blocks, 262144, feat=512 (rowgroup +21% in benchmark) ---
profile scat8x1_262144_f512_block    262144 512 8 2 2048 scattered_8x1 block
profile scat8x1_262144_f512_rowgroup 262144 512 8 2 2048 scattered_8x1 rowgroup

echo ""
echo "[job] done $(date)"
