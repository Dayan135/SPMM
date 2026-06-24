#!/bin/bash
#SBATCH --job-name=ncu_occupancy
#SBATCH --partition=rtx3090
#SBATCH --account=erant
#SBATCH --qos=erant
#SBATCH --gres=gpu:rtx_3090:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=1:30:00
#SBATCH --output=jobs/profile_occupancy/logs/%j.out
#SBATCH --error=jobs/profile_occupancy/logs/%j.err

# Profile the rowgroup kernel after the occupancy-guided target_groups fix.
#
# Before: target_groups = sm_count × 8 (hardcoded _BLOCKS_PER_SM_EST×_PIPELINE_DEPTH)
# After:  target_groups = sm_count × cudaOccupancyMaxActiveBlocksPerMultiprocessor(kernel)
#
# Expected improvements (based on pre-fix profiling):
#   dense2 f512 rowgroup:    waves/SM  0.73 → ~1.0   (grid 655→~984, feat_tiles=1)
#   scat8x1 f512 rowgroup:   waves/SM  2.78 → ~4.0   (grid 2504→~3900, feat_tiles=4)
#
# This job also prints blocks_per_sm / target_groups / num_groups from pbr_info
# so we can confirm the query returned the expected value.

echo "[job] started $(date)  job_id=$SLURM_JOB_ID  node=$SLURMD_NODENAME"

cd "$SLURM_SUBMIT_DIR"

source ~/.bashrc
conda activate fem

set -euo pipefail

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
export SPDMM_COMMIT=$(git rev-parse --short HEAD)
echo "[job] commit=$SPDMM_COMMIT"

cd tests/torch_fast_spdmm
OUT="$SLURM_SUBMIT_DIR/ncu_out/profile_occupancy"
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

    grep -A 35 '\[pbr_info\]' "$OUT/${tag}.log" || true
}

# Primary cases from the last profile run — now with the occupancy fix applied.
# Rowgroup only: block kernel is unchanged.
profile dense2_262144_f512_rowgroup   262144 512 2 2 2048 scattered_dense2 rowgroup
profile scat8x1_262144_f512_rowgroup  262144 512 8 2 2048 scattered_8x1    rowgroup

# Also profile the f128 dense2 case (was rowgroup < block before occupancy fix)
profile dense2_262144_f128_rowgroup   262144 128 2 2 2048 scattered_dense2 rowgroup

echo ""
echo "[job] done $(date)"
