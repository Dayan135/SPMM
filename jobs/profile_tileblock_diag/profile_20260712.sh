#!/bin/bash
#SBATCH --job-name=ncu_tile_diag
#SBATCH --partition=rtx3090
#SBATCH --account=erant
#SBATCH --qos=erant
#SBATCH --gres=gpu:rtx_3090:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=0:45:00
#SBATCH --output=jobs/profile_tileblock_diag/logs/%j.out
#SBATCH --error=jobs/profile_tileblock_diag/logs/%j.err

# Nsight Compute: why does 'tile' lose to 'block' on diag_b1.0 even though the
# 'grid' strategy already produces perfectly-dense tiles there (every 2x2/4x4
# tile in a fully-dense 64-wide diagonal band is 100% populated -- selection
# quality isn't the variable)?
#
# test_benchmark.py's full sweep (LAST_EXPERIMENTS.md Experiment 3) found, at
# size=262144, diag_b1.0: block=4.0ms, tile(best)=9.5ms (2.4x SLOWER), and the
# best tile SHAPE was usually 1x1 -- i.e. bigger tiles (more reuse opportunity)
# make it *worse*, not better. Profiles block vs tile at 1x1/2x2/4x4 to see
# where the per-tile overhead actually goes: occupancy (launch__waves_per_multi
# processor, sm__warps_active), register pressure (launch__registers_per_thread),
# static smem footprint (launch__shared_mem_per_block_static), atomic/sync
# pressure (stall_wait, stall_mio_throttle, stall_short_scoreboard), and DRAM
# traffic (dram__bytes_read -- does tile actually cut X reloads as designed?).
#
# All runs: size=262144, 8x8 PBR blocks, min_nnz=2, smem_size=2048, diag_b1.0
# (and diag_b0.3 as a lower-density secondary point), features in {64, 512}.

echo "[job] started $(date)  job_id=$SLURM_JOB_ID  node=$SLURMD_NODENAME"

cd "$SLURM_SUBMIT_DIR"

source ~/.bashrc
conda activate fem

set -euo pipefail

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

cd tests/torch_fast_spdmm
OUT="$SLURM_SUBMIT_DIR/ncu_out/tileblock_diag"
mkdir -p "$OUT"

METRICS="\
gpu__time_duration.sum,\
sm__throughput.avg.pct_of_peak_sustained_elapsed,\
dram__throughput.avg.pct_of_peak_sustained_elapsed,\
dram__bytes_read.sum,\
dram__bytes_write.sum,\
sm__warps_active.avg.pct_of_peak_sustained_active,\
launch__grid_size,launch__block_size,launch__waves_per_multiprocessor,\
launch__registers_per_thread,launch__shared_mem_per_block_static,\
l1tex__t_sector_hit_rate.pct,\
lts__t_sector_hit_rate.pct,\
smsp__average_warps_issue_stall_lg_throttle_per_issue_active.ratio,\
smsp__average_warps_issue_stall_wait_per_issue_active.ratio,\
smsp__average_warps_issue_stall_short_scoreboard_per_issue_active.ratio,\
smsp__average_warps_issue_stall_mio_throttle_per_issue_active.ratio,\
smsp__average_warps_issue_stall_barrier_per_issue_active.ratio,\
l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum,\
l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum"

# profile TAG KERNEL MATRIX FEATURES [TILE_ROWS TILE_COLS]
profile() {
    local tag=$1 kernel=$2 matrix=$3 features=$4 tr=${5:-2} tc=${6:-2}
    local extra=()
    [ "$kernel" = "tile" ] && extra=(--tile-rows "$tr" --tile-cols "$tc" --tile-strategy grid)
    echo "[job] profiling $tag (kernel=$kernel matrix=$matrix features=$features tile=${tr}x${tc})"
    ncu --profile-from-start off --metrics "$METRICS" --csv \
        --target-processes all \
        python -m benchmark.profile_pbr \
            --size 262144 --features "$features" \
            --block 8 --min-nnz 2 --smem-size 2048 \
            --matrix-profile "$matrix" \
            --kernel "$kernel" "${extra[@]}" \
        > "$OUT/${tag}.csv" 2> "$OUT/${tag}.log" \
        || { echo "  FAILED — see $OUT/${tag}.log"; tail -5 "$OUT/${tag}.log"; }
    echo "[job] done $tag"
}

for feat in 64 512; do
    profile diag_b1.0_f${feat}_block          block diag_b1.0 "$feat"
    profile diag_b1.0_f${feat}_tile_1x1       tile  diag_b1.0 "$feat" 1 1
    profile diag_b1.0_f${feat}_tile_2x2       tile  diag_b1.0 "$feat" 2 2
    profile diag_b1.0_f${feat}_tile_4x4       tile  diag_b1.0 "$feat" 4 4
done

# Secondary point: lower (but still no-off-diag-noise) density.
profile diag_b0.3_f512_block     block diag_b0.3 512
profile diag_b0.3_f512_tile_2x2  tile  diag_b0.3 512 2 2

echo "[job] all profiles done $(date)"
