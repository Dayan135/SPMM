#!/bin/bash
#SBATCH --job-name=ncu_bg_cg
#SBATCH --partition=rtx3090
#SBATCH --account=erant
#SBATCH --qos=erant
#SBATCH --gres=gpu:rtx_3090:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=0:45:00
#SBATCH --output=jobs/profile_blockgroup_colgroup/logs/%j.out
#SBATCH --error=jobs/profile_blockgroup_colgroup/logs/%j.err

# Nsight Compute head-to-head: blockgroup vs colgroup.
#
# Profiles 3 matrix × 2 kernels = 6 runs, all at size=262144 / 8x8 / min_nnz=2 / smem=2048.
#
# Matrices chosen to span the observed performance range:
#   scattered_dense2/f512  — blockgroup wins 1.80× (strip amortisation at high features)
#   scattered_dense2/f64   — colgroup wins  1.38× (X-load amortisation at low features)
#   scattered_2x4/f512     — both ~0.95×   (neither amortises much at this density)
#
# Metrics captured:
#   gpu__time_duration              — kernel wall time
#   sm__throughput / dram__throughput — SOL utilisation
#   dram__bytes_read                — total DRAM reads (X-load savings visible here)
#   sm__warps_active                — occupancy
#   launch__*                       — grid/block/waves geometry
#   l1tex / lts hit rates           — cache efficiency
#   stall_lg_throttle               — warp stalls waiting on global memory
#   stall_wait                      — warp stalls waiting on __syncthreads()
#   stall_short_scoreboard          — warp stalls waiting on shared memory pipe
#   stall_mio_throttle              — warp stalls from memory-IO throttle (atomicAdd)
#   shared_op_ld / shared_op_st     — smem load/store wavefronts (smem pressure)

echo "[job] started $(date)  job_id=$SLURM_JOB_ID  node=$SLURMD_NODENAME"

cd "$SLURM_SUBMIT_DIR"

source ~/.bashrc
conda activate fem

set -euo pipefail

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

cd tests/torch_fast_spdmm
OUT="$SLURM_SUBMIT_DIR/ncu_out/blockgroup_colgroup"
mkdir -p "$OUT"

METRICS="\
gpu__time_duration.sum,\
sm__throughput.avg.pct_of_peak_sustained_elapsed,\
dram__throughput.avg.pct_of_peak_sustained_elapsed,\
dram__bytes_read.sum,\
sm__warps_active.avg.pct_of_peak_sustained_active,\
launch__grid_size,launch__block_size,launch__waves_per_multiprocessor,\
l1tex__t_sector_hit_rate.pct,\
lts__t_sector_hit_rate.pct,\
smsp__average_warps_issue_stall_lg_throttle_per_issue_active.ratio,\
smsp__average_warps_issue_stall_wait_per_issue_active.ratio,\
smsp__average_warps_issue_stall_short_scoreboard_per_issue_active.ratio,\
smsp__average_warps_issue_stall_mio_throttle_per_issue_active.ratio,\
l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum,\
l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum"

# profile TAG KERNEL MATRIX FEATURES
profile() {
    local tag=$1 kernel=$2 matrix=$3 features=$4
    echo "[job] profiling $tag (kernel=$kernel  matrix=$matrix  features=$features)"
    ncu --profile-from-start off --metrics "$METRICS" --csv \
        --target-processes all \
        python -m benchmark.profile_pbr \
            --size 262144 --features "$features" \
            --block 8 --min-nnz 2 --smem-size 2048 \
            --matrix-profile "$matrix" \
            --kernel "$kernel" \
        > "$OUT/${tag}.csv" 2> "$OUT/${tag}.log" \
        || { echo "  FAILED — see $OUT/${tag}.log"; tail -5 "$OUT/${tag}.log"; }
    echo "[job] done $tag"
}

# scattered_dense2 / f512  (blockgroup wins)
profile dense2_f512_blockgroup blockgroup scattered_dense2 512
profile dense2_f512_colgroup   colgroup   scattered_dense2 512

# scattered_dense2 / f64  (colgroup wins)
profile dense2_f64_blockgroup  blockgroup scattered_dense2 64
profile dense2_f64_colgroup    colgroup   scattered_dense2 64

# scattered_2x4 / f512  (both neutral — control)
profile s2x4_f512_blockgroup   blockgroup scattered_2x4    512
profile s2x4_f512_colgroup     colgroup   scattered_2x4    512

echo "[job] all profiles done $(date)"
