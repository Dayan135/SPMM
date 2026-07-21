#!/bin/bash
#SBATCH --job-name=ncu_smem_block_data_stall
#SBATCH --partition=rtx3090
#SBATCH --account=erant
#SBATCH --qos=erant
#SBATCH --gres=gpu:rtx_3090:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=6:00:00
#SBATCH --output=jobs/profile_smem_block_data/logs/%j.out
#SBATCH --error=jobs/profile_smem_block_data/logs/%j.err

# Extends profile_20260721.sh (occupancy/static-smem comparison, baseline vs
# feat/pbr-cache-block-data's pbr_spmm_zero_idle_kernel) with the project's
# standard warp-stall-reason metric list, on the SAME config grid, so
# long_scoreboard (DRAM-latency stall) and mio_throttle can be correlated
# directly against the occupancy-limiter/waves-per-SM data already collected.
# Runs both metric passes per config (occupancy then stall-reason) off one
# build per branch, since they must stay off the same METRICS list per
# STALL_REASONS.md's methodology note (mixing counters into one ncu
# invocation changes its replay-pass structure and corrupts the numbers).
#
# Grid widened per follow-up request: smem_size now covers all three tiers
# (512/1024/2048), and a third feature point (255) sits alongside 129/517 --
# all three deliberately NOT divisible by 4 (use_vec = features % 4 == 0,
# pbr_kernel.cu:381) so every config dispatches to pbr_spmm_zero_idle_kernel,
# never pbr_spmm_vec_kernel.

echo "[job] started $(date)  job_id=$SLURM_JOB_ID  node=$SLURMD_NODENAME"

cd "$SLURM_SUBMIT_DIR"

source ~/.bashrc
conda activate fem

set -euo pipefail

export CUDA_HOME="$CONDA_PREFIX"
export CC=$(which x86_64-conda-linux-gnu-cc)
export CXX=$(which x86_64-conda-linux-gnu-c++)
export CUDAHOSTCXX=$CXX
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

OUT_OCC="$SLURM_SUBMIT_DIR/ncu_out/smem_block_data"
OUT_STALL="$SLURM_SUBMIT_DIR/ncu_out/smem_block_data_stall"
mkdir -p "$OUT_OCC" "$OUT_STALL"

OCC_METRICS="\
gpu__time_duration.sum,\
launch__shared_mem_per_block_static,\
launch__shared_mem_per_block_dynamic,\
launch__shared_mem_config_size,\
launch__occupancy_limit_shared_mem,\
launch__occupancy_limit_registers,\
launch__occupancy_limit_warps,\
launch__occupancy_limit_blocks,\
launch__waves_per_multiprocessor,\
launch__registers_per_thread,\
sm__warps_active.avg.pct_of_peak_sustained_active"

# Exact list from optimizer/performance.py METRICS -- the project's
# validated stall-reason set. Kept as its own ncu invocation, never merged
# with OCC_METRICS above.
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

MATRICES="diag_b0.3 scattered_1x8 scattered_8x1"
BLOCKS="2 4 8"
SMEMS="512 1024 2048"
FEATURES="129 255 517"
SIZE=262144
MIN_NNZ=2

profile_ref () {
    local tag="$1" ref="$2"
    echo "[job] === checking out $ref (tag=$tag) ==="
    git fetch origin
    git checkout "$ref"
    git pull origin "$ref" || true
    local commit
    commit=$(git rev-parse HEAD)
    echo "[job] building $ref @ $commit"
    pip install -e torch_fast_spdmm --no-build-isolation

    cd tests/torch_fast_spdmm
    for matrix in $MATRICES; do
        for block in $BLOCKS; do
            for smem in $SMEMS; do
                for feat in $FEATURES; do
                    cfg="${tag}_${matrix}_b${block}_s${smem}_f${feat}"
                    args="--size $SIZE --features $feat --block $block --min-nnz $MIN_NNZ --smem-size $smem --matrix-profile $matrix --kernel block"

                    echo "[job] profiling (occupancy) $cfg"
                    ncu --profile-from-start off --metrics "$OCC_METRICS" --csv --target-processes all \
                        python -m benchmark.profile_pbr $args \
                        > "$OUT_OCC/${cfg}.csv" 2> "$OUT_OCC/${cfg}.log" \
                        || { echo "  FAILED (occ) $cfg"; tail -5 "$OUT_OCC/${cfg}.log"; }

                    echo "[job] profiling (stall-reason) $cfg"
                    ncu --profile-from-start off --metrics "$STALL_METRICS" --csv --target-processes all \
                        python -m benchmark.profile_pbr $args \
                        > "$OUT_STALL/${cfg}.csv" 2> "$OUT_STALL/${cfg}.log" \
                        || { echo "  FAILED (stall) $cfg"; tail -5 "$OUT_STALL/${cfg}.log"; }
                done
            done
        done
    done
    cd "$SLURM_SUBMIT_DIR"
}

profile_ref baseline research/pbr-cusparse-scalability-gap
profile_ref smem_block_data feat/pbr-cache-block-data

echo "[job] batch complete $(date)"
