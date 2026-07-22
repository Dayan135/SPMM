#!/bin/bash
#SBATCH --job-name=ncu_mergebc_alloc_ablation
#SBATCH --partition=rtx3090
#SBATCH --account=erant
#SBATCH --qos=erant
#SBATCH --gres=gpu:rtx_3090:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=jobs/profile_mergebc_alloc_ablation/logs/%j.out
#SBATCH --error=jobs/profile_mergebc_alloc_ablation/logs/%j.err

# mergeBC allocation-ablation ncu profiling (branch feat/pbr-mergebc-alloc-ablation,
# forked from feat/pbr-zeroidle-microbench). Three scalar_diag variants of
# pbr_spmm_zero_idle_kernel:
#   baseline      -- real smem_Y read/write (the true, unmodified kernel)
#   mergeBC_alloc -- Step B/C fused (atomicAdd straight from register), but an
#                    explicit dead smem_Y pointer kept at baseline's offset/
#                    size, so the __shared__ allocation is unchanged from
#                    baseline -- isolates instruction removal from occupancy
#   mergeBC       -- Step B/C fused, no smem_Y pointer at all (the original,
#                    already-measured variant from STALL_REASONS_
#                    ZEROIDLE_MICROBENCH.md)
#
# Question: does mergeBC_alloc (allocation held constant) recover most of
# mergeBC's ~9-12% win, or does the win mostly disappear when the allocation
# isn't (as Eran suspected it might already be) freed? See
# STALL_REASONS_MERGEBC_ALLOC_ABLATION.md for the write-up.
#
# Same 5-matrix x 2-feature grid, N=262144, block=8, min_nnz=2 as
# STALL_REASONS_ZEROIDLE_MICROBENCH.md, for direct comparability.
#
# TWO SEPARATE ncu invocations per config (never mixed in one -- mixing extra
# counters into the stall-reason METRICS list has previously corrupted its
# numbers in this project; see STALL_REASONS.md's methodology note):
#   1. Occupancy pass: launch-config / static-smem / occupancy-limit metrics
#      (same list as jobs/profile_smem_block_data/profile_20260721.sh)
#   2. Stall-reason pass: the exact METRICS list from optimizer/performance.py

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

echo "[job] building"
pip install -e torch_fast_spdmm --no-build-isolation

cd tests/torch_fast_spdmm
OUT_OCC="$SLURM_SUBMIT_DIR/ncu_out/mergebc_alloc_ablation/occupancy"
OUT_STALL="$SLURM_SUBMIT_DIR/ncu_out/mergebc_alloc_ablation/stall_reasons"
mkdir -p "$OUT_OCC" "$OUT_STALL"

METRICS_OCC="\
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

METRICS_STALL="\
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

profile_one() {
    local pass_name=$1 metrics=$2 out=$3 tag=$4 matrix=$5 size=$6 features=$7 variant=$8
    echo "[job] profiling $pass_name $tag (kernel=scalar_diag, variant=$variant)"
    ncu --profile-from-start off --metrics "$metrics" --csv --target-processes all \
        python -m benchmark.profile_pbr \
            --size "$size" --features "$features" \
            --block 8 --min-nnz 2 --smem-size 2048 \
            --matrix-profile "$matrix" --kernel scalar_diag --scalar-variant "$variant" \
        > "$out/${tag}.csv" 2> "$out/${tag}.log" \
        || { echo "  FAILED — see $out/${tag}.log"; tail -5 "$out/${tag}.log"; }
    echo "[job] done $pass_name $tag"
}

MATRICES="diag_b0.1 diag_b0.3 scattered_1x8 scattered_8x1 scattered_2x2"
SIZE=262144
FEATURES="128 512"
VARIANTS="baseline mergeBC_alloc mergeBC"

for matrix in $MATRICES; do
    for features in $FEATURES; do
        for variant in $VARIANTS; do
            tag="${matrix}_${SIZE}_f${features}_${variant}"
            profile_one occupancy "$METRICS_OCC" "$OUT_OCC" "$tag" "$matrix" "$SIZE" "$features" "$variant"
            profile_one stall_reasons "$METRICS_STALL" "$OUT_STALL" "$tag" "$matrix" "$SIZE" "$features" "$variant"
        done
    done
done

echo "[job] all profiles done $(date)"
