#!/bin/bash
#SBATCH --job-name=ncu_mergebc_alloc_rep
#SBATCH --partition=rtx3090
#SBATCH --account=erant
#SBATCH --qos=erant
#SBATCH --gres=gpu:rtx_3090:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=5:00:00
#SBATCH --output=jobs/profile_mergebc_alloc_ablation/logs/rep_%j.out
#SBATCH --error=jobs/profile_mergebc_alloc_ablation/logs/rep_%j.err

# mergeBC allocation-ablation ncu profiling, REPEATED (N=5 reps per config) for
# statistical confidence -- supersedes the single-shot job (profile_20260721.sh,
# job 19483209, kept as a single reference data point). Same 3 variants
# (baseline / mergeBC_alloc / mergeBC), same 5-matrix x 2-feature grid,
# N=262144, block=8, min_nnz=2, same two-pass metric split (occupancy vs
# stall-reason, never mixed -- see STALL_REASONS.md methodology note) as
# profile_20260721.sh. Each ncu invocation is now repeated 5x back-to-back per
# (matrix, features, variant, pass), writing rep0..rep4 CSVs, so the parser can
# report mean/median/stddev per metric instead of a single point value.
#
# 5 matrices x 2 features x 3 variants x 2 passes x 5 reps = 300 ncu invocations.

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
OUT_OCC="$SLURM_SUBMIT_DIR/ncu_out/mergebc_alloc_ablation_rep/occupancy"
OUT_STALL="$SLURM_SUBMIT_DIR/ncu_out/mergebc_alloc_ablation_rep/stall_reasons"
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

N_REPS=5

profile_one() {
    local pass_name=$1 metrics=$2 out=$3 tag=$4 matrix=$5 size=$6 features=$7 variant=$8
    for rep in $(seq 0 $((N_REPS - 1))); do
        local rtag="${tag}_rep${rep}"
        echo "[job] profiling $pass_name $rtag (kernel=scalar_diag, variant=$variant)"
        ncu --profile-from-start off --metrics "$metrics" --csv --target-processes all \
            python -m benchmark.profile_pbr \
                --size "$size" --features "$features" \
                --block 8 --min-nnz 2 --smem-size 2048 \
                --matrix-profile "$matrix" --kernel scalar_diag --scalar-variant "$variant" \
            > "$out/${rtag}.csv" 2> "$out/${rtag}.log" \
            || { echo "  FAILED — see $out/${rtag}.log"; tail -5 "$out/${rtag}.log"; }
        echo "[job] done $pass_name $rtag"
    done
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

echo "[job] all repeated profiles done $(date)"
