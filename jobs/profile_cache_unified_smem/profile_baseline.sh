#!/bin/bash
#SBATCH --job-name=ncu_cache_baseline
#SBATCH --partition=rtx3090
#SBATCH --account=erant
#SBATCH --qos=erant
#SBATCH --gres=gpu:rtx_3090:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=3:00:00
#SBATCH --output=jobs/profile_cache_unified_smem/logs/%j_baseline.out
#SBATCH --error=jobs/profile_cache_unified_smem/logs/%j_baseline.err

# Three-way comparison (this job = ONE arm, baseline) for
# STALL_REASONS_CACHE_BLOCK_DATA_UNIFIED_SMEM.md: baseline (no block_data
# cache) vs. the original ADDITIVE block_data cache (feat/pbr-cache-block-data)
# vs. this new UNIFIED-smem-budget redesign
# (feat/pbr-cache-block-data-unified-smem). Same grid as the original
# occupancy/stall-reason study (jobs/profile_smem_block_data/, single-shot):
# matrices {diag_b0.3, scattered_1x8, scattered_8x1} x blocks {2,4,8} x
# smem_size {512,1024,2048} x features {129,255,517} (non-multiple-of-4, so
# kernel="block" auto-selects pbr_spmm_zero_idle_kernel, not the vec kernel),
# size=262144 -- but N=5 REPEATED trials per config this time, not
# single-shot, so mean/median/sd are real. Two separate ncu passes per rep
# (occupancy metrics, stall-reason metrics) -- never mixed, per this
# project's established methodology note (STALL_REASONS.md).
#
# 81 configs x 2 passes x 5 reps = 810 ncu invocations for this one variant.

echo "[job] started $(date)  job_id=$SLURM_JOB_ID  node=$SLURMD_NODENAME  variant=baseline"

cd "$SLURM_SUBMIT_DIR"
echo "[job] source checkout: $(pwd)  HEAD=$(git rev-parse HEAD)  branch=$(git branch --show-current)"

source ~/.bashrc
conda activate fem

set -euo pipefail

export CUDA_HOME="$CONDA_PREFIX"
export CC=$(which x86_64-conda-linux-gnu-cc)
export CXX=$(which x86_64-conda-linux-gnu-c++)
export CUDAHOSTCXX=$CXX
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

# The `fem` conda env is SHARED across all clone directories (SPMM,
# SPMM-baseline-cache-cmp, SPMM-cache-block-data). `pip install -e` for a
# given package name registers into the SAME site-packages
# __editable__.torch_fast_spdmm-*.pth regardless of which source directory
# it's building from -- concurrent installs from different clones race over
# that ONE shared file (uninstall-old / install-new can interleave across
# processes), which crashed a sibling job outright once already and could
# otherwise silently leave the WRONG clone's code registered without
# crashing. Serialize the install step specifically (everything else --
# git checkout, the actual profiling loop -- still runs in parallel per job)
# via flock against a lock file shared by ALL clones.
echo "[job] building (flock-serialized against concurrent installs from sibling clones)"
flock /home/dayanb/research/.pip_install.lock \
    pip install -e torch_fast_spdmm --no-build-isolation

# Verify the install actually resolved to THIS job's own source checkout, not
# a sibling job's, before trusting any profiling data from this run. A
# non-crashed pip install is not proof of a correct install -- the race can
# resolve "successfully" while pointing the wrong direction.
EXPECTED_DIR="$SLURM_SUBMIT_DIR"
IMPORTED_FILE=$(python -c "import torch_fast_spdmm; print(torch_fast_spdmm.__file__)")
echo "[job] imported torch_fast_spdmm from: $IMPORTED_FILE"
case "$IMPORTED_FILE" in
    "$EXPECTED_DIR"/*)
        echo "[job] identity check OK: imported module is under this job's own checkout ($EXPECTED_DIR)"
        ;;
    *)
        echo "[job] FATAL: imported torch_fast_spdmm from '$IMPORTED_FILE', which is NOT under this job's own checkout '$EXPECTED_DIR' -- install race contamination, aborting rather than profile the wrong kernel."
        exit 1
        ;;
esac

cd tests/torch_fast_spdmm
OUT_OCC="$SLURM_SUBMIT_DIR/ncu_out/cache_unified_smem/occupancy"
OUT_STALL="$SLURM_SUBMIT_DIR/ncu_out/cache_unified_smem/stall_reasons"
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
VARIANT="baseline"

profile_one() {
    local pass_name=$1 metrics=$2 out=$3 tag=$4 matrix=$5 block=$6 smem=$7 feat=$8
    for rep in $(seq 0 $((N_REPS - 1))); do
        local rtag="${tag}_rep${rep}"
        echo "[job] profiling $pass_name $rtag"
        ncu --profile-from-start off --metrics "$metrics" --csv --target-processes all \
            python -m benchmark.profile_pbr \
                --size 262144 --features "$feat" \
                --block "$block" --min-nnz 2 --smem-size "$smem" \
                --matrix-profile "$matrix" --kernel block \
            > "$out/${rtag}.csv" 2> "$out/${rtag}.log" \
            || { echo "  FAILED — see $out/${rtag}.log"; tail -5 "$out/${rtag}.log"; }
    done
    echo "[job] done $pass_name $tag"
}

MATRICES="diag_b0.3 scattered_1x8 scattered_8x1"
BLOCKS="2 4 8"
SMEMS="512 1024 2048"
FEATURES="129 255 517"

for matrix in $MATRICES; do
    for block in $BLOCKS; do
        for smem in $SMEMS; do
            for feat in $FEATURES; do
                tag="${VARIANT}_${matrix}_b${block}_s${smem}_f${feat}"
                profile_one occupancy "$METRICS_OCC" "$OUT_OCC" "$tag" "$matrix" "$block" "$smem" "$feat"
                profile_one stall_reasons "$METRICS_STALL" "$OUT_STALL" "$tag" "$matrix" "$block" "$smem" "$feat"
            done
        done
    done
done

echo "[job] all profiles done $(date)"
