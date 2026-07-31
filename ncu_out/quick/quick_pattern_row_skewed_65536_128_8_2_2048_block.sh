#!/bin/bash
#SBATCH --job-name=quick_pattern_row_skewed_65536_128_8_2_2048_block
#SBATCH --partition=rtx3090
#SBATCH --account=erant
#SBATCH --qos=erant
#SBATCH --gres=gpu:rtx_3090:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=jobs/optimizer/logs_quick_pattern_row_skewed_65536_128_8_2_2048_block/%j.out
#SBATCH --error=jobs/optimizer/logs_quick_pattern_row_skewed_65536_128_8_2_2048_block/%j.err

source ~/.bashrc
conda activate fem
set -euo pipefail
export CUDA_HOME="$CONDA_PREFIX"
export CC=$(which x86_64-conda-linux-gnu-cc)
export CXX=$(which x86_64-conda-linux-gnu-c++)
export CUDAHOSTCXX=$CXX
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

cd /home/dayanb/research/SPMM/tests/torch_fast_spdmm
OUT="/home/dayanb/research/SPMM/jobs/optimizer/logs_quick_pattern_row_skewed_65536_128_8_2_2048_block"
mkdir -p "$OUT"
METRICS="gpu__time_duration.sum,sm__throughput.avg.pct_of_peak_sustained_elapsed,sm__pipe_alu_cycles_active.avg.pct_of_peak_sustained_elapsed,sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_elapsed,sm__warps_active.avg.pct_of_peak_sustained_active,smsp__warp_issue_stalled_barrier_per_warp_active.pct,smsp__warp_issue_stalled_drain_per_warp_active.pct,smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct,smsp__warp_issue_stalled_math_pipe_throttle_per_warp_active.pct,smsp__warp_issue_stalled_mio_throttle_per_warp_active.pct,smsp__warp_issue_stalled_no_instruction_per_warp_active.pct,smsp__warp_issue_stalled_not_selected_per_warp_active.pct,smsp__warp_issue_stalled_selected_per_warp_active.pct,smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct,smsp__warp_issue_stalled_wait_per_warp_active.pct"

echo "[job] profiling pattern_row_skewed_65536_128_8_2_2048_block"
ncu --profile-from-start off --metrics "$METRICS" --csv --target-processes all \
    python -m benchmark.profile_pbr --size 65536 --features 128 --block 8 --min-nnz 2 --smem-size 2048 --matrix-profile pattern_row_skewed --kernel block \
    > "$OUT/pattern_row_skewed_65536_128_8_2_2048_block.csv" 2> "$OUT/pattern_row_skewed_65536_128_8_2_2048_block.log" || { echo "  FAILED pattern_row_skewed_65536_128_8_2_2048_block"; tail -5 "$OUT/pattern_row_skewed_65536_128_8_2_2048_block.log"; }
echo "[job] done pattern_row_skewed_65536_128_8_2_2048_block"

echo "[job] batch complete"