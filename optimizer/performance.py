"""Performance Evaluator (MULTIAGENT_PBR_OPTIMIZATION.md §2.8).

Wraps the exact ncu-wrapped profile_pbr.py invocation validated throughout
this project's stall-reason investigation (jobs/profile_stall_reasons/
profile_after_20260714.sh), generalized into a reusable batch-job renderer
and a CSV parser (formalizing the ad hoc parse_stall_reasons.py script
written by hand during that investigation).
"""

from __future__ import annotations

import csv as csv_module
import io
import re
from pathlib import Path
from typing import Optional

from optimizer import remote
from optimizer.grid import CandidateConfig

# Exact metric list validated in STALL_REASONS.md / jobs/profile_stall_reasons/
# profile_after_20260714.sh. Keep this list -- and only this list -- in any one
# ncu invocation: mixing in unrelated counters (e.g. sm__sass_inst_executed_
# op_shared.sum) was found to change ncu's replay-pass structure and corrupt
# these specific stall-reason numbers (see STALL_REASONS.md's
# "Methodology note" under the adaptive-bypass findings). If you need extra
# counters, collect them in a *separate* ncu invocation, never appended here.
METRICS = [
    "gpu__time_duration.sum",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "sm__pipe_alu_cycles_active.avg.pct_of_peak_sustained_elapsed",
    "sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_elapsed",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "smsp__warp_issue_stalled_barrier_per_warp_active.pct",
    "smsp__warp_issue_stalled_drain_per_warp_active.pct",
    "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct",
    "smsp__warp_issue_stalled_math_pipe_throttle_per_warp_active.pct",
    "smsp__warp_issue_stalled_mio_throttle_per_warp_active.pct",
    "smsp__warp_issue_stalled_no_instruction_per_warp_active.pct",
    "smsp__warp_issue_stalled_not_selected_per_warp_active.pct",
    "smsp__warp_issue_stalled_selected_per_warp_active.pct",
    "smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct",
    "smsp__warp_issue_stalled_wait_per_warp_active.pct",
]

# Metric names as they'll come back keyed in the parsed dict -- stripped of the
# smsp__/sm__ prefixing noise for readability in the ledger, but the raw ncu
# name is preserved as the source of truth in a comment alongside each use.
_METRIC_SHORT_NAMES = {
    "gpu__time_duration.sum": "gpu_time_ns",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed": "sm_throughput_pct",
    "sm__pipe_alu_cycles_active.avg.pct_of_peak_sustained_elapsed": "alu_pipe_pct",
    "sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_elapsed": "fma_pipe_pct",
    "sm__warps_active.avg.pct_of_peak_sustained_active": "warps_active_pct",
    "smsp__warp_issue_stalled_barrier_per_warp_active.pct": "barrier_pct",
    "smsp__warp_issue_stalled_drain_per_warp_active.pct": "drain_pct",
    "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct": "long_scoreboard_pct",
    "smsp__warp_issue_stalled_math_pipe_throttle_per_warp_active.pct": "math_pipe_throttle_pct",
    "smsp__warp_issue_stalled_mio_throttle_per_warp_active.pct": "mio_throttle_pct",
    "smsp__warp_issue_stalled_no_instruction_per_warp_active.pct": "no_instruction_pct",
    "smsp__warp_issue_stalled_not_selected_per_warp_active.pct": "not_selected_pct",
    "smsp__warp_issue_stalled_selected_per_warp_active.pct": "selected_pct",
    "smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct": "short_scoreboard_pct",
    "smsp__warp_issue_stalled_wait_per_warp_active.pct": "wait_pct",
}

def parse_ncu_csv(csv_text: str, kernel_name_prefix: str = "pbr_spmm") -> Optional[dict]:
    """Parse one ncu --csv capture, returning the metrics for the FIRST kernel
    launch whose name contains kernel_name_prefix (generalizes the regex-based
    parsing done ad hoc in this session's scratch scripts into a real csv.reader-
    based parser, which is more robust to embedded commas in kernel signatures)."""
    reader = csv_module.reader(io.StringIO(csv_text))
    rows = list(reader)
    if not rows:
        return None

    header = None
    for row in rows:
        if row and row[0] == "ID":
            header = row
            break
    if header is None:
        return None

    name_idx = header.index("Kernel Name")
    metric_idx = header.index("Metric Name")
    value_idx = header.index("Metric Value")

    values: dict[str, float] = {}
    for row in rows:
        if len(row) != len(header):
            continue
        if row[0] == "ID":
            continue
        if kernel_name_prefix not in row[name_idx]:
            continue
        metric = row[metric_idx]
        short = _METRIC_SHORT_NAMES.get(metric, metric)
        try:
            values[short] = float(row[value_idx].replace(",", ""))
        except ValueError:
            continue
    return values or None


def render_batch_script(configs: list[CandidateConfig], out_dir: str) -> str:
    """One sbatch script body looping over a batch of config points -- matches
    the existing jobs/profile_stall_reasons/*.sh convention of many ncu
    invocations per job rather than one job per point (Slurm allocation
    overhead makes the latter wasteful even when GPU time itself is cheap)."""
    metrics_arg = ",".join(METRICS)
    lines = [
        remote.BUILD_ENV_PREAMBLE,
        f'cd {remote.REMOTE_REPO_DIR}/tests/torch_fast_spdmm',
        f'OUT="{remote.REMOTE_REPO_DIR}/{out_dir}"',
        'mkdir -p "$OUT"',
        f'METRICS="{metrics_arg}"',
        "",
    ]
    for cfg in configs:
        tag = cfg.config_key().replace("|", "_").replace(" ", "")
        cli = " ".join(cfg.cli_args())
        lines.append(
            f'echo "[job] profiling {tag}"\n'
            f'ncu --profile-from-start off --metrics "$METRICS" --csv --target-processes all \\\n'
            f'    python -m benchmark.profile_pbr {cli} \\\n'
            f'    > "$OUT/{tag}.csv" 2> "$OUT/{tag}.log" '
            f'|| {{ echo "  FAILED {tag}"; tail -5 "$OUT/{tag}.log"; }}\n'
            f'echo "[job] done {tag}"\n'
        )
    lines.append('echo "[job] batch complete"')
    return "\n".join(lines)


def run_performance_batch(configs: list[CandidateConfig], *, job_name: str,
                           local_out_dir: Path, time_limit: str = "2:00:00") -> dict[str, Optional[dict]]:
    """Submit, wait for, and parse a batch of performance-evaluation runs.
    Returns {config_key: metrics_dict_or_None}."""
    out_dir_rel = f"jobs/optimizer/logs_{job_name}"
    script_body = render_batch_script(configs, out_dir_rel)
    job_id = remote.render_and_submit(
        script_body, job_name=job_name, out_dir=out_dir_rel,
        mem="32G", time_limit=time_limit,
    )
    remote.wait_for_job(job_id, poll_interval_s=20, timeout_s=int(_time_limit_to_seconds(time_limit)))

    remote.pull_dir(f"{remote.REMOTE_REPO_DIR}/{out_dir_rel}", local_out_dir)

    results: dict[str, Optional[dict]] = {}
    for cfg in configs:
        tag = cfg.config_key().replace("|", "_").replace(" ", "")
        csv_path = local_out_dir / f"{tag}.csv"
        if not csv_path.exists():
            results[cfg.config_key()] = None
            continue
        results[cfg.config_key()] = parse_ncu_csv(csv_path.read_text(), "pbr_spmm")
    return results


def _time_limit_to_seconds(time_limit: str) -> float:
    """'2:00:00' -> 7200.0; also accepts 'HH:MM:SS' or 'MM:SS'."""
    parts = [int(p) for p in time_limit.split(":")]
    while len(parts) < 3:
        parts.insert(0, 0)
    h, m, s = parts
    return h * 3600 + m * 60 + s
