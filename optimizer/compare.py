"""Compare a kernel-edit candidate's ledger against a baseline ledger.

Formalizes the before/after comparison this project has done by hand for
every kernel-edit experiment so far (see STALL_REASONS.md's four attempts):
join on matching config_key, report the wall-clock delta AND the stall-
reason-metric deltas side by side, so a "the target stall reason went down"
result doesn't get reported as a win without also showing whether some
OTHER stall reason absorbed the gain (the repeated whack-a-mole pattern
found in every prior attempt).

CLI: python -m optimizer.compare --baseline optimizer_ledger.sqlite3 \\
         --candidate optimizer_ledger_mycandidate.sqlite3
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Optional

from optimizer.ledger import Ledger

# The metrics worth surfacing in every comparison -- mirrors the table shape
# used throughout STALL_REASONS.md.
KEY_METRICS = [
    "gpu_time_ns",
    "sm_throughput_pct",
    "mio_throttle_pct",
    "barrier_pct",
    "long_scoreboard_pct",
    "short_scoreboard_pct",
    "math_pipe_throttle_pct",
]


@dataclass
class ComparisonRow:
    config_key: str
    matrix_profile: str
    size: int
    features: int
    block: int
    smem_size: int
    baseline_metrics: dict
    candidate_metrics: dict

    def delta(self, metric: str) -> Optional[float]:
        b = self.baseline_metrics.get(metric)
        c = self.candidate_metrics.get(metric)
        if b is None or c is None:
            return None
        return c - b

    def delta_pct(self, metric: str) -> Optional[float]:
        b = self.baseline_metrics.get(metric)
        d = self.delta(metric)
        if b is None or d is None or b == 0:
            return None
        return 100.0 * d / b

    def verdict(self) -> str:
        """Naive, generic verdict based on wall-clock time alone -- does not
        know the hypothesis's specific target metric, so this is a starting
        point for a human/agent to read the full metric-delta table, not a
        substitute for the Critic-style mechanism check."""
        dpct = self.delta_pct("gpu_time_ns")
        if dpct is None:
            return "unknown"
        if dpct <= -3.0:
            return "WIN"
        if dpct >= 3.0:
            return "REGRESSION"
        return "WASH"


def compare_ledgers(baseline_db: str, candidate_db: str) -> list[ComparisonRow]:
    baseline = {r["config_key"]: r for r in Ledger(baseline_db).all_records()
                if r["verdict"] == "measured"}
    candidate_records = [r for r in Ledger(candidate_db).all_records() if r["verdict"] == "measured"]

    rows = []
    for cand in candidate_records:
        base = baseline.get(cand["config_key"])
        if base is None:
            continue  # no matching baseline point to compare against
        rows.append(ComparisonRow(
            config_key=cand["config_key"],
            matrix_profile=cand["matrix_profile"], size=cand["size"], features=cand["features"],
            block=cand["block"], smem_size=cand["smem_size"],
            baseline_metrics=base.get("metrics") or {},
            candidate_metrics=cand.get("metrics") or {},
        ))
    return rows


def print_comparison(rows: list[ComparisonRow]) -> None:
    if not rows:
        print("No matching config points found between the two ledgers -- nothing to compare.")
        print("(Check both ledgers were populated by sweeping the SAME matrix/size/features grid.)")
        return

    for row in sorted(rows, key=lambda r: (r.matrix_profile, r.size, r.features)):
        print(f"\n=== {row.matrix_profile} / {row.size} / f{row.features} "
              f"(block={row.block}, smem={row.smem_size}) -- {row.verdict()} ===")
        for metric in KEY_METRICS:
            b = row.baseline_metrics.get(metric)
            c = row.candidate_metrics.get(metric)
            dpct = row.delta_pct(metric)
            if b is None or c is None:
                continue
            arrow = "->" if dpct is None else (f"({dpct:+.1f}%)")
            print(f"  {metric:24s} {b:>14.2f} {arrow:>10s} {c:>14.2f}")

    n_win = sum(1 for r in rows if r.verdict() == "WIN")
    n_regression = sum(1 for r in rows if r.verdict() == "REGRESSION")
    n_wash = sum(1 for r in rows if r.verdict() == "WASH")
    print(f"\n{len(rows)} matched config points: {n_win} WIN, {n_regression} REGRESSION, {n_wash} WASH")
    if n_win and (n_regression or n_wash):
        print("Mixed result -- report per-matrix, not as a single aggregate verdict "
              "(matches this project's established finding that kernel-edit effects are "
              "matrix-shape-dependent).")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline", required=True, help="path to the baseline ledger sqlite file")
    p.add_argument("--candidate", required=True, help="path to the candidate (kernel-edit) ledger sqlite file")
    args = p.parse_args()

    rows = compare_ledgers(args.baseline, args.candidate)
    print_comparison(rows)


if __name__ == "__main__":
    main()
