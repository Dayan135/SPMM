"""Single-config fast path for live kernel-edit iteration.

Wraps the same remote build + ncu-profiled performance run the full sweep
(optimizer.run) uses, but for exactly one CandidateConfig, synchronously, so
a single kernel-source edit can be tried and measured in one command instead
of waiting on a full grid sweep. Skips the full-suite correctness gate by
default (this path is for "how does it feel", not final validation) -- pass
--verify to run it first.

Usage (edit kernel source, commit, push, then):
    python -m optimizer.quick <git-ref> --matrix diag_b0.3 --size 262144 \\
        --features 512 --block 8 --min-nnz 2 --smem-size 2048

    python -m optimizer.quick <git-ref> --matrix diag_skewed --size 262144 \\
        --features 512 --verify --db optimizer_ledger_mybranch.sqlite3
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from optimizer import remote
from optimizer.correctness import verify_correctness
from optimizer.grid import CandidateConfig
from optimizer.ledger import Ledger
from optimizer.performance import run_performance_batch


def run_quick(git_ref: str, config: CandidateConfig, *, verify: bool = False,
              ledger: Optional[Ledger] = None) -> Optional[dict]:
    correctness_pass = None

    if verify:
        result = verify_correctness(git_ref)
        git_commit = result.git_commit
        correctness_pass = result.passed
        print(f"[correctness] {git_commit}: {'PASS' if result.passed else 'FAIL'} -- {result.summary}")
        if not result.passed:
            print("Correctness gate failed -- skipping performance run.")
            return None
    else:
        git_commit = remote.sync_and_build(git_ref=git_ref)
        print(f"[build] {git_commit}")

    print(f"[profiling] {config.config_key()}")
    metrics = run_performance_batch(
        [config],
        job_name=f"quick_{config.config_key().replace('|', '_').replace(' ', '')}",
        local_out_dir=Path("ncu_out") / "quick",
    ).get(config.config_key())

    if ledger is not None:
        verdict = "measured" if metrics else "failed_no_data"
        ledger.record(config, git_commit=git_commit, correctness_pass=correctness_pass,
                       metrics=metrics, verdict=verdict)

    if metrics is None:
        print("No metrics parsed -- check ncu_out/quick/*.log for the failure.")
        return None

    print(f"\n[{config.matrix_profile} / {config.size} / f{config.features} "
          f"(block={config.block}, min_nnz={config.min_nnz}, smem={config.smem_size})]")
    for key, value in metrics.items():
        print(f"  {key:24s} {value:>14.3f}")
    return metrics


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("git_ref", help="branch/commit to build and profile (must already be pushed to origin)")
    p.add_argument("--matrix", required=True, dest="matrix_profile")
    p.add_argument("--size", type=int, required=True)
    p.add_argument("--features", type=int, required=True)
    p.add_argument("--block", type=int, default=8)
    p.add_argument("--min-nnz", type=int, default=2)
    p.add_argument("--smem-size", type=int, default=2048)
    p.add_argument("--verify", action="store_true",
                   help="run the full correctness suite before profiling (slower; off by default)")
    p.add_argument("--db", default=None, help="ledger sqlite path to record this run into (optional)")
    args = p.parse_args()

    config = CandidateConfig(
        matrix_profile=args.matrix_profile, size=args.size, features=args.features,
        block=args.block, min_nnz=args.min_nnz, smem_size=args.smem_size,
    )
    ledger = Ledger(args.db) if args.db else None
    run_quick(args.git_ref, config, verify=args.verify, ledger=ledger)


if __name__ == "__main__":
    main()
