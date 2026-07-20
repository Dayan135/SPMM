"""CLI entry point: python -m optimizer.run --matrices ... --sizes ... --features ... --kernels ...

Example (the STALL_REASONS.md reference slice, block kernel only):
    python -m optimizer.run --matrices diag_b0.1 diag_b0.3 scattered_1x8 scattered_8x1 scattered_2x2 \\
        --sizes 1024 262144 --features 128 512 --kernels block
"""

from __future__ import annotations

import argparse

from optimizer.ledger import Ledger
from optimizer.orchestrator import run_sweep


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--matrices", nargs="+", default=None, help="matrix profile names (default: all)")
    p.add_argument("--sizes", nargs="+", type=int, default=None, help="system sizes (default: all)")
    p.add_argument("--features", nargs="+", type=int, default=None, help="feature counts (default: [128, 512])")
    p.add_argument("--kernels", nargs="+", default=["block"], help="kernel variants to sweep (default: block)")
    p.add_argument("--git-ref", default="research/pbr-cusparse-scalability-gap",
                   help="branch/commit to build+verify before benchmarking")
    p.add_argument("--db", default=None,
                   help="ledger sqlite path (default: repo-root optimizer_ledger.sqlite3). "
                        "IMPORTANT: config_key does not include the git commit/branch being measured, "
                        "so sweeping a kernel-edit branch against the SAME ledger as the baseline would "
                        "silently skip points already recorded there. Point kernel-edit experiments at "
                        "their own --db (e.g. optimizer_ledger_<branch-suffix>.sqlite3) and compare with "
                        "optimizer.compare instead.")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-concurrent-jobs", type=int, default=3)
    p.add_argument("--skip-correctness", action="store_true",
                   help="skip the correctness gate (only for repeat runs against an already-verified build)")
    p.add_argument("--quiet", action="store_true", help="suppress progress logging")
    args = p.parse_args()

    ledger = Ledger(args.db) if args.db else Ledger()

    summary = run_sweep(
        matrices=args.matrices, sizes=args.sizes, features=args.features,
        kernels=tuple(args.kernels), git_ref=args.git_ref, ledger=ledger,
        batch_size=args.batch_size, max_concurrent_jobs=args.max_concurrent_jobs,
        skip_correctness=args.skip_correctness, verbose=not args.quiet,
    )

    print(f"Submitted: {summary['submitted']} new config points "
          f"(skipped {summary['skipped_duplicate']} already in the ledger)")
    if summary["correctness"]:
        c = summary["correctness"]
        print(f"Correctness gate ({c.git_commit}): {'PASS' if c.passed else 'FAIL'} -- {c.summary}")
    if summary["failed_batches"]:
        print(f"WARNING: {len(summary['failed_batches'])} batch(es) failed to process:")
        for f in summary["failed_batches"]:
            print(f"  job {f['job_id']}: {f['error']}")


if __name__ == "__main__":
    main()
