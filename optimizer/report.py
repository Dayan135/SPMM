"""CLI entry point: python -m optimizer.report

Prints a best-config-per-(matrix,size,features) summary from the ledger --
the automated equivalent of eyeballing plot_benchmarks.py output.
"""

from __future__ import annotations

import argparse

from optimizer.ledger import Ledger


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--db", default=None, help="path to the ledger sqlite file (default: repo-root optimizer_ledger.sqlite3)")
    p.add_argument("--metric", default="gpu_time_ns", help="metric to rank by (default: gpu_time_ns, lower is better)")
    args = p.parse_args()

    ledger = Ledger(args.db) if args.db else Ledger()
    best = ledger.best_per_matrix(metric=args.metric)

    if not best:
        print("No measured records in the ledger yet.")
        return

    print(f"{'matrix/size/features':35s} {'kernel':10s} {'block':>6s} {'smem':>6s}  {args.metric}")
    for key in sorted(best):
        rec = best[key]
        value = rec["metrics"].get(args.metric)
        print(f"{key:35s} {rec['kernel']:10s} {rec['block']:>6d} {rec['smem_size']:>6d}  {value}")

    all_records = ledger.all_records()
    n_measured = sum(1 for r in all_records if r["verdict"] == "measured")
    n_failed = sum(1 for r in all_records if r["verdict"] != "measured")
    print(f"\n{n_measured} measured, {n_failed} failed/no-data, {len(all_records)} total records in ledger.")


if __name__ == "__main__":
    main()
