"""Phase 0 sweep-runner for the PBR parameter grid.

See MULTIAGENT_PBR_OPTIMIZATION.md for the full multi-phase design this
implements the first (no-LLM) phase of. This package automates what a human
currently does by hand: enumerate PBR kernel configs, submit ncu-profiled
benchmark jobs on the remote Slurm GPU cluster, parse the results, and record
them in a queryable SQLite ledger.
"""
