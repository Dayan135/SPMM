"""Phase-0 driver loop (MULTIAGENT_PBR_OPTIMIZATION.md §1, §3) -- plain Python,
not LangGraph (see the design doc's Phase 0 rationale: no LLM nodes, no
branching, so a graph framework would be unneeded complexity here). Ties
together grid enumeration, the ledger's dedup check, the correctness gate,
and batched/concurrent performance evaluation.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

from optimizer import performance, remote
from optimizer.correctness import verify_correctness
from optimizer.grid import CandidateConfig, enumerate_grid
from optimizer.ledger import Ledger

DEFAULT_BATCH_SIZE = 16
DEFAULT_MAX_CONCURRENT_JOBS = 3


def _batched(items: list, size: int) -> list[list]:
    return [items[i:i + size] for i in range(0, len(items), size)]


def run_sweep(
    *,
    matrices: Optional[list[str]] = None,
    sizes: Optional[list[int]] = None,
    features: Optional[list[int]] = None,
    kernels: tuple[str, ...] = ("block",),
    git_ref: str = "feat/pbr-optimizer-phase0",
    ledger: Optional[Ledger] = None,
    local_out_dir: Path = Path("ncu_out/optimizer"),
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_concurrent_jobs: int = DEFAULT_MAX_CONCURRENT_JOBS,
    skip_correctness: bool = False,
    verbose: bool = True,
) -> dict:
    """Run one sweep slice end to end: enumerate -> dedup -> correctness gate
    (once) -> batched/concurrent performance evaluation -> record.

    With verbose=True (the default), prints progress as each phase starts/ends
    and as each batch is submitted/completes -- otherwise this runs silently
    until the whole sweep finishes, which for a multi-hour run leaves a caller
    with no visibility into whether anything is actually happening.

    Returns a summary dict: {"submitted": N, "skipped_duplicate": N,
    "correctness": CorrectnessResult|None, "failed_batches": [...]}.
    """
    def _log(msg: str) -> None:
        if verbose:
            print(f"[optimizer] {time.strftime('%H:%M:%S')} {msg}", flush=True)

    ledger = ledger or Ledger()

    all_configs = enumerate_grid(matrices=matrices, sizes=sizes, features=features,
                                  kernels=kernels)
    unmeasured = [c for c in all_configs if not ledger.has_measured(c)]
    skipped = len(all_configs) - len(unmeasured)
    _log(f"enumerated {len(all_configs)} candidates -- {skipped} already in the ledger, "
         f"{len(unmeasured)} to run")

    if not unmeasured:
        return {"submitted": 0, "skipped_duplicate": skipped, "correctness": None, "failed_batches": []}

    correctness_result = None
    if not skip_correctness:
        _log(f"running correctness gate on {git_ref} (build + full pytest suite)...")
        correctness_result = verify_correctness(git_ref)
        if not correctness_result.passed:
            _log(f"correctness gate FAILED ({correctness_result.git_commit}): {correctness_result.summary}")
            raise RuntimeError(
                f"Correctness gate failed on {git_ref} ({correctness_result.git_commit}): "
                f"{correctness_result.summary} -- refusing to trust any benchmark numbers "
                f"from this build. See raw_output for detail."
            )
        _log(f"correctness gate PASSED ({correctness_result.git_commit}): {correctness_result.summary}")

    batches = _batched(unmeasured, batch_size)
    n_batches = len(batches)
    in_flight: dict[str, tuple] = {}  # job_id -> (batch, local_dir)
    failed_batches = []
    batch_queue = list(enumerate(batches))
    n_submitted = 0

    def _submit_next():
        nonlocal n_submitted
        if not batch_queue:
            return
        idx, batch = batch_queue.pop(0)
        job_name = f"sweep_{int(time.time())}_{idx}"
        local_dir = local_out_dir / job_name
        out_dir_rel = f"jobs/optimizer/logs_{job_name}"
        script_body = performance.render_batch_script(batch, out_dir_rel)
        job_id = remote.render_and_submit(script_body, job_name=job_name, out_dir=out_dir_rel,
                                           mem="32G", time_limit="2:00:00")
        n_submitted += 1
        _log(f"submitted batch {n_submitted}/{n_batches} -> Slurm job {job_id} ({len(batch)} configs)")
        in_flight[job_id] = (batch, local_dir, out_dir_rel)

    for _ in range(min(max_concurrent_jobs, len(batches))):
        _submit_next()

    n_completed_batches = 0
    while in_flight:
        for job_id in list(in_flight.keys()):
            if not remote.job_is_finished(job_id):
                continue
            batch, local_dir, out_dir_rel = in_flight.pop(job_id)
            n_completed_batches += 1
            try:
                remote.pull_dir(f"{remote.REMOTE_REPO_DIR}/{out_dir_rel}", local_dir)
                n_measured = n_failed = 0
                for cfg in batch:
                    tag = cfg.config_key().replace("|", "_").replace(" ", "")
                    csv_path = local_dir / f"{tag}.csv"
                    metrics = None
                    if csv_path.exists():
                        metrics = performance.parse_ncu_csv(csv_path.read_text(), "pbr_spmm")
                    verdict = "measured" if metrics else "failed_no_data"
                    n_measured += verdict == "measured"
                    n_failed += verdict != "measured"
                    ledger.record(
                        cfg,
                        git_commit=correctness_result.git_commit if correctness_result else None,
                        correctness_pass=correctness_result.passed if correctness_result else None,
                        metrics=metrics, verdict=verdict,
                    )
                _log(f"batch job {job_id} done ({n_completed_batches}/{n_batches} batches complete): "
                     f"{n_measured} measured, {n_failed} failed/no-data "
                     f"-- {len(in_flight) - 1 + len(batch_queue)} batches remaining")
            except Exception as exc:  # noqa: BLE001 -- record and continue, don't abort the whole sweep
                _log(f"batch job {job_id} FAILED to process: {exc}")
                failed_batches.append({"job_id": job_id, "error": str(exc)})
            _submit_next()
        if in_flight:
            time.sleep(15)

    _log(f"sweep complete: {len(unmeasured)} submitted, {skipped} skipped (duplicate), "
         f"{len(failed_batches)} batch(es) failed to process")

    return {
        "submitted": len(unmeasured),
        "skipped_duplicate": skipped,
        "correctness": correctness_result,
        "failed_batches": failed_batches,
    }
