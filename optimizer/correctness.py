"""Correctness Evaluator (tier-1, MULTIAGENT_PBR_OPTIMIZATION.md §2.7).

Phase 0 has no LLM-generated kernel code, so this is a single hard gate run
once per commit under test (not once per config point -- config choice
doesn't change which code path is exercised in a way that needs re-verifying
per grid cell, only the kernel *code itself* would). Every benchmark run in
a session must be preceded by one passing CorrectnessResult.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from optimizer import remote


@dataclass
class CorrectnessResult:
    git_commit: str
    passed: bool
    summary: str  # e.g. "3704 passed, 1 warning in 11.56s"
    raw_output: str


_SUMMARY_RE = re.compile(r"^(\d+ passed.*)$", re.MULTILINE)
_FAILED_RE = re.compile(r"^(\d+ failed.*)$", re.MULTILINE)


def verify_correctness(git_ref: str) -> CorrectnessResult:
    """Build the given git ref on the remote GPU node and run the full
    correctness suite. Hard gate: callers must not trust any benchmark number
    collected against a commit whose CorrectnessResult.passed is False."""
    commit = remote.sync_and_build(git_ref=git_ref)
    result = remote.run_pytest_suite()

    failed_match = _FAILED_RE.search(result.stdout)
    passed_match = _SUMMARY_RE.search(result.stdout)

    if failed_match:
        return CorrectnessResult(commit, False, failed_match.group(1), result.stdout)
    if result.returncode == 0 and passed_match:
        return CorrectnessResult(commit, True, passed_match.group(1), result.stdout)
    return CorrectnessResult(
        commit, False,
        f"unrecognized pytest output (exit {result.returncode})",
        result.stdout + "\n" + result.stderr,
    )
