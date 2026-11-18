"""Locked, atomic local state for independent tool invocations."""
from __future__ import annotations

import contextlib
import fcntl
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STATE = ROOT / '.agent_loop' / 'state.json'
DEFAULT = {
    'remote_host': 'bgu-slurm',
    'remote_repo': '/home/dayanb/research/SPMM',
    'agent_root': '/home/dayanb/research/spmm-agent',
    'env_name': 'fem', 'modules': [], 'preamble': '',
    'exclude_nodes': ['ise-3090-01', 'ise-3090-04'],
    'loop_branch': None, 'verified': None, 'builds': {}, 'tests_passed': {},
}


@contextlib.contextmanager
def locked(path: Path = STATE):
    path.parent.mkdir(parents=True, exist_ok=True)
    with (path.parent / 'state.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        state = {**DEFAULT, **(json.loads(path.read_text()) if path.exists() else {})}
        yield state
        tmp = path.with_suffix('.tmp')
        tmp.write_text(json.dumps(state, sort_keys=True, indent=2) + '\n')
        os.replace(tmp, path)
        fcntl.flock(lock, fcntl.LOCK_UN)


def read(path: Path = STATE) -> dict:
    with locked(path) as state:
        return dict(state)


def update(changes: dict, path: Path = STATE) -> dict:
    with locked(path) as state:
        state.update(changes)
        return dict(state)
