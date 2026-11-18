"""Local Git and retrying remote transport. Never imports CUDA libraries."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import time

from optimizer import remote
from . import state


class ArgumentError(ValueError):
    pass


class InfrastructureError(RuntimeError):
    pass


def git(*args: str, cwd: Path | None = None, check: bool = True) -> subprocess.CompletedProcess:
    result = subprocess.run(['git', *map(str, args)], cwd=cwd or state.ROOT,
                            text=True, capture_output=True)
    if check and result.returncode:
        raise ArgumentError(result.stderr.strip() or result.stdout.strip())
    return result


def head(cwd: Path | None = None) -> str:
    return git('rev-parse', 'HEAD', cwd=cwd).stdout.strip()


def ssh(command: str, *, timeout: int = 120, config: dict | None = None):
    cfg = config or state.read()
    host = os.getenv('OPTIMIZER_REMOTE_HOST', cfg['remote_host'])
    for attempt in range(3):
        try:
            result = remote.ssh(command, host=host, timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            result = remote.CommandResult(255, '', str(exc))
        if result.returncode != 255:
            return result
        if attempt < 2:
            time.sleep(2 ** attempt)
    raise InfrastructureError(f'ssh {host} unreachable after 3 attempts: {result.stderr.strip()}')


def checked_ssh(command: str, *, timeout: int = 120, config: dict | None = None):
    result = ssh(command, timeout=timeout, config=config)
    if result.returncode:
        raise InfrastructureError(f'remote command failed ({result.returncode}): {result.stderr.strip() or result.stdout.strip()}')
    return result


def short_tail(value: str, lines: int = 60) -> str:
    return '\n'.join(value.splitlines()[-lines:])


def transfer(kind: str, source, target, *, config: dict | None = None):
    cfg = config or state.read()
    host = os.getenv('OPTIMIZER_REMOTE_HOST', cfg['remote_host'])
    operation = {'push': remote.push_file, 'pull': remote.pull_file,
                 'pull_dir': remote.pull_dir}[kind]
    for attempt in range(3):
        try:
            return operation(source, target, host=host)
        except subprocess.CalledProcessError as exc:
            if exc.returncode != 255:
                raise InfrastructureError(f'{kind} failed: {exc}') from exc
            if attempt < 2:
                time.sleep(2 ** attempt)
    raise InfrastructureError(f'{kind} failed after 3 SSH attempts')
