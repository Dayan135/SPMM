"""Conservative loop commits and history-preserving workspace rollback."""
from __future__ import annotations

from pathlib import Path

from . import state
from .util import ArgumentError, git, head


def _branch(repo: Path, config: dict) -> str:
    branch = git('branch', '--show-current', cwd=repo).stdout.strip()
    expected = config.get('loop_branch')
    if branch == 'main' or not expected or branch != expected:
        raise ArgumentError('create an opt/loop-YYYYMMDD branch from the tools commit, '
                            'set state.loop_branch and state.verified to that commit, then retry')
    required = ('optimizer/agent_tools/__main__.py',
                'tests/torch_fast_spdmm/benchmark/universal_suite.py',
                'tests/torch_fast_spdmm/benchmark/agent_bench.py',
                'tests/torch_fast_spdmm/benchmark/agent_profile.py')
    existing = set(git('ls-tree', '-r', '--name-only', 'HEAD', cwd=repo).stdout.splitlines())
    if any(path not in existing for path in required):
        raise ArgumentError('loop branch must start from the tools commit containing all three drivers')
    return branch


def _status(repo: Path) -> list[tuple[str, str]]:
    raw = git('status', '--porcelain', '-z', '-uall', cwd=repo).stdout.split('\0')
    result = []
    i = 0
    while i < len(raw) and raw[i]:
        record = raw[i]
        kind, path = record[:2], record[3:]
        if 'R' in kind or 'C' in kind:
            i += 1  # rename/copy has another path entry in porcelain -z
        result.append((kind, path))
        i += 1
    return result


def _allowed(path: str) -> bool:
    if path == 'CHANGELOG.md':
        return True
    if path.startswith('torch_fast_spdmm/'):
        return True
    if path.startswith('tests/torch_fast_spdmm/'):
        parts = Path(path).parts
        return (len(parts) == 3 and path.endswith('.py')) or (
            len(parts) == 4 and parts[2] == 'benchmark' and path.endswith('.py'))
    return False


def commit_kernel_attempt(message: str, *, repo: Path | None = None,
                          config: dict | None = None) -> dict:
    if not isinstance(message, str) or not message.strip():
        raise ArgumentError('message must be nonempty')
    repo, cfg = repo or state.ROOT, config or state.read()
    branch = _branch(repo, cfg)
    changes = _status(repo)
    files = [path for _, path in changes if _allowed(path)]
    ignored = [path for _, path in changes if not _allowed(path)]
    if not files:
        return {'error': 'nothing to commit', 'ignored': ignored}
    git('add', '--', *files, cwd=repo)
    staged = git('diff', '--cached', '--name-only', cwd=repo).stdout.splitlines()
    unwanted = [path for path in staged if not _allowed(path)]
    if unwanted:
        git('restore', '--staged', '--', *unwanted, cwd=repo)
        ignored.extend(unwanted)
    staged = git('diff', '--cached', '--name-only', cwd=repo).stdout.splitlines()
    if not staged:
        return {'error': 'nothing to commit', 'ignored': ignored}
    git('commit', '-m', message, cwd=repo)
    sha = head(repo)
    git('push', 'origin', branch, cwd=repo)
    return {'sha': sha, 'branch': branch, 'files': staged, 'pushed': True,
            'ignored': sorted(set(ignored))}


def rollback_workspace(*, repo: Path | None = None, config: dict | None = None) -> dict:
    repo, cfg = repo or state.ROOT, config or state.read()
    branch = _branch(repo, cfg)
    verified = cfg.get('verified')
    if not verified:
        raise ArgumentError('state.verified is missing; set it to the tools baseline commit')
    verified = git('rev-parse', f'{verified}^{{commit}}', cwd=repo).stdout.strip()
    old_head = head(repo)
    targets = ['torch_fast_spdmm', 'tests/torch_fast_spdmm']
    discarded = [path for _, path in _status(repo)
                 if path.startswith('torch_fast_spdmm/') or path.startswith('tests/torch_fast_spdmm/')]
    git('restore', '--staged', '--worktree', '--', *targets, cwd=repo)
    git('clean', '-fd', '--', 'torch_fast_spdmm', ':(glob)tests/torch_fast_spdmm/*.py', cwd=repo)
    diff = git('diff', '--quiet', verified, 'HEAD', '--', *targets, cwd=repo, check=False)
    revert_sha = None
    if diff.returncode == 1:
        git('restore', '--source', verified, '--staged', '--worktree', '--',
            *targets, ':(exclude)tests/torch_fast_spdmm/benchmark/results', cwd=repo)
        git('commit', '-m', f'revert: restore torch_fast_spdmm to verified {verified[:7]} after failed {old_head[:7]}',
            cwd=repo)
        revert_sha = head(repo)
        git('push', 'origin', branch, cwd=repo)
    return {'reverted_from': old_head, 'restored_to': verified,
            'revert_sha': revert_sha, 'discarded': discarded}
