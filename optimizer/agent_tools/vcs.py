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


EXCLUDED_TEST_FILES = frozenset({'conftest.py', 'test_utils.py'})


def is_attempt_code_path(path: str) -> bool:
    """Paths an optimization attempt may edit: torch_fast_spdmm/** and top-level test_*.py."""
    if path.startswith('torch_fast_spdmm/'):
        return True
    parts = Path(path).parts
    return (
        len(parts) == 3
        and parts[0] == 'tests'
        and parts[1] == 'torch_fast_spdmm'
        and parts[2].startswith('test_')
        and parts[2].endswith('.py')
        and parts[2] not in EXCLUDED_TEST_FILES
    )


def _allowed(path: str) -> bool:
    return path == 'CHANGELOG.md' or is_attempt_code_path(path)


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
    changes = _status(repo)
    attempt_changes = [(kind, path) for kind, path in changes if is_attempt_code_path(path)]
    discarded = [path for _, path in attempt_changes]
    tracked_dirty = [path for kind, path in attempt_changes if kind != '??']
    untracked_dirty = [path for kind, path in attempt_changes if kind == '??']
    if tracked_dirty:
        git('restore', '--staged', '--worktree', '--', *tracked_dirty, cwd=repo)
    if (repo / 'torch_fast_spdmm').exists():
        git('clean', '-fd', '--', 'torch_fast_spdmm', cwd=repo)
    for path in untracked_dirty:
        if not path.startswith('torch_fast_spdmm/'):
            target = repo / path
            if target.exists():
                target.unlink()

    diff_res = git('diff', '--name-only', verified, 'HEAD', '--',
                   'torch_fast_spdmm', 'tests/torch_fast_spdmm', cwd=repo, check=False)
    if diff_res.returncode:
        raise ArgumentError(f'could not diff HEAD against verified {verified[:7]}: {diff_res.stderr.strip()}')
    changed_committed = [p for p in diff_res.stdout.splitlines() if is_attempt_code_path(p)]
    revert_sha = None
    if changed_committed:
        verified_tree = set(
            git('ls-tree', '-r', '--name-only', verified, '--',
                'torch_fast_spdmm', 'tests/torch_fast_spdmm', cwd=repo).stdout.splitlines()
        )
        to_restore = [p for p in changed_committed if p in verified_tree]
        to_remove = [p for p in changed_committed if p not in verified_tree]
        if to_restore:
            git('restore', '--source', verified, '--staged', '--worktree', '--', *to_restore, cwd=repo)
        if to_remove:
            git('rm', '-f', '--', *to_remove, cwd=repo)
        # Commit ONLY the rollback paths (--only): any unrelated entries already staged in the
        # index stay staged and are neither committed nor pushed.
        git('commit', '-m', f'revert: restore torch_fast_spdmm to verified {verified[:7]} after failed {old_head[:7]}',
            '--only', '--', *changed_committed, cwd=repo)
        revert_sha = head(repo)
        git('push', 'origin', branch, cwd=repo)
    return {'reverted_from': old_head, 'restored_to': verified,
            'revert_sha': revert_sha, 'discarded': discarded}


def _status_with_origins(repo: Path) -> list[tuple[str, str, str | None]]:
    """Porcelain -z status as (kind, path, original path for renames/copies)."""
    raw = git('status', '--porcelain', '-z', '-uall', cwd=repo).stdout.split('\0')
    result = []
    i = 0
    while i < len(raw) and raw[i]:
        record = raw[i]
        kind, path = record[:2], record[3:]
        orig = None
        if 'R' in kind or 'C' in kind:
            i += 1
            orig = raw[i] if i < len(raw) else None
        result.append((kind, path, orig))
        i += 1
    return result


def clean_attempt_workspace(*, repo: Path | None = None, config: dict | None = None) -> dict:
    """Discard UNCOMMITTED changes to attempt source/test paths when no attempt is active.

    Invariant: never commits, never pushes, never touches a path outside torch_fast_spdmm/** and
    top-level tests/torch_fast_spdmm/test_*.py (excluding conftest.py/test_utils.py). Refuses if
    HEAD already carries committed attempt-path changes relative to state.verified, because
    undoing committed work outside an attempt needs a user decision. (The CLI refuses this tool
    while an attempt is active; use rollback_workspace then.)
    """
    repo, cfg = repo or state.ROOT, config or state.read()
    _branch(repo, cfg)
    verified = cfg.get('verified')
    if not verified:
        raise ArgumentError('state.verified is missing; set it to the tools baseline commit')
    verified = git('rev-parse', f'{verified}^{{commit}}', cwd=repo).stdout.strip()
    drift_res = git('diff', '--name-only', verified, 'HEAD', '--',
                    'torch_fast_spdmm', 'tests/torch_fast_spdmm', cwd=repo, check=False)
    if drift_res.returncode:
        raise ArgumentError(f'could not diff HEAD against verified {verified[:7]}: {drift_res.stderr.strip()}')
    drifted = sorted(p for p in drift_res.stdout.splitlines() if is_attempt_code_path(p))
    if drifted:
        raise ArgumentError(
            f'HEAD has committed attempt-path changes relative to verified {verified[:7]}: {drifted}; '
            'clean_attempt_workspace never commits or reverts history - this needs user intervention'
        )
    tracked, untracked = set(), set()
    for kind, path, orig in _status_with_origins(repo):
        for candidate in (path, orig):
            if candidate and is_attempt_code_path(candidate):
                (untracked if kind == '??' else tracked).add(candidate)
    if tracked:
        git('restore', '--source=HEAD', '--staged', '--worktree', '--', *sorted(tracked), cwd=repo)
    for rel in sorted(untracked):
        target = repo / rel
        if target.is_file() or target.is_symlink():
            target.unlink()
    return {'discarded': sorted(tracked | untracked), 'head': head(repo),
            'committed': False, 'pushed': False}
