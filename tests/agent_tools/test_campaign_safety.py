"""Regression tests for the campaign-safety fixes (pre-attempt recovery, campaign budget,
governed CHANGELOG binding, benchmark crash classification, secret redaction, scoped rollback
commits, and audit-finalization recovery). Offline: temp git repos only, no SSH/GPU."""
from __future__ import annotations

import json
import os
from pathlib import Path
import sqlite3
import subprocess

import pytest

from optimizer.agent_tools import __main__ as cli
from optimizer.agent_tools import attempt_log, bench, state
from optimizer.agent_tools.util import InfrastructureError

from test_attempt_log import _commit_changelog, _git, _init_loop_repo, _run_cli

KERNEL = 'torch_fast_spdmm/kernels/boo_kernel.cu'


def _db(env) -> sqlite3.Connection:
    conn = sqlite3.connect(env['db_path'])
    conn.row_factory = sqlite3.Row
    return conn


def _origin_head(env) -> str:
    return _git(env['origin'], 'rev-parse', 'refs/heads/opt/loop')


def _counted_attempt(capsys, repo: Path, title: str, *, new_run: bool = False) -> dict:
    args = {'title': title, 'hypothesis': f'{title} should change something measurable.'}
    if new_run:
        args['new_run'] = True
    code, started = _run_cli(capsys, 'begin_attempt', args)
    assert code == 0, started
    (repo / KERNEL).write_text(f'// {title}\n')
    assert _run_cli(capsys, 'commit_kernel_attempt', {'message': f'kernel: {title}'})[0] == 0
    assert _run_cli(capsys, 'rollback_workspace', {})[0] == 0
    cl = _commit_changelog(repo, title)
    code, fin = _run_cli(capsys, 'finish_attempt', {
        'outcome': 'ROLLED BACK', 'decision_reason': 'flat', 'rolled_back': True, 'changelog_sha': cl,
    })
    assert code == 0, fin
    return fin


# ---------------------------------------------------------------- 1. pre-attempt recovery


def test_clean_attempt_workspace_recovers_without_commit(tmp_path, monkeypatch, capsys):
    env = _init_loop_repo(tmp_path, monkeypatch)
    repo = env['repo']
    head0, origin0 = _git(repo, 'rev-parse', 'HEAD'), _origin_head(env)

    # Dirty attempt paths with NO active attempt, plus unrelated state that must survive.
    (repo / KERNEL).write_text('// stray edit\n')
    (repo / 'tests/torch_fast_spdmm/test_scratch.py').write_text('def test_x(): pass\n')
    (repo / 'tests/torch_fast_spdmm/benchmark/agent_bench.py').write_text('# dirty harness\n')
    (repo / 'notes.txt').write_text('unrelated staged\n')
    _git(repo, 'add', 'notes.txt')
    (repo / 'outside.txt').write_text('unrelated untracked\n')

    code, res = _run_cli(capsys, 'begin_attempt', {'title': 'T', 'hypothesis': 'H.'})
    assert code == 2 and 'clean_attempt_workspace' in res['detail']
    code, res = _run_cli(capsys, 'rollback_workspace', {})
    assert code == 2 and 'requires an active attempt' in res['detail']

    code, res = _run_cli(capsys, 'clean_attempt_workspace', {})
    assert code == 0, res
    assert res['discarded'] == sorted([KERNEL, 'tests/torch_fast_spdmm/test_scratch.py'])
    assert res['committed'] is False and res['pushed'] is False
    # No commit, no push.
    assert _git(repo, 'rev-parse', 'HEAD') == head0 and _origin_head(env) == origin0
    # Attempt paths restored; unrelated files untouched (incl. the staged entry and the harness).
    assert (repo / KERNEL).read_text() == '// baseline kernel\n'
    assert not (repo / 'tests/torch_fast_spdmm/test_scratch.py').exists()
    assert (repo / 'tests/torch_fast_spdmm/benchmark/agent_bench.py').read_text() == '# dirty harness\n'
    assert (repo / 'outside.txt').exists()
    assert 'notes.txt' in _git(repo, 'diff', '--cached', '--name-only').splitlines()

    # Now begin works; the recovery tool is refused while an attempt is active.
    assert _run_cli(capsys, 'begin_attempt', {'title': 'T', 'hypothesis': 'H.'})[0] == 0
    code, res = _run_cli(capsys, 'clean_attempt_workspace', {})
    assert code == 2 and 'no active attempt' in res['detail']
    assert _run_cli(capsys, 'finish_attempt', {
        'outcome': 'ABORTED', 'decision_reason': 'test', 'counts_toward_limit': False,
    })[0] == 0


def test_clean_attempt_workspace_refuses_committed_drift(tmp_path, monkeypatch, capsys):
    env = _init_loop_repo(tmp_path, monkeypatch)
    repo = env['repo']
    (repo / KERNEL).write_text('// committed outside any attempt\n')
    _git(repo, 'commit', '-am', 'stray commit')
    head1, origin0 = _git(repo, 'rev-parse', 'HEAD'), _origin_head(env)
    (repo / 'tests/torch_fast_spdmm/test_new.py').write_text('x\n')

    code, res = _run_cli(capsys, 'clean_attempt_workspace', {})
    assert code == 2 and 'user intervention' in res['detail']
    assert _git(repo, 'rev-parse', 'HEAD') == head1 and _origin_head(env) == origin0
    assert (repo / 'tests/torch_fast_spdmm/test_new.py').exists()  # nothing discarded on refusal
    code, res = _run_cli(capsys, 'begin_attempt', {'title': 'T', 'hypothesis': 'H.'})
    assert code == 2


def test_export_output_dir_is_restricted(tmp_path, monkeypatch, capsys):
    env = _init_loop_repo(tmp_path, monkeypatch)
    repo = env['repo']
    for bad in ('torch_fast_spdmm/kernels', 'tests/torch_fast_spdmm', '../escape',
                str(tmp_path / 'abs'), 'reports/agent_loop/runs/../../..'):
        code, res = _run_cli(capsys, 'export_attempt_log', {'output_dir': bad})
        assert code == 2 and 'reports/agent_loop/runs' in res['detail'], bad
    assert not (repo / 'torch_fast_spdmm/kernels/summary.json').exists()
    code, res = _run_cli(capsys, 'export_attempt_log', {'output_dir': 'reports/agent_loop/runs/custom'})
    assert code == 0 and res['json_path'] == 'reports/agent_loop/runs/custom/summary.json'


# ---------------------------------------------------------------- 2. campaign budget


def test_budget_is_campaign_wide_across_runs(tmp_path, monkeypatch, capsys):
    env = _init_loop_repo(tmp_path, monkeypatch)
    repo = env['repo']
    for i in range(3):
        _counted_attempt(capsys, repo, f'run1 attempt {i}')
    assert _run_cli(capsys, 'close_run', {'status': 'completed', 'reason': 'early stop'})[0] == 0

    code, started = _run_cli(capsys, 'begin_attempt', {
        'title': 'run2 attempt 0', 'hypothesis': 'Second run.', 'new_run': True})
    assert code == 0 and started['counted_attempts_used'] == 3
    assert started['counted_attempts_remaining'] == 2 and started['campaign_budget'] == 5
    conn = _db(env)
    assert conn.execute('SELECT max_attempts FROM runs WHERE run_id = ?',
                        (started['run_id'],)).fetchone()[0] == 2  # only the remaining budget
    conn.close()
    assert _run_cli(capsys, 'finish_attempt', {
        'outcome': 'ABORTED', 'decision_reason': 'x', 'counts_toward_limit': False})[0] == 0
    for i in range(2):
        _counted_attempt(capsys, repo, f'run2 attempt {i + 1}')

    code, st = _run_cli(capsys, 'get_loop_status')
    assert code == 0 and st['counted_attempts_used'] == 5 and st['counted_attempts_remaining'] == 0
    assert st['run_counted_attempts'] == 2
    code, exp = _run_cli(capsys, 'export_attempt_log', {})
    summary = json.loads((repo / exp['json_path']).read_text())
    assert summary['counts']['counted_attempts_used'] == 5
    assert summary['counts']['counted_attempts_remaining'] == 0
    assert summary['counts']['run_counted_attempts'] == 2
    assert '5 / 5 (remaining: 0; this run: 2)' in (repo / exp['markdown_path']).read_text()

    assert _run_cli(capsys, 'close_run', {'status': 'completed', 'reason': 'done'})[0] == 0
    code, res = _run_cli(capsys, 'begin_attempt', {
        'title': 'run3', 'hypothesis': 'Must be refused.', 'new_run': True})
    assert code == 2 and 'attempt limit reached (5/5' in res['detail']
    # No ordinary begin_attempt argument can reset the budget.
    for extra in ({'max_attempts': 99}, {'budget': 10}, {'reset_budget': True}):
        code, res = _run_cli(capsys, 'begin_attempt', {
            'title': 'run3', 'hypothesis': 'Must be refused.', 'new_run': True, **extra})
        assert code == 2
    conn = _db(env)
    assert conn.execute('SELECT COUNT(*) FROM runs').fetchone()[0] == 2  # no third run created
    conn.close()


def test_plain_git_commit_forbids_noncounting_abort(tmp_path, monkeypatch, capsys):
    env = _init_loop_repo(tmp_path, monkeypatch)
    repo = env['repo']

    # (a) committed outside commit_kernel_attempt, then reverted: still visible in history.
    code, a = _run_cli(capsys, 'begin_attempt', {'title': 'A', 'hypothesis': 'Plain commit then revert.'})
    assert code == 0 and a['start_head_sha'] == _git(repo, 'rev-parse', 'HEAD')
    (repo / KERNEL).write_text('// plain\n')
    _git(repo, 'commit', '-am', 'plain kernel commit')
    _git(repo, 'revert', '--no-edit', 'HEAD')
    code, res = _run_cli(capsys, 'finish_attempt', {
        'outcome': 'ABORTED', 'decision_reason': 'x', 'counts_toward_limit': False, 'rolled_back': True})
    assert code == 2 and 'candidate commit was made' in res['detail']
    code, fin = _run_cli(capsys, 'finish_attempt', {
        'outcome': 'ABORTED', 'decision_reason': 'counted', 'counts_toward_limit': True, 'rolled_back': True})
    assert code == 0 and fin['counted_attempts_used'] == 1

    # (b) committed, then reset away (test repo only): still visible in HEAD's reflog.
    start = _git(repo, 'rev-parse', 'HEAD')
    assert _run_cli(capsys, 'begin_attempt', {'title': 'B', 'hypothesis': 'Plain commit then reset.'})[0] == 0
    (repo / KERNEL).write_text('// plain again\n')
    _git(repo, 'commit', '-am', 'plain kernel commit 2')
    _git(repo, 'reset', '--hard', start)
    code, res = _run_cli(capsys, 'finish_attempt', {
        'outcome': 'ABORTED', 'decision_reason': 'x', 'counts_toward_limit': False})
    assert code == 2 and 'candidate commit was made' in res['detail']

    # (c) control: a docs-only plain commit does not touch attempt paths.
    assert _run_cli(capsys, 'finish_attempt', {
        'outcome': 'ABORTED', 'decision_reason': 'counted', 'counts_toward_limit': True,
        'rolled_back': True})[0] == 0
    assert _run_cli(capsys, 'begin_attempt', {'title': 'C', 'hypothesis': 'Docs only.'})[0] == 0
    (repo / 'README.md').write_text('docs\n')
    _git(repo, 'add', 'README.md')
    _git(repo, 'commit', '-m', 'docs')
    assert _run_cli(capsys, 'finish_attempt', {
        'outcome': 'ABORTED', 'decision_reason': 'clean', 'counts_toward_limit': False})[0] == 0


def test_legacy_ledger_is_migrated(tmp_path, monkeypatch, capsys):
    env = _init_loop_repo(tmp_path, monkeypatch)
    assert _run_cli(capsys, 'get_loop_status')[0] == 0
    conn = sqlite3.connect(env['db_path'])
    run_id = conn.execute('SELECT run_id FROM runs').fetchone()[0]
    conn.execute('DROP TABLE attempt_commits')
    conn.execute('DROP TABLE campaign')
    conn.execute('ALTER TABLE attempts DROP COLUMN start_head_sha')
    conn.execute('ALTER TABLE attempts DROP COLUMN start_reflog_len')
    conn.execute(
        """INSERT INTO attempts (run_id, ordinal, title, hypothesis, baseline_sha, started_at, ended_at,
               status, outcome, rolled_back, counts_toward_limit)
           VALUES (?, 1, 'legacy', 'legacy row', ?, '2026-11-01T00:00:00Z', '2026-11-01T01:00:00Z',
               'closed', 'ROLLED BACK', 1, 1)""",
        (run_id, env['baseline']),
    )
    conn.commit()
    conn.close()

    code, st = _run_cli(capsys, 'get_loop_status')
    assert code == 0 and st['counted_attempts_used'] == 1 and st['counted_attempts_remaining'] == 4
    conn = _db(env)
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    cols = {r[1] for r in conn.execute('PRAGMA table_info(attempts)')}
    assert {'campaign', 'attempt_commits'} <= tables and {'start_head_sha', 'start_reflog_len'} <= cols
    assert conn.execute('SELECT budget FROM campaign').fetchone()[0] == 5
    conn.close()
    code, started = _run_cli(capsys, 'begin_attempt', {'title': 'after migration', 'hypothesis': 'Works.'})
    assert code == 0 and started['ordinal'] == 2 and started['counted_attempts_remaining'] == 4


# ---------------------------------------------------------------- 3. changelog binding


def test_changelog_sha_is_bound_to_the_attempt(tmp_path, monkeypatch, capsys):
    env = _init_loop_repo(tmp_path, monkeypatch)
    repo = env['repo']
    base_args = {'outcome': 'ROLLED BACK', 'decision_reason': 'flat', 'rolled_back': True}

    assert _run_cli(capsys, 'begin_attempt', {'title': 'A', 'hypothesis': 'Changelog binding.'})[0] == 0
    (repo / KERNEL).write_text('// a\n')
    assert _run_cli(capsys, 'commit_kernel_attempt', {'message': 'kernel: a'})[0] == 0
    assert _run_cli(capsys, 'rollback_workspace', {})[0] == 0

    # Old commit that touched CHANGELOG.md (the tools baseline created it): rejected.
    code, res = _run_cli(capsys, 'finish_attempt', {**base_args, 'changelog_sha': env['baseline']})
    assert code == 2 and 'not made by commit_kernel_attempt' in res['detail']
    # Plain git CHANGELOG commit outside the governed tool: rejected.
    (repo / 'CHANGELOG.md').write_text('### plain\n' + (repo / 'CHANGELOG.md').read_text())
    _git(repo, 'commit', '-am', 'docs: plain changelog')
    plain = _git(repo, 'rev-parse', 'HEAD')
    code, res = _run_cli(capsys, 'finish_attempt', {**base_args, 'changelog_sha': plain})
    assert code == 2 and 'not made by commit_kernel_attempt' in res['detail']
    # Governed commit that later became unreachable from HEAD (test repo reset): rejected.
    lost = _commit_changelog(repo, 'A lost')
    _git(repo, 'reset', '--hard', 'HEAD~1')
    _git(repo, 'push', '-f', 'origin', 'opt/loop')  # realign the throwaway test origin
    code, res = _run_cli(capsys, 'finish_attempt', {**base_args, 'changelog_sha': lost})
    assert code == 2 and 'not reachable from HEAD' in res['detail']
    # Governed commit of this attempt: accepted, with CHANGELOG.md in its recorded files.
    good = _commit_changelog(repo, 'A')
    conn = _db(env)
    row = conn.execute('SELECT files_json FROM attempt_commits WHERE sha = ?', (good,)).fetchone()
    conn.close()
    assert 'CHANGELOG.md' in json.loads(row['files_json'])
    assert _run_cli(capsys, 'finish_attempt', {**base_args, 'changelog_sha': good})[0] == 0

    # A second attempt cannot reuse attempt A's changelog commit.
    assert _run_cli(capsys, 'begin_attempt', {'title': 'B', 'hypothesis': 'Reuse.'})[0] == 0
    (repo / KERNEL).write_text('// b\n')
    assert _run_cli(capsys, 'commit_kernel_attempt', {'message': 'kernel: b'})[0] == 0
    assert _run_cli(capsys, 'rollback_workspace', {})[0] == 0
    code, res = _run_cli(capsys, 'finish_attempt', {**base_args, 'changelog_sha': good})
    assert code == 2 and 'belongs to attempt' in res['detail']
    good_b = _commit_changelog(repo, 'B')
    assert _run_cli(capsys, 'finish_attempt', {**base_args, 'changelog_sha': good_b})[0] == 0


# ---------------------------------------------------------------- 4. benchmark crash classes


def _run_runner(tmp_path: Path, order, fail_role: str | None) -> tuple[int, str, set]:
    run_dir = tmp_path / 'run'
    run_dir.mkdir(parents=True)
    body = bench.bench_runner(order, base_dir='/b', cand_dir='/c', kernel_name='boo:rowseg',
                              cases_json='[{"mat":"x","F":32}]', reps=3, warmup=1)
    body = body.replace('{{RUN_DIR}}', str(run_dir)).replace('{{SRC_DIR}}', str(tmp_path))
    stub = (
        'set -euo pipefail\n'
        'use() { :; }\n'
        f'python() {{ echo "ran $4" >> {run_dir}/calls; '
        f'if [ "$4" = "{fail_role or "none"}" ]; then return 7; fi; return 0; }}\n'
    )
    proc = subprocess.run(['bash', '-c', stub + body], capture_output=True, text=True)
    done = {p.name for p in run_dir.glob('*.done')}
    calls = (run_dir / 'calls').read_text() if (run_dir / 'calls').exists() else ''
    return proc.returncode, proc.stdout + calls, done


ORDER = [('cand', 1), ('base', 1), ('cand', 2), ('base', 2)]


def test_bench_runner_script_markers(tmp_path):
    rc, out, done = _run_runner(tmp_path / 'ok', ORDER, None)
    assert rc == 0 and 'AGENT_BENCH_VERDICT ok' in out
    assert done == {'cand_1.done', 'base_1.done', 'cand_2.done', 'base_2.done'}

    rc, out, done = _run_runner(tmp_path / 'cand', ORDER, 'cand')
    assert rc == bench.BENCH_EXIT_CANDIDATE_CRASH and 'AGENT_BENCH_VERDICT candidate_crash' in out
    assert 'AGENT_BENCH_FAILURE role=cand round=1 rc=7' in out
    assert out.count('ran cand') == 1          # the crash is not re-run in later rounds
    assert out.count('ran base') == 1 and done == {'base_1.done'}  # baseline ran to attribute it

    rc, out, done = _run_runner(tmp_path / 'base', ORDER, 'base')
    assert rc == bench.BENCH_EXIT_INTEGRITY and 'AGENT_BENCH_VERDICT integrity' in out
    assert 'AGENT_BENCH_FAILURE role=base round=1 rc=7' in out


def _job(state_name, exit_code, out_tail='', err_tail=''):
    return {'job': 9300, 'state': state_name, 'exit': exit_code, 'node': 'ise-3090-02',
            'elapsed_s': 5, 'out_tail': out_tail, 'err_tail': err_tail,
            'out_path': '/o', 'err_path': '/e', 'resume_key': 'rk', 'reattached': False}


CAND_CRASH_OUT = ('AGENT_BENCH_FAILURE role=cand round=1 rc=134\nAGENT_BENCH_VERDICT candidate_crash')
BASE_CRASH_OUT = ('AGENT_BENCH_FAILURE role=base round=1 rc=1\nAGENT_BENCH_VERDICT integrity')


def test_classify_bench_job():
    assert bench.classify_bench_job(_job('COMPLETED', 0))[0] == 'ok'
    kind, markers = bench.classify_bench_job(_job('FAILED', 86, CAND_CRASH_OUT))
    assert kind == 'candidate_crash' and markers == [{'role': 'cand', 'round': 1, 'rc': 134}]
    assert bench.classify_bench_job(_job('FAILED', 87, BASE_CRASH_OUT))[0] == 'integrity'
    for job in (_job('NODE_FAIL', 0), _job('BOOT_FAIL', 0), _job('PREEMPTED', 0),
                _job('CANCELLED', 0), _job('TIMEOUT', 0), _job('OUT_OF_MEMORY', 0),
                _job('FAILED', 1, err_tail='CUDA error: an illegal memory access'),  # no markers
                _job('FAILED', 86),                                                # code w/o verdict
                _job('NODE_FAIL', 86, CAND_CRASH_OUT)):                             # node died
        assert bench.classify_bench_job(job)[0] == 'infrastructure', job


def _bench_env(monkeypatch):
    manifest = json.dumps({'matrices': {'roadNet-CA': {
        'canonical_shape': [10, 10], 'canonical_nnz': 20, 'nonempty_cols': 10}}})
    monkeypatch.setattr(bench, 'ensure_build', lambda sha, config=None: {'ok': True, 'build_dir': '/r/b'})
    monkeypatch.setattr(bench, 'ssh', lambda cmd, config=None: type('R', (), {'returncode': 0, 'stdout': '', 'stderr': ''})())

    def fake_checked_ssh(cmd, config=None):
        if 'ls ' in cmd:
            return type('R', (), {'stdout': 'roadNet-CA_F128.json\n', 'stderr': '', 'returncode': 0})()
        return type('R', (), {'stdout': manifest, 'stderr': '', 'returncode': 0})()
    monkeypatch.setattr(bench, 'checked_ssh', fake_checked_ssh)


BENCH_ARGS = {'kernel_name': 'boo:rowseg', 'matrix_datasets': ['roadNet-CA'], 'features': [128]}


def test_run_benchmarks_failure_classes(tmp_path, monkeypatch, capsys):
    env = _init_loop_repo(tmp_path, monkeypatch)
    repo = env['repo']
    _bench_env(monkeypatch)
    assert _run_cli(capsys, 'begin_attempt', {'title': 'Crash', 'hypothesis': 'Crashes the GPU.'})[0] == 0
    (repo / KERNEL).write_text('// crashes\n')
    assert _run_cli(capsys, 'commit_kernel_attempt', {'message': 'kernel: crash'})[0] == 0
    released = []
    monkeypatch.setattr(bench, 'release_job', lambda key: released.append(key))

    def submit_returning(job):
        monkeypatch.setattr(bench, 'submit_slurm_job', lambda script, **kw: dict(job))

    # Baseline crash -> integrity (exit 3), never blamed on the candidate.
    submit_returning(_job('FAILED', 87, BASE_CRASH_OUT))
    code, res = _run_cli(capsys, 'run_benchmarks', BENCH_ARGS)
    assert code == 3 and 'integrity' in res['detail'] and 'not attributable to the candidate' in res['detail']
    # Node failure -> infrastructure (exit 3).
    submit_returning(_job('NODE_FAIL', 0))
    code, res = _run_cli(capsys, 'run_benchmarks', BENCH_ARGS)
    assert code == 3 and 'infrastructure' in res['detail']
    # SSH outage / accounting uncertainty inside the Slurm layer -> infrastructure (exit 3).

    def uncertain(script, **kw):
        raise InfrastructureError('uncertain Slurm identity for pending job 9300; sacct returned UNKNOWN')
    monkeypatch.setattr(bench, 'submit_slurm_job', uncertain)
    code, res = _run_cli(capsys, 'run_benchmarks', BENCH_ARGS)
    assert code == 3 and 'uncertain Slurm identity' in res['detail']

    # Candidate crash -> structured correctness failure, exit 0, ok=false; not infrastructure.
    submit_returning(_job('FAILED', 86, CAND_CRASH_OUT, 'CUDA error: an illegal memory access'))
    code, res = _run_cli(capsys, 'run_benchmarks', BENCH_ARGS)
    assert code == 0 and res['ok'] is False and res['failure'] == 'candidate_runtime'
    assert res['verdict'] == 'F' and res['failures'] == [{'role': 'cand', 'round': 1, 'rc': 134}]
    failure_file = repo / 'tests/torch_fast_spdmm/benchmark/results/agent_runs' / res['cand'] / '9300' / 'bench_failure.json'
    assert json.loads(failure_file.read_text())['failure'] == 'candidate_runtime'
    assert released.count('rk') == 3

    # The workflow closes it as FAIL (correctness) without re-running; it counts.
    assert _run_cli(capsys, 'rollback_workspace', {})[0] == 0
    cl = _commit_changelog(repo, 'Crash')
    code, fin = _run_cli(capsys, 'finish_attempt', {
        'outcome': 'FAIL (correctness)', 'decision_reason': 'candidate CUDA crash in benchmark',
        'rolled_back': True, 'changelog_sha': cl})
    assert code == 0 and fin['counts_toward_limit'] is True and fin['counted_attempts_used'] == 1
    code, res = _run_cli(capsys, 'begin_attempt', {'title': 'x', 'hypothesis': 'y.'})
    assert code == 0
    assert _run_cli(capsys, 'finish_attempt', {
        'outcome': 'ABORTED', 'decision_reason': 'x', 'counts_toward_limit': False})[0] == 0


# ---------------------------------------------------------------- 5. secret redaction

SECRETS = {
    'gh': 'ghp_' + 'A1b2C3d4E5f6G7h8I9j0K1l2M3n4O5p6Q7r8',
    'pat': 'github_pat_' + '11ABCDEFG0123456789_abcdefghijklmnopqrstuvwxyz',
    'sk': 'sk-proj-' + 'abcdefghijklmnopqrstuvwxyz012345',
    'url': 'hunter2pass',
    'bearer': 'bearerSecretValue123456',
    'aws': 'AKIA' + 'ABCDEFGHIJKLMNOP',
    'pem': 'MIIEowIBAAKCAQEAsecretkeymaterial',
    'envpw': 'S3cr3tDbPassw0rd',
}


def _secret_strings():
    return [
        f"use token {SECRETS['gh']} here",
        f"GH_TOKEN={SECRETS['pat']}",
        f"key={SECRETS['sk']} tail",
        f"https://alice:{SECRETS['url']}@example.com/repo.git",
        f"Authorization: Bearer {SECRETS['bearer']}",
        f"aws id {SECRETS['aws']}",
        f"-----BEGIN RSA PRIVATE KEY-----\n{SECRETS['pem']}\n-----END RSA PRIVATE KEY-----",
        f"export DB_PASSWORD={SECRETS['envpw']}",
    ]


def test_secret_redaction_everywhere(tmp_path, monkeypatch, capsys):
    env = _init_loop_repo(tmp_path, monkeypatch)
    repo = env['repo']
    strings = _secret_strings()
    joined = ' | '.join(s.replace('\n', ' ') for s in strings)

    # Free-text fields (stored in attempts, args_json and result summaries).
    code, _ = _run_cli(capsys, 'begin_attempt', {
        'title': f"Title {SECRETS['gh']}", 'hypothesis': f'Hypothesis {joined}.'})
    assert code == 0
    # Nested and ordinary string arguments + error text (unexpected kwarg -> exit 2, still logged).
    code, _ = _run_cli(capsys, 'submit_slurm_job', {
        'script_content': 'echo ' + SECRETS['sk'], 'dry_run': True,
        'note': strings[3], 'nested': {'list': strings, 'max_tokens': 5, 'environment_name': 'fem'}})
    assert code == 2
    # Error detail echoing a secret-bearing argument.
    code, res = _run_cli(capsys, 'export_attempt_log', {'output_dir': strings[3]})
    assert code == 2
    assert _run_cli(capsys, 'finish_attempt', {
        'outcome': 'ABORTED', 'decision_reason': f'Reason {joined}', 'counts_toward_limit': False})[0] == 0
    assert _run_cli(capsys, 'close_run', {'status': 'aborted', 'reason': f'close {joined}'})[0] == 0
    code, exp = _run_cli(capsys, 'export_attempt_log', {})
    assert code == 0

    conn = _db(env)
    blobs = []
    for row in conn.execute('SELECT args_json, result_summary_json, error_detail FROM tool_calls'):
        blobs.extend(v or '' for v in row)
    for row in conn.execute('SELECT title, hypothesis, decision_reason FROM attempts'):
        blobs.extend(v or '' for v in row)
    blobs.extend(r[0] or '' for r in conn.execute('SELECT close_reason FROM runs'))
    args_rows = [json.loads(r[0]) for r in conn.execute(
        "SELECT args_json FROM tool_calls WHERE tool_name = 'submit_slurm_job'")]
    conn.close()
    blobs.append((repo / exp['json_path']).read_text())
    blobs.append((repo / exp['markdown_path']).read_text())
    scripts_dir = env['db_path'].parent / 'scripts'
    blobs.extend(p.read_text() for p in scripts_dir.glob('*') if p.is_file())
    everything = '\n'.join(blobs)
    for name, secret in SECRETS.items():
        assert secret not in everything, name

    # Surrounding non-secret text and non-secret keys are preserved.
    nested = args_rows[0]['nested']
    assert nested['max_tokens'] == 5 and nested['environment_name'] == 'fem'
    assert 'https://[REDACTED]@example.com/repo.git' in args_rows[0]['note']
    assert 'GH_TOKEN=[REDACTED' in json.dumps(nested['list'])
    assert 'use token [REDACTED_GITHUB_TOKEN] here' in nested['list'][0]
    assert args_rows[0]['script_content']['redacted_type'] == 'script_omitted_sensitive'


# ---------------------------------------------------------------- 6. scoped rollback commit


def test_rollback_never_commits_unrelated_staged_file(tmp_path, monkeypatch, capsys):
    env = _init_loop_repo(tmp_path, monkeypatch)
    repo = env['repo']
    assert _run_cli(capsys, 'begin_attempt', {'title': 'R', 'hypothesis': 'Rollback scope.'})[0] == 0
    (repo / KERNEL).write_text('// candidate\n')
    assert _run_cli(capsys, 'commit_kernel_attempt', {'message': 'kernel: r'})[0] == 0
    (repo / 'notes.txt').write_text('private staged note\n')
    _git(repo, 'add', 'notes.txt')

    code, res = _run_cli(capsys, 'rollback_workspace', {})
    assert code == 0 and res['revert_sha']
    committed = _git(repo, 'show', '--name-only', '--format=', res['revert_sha']).splitlines()
    assert committed == [KERNEL]
    assert 'notes.txt' in _git(repo, 'diff', '--cached', '--name-only').splitlines()  # still staged
    pushed_tree = _git(env['origin'], 'ls-tree', '-r', '--name-only', 'refs/heads/opt/loop').splitlines()
    assert 'notes.txt' not in pushed_tree and _origin_head(env) == res['revert_sha']
    assert (repo / KERNEL).read_text() == '// baseline kernel\n'


# ---------------------------------------------------------------- 7. audit finalization failure


def _fail_finalize_once(monkeypatch, tool: str):
    real = attempt_log._finalize_tx
    fired = []

    def flaky(conn, payload, *, replay=False):
        if payload['tool_name'] == tool and not replay and not fired:
            fired.append(payload['call_id'])
            raise sqlite3.OperationalError('disk I/O error (simulated)')
        return real(conn, payload, replay=replay)
    monkeypatch.setattr(attempt_log, '_finalize_tx', flaky)
    return fired


def test_audit_failure_after_commit_is_explicit_and_reconciled(tmp_path, monkeypatch, capsys):
    env = _init_loop_repo(tmp_path, monkeypatch)
    repo = env['repo']
    assert _run_cli(capsys, 'begin_attempt', {'title': 'Audit', 'hypothesis': 'Audit recovery.'})[0] == 0
    (repo / KERNEL).write_text('// committed\n')
    fired = _fail_finalize_once(monkeypatch, 'commit_kernel_attempt')

    code, out = _run_cli(capsys, 'commit_kernel_attempt', {'message': 'kernel: audit'})
    assert code == cli.EXIT_AUDIT_PENDING == 4
    assert out['side_effect_succeeded'] is True and out['do_not_retry'] is True
    assert out['audit_finalization'] == 'pending_recovery'
    sha = out['result']['sha']
    assert sha == _git(repo, 'rev-parse', 'HEAD') == _origin_head(env)  # the push did happen
    record = Path(out['recovery_record'])
    assert record.exists() and record.name == f'{fired[0]}.json'

    # The next invocation of any tool replays the record into the ledger.
    assert _run_cli(capsys, 'get_loop_status')[0] == 0
    assert not record.exists()
    conn = _db(env)
    call = conn.execute('SELECT status, exit_code, result_summary_json FROM tool_calls WHERE call_id = ?',
                        (fired[0],)).fetchone()
    bound = conn.execute('SELECT attempt_id FROM attempt_commits WHERE sha = ?', (sha,)).fetchone()
    cand = json.loads(conn.execute("SELECT candidate_shas_json FROM attempts WHERE status = 'active'")
                      .fetchone()[0])
    conn.close()
    assert call['status'] == 'success' and call['exit_code'] == 0 and sha in call['result_summary_json']
    assert bound is not None and cand == [sha]
    # Because the commit is now recorded, a non-counting ABORTED is still forbidden.
    assert _run_cli(capsys, 'rollback_workspace', {})[0] == 0
    code, res = _run_cli(capsys, 'finish_attempt', {
        'outcome': 'ABORTED', 'decision_reason': 'x', 'counts_toward_limit': False, 'rolled_back': True})
    assert code == 2 and 'candidate commit was made' in res['detail']


def test_audit_failure_after_promotion_is_explicit_and_reconciled(tmp_path, monkeypatch, capsys):
    env = _init_loop_repo(tmp_path, monkeypatch)
    assert _run_cli(capsys, 'begin_attempt', {'title': 'Promo', 'hypothesis': 'Audit after promote.'})[0] == 0
    promoted = {'promoted': True, 'cand': 'c' * 40, 'previous_verified': env['baseline'], 'reasons': []}
    real_dispatch = cli.dispatch
    # promote_attempt's own gate/tag logic is covered in test_offline; here the mutation is stubbed
    # so the test isolates what happens when its audit record cannot be written afterwards.
    monkeypatch.setattr(cli, 'dispatch', lambda tool, args: dict(promoted) if tool == 'promote_attempt'
                        else real_dispatch(tool, args))
    fired = _fail_finalize_once(monkeypatch, 'promote_attempt')

    code, out = _run_cli(capsys, 'promote_attempt', {'bench_json': 'x/bench.json'})
    assert code == 4 and out['side_effect_succeeded'] is True and out['do_not_retry'] is True
    assert out['result'] == promoted and Path(out['recovery_record']).exists()

    assert _run_cli(capsys, 'get_loop_status')[0] == 0
    conn = _db(env)
    call = conn.execute('SELECT status, result_summary_json FROM tool_calls WHERE call_id = ?',
                        (fired[0],)).fetchone()
    conn.close()
    assert call['status'] == 'success' and json.loads(call['result_summary_json'])['promoted'] is True
    assert not Path(out['recovery_record']).exists()


def test_unwritable_recovery_record_still_reports_side_effect(tmp_path, monkeypatch, capsys):
    env = _init_loop_repo(tmp_path, monkeypatch)
    repo = env['repo']
    assert _run_cli(capsys, 'begin_attempt', {'title': 'NoRec', 'hypothesis': 'Recovery dir blocked.'})[0] == 0
    (repo / KERNEL).write_text('// c\n')
    _fail_finalize_once(monkeypatch, 'commit_kernel_attempt')
    (env['db_path'].parent / 'audit_recovery').write_text('not a directory')
    code, out = _run_cli(capsys, 'commit_kernel_attempt', {'message': 'kernel: c'})
    assert code == 4 and out['side_effect_succeeded'] is True and out['recovery_record'] is None
    assert out['audit_finalization'] == 'manual_recovery_required'
    assert out['manual_recovery_required'] is True
    assert out['do_not_retry'] is True
    assert 'STOP the campaign' in out['next_step']
    # The successful side effect's result is preserved in full.
    assert out['result']['sha'] == _git(repo, 'rev-parse', 'HEAD') == _origin_head(env)
    assert out['result']['pushed'] is True and out['result']['files'] == [KERNEL]
