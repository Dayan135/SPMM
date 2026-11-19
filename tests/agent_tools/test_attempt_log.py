"""Offline tests for durable SQLite attempt/tool-call logging and CLI integration."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import copy
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import subprocess

import pytest

from optimizer.agent_tools import __main__ as cli
from optimizer.agent_tools import attempt_log, bench, slurm, state, vcs
from optimizer.agent_tools.util import ArgumentError, InfrastructureError


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ['git', *args], cwd=repo, capture_output=True, text=True, check=True
    ).stdout.strip()


def _init_loop_repo(tmp_path: Path, monkeypatch) -> dict:
    """Create an isolated git repo + bare origin + state.json in tmp_path."""
    repo = (tmp_path / 'repo').resolve()
    origin = (tmp_path / 'origin.git').resolve()
    _git(tmp_path, 'init', '--bare', str(origin))
    _git(tmp_path, 'init', str(repo))
    _git(repo, 'config', 'user.name', 'Offline Test')
    _git(repo, 'config', 'user.email', 'offline@example.com')
    _git(repo, 'remote', 'add', 'origin', str(origin))

    real_prompt = (state.ROOT / 'reports/agent_loop/optimization_loop_prompt.md').read_text()
    required = {
        'optimizer/agent_tools/__main__.py': '# entry\n',
        'tests/torch_fast_spdmm/benchmark/universal_suite.py': '# suite\n',
        'tests/torch_fast_spdmm/benchmark/agent_bench.py': '# bench\n',
        'tests/torch_fast_spdmm/benchmark/agent_profile.py': '# profile\n',
        'torch_fast_spdmm/kernels/boo_kernel.cu': '// baseline kernel\n',
        'reports/agent_loop/optimization_loop_prompt.md': real_prompt,
        'CHANGELOG.md': '# Changelog\n',
    }
    for rel, content in required.items():
        p = repo / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
    _git(repo, 'add', '.')
    _git(repo, 'commit', '-m', 'tools baseline')
    baseline = _git(repo, 'rev-parse', 'HEAD')
    _git(repo, 'branch', '-M', 'opt/loop')
    _git(repo, 'tag', 'opt/verified', baseline)
    _git(repo, 'push', '-u', 'origin', 'opt/loop')
    _git(repo, 'push', 'origin', 'refs/tags/opt/verified')

    state_file = repo / '.agent_loop' / 'state.json'
    state_file.parent.mkdir(parents=True, exist_ok=True)
    cfg = copy.deepcopy(state.DEFAULT)
    cfg.update(
        loop_branch='opt/loop',
        verified=baseline,
        approved_skips=['cant', 'pdb1HYS', 'ASIC_680k', 'rail4284'],
    )
    state_file.write_text(json.dumps(cfg, indent=2) + '\n')

    # Point state.ROOT and state.STATE at the isolated repo.
    monkeypatch.setattr(state, 'ROOT', repo)
    monkeypatch.setattr(state, 'STATE', state_file)
    # state.locked/read/update have default arg path=STATE bound at import time;
    # rebind defaults via thin wrappers so all modules use the isolated state.json.
    orig_locked = state.locked
    monkeypatch.setattr(state, 'locked', lambda path=state_file: orig_locked(path))
    monkeypatch.setattr(state, 'read', lambda path=state_file: _read_state(orig_locked, path))
    monkeypatch.setattr(state, 'update', lambda changes, path=state_file: _update_state(orig_locked, changes, path))

    return {
        'repo': repo,
        'origin': origin,
        'baseline': baseline,
        'state_file': state_file,
        'db_path': repo / '.agent_loop' / 'attempts.sqlite3',
    }


def _read_state(orig_locked, path):
    with orig_locked(path) as st:
        return dict(st)


def _update_state(orig_locked, changes, path):
    with orig_locked(path) as st:
        st.update(changes)
        return dict(st)


def _run_cli(capsys, tool: str, args: dict | str = '{}') -> tuple[int, dict]:
    raw_args = args if isinstance(args, str) else json.dumps(args)
    code = cli.main([tool, '--args', raw_args])
    out = capsys.readouterr().out.strip()
    lines = out.splitlines()
    assert len(lines) == 1, f'expected single JSON line on stdout, got: {out!r}'
    return code, json.loads(lines[0])


def _make_bench_file(repo: Path, rel: str, *, cand: str, base: str, ok: bool = True,
                     tune_geo: float = 1.05, held_geo: float = 1.01,
                     tune_w: int = 6, tune_l: int = 0, tune_t: int = 14,
                     held_w: int = 2, held_l: int = 0, held_t: int = 13,
                     job: int = 24001) -> Path:
    real_example = json.loads(
        (Path(__file__).resolve().parents[2] / 'reports/agent_loop/schemas/examples/bench.example.json').read_text()
    )
    data = copy.deepcopy(real_example)
    raw_rel = str(Path(rel).with_name('samples.csv'))
    raw_path = repo / raw_rel
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_text('role,mat,F,arm,time_ms\ncand,roadNet-CA,128,kernel,1.0\n')
    data.update(cand=cand, base=base, ok=ok, raw=raw_rel)
    data['env'].update(job=job, reps=30, rounds=3)
    data['sum']['fail'] = 0 if ok else 1
    data['sum']['tune'].update(n=tune_w + tune_l + tune_t, geo_x_base=tune_geo, geo_x_cusp=0.82,
                               W=tune_w, L=tune_l, T=tune_t)
    data['sum']['held'].update(n=held_w + held_l + held_t, geo_x_base=held_geo, geo_x_cusp=0.79,
                               W=held_w, L=held_l, T=held_t)
    if not ok:
        data['rows'][0][11] = 'F'
    target = repo / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(data, indent=2) + '\n')
    return target


def _commit_changelog(repo: Path, title: str) -> str:
    """Commit a CHANGELOG entry through the audited tool path (as the CLI does), so the
    commit is bound to the active attempt in attempt_commits."""
    log_path = repo / 'CHANGELOG.md'
    log_path.write_text(f'### {title}\n\n' + log_path.read_text())
    args = {'message': f'docs: changelog for {title}'}
    handle = attempt_log.start_tool_call('commit_kernel_attempt', args)
    res = vcs.commit_kernel_attempt(**args)
    attempt_log.finish_tool_call(handle, tool_name='commit_kernel_attempt', exit_code=0, result=res)
    return res['sha']


def test_run_initialization_and_pragmas(tmp_path, monkeypatch, capsys):
    env = _init_loop_repo(tmp_path, monkeypatch)
    code, status = _run_cli(capsys, 'get_loop_status')
    assert code == 0
    run = status['run']
    assert run['branch'] == 'opt/loop'
    assert run['initial_verified_sha'] == env['baseline']
    assert run['status'] == 'active'
    assert run['max_attempts'] == 5
    assert len(run['prompt_sha256']) == 64
    assert status['active_attempt'] is None
    assert status['counted_attempts_used'] == 0
    assert status['counted_attempts_remaining'] == 5
    assert status['unfinished_tool_calls'] == []

    # Verify WAL mode, foreign keys, and single setup tool_call row.
    conn = sqlite3.connect(env['db_path'])
    conn.row_factory = sqlite3.Row
    assert conn.execute('PRAGMA journal_mode').fetchone()[0].lower() == 'wal'
    runs = conn.execute('SELECT * FROM runs').fetchall()
    calls = conn.execute('SELECT * FROM tool_calls').fetchall()
    conn.close()
    assert len(runs) == 1
    assert len(calls) == 1
    assert calls[0]['tool_name'] == 'get_loop_status'
    assert calls[0]['attempt_id'] is None
    assert calls[0]['status'] == 'success'
    assert calls[0]['exit_code'] == 0


def test_begin_finish_lifecycle_and_compile_fixes(tmp_path, monkeypatch, capsys):
    env = _init_loop_repo(tmp_path, monkeypatch)
    repo = env['repo']

    # Start attempt 1.
    code, started = _run_cli(capsys, 'begin_attempt', {
        'title': 'Guard trailing tile barrier',
        'hypothesis': 'Guarding the trailing syncthreads to non-final tiles reduces barrier stalls on F<=P.',
    })
    assert code == 0
    assert started['ordinal'] == 1
    assert started['baseline_sha'] == env['baseline']
    assert started['counted_attempts_remaining'] == 5

    # One active attempt at a time: starting another must fail with exit 2.
    code2, dup = _run_cli(capsys, 'begin_attempt', {
        'title': 'Second overlapping attempt',
        'hypothesis': 'Should be refused because attempt 1 is still active.',
    })
    assert code2 == 2
    assert 'already active' in dup['detail']

    # First kernel commit + compile-fix commit stay in attempt 1.
    (repo / 'torch_fast_spdmm/kernels/boo_kernel.cu').write_text('// try 1 (typo)\n')
    c1_code, c1 = _run_cli(capsys, 'commit_kernel_attempt', {'message': 'kernel: guard trailing barrier'})
    assert c1_code == 0

    (repo / 'torch_fast_spdmm/kernels/boo_kernel.cu').write_text('// try 1 (fixed)\n')
    c2_code, c2 = _run_cli(capsys, 'commit_kernel_attempt', {'message': 'kernel: fix compile error'})
    assert c2_code == 0

    # Simulate promotion: advance state.verified and write bench.json.
    bench_path = _make_bench_file(
        repo, 'tests/torch_fast_spdmm/benchmark/results/agent_runs/c2/24001/bench.json',
        cand=c2['sha'], base=env['baseline'], ok=True,
    )
    # Inconsistent finish_attempt (promoted=True before state.verified advances) must be rejected.
    cl_sha = _commit_changelog(repo, 'Guard trailing tile barrier')
    bad_code, bad = _run_cli(capsys, 'finish_attempt', {
        'outcome': 'PASS',
        'decision_reason': 'Passed promotion gate.',
        'promoted': True,
        'rolled_back': False,
        'bench_json': str(bench_path),
        'changelog_sha': cl_sha,
    })
    assert bad_code == 2
    assert 'state.verified is still equal' in bad['detail']

    # Now advance verified as promote_attempt would.
    state.update({'verified': c2['sha']})
    prof_local = repo / '.agent_loop' / 'roadNet-CA_F128.ncu.json'
    prof_local.write_text('{"v":"ncu.1"}\n')
    ok_code, finished = _run_cli(capsys, 'finish_attempt', {
        'outcome': 'PASS',
        'decision_reason': 'Passed tune (1.05x) and held (1.01x) promotion gates.',
        'promoted': True,
        'rolled_back': False,
        'bench_json': str(bench_path),
        'profile_paths': [str(prof_local), '/home/dayanb/research/spmm-agent/runs/c2/24002/ncu/roadNet-CA_F128.ncu-rep'],
        'changelog_sha': cl_sha,
    })
    assert ok_code == 0
    assert finished['ordinal'] == 1
    assert finished['promoted'] is True
    assert finished['counts_toward_limit'] is True
    assert finished['counted_attempts_used'] == 1
    assert finished['counted_attempts_remaining'] == 4
    # Compile-fix commit stays inside attempt 1; pure CHANGELOG commit is NOT in candidate_shas.
    assert finished['candidate_shas'] == [c1['sha'], c2['sha']]

    # Verify begin_attempt and finish_attempt calls are both associated with attempt 1.
    conn = sqlite3.connect(env['db_path'])
    conn.row_factory = sqlite3.Row
    rows = conn.execute('SELECT tool_name, attempt_id, status FROM tool_calls ORDER BY call_id').fetchall()
    arts = conn.execute('SELECT kind, local_path, remote_path, sha256, size_bytes FROM artifacts ORDER BY artifact_id').fetchall()
    conn.close()

    by_tool = [(r['tool_name'], r['attempt_id'], r['status']) for r in rows]
    assert ('begin_attempt', 1, 'success') in by_tool
    assert ('finish_attempt', 1, 'success') in by_tool
    assert ('commit_kernel_attempt', 1, 'success') in by_tool
    # Artifact hashing: local bench.json and local ncu.json have SHA-256; remote ncu-rep has remote_path only.
    bench_arts = [a for a in arts if a['kind'] == 'bench_all_json']
    assert len(bench_arts) == 1
    assert bench_arts[0]['sha256'] == hashlib.sha256(bench_path.read_bytes()).hexdigest()
    assert bench_arts[0]['size_bytes'] == bench_path.stat().st_size
    remote_arts = [a for a in arts if a['remote_path'] and a['remote_path'].endswith('.ncu-rep')]
    assert len(remote_arts) == 1 and remote_arts[0]['sha256'] is None


def test_five_attempt_limit_and_aborted_rule(tmp_path, monkeypatch, capsys):
    env = _init_loop_repo(tmp_path, monkeypatch)
    repo = env['repo']

    # An ABORTED attempt where no benchmark ran can have counts_toward_limit=False.
    code, a0 = _run_cli(capsys, 'begin_attempt', {
        'title': 'Aborted infra attempt',
        'hypothesis': 'Aborted before running benchmarks due to outage.',
    })
    assert code == 0 and a0['ordinal'] == 1
    code, f0 = _run_cli(capsys, 'finish_attempt', {
        'outcome': 'ABORTED',
        'decision_reason': 'Cluster outage before build.',
        'counts_toward_limit': False,
        'rolled_back': False,
    })
    assert code == 0 and f0['counts_toward_limit'] is False
    assert f0['counted_attempts_used'] == 0 and f0['counted_attempts_remaining'] == 5

    # Run 5 counted attempts (mix of ROLLED BACK, FAIL (build), FAIL (tests)).
    outcomes = ['ROLLED BACK', 'FAIL (build)', 'FAIL (tests)', 'ROLLED BACK', 'ROLLED BACK']
    for idx, outcome in enumerate(outcomes, start=1):
        code, a = _run_cli(capsys, 'begin_attempt', {
            'title': f'Attempt {idx}',
            'hypothesis': f'Hypothesis number {idx} should improve rowseg latency.',
        })
        assert code == 0 and a['ordinal'] == idx + 1
        (repo / 'torch_fast_spdmm/kernels/boo_kernel.cu').write_text(f'// attempt {idx}\n')
        assert _run_cli(capsys, 'commit_kernel_attempt', {'message': f'kernel: attempt {idx}'})[0] == 0

        # Trying to finish ROLLED BACK without calling rollback_workspace must be rejected.
        cl_sha = _commit_changelog(repo, f'Attempt {idx}')
        bad_code, bad = _run_cli(capsys, 'finish_attempt', {
            'outcome': outcome,
            'decision_reason': 'Not faster.',
            'rolled_back': True,
            'changelog_sha': cl_sha,
        })
        assert bad_code == 2 and 'rollback_workspace' in bad['detail']

        # Trying to mark a counted outcome as counts_toward_limit=False must be rejected.
        assert _run_cli(capsys, 'rollback_workspace', {})[0] == 0
        bad_cnt, bad_cnt_res = _run_cli(capsys, 'finish_attempt', {
            'outcome': outcome,
            'decision_reason': 'Not faster.',
            'counts_toward_limit': False,
            'rolled_back': True,
            'changelog_sha': cl_sha,
        })
        assert bad_cnt == 2 and 'must count' in bad_cnt_res['detail']

        ok_code, fin = _run_cli(capsys, 'finish_attempt', {
            'outcome': outcome,
            'decision_reason': f'Closed attempt {idx}.',
            'rolled_back': True,
            'changelog_sha': cl_sha,
        })
        assert ok_code == 0
        assert fin['counted_attempts_used'] == idx
        assert fin['counted_attempts_remaining'] == 5 - idx

    # 6th counted attempt must be refused.
    code6, denied = _run_cli(capsys, 'begin_attempt', {
        'title': 'Sixth attempt',
        'hypothesis': 'Should be blocked by the five-attempt cap.',
    })
    assert code6 == 2
    assert 'attempt limit reached (5/5' in denied['detail']

    # Normal 5/5 campaign completion: close_run(completed) + export_attempt_log.
    c_done, done = _run_cli(capsys, 'close_run', {
        'status': 'completed',
        'reason': 'All 5 counted optimization attempts finished.',
    })
    assert c_done == 0 and done['status'] == 'completed' and done['counted_attempts_used'] == 5
    c_exp, exp = _run_cli(capsys, 'export_attempt_log', {})
    assert c_exp == 0 and exp['run_id'] == done['run_id']
    exp_json = json.loads((repo / exp['json_path']).read_text())
    assert exp_json['run']['status'] == 'completed' and exp_json['run']['ended_at'] == done['ended_at']
    assert exp_json['counts']['counted_attempts_used'] == 5 and exp_json['counts']['counted_attempts_remaining'] == 0


def test_interrupted_call_recovery_and_live_call_preservation(tmp_path, monkeypatch, capsys):
    env = _init_loop_repo(tmp_path, monkeypatch)
    # Initialize DB via a normal call.
    assert _run_cli(capsys, 'get_loop_status')[0] == 0

    # Insert two simulated 'running' calls:
    # 1) A dead process call (dead PID, no lock) -> should become 'interrupted'.
    # 2) A live call (created via start_tool_call with lock held) -> must remain 'running'.
    conn = sqlite3.connect(env['db_path'])
    run_id = conn.execute('SELECT run_id FROM runs').fetchone()[0]
    cur = conn.execute(
        """
        INSERT INTO tool_calls (
            run_id, attempt_id, tool_name, args_json, args_sha256,
            head_sha, verified_sha, pid, started_at, status, slurm_job_id
        ) VALUES (?, NULL, 'run_benchmarks', '{}', 'abc', ?, ?, 99999999, '2026-11-19T00:00:00Z', 'running', 55555)
        """,
        (run_id, env['baseline'], env['baseline']),
    )
    dead_call_id = cur.lastrowid
    conn.commit()
    conn.close()

    live_handle = attempt_log.start_tool_call('submit_slurm_job', {'name': 'live-job'})
    try:
        code, status = _run_cli(capsys, 'get_loop_status')
        assert code == 0
        by_id = {c['call_id']: c for c in status['unfinished_tool_calls']}
        # Dead call was preserved and marked 'interrupted', never deleted.
        assert dead_call_id in by_id
        assert by_id[dead_call_id]['status'] == 'interrupted'
        assert by_id[dead_call_id]['error_class'] == 'ProcessInterrupted'
        assert by_id[dead_call_id]['slurm_job_id'] == 55555
        # Live call is still 'running' because its lock file is held.
        assert live_handle.ctx.call_id in by_id
        assert by_id[live_handle.ctx.call_id]['status'] == 'running'
    finally:
        # Simulate crash of live_handle by releasing lock without finalizing DB row.
        attempt_log._ACTIVE_CALL.reset(live_handle.token)
        attempt_log._release_call_lock(live_handle.lock_file, None)  # keep lock file on disk, unlocked

    # Next invocation detects the unlocked lock file and marks that call interrupted too.
    code2, status2 = _run_cli(capsys, 'get_loop_status')
    assert code2 == 0
    by_id2 = {c['call_id']: c for c in status2['unfinished_tool_calls']}
    assert by_id2[live_handle.ctx.call_id]['status'] == 'interrupted'


def test_argument_redaction_and_bounded_storage(tmp_path, monkeypatch, capsys):
    env = _init_loop_repo(tmp_path, monkeypatch)
    big_script = '#!/bin/bash\n' + ('echo line\n' * 1000)

    # Dispatch dry-run submit_slurm_job with nested secret fields (which will fail TypeError
    # after start_tool_call sanitizes args) and also test valid dry-run with large script_content.
    code_bad, _ = _run_cli(capsys, 'submit_slurm_job', {
        'script_content': big_script,
        'dry_run': True,
        'api_token': 'ghp_super_secret_token_12345',
        'nested': {
            'password': 'my-password',
            'client_secret': 'top-secret',
            'Authorization': 'Bearer abcdef1234567890',
            'cookie': 'session=xyz',
            'private_key': '-----BEGIN RSA PRIVATE KEY-----\nSECRET\n-----END RSA PRIVATE KEY-----',
            'credentials': {'ssh_config': 'Host bgu'},
            'env_name': 'fem',  # must NOT be redacted
        },
    })
    assert code_bad == 2  # unexpected kwargs rejected by submit_slurm_job, but logged!

    code_ok, ok_out = _run_cli(capsys, 'submit_slurm_job', {
        'script_content': big_script,
        'dry_run': True,
        'name': 'dry-bounded',
    })
    assert code_ok == 0 and 'rendered' in ok_out

    conn = sqlite3.connect(env['db_path'])
    conn.row_factory = sqlite3.Row
    rows = conn.execute('SELECT * FROM tool_calls ORDER BY call_id').fetchall()
    arts = conn.execute("SELECT * FROM artifacts WHERE kind = 'slurm_script'").fetchall()
    conn.close()

    # Secrets must never appear anywhere in args_json, result_summary_json, or error_detail.
    for row in rows:
        blob = (row['args_json'] or '') + (row['result_summary_json'] or '') + (row['error_detail'] or '')
        for secret in ('ghp_super_secret_token_12345', 'my-password', 'top-secret',
                       'abcdef1234567890', 'session=xyz', 'RSA PRIVATE KEY', 'Host bgu'):
            assert secret not in blob
        # Raw huge script must never be embedded in DB rows.
        assert 'echo line\necho line\n' not in blob
        if row['result_summary_json']:
            assert len(row['result_summary_json'].encode('utf-8')) <= attempt_log.MAX_SUMMARY_BYTES

    # Legitimate 'env_name' was preserved, while secrets were replaced with [REDACTED].
    parsed_args = json.loads(rows[0]['args_json'])
    assert parsed_args['api_token'] == '[REDACTED]'
    assert parsed_args['nested']['password'] == '[REDACTED]'
    assert parsed_args['nested']['client_secret'] == '[REDACTED]'
    assert parsed_args['nested']['Authorization'] == '[REDACTED]'
    assert parsed_args['nested']['cookie'] == '[REDACTED]'
    assert parsed_args['nested']['private_key'] == '[REDACTED]'
    assert parsed_args['nested']['credentials'] == '[REDACTED]'
    assert parsed_args['nested']['env_name'] == 'fem'

    # script_content was replaced with sha256 + safe artifact path under .agent_loop/scripts/.
    script_meta = parsed_args['script_content']
    expected_sha = hashlib.sha256(big_script.encode('utf-8')).hexdigest()
    assert script_meta['sha256'] == expected_sha
    assert script_meta['artifact_path'].startswith('.agent_loop/scripts/')
    assert (env['repo'] / script_meta['artifact_path']).read_text() == big_script
    assert len(arts) == 2 and all(a['sha256'] == expected_sha for a in arts)


def test_slurm_metadata_logging_on_success_and_terminal_failure(tmp_path, monkeypatch, capsys):
    env = _init_loop_repo(tmp_path, monkeypatch)
    submitted = []
    sacct_info = {'state': 'COMPLETED', 'exit': '0:0'}
    monkeypatch.setattr(slurm, '_submit', lambda *a, **k: submitted.append(1) or 8000 + len(submitted))
    monkeypatch.setattr(slurm, '_wait', lambda job, c: None)
    monkeypatch.setattr(slurm, '_sacct', lambda job, c: (sacct_info['state'], sacct_info['exit'], 'ise-3090-02', '00:02:15'))
    monkeypatch.setattr(slurm, '_in_queue', lambda job, c: False)
    monkeypatch.setattr(slurm, 'checked_ssh', lambda *a, **k: type('R', (), {'stdout': 'ok__AGENT_LOG_SPLIT__'})())

    code, res = _run_cli(capsys, 'submit_slurm_job', {'script_content': 'echo hi', 'name': 'spmm-meta'})
    assert code == 0 and res['job'] == 8001

    # Simulate a recorded job that died with TIMEOUT while SSH was down: retry reports it (reattached=True).
    key = res['resume_key']
    state.update({'pending_jobs': {key: 8001}})
    sacct_info.update(state='TIMEOUT', exit='0:15')
    code2, res2 = _run_cli(capsys, 'submit_slurm_job', {'script_content': 'echo hi', 'name': 'spmm-meta'})
    assert code2 == 0
    assert res2['job'] == 8001 and res2['reattached'] is True and res2['state'] == 'TIMEOUT'
    assert submitted == [1]  # NOT resubmitted!
    assert state.read()['pending_jobs'] == {}

    conn = sqlite3.connect(env['db_path'])
    conn.row_factory = sqlite3.Row
    calls = conn.execute('SELECT * FROM tool_calls ORDER BY call_id').fetchall()
    arts = conn.execute("SELECT * FROM artifacts WHERE kind IN ('slurm_out', 'slurm_err')").fetchall()
    conn.close()
    assert calls[0]['slurm_job_id'] == 8001
    assert calls[0]['slurm_resume_key'] == key
    assert calls[0]['slurm_reattached'] == 0
    assert calls[1]['slurm_job_id'] == 8001
    assert calls[1]['slurm_reattached'] == 1
    assert len(arts) == 4  # out + err for both calls


def test_audit_failure_blocks_tool_execution(tmp_path, monkeypatch, capsys):
    env = _init_loop_repo(tmp_path, monkeypatch)
    # Place a directory where attempts.sqlite3 would be opened so SQLite cannot open it.
    env['db_path'].mkdir(parents=True, exist_ok=True)
    called = []
    monkeypatch.setattr(cli, 'dispatch', lambda tool, args: called.append((tool, args)) or {})
    code, out = _run_cli(capsys, 'commit_kernel_attempt', {'message': 'should not execute'})
    assert code == 3
    assert out['error'] == 'infrastructure'
    assert 'attempt database unavailable' in out['detail']
    assert called == []  # State-changing tool MUST NOT execute when audit record cannot be written.


def test_database_concurrency_and_single_active_constraint(tmp_path, monkeypatch):
    env = _init_loop_repo(tmp_path, monkeypatch)

    def try_begin(i: int):
        try:
            res = attempt_log.begin_attempt(
                title=f'Concurrent attempt {i}',
                hypothesis=f'Concurrent hypothesis {i} should enforce single active attempt.',
                db_path=env['db_path'],
                state_path=env['state_file'],
                repo=env['repo'],
            )
            return ('ok', res['ordinal'])
        except ArgumentError as exc:
            return ('refused', str(exc))

    with ThreadPoolExecutor(max_workers=6) as pool:
        outcomes = list(pool.map(try_begin, range(6)))

    oks = [o for o in outcomes if o[0] == 'ok']
    refused = [o for o in outcomes if o[0] == 'refused']
    assert len(oks) == 1 and oks[0][1] == 1
    assert len(refused) == 5

    # Concurrent tool_call start/finish + status reads must succeed without lock errors.
    def do_logged_work(i: int):
        h = attempt_log.start_tool_call(
            'get_loop_status', {'recent_limit': 5},
            db_path=env['db_path'], state_path=env['state_file'], repo=env['repo'],
        )
        st = attempt_log.get_loop_status(
            db_path=env['db_path'], state_path=env['state_file'], repo=env['repo'],
        )
        attempt_log.finish_tool_call(h, tool_name='get_loop_status', exit_code=0, result=st)
        return st['active_attempt']['ordinal']

    with ThreadPoolExecutor(max_workers=6) as pool:
        ordinals = list(pool.map(do_logged_work, range(12)))
    assert ordinals == [1] * 12


def test_export_json_and_markdown(tmp_path, monkeypatch, capsys):
    env = _init_loop_repo(tmp_path, monkeypatch)
    repo = env['repo']

    assert _run_cli(capsys, 'begin_attempt', {
        'title': 'Skip final tile barrier',
        'hypothesis': 'Skipping the trailing syncthreads on the last tile lowers barrier stall.',
    })[0] == 0
    (repo / 'torch_fast_spdmm/kernels/boo_kernel.cu').write_text('// optimized\n')
    c_code, c_res = _run_cli(capsys, 'commit_kernel_attempt', {'message': 'kernel: skip final barrier'})
    assert c_code == 0
    bench_path = _make_bench_file(
        repo, 'tests/torch_fast_spdmm/benchmark/results/agent_runs/opt1/24100/bench.json',
        cand=c_res['sha'], base=env['baseline'], ok=True,
        tune_geo=1.061, held_geo=1.018, tune_w=7, tune_l=0, tune_t=13, held_w=3, held_l=0, held_t=12,
    )
    state.update({'verified': c_res['sha']})
    cl_sha = _commit_changelog(repo, 'Skip final tile barrier')
    assert _run_cli(capsys, 'finish_attempt', {
        'outcome': 'PASS',
        'decision_reason': 'Improved tune 1.061x and held 1.018x.',
        'promoted': True,
        'rolled_back': False,
        'bench_json': str(bench_path),
        'changelog_sha': cl_sha,
    })[0] == 0

    code, exported = _run_cli(capsys, 'export_attempt_log', {})
    assert code == 0
    assert exported['attempts_exported'] == 1
    assert exported['json_path'].startswith(f"reports/agent_loop/runs/{exported['run_id']}/")
    assert exported['markdown_path'].startswith(f"reports/agent_loop/runs/{exported['run_id']}/")

    json_file = repo / exported['json_path']
    md_file = repo / exported['markdown_path']
    assert hashlib.sha256(json_file.read_bytes()).hexdigest() == exported['json_sha256']
    assert hashlib.sha256(md_file.read_bytes()).hexdigest() == exported['markdown_sha256']

    summary = json.loads(json_file.read_text())
    assert summary['schema'] == 'attempt_log.1'
    assert summary['run']['verified_moved'] is True
    assert summary['counts']['counted_attempts_used'] == 1
    assert summary['counts']['promoted_count'] == 1
    assert len(summary['attempts']) == 1
    # Must reference raw benchmark file rather than embedding raw CSV rows.
    assert 'role,mat,F,arm,time_ms' not in json_file.read_text()

    md = md_file.read_text()
    assert '| # | Title | sha | Outcome | tune `geo_x_base` (W/L/T) | held `geo_x_base` (W/L/T) | `geo_x_cusp` tune / held | Promoted? |' in md
    assert 'Skip final tile barrier' in md
    assert '1.061× (7/0/13)' in md
    assert '1.018× (3/0/12)' in md
    assert '| Yes |' in md


def test_finish_attempt_consistency_rejections(tmp_path, monkeypatch, capsys):
    env = _init_loop_repo(tmp_path, monkeypatch)
    repo = env['repo']
    assert _run_cli(capsys, 'begin_attempt', {
        'title': 'Consistency check attempt',
        'hypothesis': 'Testing finish_attempt consistency gates.',
    })[0] == 0

    (repo / 'torch_fast_spdmm/kernels/boo_kernel.cu').write_text('// change\n')
    c_code, c_res = _run_cli(capsys, 'commit_kernel_attempt', {'message': 'kernel: test'})
    assert c_code == 0
    assert _run_cli(capsys, 'rollback_workspace', {})[0] == 0

    # 1) Passing a non-CHANGELOG commit as changelog_sha must be rejected.
    bad_cl_code, bad_cl = _run_cli(capsys, 'finish_attempt', {
        'outcome': 'ROLLED BACK',
        'decision_reason': 'Flat.',
        'rolled_back': True,
        'changelog_sha': c_res['sha'],
    })
    assert bad_cl_code == 2 and 'did not modify CHANGELOG.md' in bad_cl['detail']

    # 2) Outcome FAIL (correctness) contradicting a passing tune_bench_json must be rejected.
    cl_sha = _commit_changelog(repo, 'Consistency check attempt')
    tune_path = _make_bench_file(
        repo, 'tests/torch_fast_spdmm/benchmark/results/agent_runs/t1/24200/bench.json',
        cand=c_res['sha'], base=env['baseline'], ok=True,
        tune_geo=1.005, held_geo=1.0, tune_w=1, tune_l=1, tune_t=18, held_w=0, held_l=0, held_t=0,
    )
    bad_corr_code, bad_corr = _run_cli(capsys, 'finish_attempt', {
        'outcome': 'FAIL (correctness)',
        'decision_reason': 'Claimed correctness fail.',
        'rolled_back': True,
        'tune_bench_json': str(tune_path),
        'changelog_sha': cl_sha,
    })
    assert bad_corr_code == 2 and 'contradicts bench.json' in bad_corr['detail']

    # 3) ABORTED with counts_toward_limit=False when tune_bench_json ran must be rejected.
    bad_abort_code, bad_abort = _run_cli(capsys, 'finish_attempt', {
        'outcome': 'ABORTED',
        'decision_reason': 'Aborted after tune bench.',
        'counts_toward_limit': False,
        'rolled_back': True,
        'tune_bench_json': str(tune_path),
    })
    assert bad_abort_code == 2 and 'benchmark ran' in bad_abort['detail']

    # 4) Valid ROLLED BACK with tune_bench_json succeeds and exports held as em-dash.
    ok_code, _ = _run_cli(capsys, 'finish_attempt', {
        'outcome': 'ROLLED BACK',
        'decision_reason': 'Stage 6a tune geo_x_base 1.005x < 1.02x.',
        'rolled_back': True,
        'tune_bench_json': str(tune_path),
        'changelog_sha': cl_sha,
    })
    assert ok_code == 0
    exp_code, exp = _run_cli(capsys, 'export_attempt_log', {})
    assert exp_code == 0
    md = (repo / exp['markdown_path']).read_text()
    assert '1.005× (1/1/18) | — |' in md


def test_run_benchmarks_logging_and_failure_cleanup(tmp_path, monkeypatch, capsys):
    env = _init_loop_repo(tmp_path, monkeypatch)
    repo = env['repo']
    assert _run_cli(capsys, 'begin_attempt', {
        'title': 'Benchmark logging test',
        'hypothesis': 'Verify benchmark tool call logging and cleanup.',
    })[0] == 0

    manifest_obj = {
        'matrices': {
            m: {'canonical_shape': [100, 100], 'canonical_nnz': 1000, 'nonempty_cols': 100}
            for m in ('web-Google', 'roadNet-CA', 'uniform16', 'blockdiag32')
        }
    }
    manifest_text = json.dumps(manifest_obj)
    expected_manifest_sha = hashlib.sha256(manifest_text.encode('utf-8')).hexdigest()

    monkeypatch.setattr(bench, 'ensure_build', lambda sha, config=None: {'ok': True, 'build_dir': '/r/b'})
    monkeypatch.setattr(bench, 'ssh', lambda cmd, config=None: type('R', (), {'returncode': 0, 'stdout': '', 'stderr': ''})())

    def fake_checked_ssh(cmd, config=None):
        if 'ls ' in cmd:
            return type('R', (), {'stdout': 'roadNet-CA_F128.json\n', 'stderr': '', 'returncode': 0})()
        if 'cat ' in cmd:
            return type('R', (), {'stdout': manifest_text, 'stderr': '', 'returncode': 0})()
        return type('R', (), {'stdout': 'ok__AGENT_LOG_SPLIT__', 'stderr': '', 'returncode': 0})()

    monkeypatch.setattr(bench, 'checked_ssh', fake_checked_ssh)

    # First: Slurm job ends in terminal FAILED -> run_benchmarks exits 3, releases pending_jobs,
    # and logs slurm_job_id + error in tool_calls.
    def fake_submit_fail(script, **kwargs):
        attempt_log.attach_slurm_info(
            job_id=9101, resume_key='rk_fail', reattached=True,
            out_path='/r/runs/9101.out', err_path='/r/runs/9101.err',
        )
        return {
            'job': 9101, 'state': 'FAILED', 'exit': 1, 'node': 'ise-3090-02',
            'elapsed_s': 12, 'out_tail': '', 'err_tail': 'CUDA illegal memory access',
            'out_path': '/r/runs/9101.out', 'err_path': '/r/runs/9101.err',
            'resume_key': 'rk_fail', 'reattached': True,
        }

    state.update({'pending_jobs': {'rk_fail': 9101}})
    monkeypatch.setattr(bench, 'submit_slurm_job', fake_submit_fail)
    code_fail, out_fail = _run_cli(capsys, 'run_benchmarks', {
        'kernel_name': 'boo:rowseg', 'matrix_datasets': ['roadNet-CA'], 'features': [128],
    })
    assert code_fail == 3
    assert '9101' in out_fail['detail'] and 'FAILED' in out_fail['detail']
    assert state.read()['pending_jobs'] == {}

    # Second: Slurm job succeeds -> records suite_manifest_sha256, bench_json/samples.csv artifacts.
    def fake_submit_ok(script, **kwargs):
        attempt_log.attach_slurm_info(
            job_id=9102, resume_key='rk_ok', reattached=False,
            out_path='/r/runs/9102.out', err_path='/r/runs/9102.err',
        )
        return {
            'job': 9102, 'state': 'COMPLETED', 'exit': 0, 'node': 'ise-3090-02',
            'elapsed_s': 30, 'out_tail': 'ok', 'err_tail': '',
            'out_path': '/r/runs/9102.out', 'err_path': '/r/runs/9102.err',
            'resume_key': 'rk_ok', 'reattached': False,
        }

    def fake_pull_dir(kind, src, dst, config=None):
        dst = Path(dst)
        dst.mkdir(parents=True, exist_ok=True)
        (dst / 'samples.csv').write_text(
            'role,mat,F,arm,time_ms\n'
            'cand,roadNet-CA,128,kernel,1.0\n'
            'cand,roadNet-CA,128,kernel_dup,1.01\n'
            'cand,roadNet-CA,128,kernel_api,1.1\n'
            'cand,roadNet-CA,128,cusparse,2.0\n'
            'base,roadNet-CA,128,kernel,1.1\n'
            'base,roadNet-CA,128,cusparse,2.01\n'
        )
        (dst / 'summary.csv').write_text(
            'role,mat,F,arm,status,reason,mem_mib,prep_ms\n'
            'cand,roadNet-CA,128,kernel,measured,,100,10\n'
            'cand,roadNet-CA,128,kernel_dup,measured,,100,10\n'
            'cand,roadNet-CA,128,kernel_api,measured,,100,10\n'
            'cand,roadNet-CA,128,cusparse,measured,,100,10\n'
            'base,roadNet-CA,128,kernel,measured,,100,10\n'
            'base,roadNet-CA,128,cusparse,measured,,100,10\n'
        )

    monkeypatch.setattr(bench, 'submit_slurm_job', fake_submit_ok)
    monkeypatch.setattr(bench, 'transfer', fake_pull_dir)
    code_ok, out_ok = _run_cli(capsys, 'run_benchmarks', {
        'kernel_name': 'boo:rowseg', 'matrix_datasets': ['roadNet-CA'], 'features': [128],
    })
    assert code_ok == 0 and out_ok['v'] == 'bench.1' and out_ok['ok'] is True

    conn = sqlite3.connect(env['db_path'])
    conn.row_factory = sqlite3.Row
    run_row = conn.execute('SELECT * FROM runs').fetchone()
    calls = conn.execute('SELECT * FROM tool_calls ORDER BY call_id').fetchall()
    arts = conn.execute('SELECT * FROM artifacts ORDER BY artifact_id').fetchall()
    conn.close()

    assert run_row['suite_manifest_sha256'] == expected_manifest_sha
    bench_calls = [c for c in calls if c['tool_name'] == 'run_benchmarks']
    assert len(bench_calls) == 2
    assert bench_calls[0]['status'] == 'failure' and bench_calls[0]['slurm_job_id'] == 9101 and bench_calls[0]['slurm_reattached'] == 1
    assert bench_calls[1]['status'] == 'success' and bench_calls[1]['slurm_job_id'] == 9102 and bench_calls[1]['slurm_reattached'] == 0
    # Full rows array must NOT be duplicated in result_summary_json.
    summary = json.loads(bench_calls[1]['result_summary_json'])
    assert 'rows' not in summary and summary['row_count'] == 1
    art_kinds = {a['kind'] for a in arts}
    assert {'bench_json', 'bench_raw_csv', 'slurm_out', 'slurm_err'} <= art_kinds
    bench_art = next(a for a in arts if a['kind'] == 'bench_json')
    assert bench_art['sha256'] == hashlib.sha256((repo / bench_art['local_path']).read_bytes()).hexdigest()


def test_sigkill_process_interruption_recovery(tmp_path, monkeypatch, capsys):
    """A real subprocess killed with SIGKILL mid-call is preserved and marked interrupted."""
    import signal
    import sys
    import time

    env = _init_loop_repo(tmp_path, monkeypatch)
    ready = tmp_path / 'ready.txt'
    child_code = (
        'import pathlib, time; '
        'from optimizer.agent_tools import attempt_log; '
        f'h = attempt_log.start_tool_call("submit_slurm_job", {{"name": "killed-job"}}, '
        f'db_path=pathlib.Path({str(env["db_path"])!r}), '
        f'state_path=pathlib.Path({str(env["state_file"])!r}), '
        f'repo=pathlib.Path({str(env["repo"])!r})); '
        f'pathlib.Path({str(ready)!r}).write_text(str(h.ctx.call_id)); '
        'time.sleep(60)'
    )
    proc = subprocess.Popen(
        [sys.executable, '-c', child_code],
        cwd=Path(__file__).resolve().parents[2],
    )
    try:
        deadline = time.monotonic() + 10.0
        while not ready.exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        assert ready.exists(), 'child did not start tool call in time'
        killed_call_id = int(ready.read_text().strip())

        # While child is alive and holding its lock, get_loop_status sees status == 'running'.
        c1, st1 = _run_cli(capsys, 'get_loop_status')
        assert c1 == 0
        by_id1 = {c['call_id']: c for c in st1['unfinished_tool_calls']}
        assert by_id1[killed_call_id]['status'] == 'running'

        os.kill(proc.pid, signal.SIGKILL)
        proc.wait(timeout=5)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5)

    # After SIGKILL, the next invocation marks the call 'interrupted' without deleting it.
    c2, st2 = _run_cli(capsys, 'get_loop_status')
    assert c2 == 0
    by_id2 = {c['call_id']: c for c in st2['unfinished_tool_calls']}
    assert by_id2[killed_call_id]['status'] == 'interrupted'
    assert by_id2[killed_call_id]['error_class'] == 'ProcessInterrupted'


def test_guarded_tools_require_active_attempt(tmp_path, monkeypatch, capsys):
    env = _init_loop_repo(tmp_path, monkeypatch)
    dispatched = []
    real_dispatch = cli.dispatch

    def spy_dispatch(tool, arguments):
        dispatched.append(tool)
        return real_dispatch(tool, arguments)

    monkeypatch.setattr(cli, 'dispatch', spy_dispatch)

    guarded = [
        ('commit_kernel_attempt', {'message': 'kernel: unauthorized'}),
        ('rollback_workspace', {}),
        ('run_benchmarks', {'kernel_name': 'boo:rowseg', 'matrix_datasets': ['tune']}),
        ('promote_attempt', {'bench_json': 'bench.json'}),
    ]
    for tool, args in guarded:
        code, res = _run_cli(capsys, tool, args)
        assert code == 2
        assert 'requires an active attempt' in res['detail']
    assert dispatched == []

    # Setup/status/baseline-profile tools MAY run without an active attempt.
    assert _run_cli(capsys, 'get_loop_status')[0] == 0
    assert _run_cli(capsys, 'submit_slurm_job', {'script_content': 'echo baseline', 'dry_run': True})[0] == 0
    assert _run_cli(capsys, 'export_attempt_log', {})[0] == 0

    conn = sqlite3.connect(env['db_path'])
    conn.row_factory = sqlite3.Row
    failed_rows = conn.execute(
        "SELECT tool_name, status, exit_code, attempt_id FROM tool_calls WHERE status = 'failure' ORDER BY call_id"
    ).fetchall()
    conn.close()
    assert [(r['tool_name'], r['exit_code'], r['attempt_id']) for r in failed_rows] == [
        (tool, 2, None) for tool, _ in guarded
    ]


def test_clean_verified_starting_point_and_contamination(tmp_path, monkeypatch, capsys):
    env = _init_loop_repo(tmp_path, monkeypatch)
    repo = env['repo']
    (repo / 'tests/torch_fast_spdmm/test_boo.py').write_text('def test_ok(): pass\n')
    _git(repo, 'add', 'tests/torch_fast_spdmm/test_boo.py')
    _git(repo, 'commit', '-m', 'add baseline test')
    new_base = _git(repo, 'rev-parse', 'HEAD')
    state.update({'verified': new_base})
    _git(repo, 'tag', '-f', 'opt/verified', new_base)
    _git(repo, 'push', 'origin', 'opt/loop')

    # 1) Wrong branch / main must block begin_attempt.
    _git(repo, 'checkout', '-b', 'main')
    code_main, res_main = _run_cli(capsys, 'begin_attempt', {
        'title': 'On main', 'hypothesis': 'Should fail on main branch.',
    })
    assert code_main == 2 and 'loop_branch' in res_main['detail']
    _git(repo, 'checkout', 'opt/loop')

    # 2) Dirty edit in torch_fast_spdmm/ or allowed test_*.py blocks begin_attempt.
    (repo / 'torch_fast_spdmm/kernels/boo_kernel.cu').write_text('// dirty edit\n')
    c_dirty, r_dirty = _run_cli(capsys, 'begin_attempt', {
        'title': 'Dirty start', 'hypothesis': 'Should fail with dirty kernel file.',
    })
    assert c_dirty == 2 and 'uncommitted changes' in r_dirty['detail']
    _git(repo, 'checkout', '--', 'torch_fast_spdmm/kernels/boo_kernel.cu')

    # 3) Unrolled-back committed candidate at HEAD blocks begin_attempt.
    (repo / 'torch_fast_spdmm/kernels/boo_kernel.cu').write_text('// unrolled-back\n')
    _git(repo, 'add', 'torch_fast_spdmm/kernels/boo_kernel.cu')
    _git(repo, 'commit', '-m', 'unrolled-back commit')
    c_drift, r_drift = _run_cli(capsys, 'begin_attempt', {
        'title': 'Drifted start', 'hypothesis': 'Should fail when HEAD != verified.',
    })
    assert c_drift == 2 and 'HEAD differs from state.verified' in r_drift['detail']
    _git(repo, 'reset', '--hard', new_base)

    # 4) Docs-only CHANGELOG commit after promotion is ALLOWED by begin_attempt.
    (repo / 'CHANGELOG.md').write_text('# Changelog\n\n### Prior entry\n')
    _git(repo, 'add', 'CHANGELOG.md')
    _git(repo, 'commit', '-m', 'docs: changelog entry')
    c_ok, _ = _run_cli(capsys, 'begin_attempt', {
        'title': 'Contamination attempt',
        'hypothesis': 'Testing abort contamination guards.',
    })
    assert c_ok == 0

    # 5) Abort after dirty edit must fail.
    (repo / 'torch_fast_spdmm/kernels/boo_kernel.cu').write_text('// dirty during attempt\n')
    c_ab_dirty, r_ab_dirty = _run_cli(capsys, 'finish_attempt', {
        'outcome': 'ABORTED',
        'decision_reason': 'Trying to abort with dirty tree.',
        'counts_toward_limit': False,
    })
    assert c_ab_dirty == 2 and 'uncommitted changes' in r_ab_dirty['detail']

    # 6) Abort after candidate commit must fail (before rollback AND non-counting after rollback).
    assert _run_cli(capsys, 'commit_kernel_attempt', {'message': 'kernel: committed edit'})[0] == 0
    c_ab_comm, r_ab_comm = _run_cli(capsys, 'finish_attempt', {
        'outcome': 'ABORTED',
        'decision_reason': 'Trying to abort after commit without rollback.',
        'counts_toward_limit': False,
    })
    assert c_ab_comm == 2 and 'differ from baseline' in r_ab_comm['detail']

    assert _run_cli(capsys, 'rollback_workspace', {})[0] == 0
    c_ab_noncount, r_ab_noncount = _run_cli(capsys, 'finish_attempt', {
        'outcome': 'ABORTED',
        'decision_reason': 'Trying non-counting abort after a committed candidate.',
        'counts_toward_limit': False,
        'rolled_back': True,
    })
    assert c_ab_noncount == 2 and 'candidate commit was made' in r_ab_noncount['detail']


def test_close_run_lifecycle_and_future_new_run(tmp_path, monkeypatch, capsys):
    env = _init_loop_repo(tmp_path, monkeypatch)
    repo = env['repo']

    # Start attempt 1 -> close_run must refuse while an attempt is active.
    assert _run_cli(capsys, 'begin_attempt', {
        'title': 'Early-stop attempt 1',
        'hypothesis': 'Single attempt before early-stop campaign close.',
    })[0] == 0
    c_act, r_act = _run_cli(capsys, 'close_run', {'status': 'completed', 'reason': 'premature'})
    assert c_act == 2 and 'is active' in r_act['detail']

    # Close attempt 1 cleanly (early-stop after 1 counted attempt).
    (repo / 'torch_fast_spdmm/kernels/boo_kernel.cu').write_text('// a1\n')
    assert _run_cli(capsys, 'commit_kernel_attempt', {'message': 'kernel: a1'})[0] == 0
    assert _run_cli(capsys, 'rollback_workspace', {})[0] == 0
    cl_sha = _commit_changelog(repo, 'Early-stop attempt 1')
    assert _run_cli(capsys, 'finish_attempt', {
        'outcome': 'ROLLED BACK',
        'decision_reason': 'Flat on tune.',
        'rolled_back': True,
        'changelog_sha': cl_sha,
    })[0] == 0

    # Inject a live unfinished running call -> close_run(completed) must refuse.
    live = attempt_log.start_tool_call('submit_slurm_job', {'name': 'still-running'})
    try:
        c_unf, r_unf = _run_cli(capsys, 'close_run', {'status': 'completed', 'reason': 'done'})
        assert c_unf == 2 and 'live running tool calls' in r_unf['detail']
    finally:
        attempt_log.finish_tool_call(live, tool_name='submit_slurm_job', exit_code=0, result={'job': 1})

    # Early-stop completion succeeds now, records ended_at & close_reason.
    c_close, closed = _run_cli(capsys, 'close_run', {
        'status': 'completed',
        'reason': 'Early stop after 1 attempt per review decision.',
    })
    assert c_close == 0
    run1_id = closed['run_id']
    assert closed['status'] == 'completed'
    assert closed['ended_at']
    assert closed['counted_attempts_used'] == 1

    # New attempts on the closed run must be refused (with or without explicit run_id).
    c_no1, r_no1 = _run_cli(capsys, 'begin_attempt', {
        'title': 'After close', 'hypothesis': 'Should fail on closed run.',
    })
    assert c_no1 == 2 and 'is closed' in r_no1['detail']
    c_no2, r_no2 = _run_cli(capsys, 'begin_attempt', {
        'title': 'Explicit closed', 'hypothesis': 'Should fail on explicit closed run.',
        'run_id': run1_id,
    })
    assert c_no2 == 2 and 'is closed' in r_no2['detail']

    # Export immediately after close_run exports the CLOSED run with closed status, end time, close_reason.
    c_exp, exp = _run_cli(capsys, 'export_attempt_log', {})
    assert c_exp == 0 and exp['run_id'] == run1_id
    exported_json = json.loads((repo / exp['json_path']).read_text())
    exported_md = (repo / exp['markdown_path']).read_text()
    assert exported_json['run']['status'] == 'completed'
    assert exported_json['run']['ended_at'] == closed['ended_at']
    assert exported_json['run']['close_reason'] == 'Early stop after 1 attempt per review decision.'
    assert f"**Ended:** `{closed['ended_at']}` · **Status:** `completed`" in exported_md
    assert 'Early stop after 1 attempt per review decision.' in exported_md

    # Verify close_run call remained associated with run1_id's audit log.
    conn = sqlite3.connect(env['db_path'])
    conn.row_factory = sqlite3.Row
    close_calls = conn.execute(
        "SELECT run_id, status FROM tool_calls WHERE tool_name = 'close_run' AND status = 'success'"
    ).fetchall()
    conn.close()
    assert len(close_calls) == 1 and close_calls[0]['run_id'] == run1_id

    # Begin a NEW run after run1 is closed -> new run_id, ordinal=1, but the budget is shared
    # across the campaign ledger: run1 used 1 counted attempt, so only 4 remain (not a fresh 5).
    c_new, new_att = _run_cli(capsys, 'begin_attempt', {
        'title': 'Campaign 2 attempt 1',
        'hypothesis': 'Fresh campaign after closing previous run.',
        'new_run': True,
    })
    assert c_new == 0
    run2_id = new_att['run_id']
    run2_att_id = new_att['attempt_id']
    assert run2_id != run1_id and new_att['ordinal'] == 1 and new_att['counted_attempts_remaining'] == 4

    # Verify the begin_attempt(new_run=True) tool_call row belongs to run2_id and run2_att_id,
    # appears in run2's export, does NOT appear in run1's export, and cross-run FKs are rejected.
    conn = sqlite3.connect(env['db_path'])
    conn.execute('PRAGMA foreign_keys = ON')
    conn.row_factory = sqlite3.Row
    new_begin_row = conn.execute(
        "SELECT call_id, run_id, attempt_id FROM tool_calls WHERE tool_name = 'begin_attempt' AND status = 'success' ORDER BY call_id DESC LIMIT 1"
    ).fetchone()
    assert new_begin_row['run_id'] == run2_id
    assert new_begin_row['attempt_id'] == run2_att_id
    new_begin_call_id = int(new_begin_row['call_id'])
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            """
            INSERT INTO tool_calls (
                run_id, attempt_id, tool_name, args_json, args_sha256,
                head_sha, verified_sha, pid, started_at, status
            ) VALUES (?, ?, 'get_loop_status', '{}', 'x', ?, ?, 1, '2026-11-19T00:00:00Z', 'success')
            """,
            (run1_id, run2_att_id, env['baseline'], env['baseline']),
        )
    conn.close()

    c_exp1, exp1 = _run_cli(capsys, 'export_attempt_log', {'run_id': run1_id})
    c_exp2, exp2 = _run_cli(capsys, 'export_attempt_log', {'run_id': run2_id})
    assert c_exp1 == 0 and c_exp2 == 0
    run1_summary = json.loads((repo / exp1['json_path']).read_text())
    run2_summary = json.loads((repo / exp2['json_path']).read_text())
    run1_call_ids = {c['call_id'] for c in run1_summary['setup_tool_calls']} | {
        c['call_id'] for a in run1_summary['attempts'] for c in a['tool_calls']
    }
    run2_att_call_ids = {
        c['call_id'] for a in run2_summary['attempts'] for c in a['tool_calls']
    }
    assert new_begin_call_id not in run1_call_ids
    assert new_begin_call_id in run2_att_call_ids

    # Close attempt 1 as clean non-counting ABORTED, then close campaign 2 as 'aborted' (even with interrupted call).
    assert _run_cli(capsys, 'finish_attempt', {
        'outcome': 'ABORTED',
        'decision_reason': 'Aborted before editing code.',
        'counts_toward_limit': False,
    })[0] == 0
    conn = sqlite3.connect(env['db_path'])
    conn.execute(
        """
        INSERT INTO tool_calls (
            run_id, attempt_id, tool_name, args_json, args_sha256,
            head_sha, verified_sha, pid, started_at, status
        ) VALUES (?, NULL, 'submit_slurm_job', '{}', 'dead', ?, ?, 99999998, '2026-11-19T00:00:00Z', 'running')
        """,
        (run2_id, env['baseline'], env['baseline']),
    )
    conn.commit()
    conn.close()
    c_ab, ab = _run_cli(capsys, 'close_run', {
        'status': 'aborted',
        'reason': 'Cluster outage exceeded 60 minutes.',
    })
    assert c_ab == 0 and ab['run_id'] == run2_id and ab['status'] == 'aborted' and ab['ended_at']


def test_close_run_completed_with_interrupted_call_and_retry(tmp_path, monkeypatch, capsys):
    env = _init_loop_repo(tmp_path, monkeypatch)
    repo = env['repo']

    c_beg, att = _run_cli(capsys, 'begin_attempt', {
        'title': 'Retry after crash',
        'hypothesis': 'Interrupted call recovered and retried should not block completed closure.',
    })
    assert c_beg == 0
    run_id, att_id = att['run_id'], att['attempt_id']

    # Create a crashed/dead 'running' tool call -> next CLI call marks it 'interrupted'.
    conn = sqlite3.connect(env['db_path'])
    cur = conn.execute(
        """
        INSERT INTO tool_calls (
            run_id, attempt_id, tool_name, args_json, args_sha256,
            head_sha, verified_sha, pid, started_at, status, slurm_job_id
        ) VALUES (?, ?, 'submit_slurm_job', '{}', 'crash', ?, ?, 99999997, '2026-11-19T00:00:00Z', 'running', 77777)
        """,
        (run_id, att_id, env['baseline'], env['baseline']),
    )
    interrupted_call_id = cur.lastrowid
    conn.commit()
    conn.close()

    # Record a successful retry/replacement tool call, then close the attempt cleanly.
    assert _run_cli(capsys, 'submit_slurm_job', {'script_content': 'echo retry ok', 'dry_run': True})[0] == 0
    (repo / 'torch_fast_spdmm/kernels/boo_kernel.cu').write_text('// tried\n')
    assert _run_cli(capsys, 'commit_kernel_attempt', {'message': 'kernel: tried'})[0] == 0
    assert _run_cli(capsys, 'rollback_workspace', {})[0] == 0
    cl_sha = _commit_changelog(repo, 'Retry after crash')
    assert _run_cli(capsys, 'finish_attempt', {
        'outcome': 'ROLLED BACK',
        'decision_reason': 'Measured after retry; no gain.',
        'rolled_back': True,
        'changelog_sha': cl_sha,
    })[0] == 0

    # Interrupted historical call must NOT block close_run(status="completed").
    c_close, closed = _run_cli(capsys, 'close_run', {
        'status': 'completed',
        'reason': 'Completed cleanly after recovering from an interrupted call.',
    })
    assert c_close == 0 and closed['status'] == 'completed' and closed['ended_at']

    # Export preserves the interrupted call and reports interrupted_call_count == 1.
    c_exp, exp = _run_cli(capsys, 'export_attempt_log', {})
    assert c_exp == 0 and exp['run_id'] == run_id
    summary = json.loads((repo / exp['json_path']).read_text())
    md = (repo / exp['markdown_path']).read_text()
    assert summary['run']['status'] == 'completed'
    assert summary['counts']['interrupted_call_count'] == 1
    assert any(
        c['call_id'] == interrupted_call_id and c['status'] == 'interrupted'
        for c in summary['unfinished_tool_calls']
    )
    assert '- **Interrupted Tool Calls:** 1' in md
    assert f'call `{interrupted_call_id}`' in md


def test_suite_manifest_drift_raises_integrity_error(tmp_path, monkeypatch, capsys):
    env = _init_loop_repo(tmp_path, monkeypatch)
    assert _run_cli(capsys, 'begin_attempt', {
        'title': 'Manifest drift test',
        'hypothesis': 'Detect mid-run suite manifest modification.',
    })[0] == 0

    manifest_state = {'text': json.dumps({'version': 1, 'matrices': {'roadNet-CA': {'canonical_shape': [10, 10], 'canonical_nnz': 20, 'nonempty_cols': 10}}})}
    submitted = []
    monkeypatch.setattr(bench, 'ensure_build', lambda sha, config=None: {'ok': True, 'build_dir': '/r/b'})
    monkeypatch.setattr(bench, 'ssh', lambda cmd, config=None: type('R', (), {'returncode': 0, 'stdout': '', 'stderr': ''})())

    def fake_checked_ssh(cmd, config=None):
        if 'ls ' in cmd:
            return type('R', (), {'stdout': 'roadNet-CA_F128.json\n', 'stderr': '', 'returncode': 0})()
        if 'cat ' in cmd:
            return type('R', (), {'stdout': manifest_state['text'], 'stderr': '', 'returncode': 0})()
        return type('R', (), {'stdout': 'ok__AGENT_LOG_SPLIT__', 'stderr': '', 'returncode': 0})()

    monkeypatch.setattr(bench, 'checked_ssh', fake_checked_ssh)

    def fake_submit(script, **kwargs):
        submitted.append(1)
        return {
            'job': 9200, 'state': 'FAILED', 'exit': 1, 'node': 'n1', 'elapsed_s': 1,
            'out_tail': '', 'err_tail': 'stop early', 'out_path': '/o', 'err_path': '/e',
            'resume_key': 'rk', 'reattached': False,
        }

    monkeypatch.setattr(bench, 'submit_slurm_job', fake_submit)

    # First call records initial manifest SHA-256 on the run (even though job fails later).
    c1, _ = _run_cli(capsys, 'run_benchmarks', {
        'kernel_name': 'boo:rowseg', 'matrix_datasets': ['roadNet-CA'], 'features': [128],
    })
    assert c1 == 3 and submitted == [1]

    # Mutate remote manifest.json -> second call MUST abort with suite manifest drift before submitting!
    manifest_state['text'] = json.dumps({'version': 2, 'matrices': {'roadNet-CA': {'canonical_shape': [10, 10], 'canonical_nnz': 25, 'nonempty_cols': 10}}})
    c2, r2 = _run_cli(capsys, 'run_benchmarks', {
        'kernel_name': 'boo:rowseg', 'matrix_datasets': ['roadNet-CA'], 'features': [128],
    })
    assert c2 == 3
    assert 'suite manifest drift' in r2['detail']
    assert submitted == [1]  # Must NOT submit a Slurm job after detecting drift!


def test_embedded_script_secret_omitted_and_owner_only_permissions(tmp_path, monkeypatch, capsys):
    env = _init_loop_repo(tmp_path, monkeypatch)
    secret_script = (
        '#!/bin/bash\n'
        'export GITHUB_TOKEN="ghp_1234567890abcdefghijklmnop"\n'
        'curl -H "Authorization: Bearer super_secret_bearer_token_999" https://example.com\n'
        'echo done\n'
    )
    benign_script = '#!/bin/bash\necho hello_benign_world\n'

    c1, _ = _run_cli(capsys, 'submit_slurm_job', {'script_content': secret_script, 'dry_run': True})
    c2, _ = _run_cli(capsys, 'submit_slurm_job', {'script_content': benign_script, 'dry_run': True})
    assert c1 == 0 and c2 == 0

    conn = sqlite3.connect(env['db_path'])
    conn.row_factory = sqlite3.Row
    rows = conn.execute('SELECT args_json, result_summary_json FROM tool_calls ORDER BY call_id').fetchall()
    arts = conn.execute("SELECT * FROM artifacts WHERE kind = 'slurm_script'").fetchall()
    conn.close()

    # Sensitive script MUST NOT be written to .agent_loop/scripts/ and MUST NOT leak into DB.
    a1 = json.loads(rows[0]['args_json'])['script_content']
    assert a1['redacted_type'] == 'script_omitted_sensitive'
    assert a1['artifact_path'] is None
    assert a1['sha256'] == hashlib.sha256(secret_script.encode('utf-8')).hexdigest()
    assert a1['bytes'] == len(secret_script.encode('utf-8'))
    assert 'ghp_1234567890' not in rows[0]['args_json']
    assert 'super_secret_bearer_token_999' not in rows[0]['args_json']

    scripts_dir = env['db_path'].parent / 'scripts'
    saved_files = list(scripts_dir.glob('*.sh'))
    assert len(saved_files) == 1  # ONLY the benign script was persisted.
    assert 'ghp_1234567890' not in saved_files[0].read_text()
    assert len(arts) == 1

    # Owner-only permissions on DB, scripts dir, saved script, and state.json.
    assert (env['db_path'].stat().st_mode & 0o777) == 0o600
    assert (env['db_path'].parent.stat().st_mode & 0o777) == 0o700
    assert (scripts_dir.stat().st_mode & 0o777) == 0o700
    assert (saved_files[0].stat().st_mode & 0o777) == 0o600
    assert (env['state_file'].stat().st_mode & 0o777) == 0o600


