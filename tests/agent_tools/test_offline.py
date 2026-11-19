"""Offline contracts; no SSH, torch or GPU is imported by these tests."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import shutil
import subprocess

from optimizer.agent_tools import state
from optimizer.agent_tools.bench import COLS, aggregate
from optimizer.agent_tools.ncu import parse_ncu_profile, FLOP
from optimizer.agent_tools.schema import validate
from optimizer.agent_tools.slurm import submit_slurm_job
from optimizer.agent_tools.vcs import commit_kernel_attempt, rollback_workspace

FIXTURES = Path(__file__).parent / 'fixtures'


def test_schemas_examples():
    root = state.ROOT/'reports/agent_loop/schemas'
    for kind in ('bench', 'ncu'):
        schema = json.loads((root/f'{kind}.v1.json').read_text())
        value = json.loads((root/'examples'/f'{kind}.example.json').read_text())
        validate(value, schema)


def test_ncu_fixture(tmp_path):
    source = tmp_path/'fp512_win_raw.csv'
    shutil.copyfile(FIXTURES/source.name, source)
    source.with_name('fp512_win.case.json').write_text(json.dumps({
        'mat': 'cant', 'F': 512, 'fmt': 'boo', 'kernel': 'rowseg', 'commit': 'a079cba'}))
    result = parse_ncu_profile(str(source))
    assert result['k'][0]['roof']['src'] == 'ffma_only'
    assert result['k'][0]['roof']['ridge_ai'] == 32.1
    assert set(result['missing']) == set(FLOP)
    assert len(result['k']) == 1


def test_ncu_multiple_kernels(tmp_path):
    source = tmp_path/'dd512_win_raw.csv'
    shutil.copyfile(FIXTURES/source.name, source)
    source.with_name('dd512_win.case.json').write_text(json.dumps({
        'mat': 'cant', 'F': 512, 'fmt': 'boo', 'kernel': 'rowseg', 'commit': 'a079cba'}))
    result = parse_ncu_profile(str(source))
    assert len(result['k']) >= 2
    assert result['k'][0]['share'] >= result['k'][1]['share']
    assert abs(sum(k['share'] for k in result['k']) - 1) < .01


def test_bench_aggregation(tmp_path):
    sample = tmp_path/'samples.csv'
    summary = tmp_path/'summary.csv'
    with sample.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['role','mat','F','arm','time_ms'])
        writer.writeheader()
        for mat, candidate, baseline in [('cant', 1., 1.1), ('soc-Pokec', 2., 1.8)]:
            for role, arm, value in [('cand','kernel',candidate), ('cand','kernel_dup',candidate*1.01),
                                     ('cand','kernel_api',candidate*1.1), ('cand','cusparse',3.),
                                     ('base','kernel',baseline), ('base','cusparse',3.03)]:
                for _ in range(3):
                    writer.writerow(dict(role=role, mat=mat, F=128, arm=arm, time_ms=value))
    with summary.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['role','mat','F','arm','status','reason','mem_mib','prep_ms'])
        writer.writeheader()
        for mat in ('cant','soc-Pokec'):
            for role in ('cand','base'):
                for arm in ('kernel','kernel_dup','kernel_api','cusparse'):
                    writer.writerow(dict(role=role,mat=mat,F=128,arm=arm,status='measured',
                                         reason='',mem_mib=100,prep_ms=10))
    manifest = {'matrices': {mat: {'canonical_shape':[100,100], 'canonical_nnz':1000,
                                  'nonempty_cols':100} for mat in ('cant','soc-Pokec')}}
    # aggregate's raw path is deliberately repo-relative.
    target = state.ROOT/'tests/torch_fast_spdmm/benchmark/results/agent_runs/offline'
    target.mkdir(parents=True, exist_ok=True)
    local_sample, local_summary = target/'samples.csv', target/'summary.csv'
    shutil.copyfile(sample, local_sample)
    shutil.copyfile(summary, local_summary)
    result = aggregate(local_sample, local_summary, manifest, fmt='boo',kernel='rowseg',
                       cand='a079cba',base='a079cba',job=1,node='test',reps=3,warmup=0,rounds=1)
    assert result['noise'] == .01
    assert [r[11] for r in result['rows']] == ['W','L']
    assert result['rows'][0][7] == 1.1
    assert result['sum']['tune']['geo_x_base'] == 1.1
    assert result['sum']['held']['geo_x_base'] == .9
    with local_summary.open('a', newline='') as f:
        csv.writer(f).writerow(['cand','cant',128,'kernel','failed','numerical',100,10])
    failed = aggregate(local_sample, local_summary, manifest, fmt='boo',kernel='rowseg',
                       cand='a079cba',base='a079cba',job=1,node='test',reps=3,warmup=0,rounds=1)
    assert not failed['ok'] and failed['sum']['fail'] == 1


def test_slurm_dry_run():
    config = dict(state.DEFAULT, preamble='echo ENV_READY', verified=None)
    result = submit_slurm_job('echo {{RUN_DIR}}', dry_run=True, config=config)
    script = result['rendered']
    assert '--exclude=ise-3090-01,ise-3090-04' in script
    assert 'ENV_READY' in script
    assert '{{' not in script


def _git(repo, *args):
    return subprocess.run(['git',*args], cwd=repo, capture_output=True, text=True, check=True).stdout.strip()


def test_commit_and_rollback(tmp_path):
    repo = tmp_path/'repo'
    origin = tmp_path/'origin.git'
    _git(tmp_path, 'init', '--bare', str(origin))
    _git(tmp_path, 'init', str(repo))
    _git(repo, 'config', 'user.name', 'Offline Test')
    _git(repo, 'config', 'user.email', 'offline@example.com')
    _git(repo, 'remote', 'add', 'origin', str(origin))
    required = ['optimizer/agent_tools/__main__.py',
                'tests/torch_fast_spdmm/benchmark/universal_suite.py',
                'tests/torch_fast_spdmm/benchmark/agent_bench.py',
                'tests/torch_fast_spdmm/benchmark/agent_profile.py',
                'tests/torch_fast_spdmm/conftest.py',
                'tests/torch_fast_spdmm/test_utils.py',
                'tests/torch_fast_spdmm/test_boo.py',
                'torch_fast_spdmm/kernels/kernel.cu', 'CHANGELOG.md']
    for path in required:
        file = repo/path
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text('base\n')
    _git(repo, 'add', '.')
    _git(repo, 'commit', '-m', 'tools baseline')
    baseline = _git(repo, 'rev-parse', 'HEAD')
    _git(repo, 'branch', '-M', 'opt/loop-20261118')
    _git(repo, 'push', '-u', 'origin', 'opt/loop-20261118')

    # Advance harness in a separate tooling commit (simulating opt/verified behind HEAD on harness).
    (repo/'tests/torch_fast_spdmm/benchmark/agent_bench.py').write_text('newer harness\n')
    _git(repo, 'add', 'tests/torch_fast_spdmm/benchmark/agent_bench.py')
    _git(repo, 'commit', '-m', 'tools: newer benchmark harness')
    _git(repo, 'push', 'origin', 'opt/loop-20261118')

    config = dict(state.DEFAULT, loop_branch='opt/loop-20261118', verified=baseline)
    # Allowed attempt files:
    (repo/'torch_fast_spdmm/kernels/kernel.cu').write_text('candidate\n')
    (repo/'tests/torch_fast_spdmm/test_boo.py').write_text('base\nextra test\n')
    (repo/'tests/torch_fast_spdmm/test_added.py').write_text('new test\n')
    (repo/'CHANGELOG.md').write_text('unrelated log\n')
    # Disallowed harness / outside files (must NEVER be staged by commit_kernel_attempt):
    (repo/'optimizer/agent_tools/__main__.py').write_text('dirty optimizer\n')
    (repo/'tests/torch_fast_spdmm/benchmark/agent_bench.py').write_text('dirty harness\n')
    (repo/'tests/torch_fast_spdmm/conftest.py').write_text('dirty conftest\n')
    (repo/'tests/torch_fast_spdmm/test_utils.py').write_text('dirty test_utils\n')
    (repo/'outside.txt').write_text('ignored\n')

    result = commit_kernel_attempt('kernel: try variant', repo=repo, config=config)
    assert result['pushed']
    assert set(result['files']) == {
        'torch_fast_spdmm/kernels/kernel.cu',
        'tests/torch_fast_spdmm/test_boo.py',
        'tests/torch_fast_spdmm/test_added.py',
        'CHANGELOG.md',
    }
    assert set(result['ignored']) == {
        'optimizer/agent_tools/__main__.py',
        'tests/torch_fast_spdmm/benchmark/agent_bench.py',
        'tests/torch_fast_spdmm/conftest.py',
        'tests/torch_fast_spdmm/test_utils.py',
        'outside.txt',
    }
    assert result['sha'] != baseline
    # Also add an untracked attempt test file before rollback.
    (repo/'tests/torch_fast_spdmm/test_untracked.py').write_text('scratch\n')

    before = _git(repo, 'rev-parse', 'HEAD')
    back = rollback_workspace(repo=repo, config=config)
    assert back['revert_sha'] and back['revert_sha'] != before
    assert (repo/'torch_fast_spdmm/kernels/kernel.cu').read_text() == 'base\n'
    assert (repo/'tests/torch_fast_spdmm/test_boo.py').read_text() == 'base\n'
    assert not (repo/'tests/torch_fast_spdmm/test_added.py').exists()
    assert not (repo/'tests/torch_fast_spdmm/test_untracked.py').exists()
    assert (repo/'CHANGELOG.md').read_text() == 'unrelated log\n'
    # Harness and outside files MUST remain untouched by rollback_workspace.
    assert (repo/'optimizer/agent_tools/__main__.py').read_text() == 'dirty optimizer\n'
    assert (repo/'tests/torch_fast_spdmm/benchmark/agent_bench.py').read_text() == 'dirty harness\n'
    assert (repo/'tests/torch_fast_spdmm/conftest.py').read_text() == 'dirty conftest\n'
    assert (repo/'tests/torch_fast_spdmm/test_utils.py').read_text() == 'dirty test_utils\n'
    # Committed harness version at HEAD must still be 'newer harness\n', NOT reverted to baseline!
    assert _git(repo, 'show', 'HEAD:tests/torch_fast_spdmm/benchmark/agent_bench.py') == 'newer harness'
    assert _git(repo, 'merge-base', '--is-ancestor', before, 'HEAD') == ''


def test_slurm_header_merge():
    config = dict(state.DEFAULT, preamble='', verified=None)
    cpu = submit_slurm_job('#SBATCH --partition=main\n#SBATCH --time=00:10:00\necho x',
                           gpu=False, dry_run=True, config=config)['rendered']
    assert '--partition=main' in cpu and '--time=00:10:00' in cpu
    assert '--account=erant' in cpu and '--qos=normal' in cpu and '--mem=' in cpu
    gpu = submit_slurm_job('#SBATCH --partition=foo\n#SBATCH --gres=gpu:2\necho x',
                           dry_run=True, config=config)['rendered']
    assert '--partition=rtx3090' in gpu and '--gres=gpu:rtx_3090:1' in gpu
    assert 'foo' not in gpu and 'gpu:2' not in gpu and '--qos=erant' in gpu


def test_promote_gate(tmp_path, monkeypatch):
    import contextlib, copy, hashlib
    import pytest
    from optimizer.agent_tools import bench
    from optimizer.agent_tools.util import InfrastructureError
    root = tmp_path.resolve()
    example = json.loads((state.ROOT/'reports/agent_loop/schemas/examples/bench.example.json').read_text())
    cand, base = 'c'*40, 'b'*40
    approved = ['cant', 'pdb1HYS', 'ASIC_680k', 'rail4284']
    cfg = {'verified': base, 'tests_passed': {cand: True}, 'approved_skips': approved,
           'bench_runs': {}}
    calls, git_rc = [], {}
    monkeypatch.setattr(bench.state, 'ROOT', root)
    monkeypatch.setattr(bench.state, 'read', lambda *a, **k: copy.deepcopy(cfg))
    monkeypatch.setattr(bench, 'head', lambda *a, **k: cand)

    def fake_git(*args, **kwargs):
        calls.append(args)
        out = base + '\n' if args[0] == 'rev-parse' else git_rc.get('numstat', '') if '--numstat' in args else ''
        return type('R', (), {'stdout': out, 'stderr': 'rejected', 'returncode': git_rc.get(args[0], 0)})()
    monkeypatch.setattr(bench, 'git', fake_git)

    @contextlib.contextmanager
    def fake_locked(*a, **k):
        yield cfg
    monkeypatch.setattr(bench.state, 'locked', fake_locked)

    grid = [(m, f) for m in bench.SUBSETS['all'] if m not in approved for f in bench.PROMOTE_FEATURES]
    default_skips = [(m, f, 'missing: reference not in suite') for m in approved for f in bench.PROMOTE_FEATURES]

    def run(drop=0, skipped=None, extra_rows=(), fmt='boo', kernel='rowseg',
            record=True, tamper=False, **sums):
        result = copy.deepcopy(example)
        result.update(cand=cand, base=base, fmt=fmt, kernel=kernel)
        result['rows'] = [[m, f, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 'T']
                          for m, f in grid[drop:]]
        result['rows'].extend(list(r) for r in extra_rows)
        use_skips = default_skips if skipped is None else skipped
        result['skipped'] = [list(s) for s in use_skips]
        result['sum']['skip'] = len(result['skipped'])
        result['env'].update(reps=30, rounds=3)
        result['ok'], result['sum']['fail'] = True, 0
        for subset, values in sums.items():
            result['sum'][subset].update(values)
        path = root/'bench.json'
        path.write_text(json.dumps(result))
        cfg['bench_runs'] = {'bench.json': hashlib.sha256(path.read_bytes()).hexdigest()} if record else {}
        if tamper:
            path.write_text(path.read_text().replace('"W": 5', '"W": 6'))
        return bench.promote_attempt(str(path))

    good_tune = {'n': 20, 'geo_x_base': 1.05, 'W': 5, 'L': 0, 'T': 15}
    good_held = {'n': 15, 'geo_x_base': 1.00, 'W': 1, 'L': 0, 'T': 14}
    good = dict(tune=good_tune, held=good_held)

    def refused(phrase, **kw):
        out = run(**{**good, **kw})
        assert not out['promoted'] and any(phrase in r for r in out['reasons']), out
    refused('tune geo', tune={**good_tune, 'geo_x_base': 1.01})
    refused('held geo', held={**good_held, 'L': 2, 'W': 1})
    refused('incomplete grid', drop=1)
    refused('unapproved skips', skipped=[*default_skips, ('soc-Pokec', 32, 'missing: reference')])
    refused('wrong benchmark identity', fmt='pbr')
    refused('wrong benchmark identity', kernel='nnzflat')
    refused('wrong benchmark identity', kernel='rowseg?p=128')
    refused('missing approved-skip', skipped=default_skips[1:])
    refused('duplicate cases', extra_rows=[['roadNet-CA', 32, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 'T']])
    refused('duplicate cases', skipped=[*default_skips, default_skips[0]])
    refused('duplicate cases', extra_rows=[['cant', 32, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 'T']])
    refused('extra matrices or features', extra_rows=[['roadNet-CA', 33, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 'T']])
    refused('extra matrices or features', extra_rows=[['extra-mat', 128, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 'T']])
    refused('extra matrices or features', skipped=[*default_skips, ('extra-mat', 128, 'missing')])
    refused('not written by run_benchmarks', record=False)
    refused('not written by run_benchmarks', tamper=True)
    git_rc['diff'] = 1
    refused('harness')
    git_rc['diff'] = 0
    git_rc['numstat'] = '3\t1\ttests/torch_fast_spdmm/test_boo.py\n'
    refused('deletes lines')
    git_rc.pop('numstat')
    # a malformed file is an argument error, not a silent pass
    (root/'bad.json').write_text(json.dumps({'v': 'bench.1', 'ok': True}))
    with pytest.raises(ValueError, match='not a valid bench.1'):
        bench.promote_attempt(str(root/'bad.json'))
    assert not any(c[0] in ('tag', 'push') for c in calls)
    # a failed lease push leaves local tag and state untouched
    git_rc['push'] = 1
    with pytest.raises(InfrastructureError):
        run(**good)
    assert cfg['verified'] == base and not any(c[0] == 'tag' for c in calls)
    git_rc['push'] = 0
    ok = run(**good)
    assert ok['promoted'] and cfg['verified'] == cand
    push = next(c for c in calls if c[0] == 'push')
    assert f'--force-with-lease=refs/tags/opt/verified:{base}' in push and '-f' not in push
    assert ('tag', '-f', 'opt/verified', cand) in calls


def test_validator_bounds():
    import pytest
    schema = {'type': 'number', 'minimum': 0, 'maximum': 100}
    validate(50, schema)
    with pytest.raises(ValueError, match='above maximum'):
        validate(101, schema)
    with pytest.raises(TypeError):
        validate(1, 'bench')          # swapped arguments must fail loudly
    with pytest.raises(TypeError, match='unsupported'):
        validate(1, {'type': 'number', 'multipleOf': 2})
    ncu = json.loads((state.ROOT/'optimizer/agent_tools/schemas/ncu.v1.json').read_text())
    example = json.loads((state.ROOT/'reports/agent_loop/schemas/examples/ncu.example.json').read_text())
    validate(example, ncu)
    example['k'][0]['share'] = 1.5      # schema maximum 1; accepted before the fix
    with pytest.raises(ValueError):
        validate(example, ncu)


def test_slurm_reattach(monkeypatch):
    """A retry after SSH loss reattaches, reports terminal failures, and fails closed on uncertain identity."""
    import contextlib
    import pytest
    from optimizer.agent_tools import slurm
    cfg = {'agent_root': '/r', 'exclude_nodes': [], 'preamble': '', 'verified': None,
           'tests_passed': {}, 'pending_jobs': {}}
    submitted = []
    sacct_info = {'state': 'RUNNING', 'exit': '0:0'}
    in_queue = {'value': False}
    logs_exist = {'value': True}
    wait_calls = []
    sha = slurm.head()

    @contextlib.contextmanager
    def fake_locked(*a, **k):
        yield cfg
    monkeypatch.setattr(slurm.state, 'locked', fake_locked)
    monkeypatch.setattr(slurm, 'SACCT_GRACE_ATTEMPTS', 3)
    monkeypatch.setattr(slurm, 'SACCT_GRACE_SLEEP_S', 0)
    monkeypatch.setattr(slurm, '_submit', lambda *a, **k: submitted.append(1) or 100 + len(submitted))
    monkeypatch.setattr(slurm, '_sacct', lambda job, c: (sacct_info['state'], sacct_info['exit'], 'n1', '00:01:00'))
    monkeypatch.setattr(slurm, '_in_queue', lambda job, c: in_queue['value'])
    monkeypatch.setattr(slurm, '_logs_exist', lambda out_dir, job, c: logs_exist['value'])
    monkeypatch.setattr(slurm, 'checked_ssh', lambda *a, **k: type('R', (), {'stdout': 'out__AGENT_LOG_SPLIT__err'})())

    def lost(*a, **k):
        raise slurm.InfrastructureError('ssh down')
    monkeypatch.setattr(slurm, '_wait', lost)
    try:
        submit_slurm_job('echo hi', sha=sha, config=cfg)
    except slurm.InfrastructureError:
        pass
    assert submitted == [1] and list(cfg['pending_jobs'].values()) == [101]
    monkeypatch.setattr(slurm, '_wait', lambda job, c: wait_calls.append(job))
    sacct_info.update(state='COMPLETED', exit='0:0')
    out = submit_slurm_job('echo hi', sha=sha, config=cfg)
    assert out['reattached'] and out['job'] == 101 and submitted == [1]
    assert cfg['pending_jobs'] == {}

    # A recorded job that genuinely reached a terminal failure must be reported, NOT resubmitted,
    # and its pending record must be released even when hold=True.
    cfg['pending_jobs'] = {out['resume_key']: 101}
    sacct_info.update(state='FAILED', exit='1:0')
    wait_calls.clear()
    failed = submit_slurm_job('echo hi', sha=sha, config=cfg, hold=True)
    assert failed['reattached'] and failed['job'] == 101
    assert failed['state'] == 'FAILED' and failed['exit'] == 1
    assert submitted == [1] and wait_calls == []
    assert cfg['pending_jobs'] == {}

    # After the failure was reported and released, a deliberate new call submits a fresh job.
    sacct_info.update(state='COMPLETED', exit='0:0')
    fresh = submit_slurm_job('echo hi', sha=sha, config=cfg)
    assert not fresh['reattached'] and fresh['job'] == 102 and submitted == [1, 1]

    # Job completed and left squeue, while sacct temporarily returns UNKNOWN then resolves
    # during the grace window: MUST reattach and MUST NOT call _submit.
    cfg['pending_jobs'] = {out['resume_key']: 102}
    sacct_seq = iter([
        ('UNKNOWN', '0:0', '', '0:0'),   # initial probe
        ('UNKNOWN', '0:0', '', '0:0'),   # grace attempt 1
        ('COMPLETED', '0:0', 'n1', '00:00:10'),  # grace attempt 2 resolves
        ('COMPLETED', '0:0', 'n1', '00:00:10'),  # final read
    ])
    monkeypatch.setattr(slurm, '_sacct', lambda job, c: next(sacct_seq))
    in_queue['value'] = False
    graced = submit_slurm_job('echo hi', sha=sha, config=cfg)
    assert graced['reattached'] and graced['job'] == 102 and submitted == [1, 1]
    assert cfg['pending_jobs'] == {}

    # If identity remains UNKNOWN after the full grace period (absent from both squeue and sacct),
    # fail closed: keep pending_jobs, raise InfrastructureError with recovery instructions, never resubmit.
    cfg['pending_jobs'] = {out['resume_key']: 999}
    monkeypatch.setattr(slurm, '_sacct', lambda job, c: ('UNKNOWN', '0:0', '', '0:0'))
    in_queue['value'] = False
    logs_exist['value'] = True
    with pytest.raises(slurm.InfrastructureError, match='clear_pending_job'):
        submit_slurm_job('echo hi', sha=sha, config=cfg)
    assert cfg['pending_jobs'] == {out['resume_key']: 999}
    assert submitted == [1, 1]

    # Only after an explicit administrative clear_pending_job may a fresh job be submitted.
    cleared = slurm.clear_pending_job(out['resume_key'], 'verified stale via sacct on cluster')
    assert cleared['cleared'] and cleared['job'] == 999 and cfg['pending_jobs'] == {}
    monkeypatch.setattr(slurm, '_sacct', lambda job, c: ('COMPLETED', '0:0', 'n1', '00:00:10'))
    resub = submit_slurm_job('echo hi', sha=sha, config=cfg)
    assert not resub['reattached'] and resub['job'] == 103 and submitted == [1, 1, 1]
