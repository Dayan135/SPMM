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
    config = dict(state.DEFAULT, loop_branch='opt/loop-20261118', verified=baseline)
    (repo/'torch_fast_spdmm/kernels/kernel.cu').write_text('candidate\n')
    (repo/'CHANGELOG.md').write_text('unrelated log\n')
    (repo/'outside.txt').write_text('ignored\n')
    result = commit_kernel_attempt('kernel: try variant', repo=repo, config=config)
    assert result['pushed'] and result['ignored'] == ['outside.txt']
    assert result['sha'] != baseline
    before = _git(repo, 'rev-parse', 'HEAD')
    back = rollback_workspace(repo=repo, config=config)
    assert back['revert_sha'] and back['revert_sha'] != before
    assert (repo/'torch_fast_spdmm/kernels/kernel.cu').read_text() == 'base\n'
    assert (repo/'CHANGELOG.md').read_text() == 'unrelated log\n'
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
