"""Render, submit and inspect isolated Slurm jobs."""
from __future__ import annotations

import os
from pathlib import Path
import re
import shlex
import tempfile
import time as clock

from optimizer import remote
from . import state
from .util import ArgumentError, InfrastructureError, checked_ssh, head, short_tail, ssh, git, transfer


def _elapsed(value: str) -> int:
    days, _, tail = value.partition('-')
    if not tail:
        tail, days = days, '0'
    pieces = [int(p) for p in tail.split(':')]
    return int(days) * 86400 + sum(v * 60 ** i for i, v in enumerate(reversed(pieces)))


def render(script_content: str, *, time='02:00:00', mem='64G', cpus=8,
           name='spmm-agent', gpu=True, config=None, sha=None) -> str:
    cfg = config or state.read()
    if not isinstance(script_content, str) or not script_content.strip():
        raise ArgumentError('script_content must be nonempty')
    if not isinstance(cpus, int) or cpus < 1:
        raise ArgumentError('cpus must be a positive integer')
    sha = sha or head()
    root = cfg['agent_root'].rstrip('/')
    src_hash = git('rev-parse', f'{sha}:torch_fast_spdmm').stdout.strip()[:12]
    base_sha = cfg.get('verified') or sha
    base_hash = git('rev-parse', f'{base_sha}:torch_fast_spdmm').stdout.strip()[:12]
    build_dir = f'{root}/builds/{src_hash}'
    out_dir = f'{root}/runs/{sha}'
    lines = script_content.splitlines()
    # Script-supplied #SBATCH options override the defaults, except the ones the
    # tool must control: log paths always, and partition/gres/exclude for GPU jobs.
    merged = {'job-name': name, 'account': 'erant', 'qos': 'erant' if gpu else 'normal',
              'cpus-per-task': str(cpus), 'mem': mem, 'time': time}
    enforced = {'output', 'error', 'gres', 'exclude'} | ({'partition'} if gpu else set())
    for line in lines:
        match = re.match(r'#SBATCH\s+--([A-Za-z0-9-]+)(?:[=\s]\s*(\S.*?))?\s*$', line)
        if match and match[1] not in enforced:
            merged[match[1]] = match[2]
    if gpu:
        merged.update({'partition': 'rtx3090', 'gres': 'gpu:rtx_3090:1',
                       'exclude': ','.join(cfg['exclude_nodes'])})
    else:
        merged.setdefault('partition', 'cpu')
    merged.update({'output': f'{out_dir}/%j.out', 'error': f'{out_dir}/%j.err'})
    directives = [f'#SBATCH --{key}' + ('' if value is None else f'={value}')
                  for key, value in merged.items()]
    body = [line for line in lines if not line.startswith('#SBATCH') and not line.startswith('#!')]
    preamble = '' if '# AGENT_NO_PREAMBLE' in script_content else cfg.get('preamble', '')
    script = '\n'.join(['#!/bin/bash', *directives, preamble,
                        f'mkdir -p {shlex.quote(root)}/runs/{sha}/$SLURM_JOB_ID',
                        *body, ''])
    from .ncu import NCU_FLAGS
    mapping = {
        'BUILD_DIR': build_dir,
        'BASE_BUILD_DIR': f'{root}/builds/{base_hash}',
        'SRC_DIR': f'{root}/src/{sha}',
        'RUN_DIR': f'{root}/runs/{sha}/$SLURM_JOB_ID',
        'AGENT_ROOT': root,
        'NCU': '$CONDA_PREFIX/nsight-compute-2025.2.1/ncu',
        'NCU_FLAGS': NCU_FLAGS, 'SHA': sha,
    }
    for key, value in mapping.items():
        script = script.replace('{{' + key + '}}', value)
    unresolved = re.findall(r'{{[^}]+}}', script)
    if unresolved:
        raise ArgumentError(f'unresolved placeholders: {unresolved}')
    return script


def submit_slurm_job(script_content: str, *, time='02:00:00', mem='64G', cpus=8,
                     name='spmm-agent', gpu=True, dry_run=False, sha=None,
                     config=None) -> dict:
    cfg = config or state.read()
    sha = sha or head()
    script = render(script_content, time=time, mem=mem, cpus=cpus,
                    name=name, gpu=gpu, config=cfg, sha=sha)
    if dry_run:
        return {'rendered': script}
    root = cfg['agent_root'].rstrip('/')
    out_dir = f'{root}/runs/{sha}'
    checked_ssh(f'mkdir -p {shlex.quote(out_dir)}', config=cfg)
    with tempfile.NamedTemporaryFile('w', suffix='.sh', delete=False) as file:
        file.write(script)
        local = Path(file.name)
    remote_script = f'{out_dir}/{name}-{int(clock.time()*1000)}.sh'
    try:
        transfer('push', local, remote_script, config=cfg)
    except Exception as exc:
        raise InfrastructureError(f'could not upload Slurm script: {exc}') from exc
    finally:
        local.unlink(missing_ok=True)
    submitted = ssh(f'sbatch {shlex.quote(remote_script)}', config=cfg)
    if submitted.returncode:
        raise InfrastructureError(f'Slurm rejected job: {submitted.stderr.strip()}')
    match = re.search(r'Submitted batch job (\d+)', submitted.stdout)
    if not match:
        raise InfrastructureError(f'cannot read job id: {submitted.stdout.strip()}')
    job = int(match.group(1))
    while True:
        queue = checked_ssh(f'squeue -j {job} -h', config=cfg)
        if not queue.stdout.strip():
            break
        clock.sleep(20)
    accounting = checked_ssh(f'sacct -j {job} -n -P -o State,ExitCode,NodeList,Elapsed', config=cfg)
    record = next((line for line in accounting.stdout.splitlines() if line.strip()), '')
    if not record:
        raise InfrastructureError(f'sacct has no record for job {job}')
    fields = record.split('|')
    if len(fields) < 4:
        raise InfrastructureError(f'malformed sacct record: {record}')
    state_name, exit_code, node, elapsed = fields[:4]
    out_path, err_path = f'{out_dir}/{job}.out', f'{out_dir}/{job}.err'
    logs = checked_ssh(f'for f in {shlex.quote(out_path)} {shlex.quote(err_path)}; do '
                       'if [ -f "$f" ]; then cat "$f"; fi; '
                       'echo __AGENT_LOG_SPLIT__; done', config=cfg)
    parts = logs.stdout.split('__AGENT_LOG_SPLIT__')
    out, err = parts[0], parts[1] if len(parts) > 1 else ''
    result = {'job': job, 'state': state_name, 'exit': int(exit_code.split(':')[0]),
              'node': node, 'elapsed_s': _elapsed(elapsed),
              'out_tail': short_tail(out), 'err_tail': short_tail(err),
              'out_path': out_path, 'err_path': err_path}
    if f'AGENT_TESTS_PASSED {sha}' in out:
        passed = state_name == 'COMPLETED' and result['exit'] == 0
        result['tests_passed'] = passed
        if passed:
            with state.locked() as current:
                current['tests_passed'][sha] = True
    return result
