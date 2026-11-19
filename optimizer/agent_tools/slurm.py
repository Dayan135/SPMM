"""Render, submit and inspect isolated Slurm jobs."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import re
import shlex
import tempfile
import time as clock

from optimizer import remote
from . import attempt_log, state
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
                     config=None, hold=False) -> dict:
    cfg = config or state.read()
    sha = sha or head()
    script = render(script_content, time=time, mem=mem, cpus=cpus,
                    name=name, gpu=gpu, config=cfg, sha=sha)
    if dry_run:
        return {'rendered': script}
    root = cfg['agent_root'].rstrip('/')
    out_dir = f'{root}/runs/{sha}'
    # Idempotency: the rendered script is deterministic, so its hash names the job. If a
    # previous call submitted it and then lost SSH while polling, reattach to that job
    # instead of submitting a duplicate. A recorded job that genuinely reached any terminal
    # state (including FAILED, TIMEOUT, OUT_OF_MEMORY, CANCELLED, etc.) is reattached and
    # reported rather than silently resubmitted; its pending record is released once reported.
    # If a recorded job is absent from squeue and sacct returns UNKNOWN, poll sacct for a
    # bounded accounting grace window and check whether log files exist; if identity remains
    # uncertain, fail closed (keep pending_jobs and never auto-resubmit).
    key = hashlib.sha256(script.encode()).hexdigest()[:16]
    with state.locked() as current:
        job = current.get('pending_jobs', {}).get(key)
    reattached = False
    already_terminal = False
    if job is not None:
        prior_state = _sacct(job, cfg)[0]
        if prior_state != 'UNKNOWN':
            reattached = True
            already_terminal = prior_state in TERMINAL
        elif _in_queue(job, cfg):
            reattached = True
        else:
            graced_state = _sacct_with_grace(job, cfg)[0]
            if graced_state != 'UNKNOWN':
                reattached = True
                already_terminal = graced_state in TERMINAL
            else:
                attempt_log.attach_slurm_info(job_id=job, resume_key=key, reattached=True)
                has_logs = _logs_exist(out_dir, job, cfg)
                raise InfrastructureError(_uncertain_job_error(job, key, out_dir, has_logs))
    if not reattached:
        job = _submit(script, out_dir, name, cfg)
        with state.locked() as current:
            current.setdefault('pending_jobs', {})[key] = job
    attempt_log.attach_slurm_info(job_id=job, resume_key=key, reattached=reattached)
    if not already_terminal:
        _wait(job, cfg)
    state_name, exit_code, node, elapsed = _sacct_with_grace(job, cfg)
    if state_name == 'UNKNOWN':
        has_logs = _logs_exist(out_dir, job, cfg)
        raise InfrastructureError(_uncertain_job_error(job, key, out_dir, has_logs))
    out_path, err_path = f'{out_dir}/{job}.out', f'{out_dir}/{job}.err'
    attempt_log.attach_slurm_info(
        job_id=job, resume_key=key, reattached=reattached,
        out_path=out_path, err_path=err_path,
    )
    logs = checked_ssh(f'for f in {shlex.quote(out_path)} {shlex.quote(err_path)}; do '
                       'if [ -f "$f" ]; then cat "$f"; fi; '
                       'echo __AGENT_LOG_SPLIT__; done', config=cfg)
    parts = logs.stdout.split('__AGENT_LOG_SPLIT__')
    out, err = parts[0], parts[1] if len(parts) > 1 else ''
    result = {'job': job, 'state': state_name, 'exit': int(exit_code.split(':')[0]),
              'node': node, 'elapsed_s': _elapsed(elapsed),
              'out_tail': short_tail(out), 'err_tail': short_tail(err),
              'out_path': out_path, 'err_path': err_path,
              'resume_key': key, 'reattached': reattached}
    if f'AGENT_TESTS_PASSED {sha}' in out:
        passed = state_name == 'COMPLETED' and result['exit'] == 0
        result['tests_passed'] = passed
        if passed:
            with state.locked() as current:
                current['tests_passed'][sha] = True
    succeeded = state_name == 'COMPLETED' and result['exit'] == 0
    if not hold or not succeeded:
        release_job(key)
    return result


# Terminal Slurm states (both success and failure); a recorded job in any of these
# genuinely finished and must be reported rather than silently resubmitted.
TERMINAL = {'COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT', 'NODE_FAIL', 'OUT_OF_MEMORY',
            'PREEMPTED', 'BOOT_FAIL', 'DEADLINE'}
OUTAGE_S = 3600
SACCT_GRACE_ATTEMPTS = 6
SACCT_GRACE_SLEEP_S = 5


def release_job(key: str) -> None:
    with state.locked() as current:
        current.get('pending_jobs', {}).pop(key, None)


def clear_pending_job(resume_key: str, reason: str) -> dict:
    """Explicit administrative action to clear a verified-stale pending_jobs record."""
    if not isinstance(resume_key, str) or not resume_key.strip():
        raise ArgumentError('resume_key must be a nonempty string')
    if not isinstance(reason, str) or not reason.strip():
        raise ArgumentError('reason must be a nonempty string')
    with state.locked() as current:
        pending = current.get('pending_jobs', {})
        if resume_key not in pending:
            raise ArgumentError(f'no pending job recorded for resume_key {resume_key!r}')
        cleared_job = pending.pop(resume_key)
    return {'cleared': True, 'resume_key': resume_key, 'job': cleared_job, 'reason': reason.strip()}


def _in_queue(job: int, cfg: dict) -> bool:
    queue = checked_ssh(f'squeue -j {job} -h', config=cfg)
    return bool(queue.stdout.strip())


def _logs_exist(out_dir: str, job: int, cfg: dict) -> bool:
    out_p, err_p = f'{out_dir}/{job}.out', f'{out_dir}/{job}.err'
    res = checked_ssh(
        f'if [ -f {shlex.quote(out_p)} ] || [ -f {shlex.quote(err_p)} ]; then echo YES; else echo NO; fi',
        config=cfg,
    )
    return res.stdout.strip() == 'YES'


def _sacct_with_grace(job: int, cfg: dict) -> tuple[str, str, str, str]:
    """Poll sacct through brief post-completion accounting lag before giving up."""
    rec = _sacct(job, cfg)
    for _ in range(max(0, SACCT_GRACE_ATTEMPTS - 1)):
        if rec[0] != 'UNKNOWN':
            return rec
        if SACCT_GRACE_SLEEP_S > 0:
            clock.sleep(SACCT_GRACE_SLEEP_S)
        rec = _sacct(job, cfg)
    return rec


def _uncertain_job_error(job: int, key: str, out_dir: str, has_logs: bool) -> str:
    log_note = 'expected log files exist on the cluster' if has_logs else 'no log files found yet'
    return (
        f'uncertain Slurm identity for pending job {job} (resume_key={key}): '
        f'absent from squeue and sacct still returned UNKNOWN after {SACCT_GRACE_ATTEMPTS} grace checks '
        f'({log_note} under {out_dir}/{job}.{{out,err}}). '
        f'Keeping pending_jobs[{key!r}] and refusing to auto-resubmit. '
        f'Inspect `sacct -j {job}` and the log files on the cluster; if the record is genuinely stale, '
        f'explicitly clear it with `clear_pending_job` (args: {{"resume_key": "{key}", "reason": "..."}}) '
        f'before retrying.'
    )


def _sacct(job: int, cfg: dict) -> tuple[str, str, str, str]:
    accounting = checked_ssh(f'sacct -j {job} -n -P -X -o State,ExitCode,NodeList,Elapsed',
                             config=cfg)
    record = next((line for line in accounting.stdout.splitlines() if line.strip()), '')
    fields = record.split('|')
    if len(fields) < 4:
        return 'UNKNOWN', '0:0', '', '0:0'
    return fields[0].split()[0], *fields[1:4]


def _submit(script: str, out_dir: str, name: str, cfg: dict) -> int:
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
    return int(match.group(1))


def _wait(job: int, cfg: dict) -> None:
    """Poll until the job leaves the queue, riding out SSH outages up to OUTAGE_S."""
    down_since = None
    while True:
        try:
            queue = checked_ssh(f'squeue -j {job} -h', config=cfg)
            down_since = None
        except InfrastructureError:
            down_since = down_since or clock.monotonic()
            if clock.monotonic() - down_since > OUTAGE_S:
                raise
            clock.sleep(60)
            continue
        if not queue.stdout.strip():
            return
        clock.sleep(20)
