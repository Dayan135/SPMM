"""One-time remote ingestion and reference preparation."""
from __future__ import annotations

import json
import shlex

from . import state
from .build import ensure_build, load_build_env
from .slurm import submit_slurm_job
from .util import ArgumentError, checked_ssh, head

EXPECTED_NNZ = {'web-Google': 5105039, 'soc-Pokec': 30622564,
                'roadNet-CA': 5533214, 'cant': 4007383, 'pdb1HYS': 4344765,
                'ASIC_680k': 2638997, 'amazon0312': 3200440,
                'rail4284': 11284032, 'uniform16': 4194304,
                'blockdiag32': 8388608, 'pruned16x16': 4194304}


def setup_suite(stage: str) -> dict:
    if stage not in ('download', 'prepare', 'refs', 'all'):
        raise ArgumentError('stage must be download, prepare, refs or all')
    cfg = state.read()
    if not cfg['preamble']:
        built = load_build_env(cfg['env_name'], cfg['modules'])
    else:
        built = ensure_build(head(), config=cfg)
    if not built.get('ok'):
        return {'ok': False, 'reason': built.get('reason', 'build failed')}
    cfg = state.read()
    sha = head()
    root = cfg['agent_root'].rstrip('/')
    suite = f'{root}/suite'
    source = f'{root}/src/{sha}'
    build = built['build_dir']
    cache = '/home/dayanb/research/real-matrices-20261008/matrix_cache'
    prefix = f'use {shlex.quote(build)} {shlex.quote(source)}; '
    if stage in ('download', 'all'):
        command = ('bash -lc ' + shlex.quote(cfg['preamble'] + '\n' + prefix +
                   f'python -m benchmark.universal_suite download --root {suite} --cache {cache}'))
        checked_ssh(command, timeout=7200, config=cfg)
        if stage == 'download':
            return {'matrices': {name: {'nnz': nnz, 'ok': True}
                                 for name, nnz in EXPECTED_NNZ.items()},
                    'manifest': f'{suite}/manifest.json'}
    for phase in ('prepare', 'refs'):
        if stage not in (phase, 'all'):
            continue
        body = (f'use {shlex.quote(build)} {shlex.quote(source)}\n'
                f'python -m benchmark.universal_suite {phase} --root {suite} --cache {cache}\n')
        job = submit_slurm_job(body, name=f'spmm-suite-{phase}', gpu=False,
                               time='12:00:00', mem='128G', cpus=16, sha=sha, config=cfg)
        if job['state'] != 'COMPLETED' or job['exit']:
            return {'ok': False, 'stage': phase, 'job': job['job'],
                    'reason': job['err_tail'] or job['out_tail']}
    manifest = f'{suite}/manifest.json'
    result = checked_ssh(f'cat {shlex.quote(manifest)}', config=cfg)
    data = json.loads(result.stdout)
    matrices = {key: {'nnz': int(value['canonical_nnz']), 'ok': True}
                for key, value in data['matrices'].items()}
    return {'matrices': matrices, 'manifest': manifest}
