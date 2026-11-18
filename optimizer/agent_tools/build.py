"""Fetch immutable source snapshots and compile extensions on GPU nodes."""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import shlex

from optimizer.remote import BUILD_ENV_PREAMBLE
from . import state
from .slurm import submit_slurm_job
from .util import ArgumentError, checked_ssh, git, head, ssh


def _preamble(env_name: str, modules: list[str], root: str) -> str:
    if not re.fullmatch(r'[A-Za-z0-9_.-]+', env_name):
        raise ArgumentError('invalid conda environment name')
    if any(not re.fullmatch(r'[A-Za-z0-9_./+-]+', m) for m in modules):
        raise ArgumentError('invalid module name')
    source = BUILD_ENV_PREAMBLE.replace('conda activate fem', f'conda activate {env_name}')
    return '\n'.join(['set -eo pipefail', 'source ~/.bashrc',
                      *(f'module load {shlex.quote(m)}' for m in modules),
                      source.replace('source ~/.bashrc', '').replace(
                          '$CONDA_PREFIX/lib:$LD_LIBRARY_PATH',
                          '$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}'),
                      'export TORCH_CUDA_ARCH_LIST=8.6 OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1',
                      'export SPMM_MATRIX_CACHE=/home/dayanb/research/real-matrices-20261008/matrix_cache',
                      f'export AGENT_ROOT={shlex.quote(root)}',
                      'use() {',
                      '  local build_dir="$1" src_dir="$2"',
                      '  export PYTHONPATH="$build_dir/torch_fast_spdmm:$src_dir/tests/torch_fast_spdmm:$src_dir"',
                      '  cd "$src_dir/tests/torch_fast_spdmm"',
                      '}', ''])


def _pushed(sha: str) -> bool:
    refs = git('branch', '-r', '--contains', sha, check=False)
    return refs.returncode == 0 and any('origin/' in line for line in refs.stdout.splitlines())


def _snapshot(sha: str, config: dict) -> tuple[str, str]:
    root = config['agent_root'].rstrip('/')
    repo = os.getenv('OPTIMIZER_REMOTE_DIR', config['remote_repo'])
    tree = git('rev-parse', f'{sha}:torch_fast_spdmm').stdout.strip()[:12]
    build_dir, src_dir = f'{root}/builds/{tree}', f'{root}/src/{sha}'
    command = (
        f'mkdir -p {shlex.quote(build_dir)} {shlex.quote(src_dir)} && '
        f'git -C {shlex.quote(repo)} fetch origin && '
        f'git -C {shlex.quote(repo)} cat-file -e {shlex.quote(sha)}^{{commit}} && '
        f'if [ ! -f {shlex.quote(src_dir)}/.agent_snapshot_ok ]; then '
        f'git -C {shlex.quote(repo)} archive {shlex.quote(sha)} | '
        f'tar -x -C {shlex.quote(src_dir)} --exclude=torch_fast_spdmm '
        f'--exclude=tests/torch_fast_spdmm/benchmark/results && '
        f'rm -rf {shlex.quote(src_dir)}/torch_fast_spdmm '
        f'{shlex.quote(src_dir)}/tests/torch_fast_spdmm/benchmark/results && '
        f'touch {shlex.quote(src_dir)}/.agent_snapshot_ok; fi && '
        f'if [ ! -d {shlex.quote(build_dir)}/torch_fast_spdmm ]; then '
        f'git -C {shlex.quote(repo)} archive {shlex.quote(sha)} torch_fast_spdmm | '
        f'tar -x -C {shlex.quote(build_dir)}; fi'
    )
    checked_ssh('bash -o pipefail -c ' + shlex.quote(command), timeout=600, config=config)
    return tree, build_dir


def ensure_build(sha: str, *, config: dict | None = None) -> dict:
    cfg = config or state.read()
    sha = git('rev-parse', f'{sha}^{{commit}}').stdout.strip()
    if not _pushed(sha):
        return {'ok': False, 'sha': sha, 'reason': 'commit is not pushed to origin; push it first'}
    tree, build_dir = _snapshot(sha, cfg)
    marker = f'{build_dir}/BUILD_OK.json'
    cached = ssh(f'cat {shlex.quote(marker)}', config=cfg)
    if cached.returncode == 0:
        info = json.loads(cached.stdout)
        info.update(ok=True, cached=True, build_dir=build_dir, err_tail=None)
        return info
    template = (Path(__file__).parent / 'templates' / 'build.sh').read_text()
    script = template.replace('@@BUILD_DIR@@', build_dir).replace('@@SHA@@', sha).replace('@@SRC_HASH@@', tree)
    result = submit_slurm_job(script, name=f'spmm-build-{tree}', time='00:45:00',
                              mem='64G', sha=sha, config=cfg)
    if result['state'] != 'COMPLETED' or result['exit']:
        checked_ssh(f'rm -rf {shlex.quote(build_dir)}', config=cfg)
        return {'ok': False, 'cached': False, 'sha': sha, 'src_hash': tree,
                'build_dir': build_dir, 'reason': 'build failed; rollback_workspace or fix and recommit',
                'warnings': result['err_tail'].lower().count('warning'),
                'err_tail': result['err_tail'], 'job': result['job']}
    info = json.loads(checked_ssh(f'cat {shlex.quote(marker)}', config=cfg).stdout)
    info.update(ok=True, cached=False, build_dir=build_dir, err_tail=None)
    with state.locked() as current:
        current['builds'][tree] = {'sha': sha, 'so_sha256': info['so_sha256']}
    return info


def load_build_env(env_name: str, modules: list[str]) -> dict:
    if not isinstance(modules, list) or any(not isinstance(m, str) for m in modules):
        raise ArgumentError('modules must be a list of names')
    cfg = state.read()
    preamble = _preamble(env_name, modules, cfg['agent_root'])
    probe = '\n'.join(['source ~/.bashrc', *(f'module load {shlex.quote(m)}' for m in modules),
                       f'conda activate {shlex.quote(env_name)}',
                       'which nvcc', 'nvcc --version | tail -1',
                       'python -c "import torch;print(torch.__version__)"'])
    tested = ssh('bash -lc ' + shlex.quote(probe), timeout=120, config=cfg)
    if tested.returncode:
        return {'ok': False, 'reason': f'env probe failed: {tested.stderr.strip() or tested.stdout.strip()}'}
    with state.locked() as current:
        current.update(env_name=env_name, modules=modules, preamble=preamble)
    sha = head()
    dirty = git('status', '--porcelain', '--', 'torch_fast_spdmm').stdout.strip()
    if dirty:
        return {'ok': False, 'sha': sha, 'reason': 'torch_fast_spdmm has uncommitted changes; commit_kernel_attempt first'}
    return ensure_build(sha, config=state.read())
