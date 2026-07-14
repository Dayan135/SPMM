"""SSH/Slurm wrapper -- formalizes the exact manual pattern used throughout this
project's own investigation sessions (ssh <alias> "sbatch ...", poll squeue,
rsync results back). No new remote-execution mechanism is introduced; this is
a thin, reusable layer over the same commands a human would type.
"""

from __future__ import annotations

import os
import re
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

REMOTE_HOST = os.environ.get("OPTIMIZER_REMOTE_HOST", "bgu-slurm")
REMOTE_REPO_DIR = os.environ.get("OPTIMIZER_REMOTE_DIR", "/home/dayanb/research/SPMM")

SBATCH_HEADER = """#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --partition=rtx3090
#SBATCH --account=erant
#SBATCH --qos=erant
#SBATCH --gres=gpu:rtx_3090:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={cpus}
#SBATCH --mem={mem}
#SBATCH --time={time_limit}
#SBATCH --output={out_dir}/%j.out
#SBATCH --error={out_dir}/%j.err
"""

BUILD_ENV_PREAMBLE = """
source ~/.bashrc
conda activate fem
set -euo pipefail
export CUDA_HOME="$CONDA_PREFIX"
export CC=$(which x86_64-conda-linux-gnu-cc)
export CXX=$(which x86_64-conda-linux-gnu-c++)
export CUDAHOSTCXX=$CXX
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
"""


class RemoteError(RuntimeError):
    pass


@dataclass
class CommandResult:
    returncode: int
    stdout: str
    stderr: str

    def check(self, context: str = "") -> "CommandResult":
        if self.returncode != 0:
            raise RemoteError(f"{context}: exit {self.returncode}\nstdout: {self.stdout}\nstderr: {self.stderr}")
        return self


def ssh(command: str, *, host: str = REMOTE_HOST, timeout: Optional[int] = 120) -> CommandResult:
    """Run a single command on the remote host over SSH."""
    proc = subprocess.run(
        ["ssh", host, command],
        capture_output=True, text=True, timeout=timeout,
    )
    return CommandResult(proc.returncode, proc.stdout, proc.stderr)


def push_file(local_path: Path, remote_path: str, *, host: str = REMOTE_HOST) -> None:
    subprocess.run(["scp", "-q", str(local_path), f"{host}:{remote_path}"], check=True)


def pull_file(remote_path: str, local_path: Path, *, host: str = REMOTE_HOST) -> None:
    local_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["scp", "-q", f"{host}:{remote_path}", str(local_path)], check=True)


def pull_dir(remote_dir: str, local_dir: Path, *, host: str = REMOTE_HOST) -> None:
    local_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(["rsync", "-a", f"{host}:{remote_dir}/", f"{local_dir}/"], check=True)


def render_and_submit(script_body: str, *, job_name: str, out_dir: str,
                       cpus: int = 8, mem: str = "16G", time_limit: str = "1:00:00",
                       host: str = REMOTE_HOST) -> str:
    """Render an sbatch script (header + body), push it to the remote host under
    REMOTE_REPO_DIR, submit it, and return the Slurm job ID."""
    header = SBATCH_HEADER.format(job_name=job_name, cpus=cpus, mem=mem,
                                   time_limit=time_limit, out_dir=out_dir)
    full_script = header + script_body

    with tempfile.NamedTemporaryFile("w", suffix=".sh", delete=False) as f:
        f.write(full_script)
        local_tmp = Path(f.name)

    remote_script_path = f"{REMOTE_REPO_DIR}/{out_dir}/{job_name}.sh"
    ssh(f"mkdir -p {REMOTE_REPO_DIR}/{out_dir}", host=host).check("mkdir remote out_dir")
    push_file(local_tmp, remote_script_path, host=host)
    local_tmp.unlink()

    result = ssh(f"cd {REMOTE_REPO_DIR} && sbatch {remote_script_path}", host=host)
    result.check(f"sbatch {job_name}")
    match = re.search(r"Submitted batch job (\d+)", result.stdout)
    if not match:
        raise RemoteError(f"could not parse job id from sbatch output: {result.stdout!r}")
    return match.group(1)


def job_is_finished(job_id: str, *, host: str = REMOTE_HOST) -> bool:
    result = ssh(f"squeue -j {job_id} -h", host=host)
    # squeue returns empty stdout (job left the queue) once it's done, whether
    # it succeeded or failed -- caller must inspect the log file for outcome.
    return result.stdout.strip() == ""


def wait_for_job(job_id: str, *, poll_interval_s: int = 20, timeout_s: int = 4 * 3600,
                  host: str = REMOTE_HOST) -> None:
    """Blocking poll loop -- fine for Phase 0's synchronous orchestrator; a
    future async orchestrator can call job_is_finished() directly instead."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if job_is_finished(job_id, host=host):
            return
        time.sleep(poll_interval_s)
    raise RemoteError(f"job {job_id} did not finish within {timeout_s}s")


def read_remote_text(remote_path: str, *, host: str = REMOTE_HOST) -> str:
    result = ssh(f"cat {remote_path}", host=host)
    return result.stdout


def sync_and_build(*, git_ref: str, host: str = REMOTE_HOST, timeout_s: int = 1800) -> str:
    """Fetch+checkout the given ref on the remote repo and build the CUDA
    extension via a GPU-node srun allocation (matches CLAUDE.md's build
    recipe exactly). Returns the resolved commit hash actually built, so the
    ledger can record precisely which build produced a given measurement --
    this is the direct fix for the stale-remote-build bug hit by hand this
    session (a shared clone left unrebuilt after a revert silently corrupted
    a later measurement)."""
    ssh(f"cd {REMOTE_REPO_DIR} && git fetch origin && git checkout {git_ref} && "
        f"git pull origin {git_ref} || true", host=host, timeout=120).check("git sync")

    commit = ssh(f"cd {REMOTE_REPO_DIR} && git rev-parse HEAD", host=host).check("rev-parse").stdout.strip()

    build_cmd = (
        f"cd {REMOTE_REPO_DIR} && srun --partition=rtx3090 --account=erant --qos=erant "
        f"--gres=gpu:rtx_3090:1 --time=00:30:00 --mem=16G bash -lc '"
        f"{BUILD_ENV_PREAMBLE} pip install -e torch_fast_spdmm --no-build-isolation'"
    )
    ssh(build_cmd, host=host, timeout=timeout_s).check("build")
    return commit


def run_pytest_suite(*, host: str = REMOTE_HOST, timeout_s: int = 1800) -> CommandResult:
    """Runs the existing correctness suite, ignoring the pre-existing,
    unrelated test_sparse_block_mat.py collection error (see STALL_REASONS.md
    / session history -- that file has a fixture bug unrelated to any kernel
    or optimizer work)."""
    cmd = (
        f"cd {REMOTE_REPO_DIR} && srun --partition=rtx3090 --account=erant --qos=erant "
        f"--gres=gpu:rtx_3090:1 --time=00:30:00 --mem=16G bash -lc '"
        f"{BUILD_ENV_PREAMBLE} pytest tests/torch_fast_spdmm/test_pbr.py "
        f"tests/torch_fast_spdmm/test_csr.py -q'"
    )
    return ssh(cmd, host=host, timeout=timeout_s)
