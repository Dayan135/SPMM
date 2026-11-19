"""JSON-only command line entry point: python -m optimizer.agent_tools TOOL --args JSON."""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
import traceback

from .util import ArgumentError, InfrastructureError

TOOLS = ('load_build_env', 'submit_slurm_job', 'commit_kernel_attempt',
         'rollback_workspace', 'run_benchmarks', 'parse_ncu_profile', 'setup_suite',
         'promote_attempt')


class Parser(argparse.ArgumentParser):
    def error(self, message):
        raise ArgumentError(message)


def dispatch(tool: str, arguments: dict) -> dict:
    if tool == 'load_build_env':
        from .build import load_build_env
        return load_build_env(**arguments)
    if tool == 'submit_slurm_job':
        from .slurm import submit_slurm_job
        return submit_slurm_job(**arguments)
    if tool == 'commit_kernel_attempt':
        from .vcs import commit_kernel_attempt
        return commit_kernel_attempt(**arguments)
    if tool == 'rollback_workspace':
        from .vcs import rollback_workspace
        return rollback_workspace(**arguments)
    if tool == 'run_benchmarks':
        from .bench import run_benchmarks
        return run_benchmarks(**arguments)
    if tool == 'parse_ncu_profile':
        from .ncu import parse_ncu_profile
        return parse_ncu_profile(**arguments)
    if tool == 'promote_attempt':
        from .bench import promote_attempt
        return promote_attempt(**arguments)
    from .suite import setup_suite
    return setup_suite(**arguments)


def main(argv=None) -> int:
    parser = Parser(add_help=False)
    parser.add_argument('tool', choices=TOOLS)
    parser.add_argument('--args', default='{}')
    try:
        opts = parser.parse_args(argv)
        arguments = json.loads(opts.args)
        if not isinstance(arguments, dict):
            raise ArgumentError('--args must be a JSON object')
        with contextlib.redirect_stdout(sys.stderr):
            result = dispatch(opts.tool, arguments)
        code = 0
    except (ArgumentError, TypeError, ValueError, FileNotFoundError, json.JSONDecodeError) as exc:
        result, code = {'tool': locals().get('opts').tool if 'opts' in locals() else 'unknown',
                        'error': 'bad arguments', 'detail': str(exc)}, 2
    except InfrastructureError as exc:
        result, code = {'tool': opts.tool, 'error': 'infrastructure', 'detail': str(exc)}, 3
    except Exception as exc:
        traceback.print_exc(file=sys.stderr)
        result, code = {'tool': locals().get('opts').tool if 'opts' in locals() else 'unknown',
                        'error': 'internal', 'detail': str(exc)}, 3
    print(json.dumps(result, separators=(',', ':'), allow_nan=False))
    return code


if __name__ == '__main__':
    raise SystemExit(main())
