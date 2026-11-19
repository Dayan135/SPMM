"""JSON-only command line entry point: python -m optimizer.agent_tools TOOL --args JSON.

Exit codes: 0 success; 2 bad arguments / refused precondition; 3 infrastructure (retry is safe
for interrupted work); 4 the tool's side effects SUCCEEDED but its audit record could not be
finalized - do NOT retry the call. With a recovery record (audit_finalization=pending_recovery) the
next invocation replays it; without one (audit_finalization=manual_recovery_required) the campaign
must stop for user reconciliation.
"""
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
         'setup_controls', 'synthetic_controls',
         'promote_attempt', 'begin_attempt', 'finish_attempt', 'close_run',
         'get_loop_status', 'export_attempt_log', 'clear_pending_job',
         'clean_attempt_workspace')
ATTEMPT_REQUIRED_TOOLS = frozenset({
    'commit_kernel_attempt', 'rollback_workspace', 'run_benchmarks', 'synthetic_controls',
    'promote_attempt',
})
# Pre-attempt recovery only: refused while an attempt is active (rollback_workspace covers that).
NO_ACTIVE_ATTEMPT_TOOLS = frozenset({'clean_attempt_workspace'})
EXIT_AUDIT_PENDING = 4


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
    if tool == 'begin_attempt':
        from .attempt_log import begin_attempt
        return begin_attempt(**arguments)
    if tool == 'finish_attempt':
        from .attempt_log import finish_attempt
        return finish_attempt(**arguments)
    if tool == 'close_run':
        from .attempt_log import close_run
        return close_run(**arguments)
    if tool == 'get_loop_status':
        from .attempt_log import get_loop_status
        return get_loop_status(**arguments)
    if tool == 'export_attempt_log':
        from .attempt_log import export_attempt_log
        return export_attempt_log(**arguments)
    if tool == 'clean_attempt_workspace':
        from .vcs import clean_attempt_workspace
        return clean_attempt_workspace(**arguments)
    if tool == 'clear_pending_job':
        from .slurm import clear_pending_job
        return clear_pending_job(**arguments)
    if tool == 'setup_controls':
        from .controls import setup_controls
        return setup_controls(**arguments)
    if tool == 'synthetic_controls':
        from .controls import synthetic_controls
        return synthetic_controls(**arguments)
    from .suite import setup_suite
    return setup_suite(**arguments)


def main(argv=None) -> int:
    from .attempt_log import AuditFinalizationError, finish_tool_call, start_tool_call

    parser = Parser(add_help=False)
    parser.add_argument('tool', choices=TOOLS)
    parser.add_argument('--args', default='{}')
    handle = None
    tool_name = 'unknown'
    error_obj: BaseException | None = None
    try:
        opts = parser.parse_args(argv)
        tool_name = opts.tool
        try:
            arguments = json.loads(opts.args)
            if not isinstance(arguments, dict):
                raise ArgumentError('--args must be a JSON object')
        except (ArgumentError, json.JSONDecodeError) as exc:
            handle = start_tool_call(tool_name, {'_parse_error': 'invalid --args JSON'})
            raise exc
        handle = start_tool_call(tool_name, arguments)
        if tool_name in ATTEMPT_REQUIRED_TOOLS and handle.ctx.attempt_id is None:
            raise ArgumentError(
                f'{tool_name} requires an active attempt; call begin_attempt first'
            )
        if tool_name in NO_ACTIVE_ATTEMPT_TOOLS and handle.ctx.attempt_id is not None:
            raise ArgumentError(
                f'{tool_name} is only for recovery with no active attempt; '
                'use rollback_workspace inside an attempt'
            )
        with contextlib.redirect_stdout(sys.stderr):
            result = dispatch(tool_name, arguments)
        code = 0
    except (ArgumentError, TypeError, ValueError, FileNotFoundError, json.JSONDecodeError) as exc:
        error_obj = exc
        result, code = {'tool': tool_name, 'error': 'bad arguments', 'detail': str(exc)}, 2
    except InfrastructureError as exc:
        error_obj = exc
        result, code = {'tool': tool_name, 'error': 'infrastructure', 'detail': str(exc)}, 3
    except Exception as exc:
        error_obj = exc
        traceback.print_exc(file=sys.stderr)
        result, code = {'tool': tool_name, 'error': 'internal', 'detail': str(exc)}, 3
    if handle is not None:
        try:
            finish_tool_call(handle, tool_name=tool_name, exit_code=code,
                             result=result, error=error_obj)
        except AuditFinalizationError as exc:
            traceback.print_exc(file=sys.stderr)
            if code == 0:
                # The work (commit, push, promotion, job, ...) already happened. Say so explicitly
                # and hand back its result, so the caller never retries it as if it had not.
                manual = exc.recovery_record is None
                result = {
                    'tool': tool_name,
                    'side_effect_succeeded': True,
                    'audit_finalization': 'manual_recovery_required' if manual else 'pending_recovery',
                    'manual_recovery_required': manual,
                    'detail': str(exc),
                    'recovery_record': exc.recovery_record,
                    'do_not_retry': True,
                    'next_step': (
                        'STOP the campaign: the side effect succeeded but no recovery record could be '
                        'written, so the audit ledger needs user reconciliation; report this full '
                        'result to the user'
                        if manual else
                        'do not retry; the next agent_tools call replays the recovery record'
                    ),
                    'result': result,
                }
                code = EXIT_AUDIT_PENDING
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            if code == 0:
                result, code = {'tool': tool_name, 'error': 'internal',
                                'detail': f'audit log finalization failed: {exc}'}, 3
    print(json.dumps(result, separators=(',', ':'), allow_nan=False))
    return code


if __name__ == '__main__':
    raise SystemExit(main())
