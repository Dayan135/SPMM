"""Run one independent allocation of the frozen Eran confirmation panel."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from scripts.research_panel.full_sweep import run


CONTROL_CASES = ('pwtk', 'webbase', 'roadnet', 'fem2d', 'fem2d_1m',
                 'fem2d_permuted', 'fem2d_permuted_1m', 'diag_b1.0',
                 'diag_b1.0_1m', 'diag_dd0.05')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True,
                        help='screening root containing confirmation_manifest.json')
    parser.add_argument('--allocation-index', type=int, choices=(1, 2, 3), required=True)
    parser.add_argument('--cache', type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.root / 'confirmation_manifest.json').read_text())
    root = args.root / 'confirm' / f'alloc{args.allocation_index}'
    root.mkdir(parents=True, exist_ok=False)
    (root / 'labels').symlink_to(args.root / 'labels', target_is_directory=True)
    metadata = dict(status='running', allocation_index=args.allocation_index,
                    slurm_job_id=os.environ.get('SLURM_JOB_ID'),
                    order_seed=100 + args.allocation_index, completed_tasks=0,
                    tasks=manifest['tasks'])
    path = root / 'status.json'
    path.write_text(json.dumps(metadata, indent=2) + '\n')
    try:
        for index, task in enumerate(manifest['tasks']):
            print(f'confirmation allocation {args.allocation_index} '
                  f'task {index+1}/{len(manifest["tasks"])} {task}', flush=True)
            run(root, task, 'matmul', args.cache, index,
                features=manifest['features'], groups=manifest['groups'],
                warmup=manifest['warmup'], repeats=manifest['repeats'],
                order_seed=metadata['order_seed'])
            metadata['completed_tasks'] = index + 1
            path.write_text(json.dumps(metadata, indent=2) + '\n')
        command = [sys.executable, '-m', 'benchmark.bench_tiles',
                   '--labels', str(args.root / 'labels'), '--cache', str(args.cache),
                   '--out', str(root / 'control'), '--allocation', str(args.allocation_index),
                   '--cases', *CONTROL_CASES, '--features', '128', '256', '512',
                   '--configs', '64:4:512', '64:4:1024', '--no-anchor',
                   '--order', 'shuffle', '--order-seed', str(metadata['order_seed']),
                   '--warmup', '5', '--repeats', '20']
        subprocess.run(command, check=True)
        metadata['status'] = 'complete'
    except BaseException as exc:
        metadata.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        path.write_text(json.dumps(metadata, indent=2) + '\n')


if __name__ == '__main__':
    main()
