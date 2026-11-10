"""Summarize the block-shape CPU gate. Ratios are against the same input and method at 32x32.

  python scripts/research_panel/shape_gate_report.py --gate RUN/gate --out ANALYSIS

Pass rule, fixed before the panel finished. A rectangular shape passes on an (input, method) when
  x_staged <= 0.80, y_commits <= 1.10, 100% threads/SM is reachable at P64, and the
  remainder fraction does not rise by more than 0.005.
A pass selects a shape for GPU timing. It says nothing about time.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

X_MAX, Y_MAX, REM_SLACK, FULL_THREADS = 0.80, 1.10, 0.005, 1536
RECT = ['64x32', '128x32', '256x32', '128x64', '256x64', '64x16', '128x16', '256x16']


def geomean(s):
    s = s[s > 0]
    return float(np.exp(np.log(s).mean())) if len(s) else float('nan')


def load(gate):
    frames = [pd.read_csv(p) for p in sorted(Path(gate).glob('*.csv'))]
    d = pd.concat(frames, ignore_index=True)
    bad = d[d.status != 'ok']
    d = d[d.status == 'ok'].copy()
    d['shape'] = d.kr.astype(int).astype(str) + 'x' + d.kc.astype(int).astype(str)
    return d, bad


def relative(d):
    base = d[d['shape'] == '32x32'].set_index(['matrix', 'method'])
    out = d.join(base[['x_staged', 'x_read', 'y_commits', 'arena_p64', 'remainder_fraction', 'num_blocks',
                       'busiest_sum_t256_p64_f2', 'busiest_sum_t256_p64_f4']],
                 on=['matrix', 'method'], rsuffix='_base')
    for col in ('x_staged', 'x_read', 'y_commits', 'arena_p64', 'num_blocks'):
        out[f'{col}_rel'] = out[col] / out[f'{col}_base']
    out['x_read_of_base_staged'] = out.x_read / out.x_staged_base
    out['read_fraction'] = out.x_read / out.x_staged
    out['remainder_delta'] = out.remainder_fraction - out.remainder_fraction_base
    out['busiest_f2_rel'] = out.busiest_sum_t256_p64_f2 / out.busiest_sum_t256_p64_f2_base
    out['passes'] = ((out.x_staged_rel <= X_MAX) & (out.y_commits_rel <= Y_MAX)
                     & (out.best_threads_sm_p64 >= FULL_THREADS) & (out.remainder_delta <= REM_SLACK))
    return out


def family(matrix):
    if matrix.startswith('generated/'):
        return 'synthetic_1m' if matrix.endswith('_n1048576') else 'synthetic_262k'
    return 'real'


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--gate', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    d, bad = load(a.gate)
    r = relative(d)
    r['family'] = r.matrix.map(family)
    r.to_csv(a.out / 'gate_relative.csv', index=False)
    if len(bad):
        bad.to_csv(a.out / 'gate_unsupported.csv', index=False)

    pd.set_option('display.width', 230, 'display.max_columns', 40, 'display.max_rows', 400)
    print(f'{len(d)} shapes, {d.matrix.nunique()} inputs, {len(bad)} unsupported')

    by_shape = (r[r['shape'].isin(['32x32', '64x64', '128x128', '32x64', '32x128', '16x16'] + RECT)]
                .groupby(['family', 'shape'])
                .agg(n=('passes', 'size'), passes=('passes', 'sum'),
                     x_staged=('x_staged_rel', geomean), x_read=('x_read_of_base_staged', geomean),
                     y_commits=('y_commits_rel', geomean), arena=('arena_p64_rel', geomean),
                     full_thread_cells=('best_threads_sm_p64', lambda s: int((s >= FULL_THREADS).sum())))
                .round(3))
    by_shape.to_csv(a.out / 'gate_by_shape.csv')
    print(by_shape.to_string())

    real = r[(r.family == 'real') & r['shape'].isin(['32x32', '64x32', '128x32', '256x32', '128x16', '128x128'])]
    cols = ['matrix', 'method', 'shape', 'num_blocks', 'nnz_per_block_p50', 'max_nnz_blk', 'arena_p64',
            'best_threads_sm_p64', 'x_staged_rel', 'x_read_of_base_staged', 'read_fraction', 'y_commits_rel',
            'remainder_fraction', 'balance_gain_t256_p64_f2', 'passes']
    table = real[cols].sort_values(['matrix', 'method', 'shape']).round(3)
    table.to_csv(a.out / 'gate_real.csv', index=False)
    print(table.to_string(index=False))

    frozen = {'Boeing/pwtk': 'identity', 'Williams/webbase-1M': 'identity', 'SNAP/roadNet-CA': 'metis',
              'generated/fem2d_n262144': 'hem', 'generated/fem2d_permuted_n262144': 'hem',
              'generated/diag_b1.0_n262144': 'identity', 'generated/diag_dd0.05_n262144': 'identity',
              'generated/diag_skewed_n262144': 'identity'}
    rows = r[r.apply(lambda x: frozen.get(x.matrix) == x.method, axis=1)
             & r['shape'].isin(['32x32', '64x32', '128x32', '256x32', '128x64', '128x16', '128x128'])]
    panel = rows[cols].sort_values(['matrix', 'shape']).round(3)
    panel.to_csv(a.out / 'gate_frozen_panel.csv', index=False)
    print('\nFrozen labeling per input (2026-10-14 screen):')
    print(panel.to_string(index=False))

    win = (r[r.passes].groupby(['matrix'])['shape'].agg(lambda s: ','.join(sorted(set(s)))).rename('passing_shapes'))
    no = sorted(set(r.matrix) - set(win.index))
    win.to_csv(a.out / 'gate_passing_shapes.csv')
    print(win.to_string())
    print('inputs with no passing shape:', no)


if __name__ == '__main__':
    main()
