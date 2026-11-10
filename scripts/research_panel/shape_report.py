"""Report the block-shape GPU runs against the rules fixed in the experiment README.

  python scripts/research_panel/shape_report.py --results RUN/results --audits RUN/launch_audit \
      --out ANALYSIS [--relative GATE.csv]

Speedup is time(reference) / time(arm), so above 1 is faster than the reference.

Accepted arms: correct, and the launched kernel verified by Kineto in the timed run or by the
Nsight Systems audit of the same cell (Kineto captures nothing in some long F512 cells).
Valid cells: every _dup and _pad0 control within [0.95, 1.05] of its base arm. This rule was added
after the first screen, when one allocation showed 2x swings between identical arms; a failing
(stage, allocation, cell) is dropped whole and listed.
Occupancy: _ctaN against _pad0 for the same arm, cell and allocation. Identical work.
Screen: screen allocations plus the kp allocations restricted to P32-P128 (the screen grid). Per
  (input, F, allocation) the best arm of each shape against the best 32x32 arm. A finalist exceeds
  1.05 in every allocation and has at least two distinct GPUs. The cross-allocation columns pick both
  arms on one allocation and time them on another, removing the selection bias of a minimum.
k by P: kp allocations only. Best arm per (shape, P) against the best 32x32 arm at any P.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

MARGIN = 1.05
CONTROL_BAND = (0.95, 1.05)


def load(results, stage):
    frames = []
    for path in sorted(Path(results, stage).glob('alloc*/task_*/summary.csv')):
        d = pd.read_csv(path, dtype={'k': str})
        d['alloc'] = f'{stage}:{path.parts[-3]}'
        d['stage_name'] = stage
        frames.append(d)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def nsys_verified(audits, stage):
    ok = set()
    for path in Path(audits, stage).glob('task_*/audit.json'):
        a = json.loads(path.read_text())
        ok |= {r['arm'] for r in a['arms'] if r['status'] == 'verified-nsys'}
    return ok


def arm_tag(d):
    d = d.copy()
    d['tag'] = d.arm.str.split(':', n=1).str[1]
    d['base'] = d.tag.str.replace(r'_(dup|pad0|cta\d+)$', '', regex=True)
    d['suffix'] = d.tag.str.extract(r'_(dup|pad0|cta\d+)$')[0].fillna('')
    return d


def accept(d, audits, name, out):
    stages = d.stage_name.unique()
    nsys = set().union(*(nsys_verified(audits, s) for s in stages)) if audits else set()
    m = d[d.status == 'measured'].copy()
    m['launch_ok'] = (m.launch_verified == 'verified') | m.arm.isin(nsys)
    gpus = d.groupby('alloc').agg(gpus=('gpu_uuid', 'nunique'), tasks=('task', 'nunique'), cells=('cell', 'nunique'))
    t = gpus.join(d.groupby(['alloc', 'status']).size().unstack(fill_value=0))
    t['kineto_verified'] = m[m.launch_verified == 'verified'].groupby('alloc').size()
    t['nsys_only'] = m[m.launch_ok & (m.launch_verified != 'verified')].groupby('alloc').size()
    t['unverified'] = m[~m.launch_ok].groupby('alloc').size()
    t = t.fillna(0).astype(int)
    t.to_csv(out / f'{name}_audit.csv')
    print(f'\n== {name} audit\n{t.to_string()}')
    problems = d[~d.status.isin(['measured', 'unsupported'])]
    if len(problems):
        problems.to_csv(out / f'{name}_failed.csv', index=False)
        print(problems[['alloc', 'arm', 'features', 'status', 'reason']].to_string(index=False))
    un = m[~m.launch_ok]
    if len(un):
        un.to_csv(out / f'{name}_unverified.csv', index=False)
        print('still unverified, by alloc/input/F:', un.groupby(['alloc', 'input', 'features']).size().to_dict())
    return arm_tag(m[m.launch_ok])


def valid_cells(m, name, out):
    key = ['alloc', 'input', 'k', 'features', 'base']
    base = m[m.suffix == ''].set_index(key).median_ms
    c = m[m.suffix.isin(['dup', 'pad0'])].join(base.rename('base_ms'), on=key)
    c['ratio'] = c.median_ms / c.base_ms
    c.to_csv(out / f'{name}_controls.csv', index=False)
    print(f'\n== {name} controls, arm / base time')
    print(c.groupby('suffix').ratio.describe()[['count', 'min', '50%', 'max']].round(4).to_string())
    lo, hi = CONTROL_BAND
    bad = c[(c.ratio < lo) | (c.ratio > hi)][['alloc', 'cell', 'gpu_uuid', 'tag', 'ratio']]
    if len(bad):
        bad.to_csv(out / f'{name}_invalid_cells.csv', index=False)
        print('dropped (alloc, cell) for controls outside the band:')
        print(bad.round(3).to_string(index=False))
    drop = set(zip(bad.alloc, bad.cell))
    keep = ~pd.Series([(a, c) in drop for a, c in zip(m.alloc, m.cell)], index=m.index)
    return m[keep]


def occupancy(m, out):
    key = ['alloc', 'input', 'features', 'base']
    base = m[m.suffix == 'pad0'].set_index(key)
    p = m[m.suffix.str.startswith('cta')].join(base[['median_ms', 'max_active_blocks_per_sm']].rename(
        columns={'median_ms': 'pad0_ms', 'max_active_blocks_per_sm': 'pad0_cta'}), on=key)
    p['slowdown'] = p.median_ms / p.pad0_ms
    p['thread_fraction'] = (p.max_active_blocks_per_sm / p.pad0_cta).round(3)
    p.to_csv(out / 'occupancy_all.csv', index=False)
    print('\n== occupancy, padded / pad0 time for identical work (fraction of pad0 threads/SM kept)')
    print(p.groupby(['features', 'threads', 'thread_fraction']).slowdown
          .agg(['size', 'min', 'median', 'max']).round(3).to_string())
    t = p.groupby(['input', 'features', 'threads', 'thread_fraction']).slowdown.median().round(3).unstack('thread_fraction')
    t.to_csv(out / 'occupancy_by_input.csv')
    print(t.to_string())


def best_per(d, keys):
    return d.loc[d.groupby(keys).median_ms.idxmin()]


def screen(m, out, relative):
    d = m[(m.suffix == '') & (m.p >= 32)].rename(columns={'k': 'shape'})
    best = best_per(d, ['alloc', 'input', 'features', 'shape'])
    ref = best[best['shape'] == '32x32'].set_index(['alloc', 'input', 'features'])
    b = best.join(ref[['median_ms', 'base']].rename(columns={'median_ms': 'ref_ms', 'base': 'ref_arm'}),
                  on=['alloc', 'input', 'features'])
    b['speedup'] = b.ref_ms / b.median_ms
    b.to_csv(out / 'screen_best.csv', index=False)

    pivot = d.set_index(['alloc', 'input', 'features', 'shape', 'base']).median_ms
    cross = []
    for r in b.itertuples():
        for other in b[(b.input == r.input) & (b.features == r.features)].alloc.unique():
            if other == r.alloc:
                continue
            try:
                t = pivot.loc[(other, r.input, r.features, r.shape, r.base)]
                t0 = pivot.loc[(other, r.input, r.features, '32x32', r.ref_arm)]
            except KeyError:
                continue
            cross.append(dict(input=r.input, features=r.features, shape=r.shape, cross_speedup=t0 / t))
    table = (b.groupby(['input', 'features', 'shape'])
             .agg(best_min=('speedup', 'min'), best_med=('speedup', 'median'), best_max=('speedup', 'max'),
                  allocs=('alloc', 'nunique'), gpus=('gpu_uuid', 'nunique'),
                  arm=('base', lambda s: '/'.join(sorted(set(s)))))
             .reset_index())
    if cross:
        cs = (pd.DataFrame(cross).groupby(['input', 'features', 'shape']).cross_speedup
              .agg(cross_min='min', cross_med='median').reset_index())
        table = table.merge(cs, how='left', on=['input', 'features', 'shape'])
    table['finalist'] = (table.best_min > MARGIN) & (table.gpus >= 2) & (table.allocs >= 2)
    if relative is not None:
        g = pd.read_csv(relative)
        frozen = d[['matrix', 'method', 'input']].drop_duplicates()
        g = g.merge(frozen, on=['matrix', 'method'])[['input', 'shape', 'x_staged_rel', 'y_commits_rel', 'passes']]
        table = table.merge(g, how='left', on=['input', 'shape'])
    table = table.round(3)
    table.to_csv(out / 'screen_table.csv', index=False)
    print('\n== screen (P32-P128), best arm vs best 32x32 arm per allocation')
    print(table[table['shape'] != '32x32'].to_string(index=False))
    print('\n== min over allocations')
    print(table.pivot_table(index=['input', 'features'], columns='shape', values='best_min').round(3).to_string())
    fin = table[table.finalist]
    fin.to_csv(out / 'screen_finalists.csv', index=False)
    print(f"\nfinalists: {len(fin)} of {len(table[table['shape'] != '32x32'])} (input, F, shape)")
    print(fin[['input', 'features', 'shape', 'best_min', 'best_med', 'cross_min', 'allocs', 'gpus', 'arm']].to_string(index=False))
    t = table[table['shape'] != '32x32'].dropna(subset=['x_staged_rel'])
    if len(t):
        print('spearman best_med vs x_staged_rel:', round(t.best_med.corr(t.x_staged_rel, method='spearman'), 3))
        print(t.groupby('passes').best_med.agg(['size', 'median', 'max']).round(3).to_string())


def kp_matrix(m, out):
    d = m[(m.stage_name == 'kp') & (m.suffix == '')].rename(columns={'k': 'shape'})
    if not len(d):
        return
    best = best_per(d, ['alloc', 'input', 'features', 'shape', 'p'])
    ref = best[best['shape'] == '32x32'].groupby(['alloc', 'input', 'features']).median_ms.min().rename('ref_ms')
    best = best.join(ref, on=['alloc', 'input', 'features'])
    best['speedup'] = best.ref_ms / best.median_ms
    best['threads_sm'] = best.max_active_blocks_per_sm * best.threads
    best.to_csv(out / 'kp_best.csv', index=False)
    t = (best.groupby(['input', 'features', 'shape', 'p'])
         .agg(speedup_min=('speedup', 'min'), speedup_med=('speedup', 'median'), allocs=('alloc', 'nunique'),
              threads_sm=('threads_sm', 'first'), arena=('arena_bytes', 'first'))
         .reset_index().round(3))
    t.to_csv(out / 'kp_matrix.csv', index=False)
    print('\n== k by P: best arm per (shape, P) vs best 32x32 at any P, min over kp allocations')
    print(t.pivot_table(index=['input', 'features', 'shape'], columns='p', values='speedup_min').round(3).to_string())
    win = best_per(best, ['alloc', 'input', 'features'])
    print('\nwinning (shape, P) per input and F, by allocation:')
    print(win.assign(w=win['shape'] + '/P' + win.p.astype(str) + ' ' + win.speedup.round(3).astype(str))
          .pivot_table(index=['input', 'features'], columns='alloc', values='w', aggfunc='first').to_string())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--results', type=Path, required=True)
    ap.add_argument('--audits', type=Path, default=None)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--relative', type=Path, default=None)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    pd.set_option('display.width', 240, 'display.max_columns', 40, 'display.max_rows', 500, 'display.max_colwidth', 44)
    occ = load(a.results, 'occupancy')
    if len(occ):
        m = valid_cells(accept(occ, a.audits, 'occupancy', a.out), 'occupancy', a.out)
        occupancy(m, a.out)
    scr = pd.concat([load(a.results, 'screen'), load(a.results, 'kp')], ignore_index=True)
    if len(scr):
        m = valid_cells(accept(scr, a.audits, 'screen', a.out), 'screen', a.out)
        screen(m, a.out, a.relative)
        kp_matrix(m, a.out)


if __name__ == '__main__':
    main()
