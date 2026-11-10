"""Summarize the frozen Eran confirmation across three Slurm allocations."""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import median


def read(path):
    if not path.exists():
        return []
    with path.open(newline='') as handle:
        return list(csv.DictReader(handle))


def write(path, rows, fields):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value):
    return f'{value:.3f}' if value is not None else '-'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    args = parser.parse_args()
    manifest = json.loads((args.root / 'confirmation_manifest.json').read_text())
    out = args.root / 'confirm' / 'analysis'
    out.mkdir(parents=True, exist_ok=True)
    cells = defaultdict(list)
    allocations = []
    controls = []
    for allocation in (1, 2, 3):
        base = args.root / 'confirm' / f'alloc{allocation}'
        status_path = base / 'status.json'
        status = json.loads(status_path.read_text()) if status_path.exists() else {}
        allocations.append(dict(allocation=allocation, status=status.get('status', 'missing'),
                                completed_tasks=status.get('completed_tasks', 0),
                                slurm_job_id=status.get('slurm_job_id', ''),
                                error=status.get('error', '')))
        for index, task in enumerate(manifest['tasks']):
            rows = read(base / 'results' / f'task_{index:03d}' / 'summary.csv')
            by_cell = defaultdict(list)
            for row in rows:
                by_cell[(row['labeling'], row['features'])].append(row)
            for (labeling, features), entries in by_cell.items():
                direct = [float(r['median_ms']) for r in entries
                          if r['algorithm'] == 'cusparse_direct' and
                          r['scope'] == 'full_call' and
                          r['config'].startswith('direct_') and
                          not r['config'].startswith('direct_perm_') and
                          r['median_ms']]
                baseline = min(direct) if direct else None
                for row in entries:
                    if (row['algorithm'] != 'boo' or row['scope'] != 'full_call'
                            or not row['median_ms']):
                        continue
                    key = (row['matrix'], int(features), row['cluster'],
                           int(row['k']), int(row['features_per_thread']))
                    boo = float(row['median_ms'])
                    cells[key].append(dict(allocation=allocation, boo_ms=boo,
                                           direct_ms=baseline,
                                           speedup=baseline/boo if baseline else None,
                                           p25_ms=float(row['p25_ms']),
                                           p75_ms=float(row['p75_ms']),
                                           label_sha256=row['label_sha256']))
        ctrl = read(base / 'control' / 'summary.csv')
        ctrl_by_cell = defaultdict(dict)
        for row in ctrl:
            if (row['scope'] == 'full_call' and row['median_ms'] and
                    row['config'] in ('boo_p64_fpt4_s512', 'boo_p64_fpt4_s1024')):
                ctrl_by_cell[(row['case'], int(row['features']))][row['config']] = row
        for (case, features), pair in ctrl_by_cell.items():
            if len(pair) != 2:
                continue
            a = float(pair['boo_p64_fpt4_s512']['median_ms'])
            b = float(pair['boo_p64_fpt4_s1024']['median_ms'])
            controls.append(dict(allocation=allocation, case=case, features=features,
                                 slab512_ms=a, slab1024_ms=b, ratio_512_over_1024=a/b))
    write(out / 'allocation_status.csv', allocations, list(allocations[0]))
    write(out / 'control.csv', controls,
          list(controls[0]) if controls else ['allocation', 'case', 'features',
                                              'slab512_ms', 'slab1024_ms',
                                              'ratio_512_over_1024'])
    summaries = []
    for (matrix, features, method, k, fpt), records in sorted(cells.items()):
        ratios = [r['speedup'] for r in records if r['speedup'] is not None]
        summaries.append(dict(matrix=matrix, features=features, method=method,
                              k=k, fpt=fpt, allocations=len(records),
                              median_boo_ms=median(r['boo_ms'] for r in records),
                              min_boo_ms=min(r['boo_ms'] for r in records),
                              max_boo_ms=max(r['boo_ms'] for r in records),
                              median_direct_over_boo=median(ratios) if ratios else '',
                              min_direct_over_boo=min(ratios) if ratios else '',
                              max_direct_over_boo=max(ratios) if ratios else '',
                              direct_allocations=len(ratios)))
    write(out / 'confirmation.csv', summaries,
          list(summaries[0]) if summaries else ['matrix', 'features', 'method',
                                               'k', 'fpt', 'allocations'])
    expected = len(manifest['tasks']) * 3 * 3 * 3
    control_ratio = [r['ratio_512_over_1024'] for r in controls]
    lines = ['# Eran FPT independent confirmation', '',
             f'Confirmed BOO cells with at least one allocation: {len(summaries)}/{expected}.',
             'Allocation statuses: ' + ', '.join(
                 f"{r['allocation']}={r['status']} ({r['completed_tasks']}/{len(manifest['tasks'])} tasks)"
                 for r in allocations) + '.', '',
             'Each row in confirmation.csv reports the median and range of three '
             'independent allocation medians. IQR within one allocation is a '
             'different source of variation; raw samples retain it.', '',
             f'Identical-work slab512/slab1024 control: {len(controls)} paired '
             f'case/F/allocation observations; ratio range '
             f'{fmt(min(control_ratio) if control_ratio else None)}–'
             f'{fmt(max(control_ratio) if control_ratio else None)}. '
             'Both use P64/FPT4 and at F<=512 cover the same feature work; '
             'launch.csv in each control run records the exact launch.', '',
             '## Focused panel', '',
             '| Input | F | Method | k | FPT | BOO ms (allocation median, range) | '
             'Direct / BOO (median, range) | n |',
             '|---|---:|---|---:|---:|---:|---:|---:|']
    for row in summaries:
        if (row['matrix'] in ('Boeing/pwtk', 'Williams/webbase-1M',
                              'SNAP/roadNet-CA') or
                any(x in row['matrix'] for x in ('fem2d_', 'diag_b1.0_',
                                                 'diag_dd0.05_'))):
            if row['fpt'] != 4 or row['k'] != 64:
                continue
            center = row['median_direct_over_boo']
            low = row['min_direct_over_boo']
            high = row['max_direct_over_boo']
            lines.append(f"| {row['matrix']} | {row['features']} | {row['method']} | "
                         f"{row['k']} | {row['fpt']} | {row['median_boo_ms']:.3f} "
                         f"[{row['min_boo_ms']:.3f}, {row['max_boo_ms']:.3f}] | "
                         f"{fmt(center if center != '' else None)} "
                         f"[{fmt(low if low != '' else None)}, "
                         f"{fmt(high if high != '' else None)}] | "
                         f"{row['allocations']} |")
    lines += ['', 'The table is a compact view. confirmation.csv contains all '
              'requested F, k and FPT cells. Method arms share a GPU allocation '
              'but run sequentially, so small method rankings remain sensitive '
              'to within-allocation drift. Screening chose the nonidentity '
              'method at F512 before these measurements.']
    (out / 'report.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines[:4]))


if __name__ == '__main__':
    main()
