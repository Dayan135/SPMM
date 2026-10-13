"""Aggregate full-sweep shards; incomplete tasks remain explicit in report.md."""
import argparse
import csv
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('root', type=Path)
    a = p.parse_args()
    tasks = json.loads((a.root / 'sweep.json').read_text())['tasks']
    all_rows, status = [], []
    for i, task in enumerate(tasks):
        folder = a.root / 'results' / f'task_{i:03d}'
        meta = json.loads((folder / 'metadata.json').read_text()) if (folder / 'metadata.json').exists() else {}
        status.append(dict(task_index=i, **task, status=meta.get('status', 'not_started'),
                           completed_cells=meta.get('completed_cells', 0), error=meta.get('error')))
        if (folder / 'summary.csv').exists():
            with (folder / 'summary.csv').open() as f:
                all_rows.extend(dict(task_index=i, **r) for r in csv.DictReader(f))
    out = a.root / 'aggregate'
    out.mkdir(exist_ok=True)
    (out / 'status.json').write_text(json.dumps(status, indent=2) + '\n')
    if all_rows:
        with (out / 'summary.csv').open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(all_rows[0]))
            w.writeheader()
            w.writerows(all_rows)
    passed = [r for r in all_rows if r.get('correctness') == 'passed' and r.get('median_ms')]
    boo = [r for r in passed if r['algorithm'] == 'boo' and r['scope'] == 'full_call']
    lines = ['# Full FPT sweep', '',
             f"Completed tasks: {sum(r['status'] == 'complete' for r in status)}/{len(tasks)}. "
             f'Validated BOO full-call rows: {len(boo)}/1872 requested combinations.', '',
             'Requested duplicates, unsupported variants and incomplete tasks are retained in raw data/status. '
             'Each ratio below uses original-order direct cuSPARSE in the same task, labeling and feature cell. '
             'Winners are post-hoc; this is one allocation per task, not independent replication.', '',
             '| Matrix | F | Clustering | k | FPT | BOO ms | Best measured direct ms | Direct / BOO | Direct algorithms measured |',
             '|---|---:|---|---:|---:|---:|---:|---:|---:|']
    for matrix, features in sorted({(r['matrix'], int(r['features'])) for r in boo}):
        candidates = [r for r in boo if r['matrix'] == matrix and int(r['features']) == features]
        b = min(candidates, key=lambda r: float(r['median_ms']))
        direct = [r for r in passed if r['task_index'] == b['task_index'] and
                  r['matrix'] == matrix and r['features'] == b['features'] and
                  r['labeling'] == b['labeling'] and r['scope'] == 'full_call' and
                  r['algorithm'] == 'cusparse_direct' and r['config'].startswith('direct_') and
                  not r['config'].startswith('direct_perm_')]
        bt = float(b['median_ms'])
        dt = min((float(r['median_ms']) for r in direct), default=None)
        lines.append(f"| {matrix} | {features} | {b['cluster']} | {b['k']} | {b['features_per_thread']} | "
                     f"{bt:.4f} | {f'{dt:.4f}' if dt else '-'} | {f'{dt/bt:.3f}' if dt else '-'} | {len(direct)} / 4 |")
    lines += ['', 'See summary.csv for every configuration, matching C++ scopes, reordered baselines, '
              'permutation costs, validation and skip reasons; status.json identifies missing/failed tasks.', '']
    (out / 'report.md').write_text('\n'.join(lines))


if __name__ == '__main__':
    main()
