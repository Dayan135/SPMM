"""Summarize unprofiled timings and diagnostic traces without mixing scopes.

Usage: python scripts/research_panel/report.py RESULTS_DIRECTORY
The directory contains panel/summary.csv and optional *_nsys.sqlite traces.
"""
import argparse
import csv
import io
import json
from pathlib import Path
import sqlite3


def records(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    if rows:
        with path.open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=sorted(set().union(*(r.keys() for r in rows))))
            writer.writeheader()
            writer.writerows(rows)


def nsys_kernels(path):
    # Read-only: reports must never alter the trace database.
    with sqlite3.connect(f'file:{path.resolve()}?mode=ro', uri=True) as db:
        db.row_factory = sqlite3.Row
        tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if 'CUPTI_ACTIVITY_KIND_KERNEL' not in tables:
            return []
        strings = dict(db.execute('SELECT id, value FROM StringIds'))
        return [dict(trace=path.stem, kernel=strings[r['demangledName']],
                     duration_ms=(r['end'] - r['start']) / 1e6, stream=r['streamId'])
                for r in db.execute('SELECT * FROM CUPTI_ACTIVITY_KIND_KERNEL ORDER BY start')]


def ncu_records(path):
    text = path.read_text()
    lines = text.splitlines()
    start = next((i for i, line in enumerate(lines)
                  if '"Metric Name"' in line and '"Metric Value"' in line), None)
    if start is None:
        return []
    result = []
    for row in csv.DictReader(io.StringIO('\n'.join(lines[start:]))):
        metric = row.get('Metric Name')
        if not metric or not row.get('Kernel Name'):
            continue
        result.append(dict(trace=path.stem, kernel=row['Kernel Name'],
                           launch_id=row.get('ID'), metric=metric,
                           unit=row.get('Metric Unit'), value=row.get('Metric Value')))
    return result


def nsys_activity(path):
    with sqlite3.connect(f'file:{path.resolve()}?mode=ro', uri=True) as db:
        tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        spans = []
        summary = dict(trace=path.stem)
        for table, label in [('CUPTI_ACTIVITY_KIND_KERNEL', 'kernel'),
                             ('CUPTI_ACTIVITY_KIND_MEMSET', 'memset'),
                             ('CUPTI_ACTIVITY_KIND_MEMCPY', 'memcpy')]:
            entries = list(db.execute(f'SELECT start, end FROM {table}')) if table in tables else []
            summary[label + '_count'] = len(entries)
            summary[label + '_sum_ms'] = sum(end - start for start, end in entries) / 1e6
            spans.extend(entries)
        if spans:
            spans.sort()
            total = 0
            start, end = spans[0]
            for next_start, next_end in spans[1:]:
                if next_start > end:
                    total += end - start
                    start, end = next_start, next_end
                else:
                    end = max(end, next_end)
            total += end - start
            summary['gpu_activity_union_ms'] = total / 1e6
            summary['gpu_first_to_last_ms'] = (max(e for _, e in spans) - spans[0][0]) / 1e6
        return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results', type=Path)
    a = parser.parse_args()
    path = a.results / 'panel' / 'summary.csv'
    if not path.exists():
        path = a.results / 'smoke' / 'summary.csv'
    rows = records(path)
    lines = ['# BOO profiling panel', '',
             'Single-session results. Best configurations are post-hoc selections; '
             'speedups use matching timing scopes. Profiler timings are diagnostic.', '',
             '| Profile | N | F | Scope | BOO config | BOO ms | Direct config | Direct ms | Direct / BOO |',
             '|---|---:|---:|---|---|---:|---|---:|---:|']
    cells = sorted({(r['profile'], int(r['requested_size']), int(r['features'])) for r in rows})
    for name, n, f in cells:
        cell = [r for r in rows if (r['profile'], int(r['requested_size']), int(r['features'])) == (name, n, f)
                and r.get('median_ms') and r.get('correctness') == 'passed']
        for scope, ds in [('full_call', 'full_call'), ('cpp_only', 'cusparseSpMM_only')]:
            boo = [r for r in cell if r['algorithm'] == 'boo' and r['timing_scope'] == scope]
            direct = [r for r in cell if r['algorithm'] == 'cusparse_direct' and r['timing_scope'] == ds]
            if not boo or not direct:
                continue
            b, d = (min(v, key=lambda r: float(r['median_ms'])) for v in (boo, direct))
            bt, dt = float(b['median_ms']), float(d['median_ms'])
            lines.append(f"| {name} | {n} | {f} | {scope} | {b['config']} | {bt:.4f} | "
                         f"{d['config']} | {dt:.4f} | {dt / bt:.3f} |")
    fpt_rows = [r for r in rows if r.get('features_per_thread') in ('2', '4') and r.get('median_ms')]
    if fpt_rows:
        lines += ['', '## Paired features-per-thread controls', '',
                  '| Profile | F | Scope | Variant | FPT1 ms | Variant ms | FPT1 / variant |',
                  '|---|---:|---|---|---:|---:|---:|']
        for r in fpt_rows:
            base_name = r['config'].replace('_fpt' + r['features_per_thread'], '')
            base = next(b for b in rows if b['profile'] == r['profile']
                        and b['requested_size'] == r['requested_size'] and b['features'] == r['features']
                        and b['config'] == base_name and b['timing_scope'] == r['timing_scope'])
            bt, vt = float(base['median_ms']), float(r['median_ms'])
            lines.append(f"| {r['profile']} | {r['features']} | {r['timing_scope']} | {r['config']} | "
                         f'{bt:.4f} | {vt:.4f} | {bt / vt:.3f} |')
    lines += ['', 'API BOO uses auto launch. The existing preallocated C++ BOO harness uses '
              'the legacy stream fork/join; both include required output initialization and remainder work.', '',
              '## Block structure', '',
              '| Matrix | Config | Blocks | Staged X rows | Active X rows | Unused X % | Active output rows across blocks |',
              '|---|---|---:|---:|---:|---:|---:|']
    for structure in sorted(path.parent.glob('*_structure.json')):
        for name, stats in json.loads(structure.read_text())['boo'].items():
            lines.append(f"| {structure.stem} | {name} | {stats['num_blocks']} | "
                         f"{stats['logical_x_rows']} | {stats['active_x_rows']} | "
                         f"{100 * stats['unused_x_fraction']:.2f} | {stats['active_y_rows']} |")
    kernels = [row for path in sorted(a.results.glob('*_nsys.sqlite')) for row in nsys_kernels(path)]
    activity = [nsys_activity(path) for path in sorted(a.results.glob('*_nsys.sqlite'))]
    metrics = [row for path in sorted(a.results.glob('*_stalls.csv')) for row in ncu_records(path)]
    metrics += [row for path in sorted(a.results.glob('*_memory.csv')) for row in ncu_records(path)]
    write_csv(a.results / 'profile_kernels.csv', kernels)
    write_csv(a.results / 'profile_activity.csv', activity)
    write_csv(a.results / 'profile_metrics.csv', metrics)
    lines += ['', f'Extracted {len(kernels)} Nsight Systems kernel records and {len(metrics)} '
              'Nsight Compute metric records. Raw traces and per-profile provenance remain beside this report.',
              '', 'Active support is structural: cancellation can reduce actual output atomics. '
              'Logical X counts do not imply equivalent DRAM traffic.', '']
    (a.results / 'report.md').write_text('\n'.join(lines))


if __name__ == '__main__':
    main()
