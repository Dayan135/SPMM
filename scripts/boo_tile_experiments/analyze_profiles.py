"""Parse a profile_cases.sh output directory into tidy CSVs.

  python analyze_profiles.py <profile dir>     -> <dir>/analysis/{ncu_long,ncu_wide,nsys_kernels,nsys_calls}.csv

ncu: every (stem, pass, launch ID, metric) is kept in ncu_long.csv; ncu_wide.csv has
one row per (stem, kernel launch) with the metrics from all passes joined by kernel
name and launch order (passes profile the same deterministic call sequence).
nsys: kernel/memset intervals inside the captured range. Per call: GPU span (union of
intervals, NOT the sum), sum of kernel times, overlap = sum - union, memset time,
idle gaps inside the span, and the NVTX range duration (host-inclusive call time).
Profiler-perturbed durations: diagnosis only, never the performance result.
"""
import csv
import re
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

PASSES = ('stalls', 'occ', 'mem', 'instmix', 'atomics')
STEM = re.compile(r'(.+)_(stalls|occ|mem|instmix|atomics)\.csv$')


def kernel_role(name):
    if 'boo_spmm_rowseg' in name or 'boo_spmm_nnzflat' in name:
        return 'boo_main'
    if 'boo_spmm_tail' in name:
        return 'boo_tail'
    if 'csr_spmm' in name:
        return 'csr_remainder'
    if 'Memset' in name or 'memset' in name:
        return 'memset'
    if 'cusparse' in name.lower() or 'csrmm' in name.lower() or 'spmm' in name.lower():
        return 'cusparse'
    return 'other'


def number(text):
    try:
        return float(text.replace(',', ''))
    except ValueError:
        return None


def read_ncu(path):
    lines = path.read_text(errors='replace').splitlines()
    start = next((i for i, l in enumerate(lines) if l.startswith('"ID"')), None)
    if start is None:
        return []
    out = []
    for row in csv.DictReader(lines[start:]):
        if not (row.get('ID') or '').isdigit():  # interleaved ==WARNING== lines
            continue
        out.append(dict(id=int(row['ID']), kernel=row['Kernel Name'], section=row.get('Section Name', ''),
                        metric=row['Metric Name'], unit=row.get('Metric Unit', ''),
                        value=row['Metric Value'], grid=row.get('Grid Size', ''),
                        block=row.get('Block Size', '')))
    return out


def ncu(directory, outdir):
    long_rows, wide = [], defaultdict(dict)
    for path in sorted(directory.glob('*.csv')):
        m = STEM.match(path.name)
        if not m:
            continue
        stem, pas = m.group(1), m.group(2)
        rows = read_ncu(path)
        ids = sorted({r['id'] for r in rows})
        order = {i: n for n, i in enumerate(ids)}
        for r in rows:
            long_rows.append(dict(stem=stem, pass_=pas, launch=order[r['id']], role=kernel_role(r['kernel']),
                                  **{k: r[k] for k in ('kernel', 'section', 'metric', 'unit', 'value',
                                                       'grid', 'block')}))
            key = (stem, order[r['id']])
            w = wide[key]
            w.update(stem=stem, launch=order[r['id']], kernel=r['kernel'][:160],
                     role=kernel_role(r['kernel']), grid=r['grid'], block=r['block'])
            name = r['metric'] if pas != 'mem' else f"{r['section']}:{r['metric']}"
            if name in w and w.get(f'_{name}_pass') != pas:
                name = f'{pas}:{name}'
            w[name] = r['value']
            w[f'_{name}_pass'] = pas
    with (outdir / 'ncu_long.csv').open('w', newline='') as f:
        fields = ['stem', 'pass_', 'launch', 'role', 'kernel', 'section', 'metric', 'unit', 'value',
                  'grid', 'block']
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(long_rows)
    rows = [{k: v for k, v in w.items() if not k.startswith('_')} for w in wide.values()]
    fields = sorted({k for r in rows for k in r}, key=lambda k: (k not in ('stem', 'launch', 'role',
                                                                          'kernel', 'grid', 'block'), k))
    with (outdir / 'ncu_wide.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    return rows


def union_length(intervals):
    total, cur_s, cur_e = 0, None, None
    for s, e in sorted(intervals):
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                total += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    if cur_e is not None:
        total += cur_e - cur_s
    return total


def nsys(directory, outdir):
    kernels, calls = [], []
    for path in sorted(directory.glob('*_nsys.sqlite')):
        stem = path.name[:-len('_nsys.sqlite')]
        con = sqlite3.connect(path)
        tables = {r[0] for r in con.execute("select name from sqlite_master where type='table'")}
        items = []
        if 'CUPTI_ACTIVITY_KIND_KERNEL' in tables:
            for s, e, stream, name in con.execute(
                    'select k.start, k.end, k.streamId, s.value from CUPTI_ACTIVITY_KIND_KERNEL k '
                    'join StringIds s on k.demangledName = s.id'):
                items.append((s, e, stream, name, 'kernel'))
        if 'CUPTI_ACTIVITY_KIND_MEMSET' in tables:
            for s, e, stream in con.execute('select start, end, streamId from CUPTI_ACTIVITY_KIND_MEMSET'):
                items.append((s, e, stream, 'Memset', 'memset'))
        nvtx = []
        if 'NVTX_EVENTS' in tables:
            nvtx = [r for r in con.execute('select start, end from NVTX_EVENTS where end is not null')]
        api = []
        if 'CUPTI_ACTIVITY_KIND_RUNTIME' in tables:
            api = list(con.execute('select r.start, r.end, s.value from CUPTI_ACTIVITY_KIND_RUNTIME r '
                                   'join StringIds s on r.nameId = s.id'))
        con.close()
        if not items:
            calls.append(dict(stem=stem, status='no GPU activity captured'))
            continue
        t0 = min(i[0] for i in items)
        for s, e, stream, name, kind in sorted(items):
            kernels.append(dict(stem=stem, kind=kind, role=kernel_role(name), stream=stream,
                                start_us=(s - t0) / 1e3, dur_us=(e - s) / 1e3, name=name[:160]))
        iv = [(i[0], i[1]) for i in items]
        span = max(i[1] for i in items) - t0
        busy = union_length(iv)
        ksum = sum(e - s for s, e, *_ in items)
        main = [i for i in items if kernel_role(i[3]) in ('boo_main', 'cusparse')]
        rem = [i for i in items if kernel_role(i[3]) == 'csr_remainder']
        mset = [i for i in items if i[4] == 'memset']
        row = dict(stem=stem, status='ok', streams=len({i[2] for i in items}), launches=len(items),
                   gpu_span_us=span / 1e3, gpu_busy_union_us=busy / 1e3, sum_of_durations_us=ksum / 1e3,
                   overlap_us=(ksum - busy) / 1e3, idle_gaps_in_span_us=(span - busy) / 1e3,
                   memset_us=sum(e - s for s, e, *_ in mset) / 1e3,
                   main_us=sum(e - s for s, e, *_ in main) / 1e3,
                   remainder_us=sum(e - s for s, e, *_ in rem) / 1e3)
        if main and rem:
            ms, me = min(i[0] for i in main), max(i[1] for i in main)
            rs, re_ = min(i[0] for i in rem), max(i[1] for i in rem)
            row['main_remainder_overlap_us'] = max(0, min(me, re_) - max(ms, rs)) / 1e3
            row['remainder_ends_after_main_us'] = (re_ - me) / 1e3
        if mset and main:
            row['memset_end_to_main_start_us'] = (min(i[0] for i in main) - max(i[1] for i in mset)) / 1e3
        if nvtx:
            s, e = max(nvtx, key=lambda r: r[1] - r[0])
            row['nvtx_call_us'] = (e - s) / 1e3
            row['host_before_first_gpu_us'] = (t0 - s) / 1e3
            row['nvtx_end_minus_gpu_end_us'] = (e - max(i[1] for i in items)) / 1e3
            inside = [a for a in api if a[0] >= s and a[1] <= e]
            row['runtime_api_calls'] = len(inside)
            row['runtime_api_us'] = sum(a[1] - a[0] for a in inside) / 1e3
            row['runtime_api_names'] = ';'.join(sorted({a[2].split('_v')[0] for a in inside}))
        calls.append(row)
    for name, rows in (('nsys_kernels.csv', kernels), ('nsys_calls.csv', calls)):
        fields = list(dict.fromkeys(k for r in rows for k in r))
        with (outdir / name).open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
    return calls


def main(directory):
    directory = Path(directory)
    outdir = directory / 'analysis'
    outdir.mkdir(exist_ok=True)
    rows = ncu(directory, outdir)
    calls = nsys(directory, outdir)
    print(f'{len(rows)} ncu kernel rows, {len(calls)} nsys calls -> {outdir}')


if __name__ == '__main__':
    main(sys.argv[1])
