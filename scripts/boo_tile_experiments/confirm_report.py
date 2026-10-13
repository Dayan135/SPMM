"""Three-allocation confirmation summary (bench_tiles outputs under confirm/alloc*/<process>/).

  python confirm_report.py <confirm dir> <out dir>

Pairing rules:
- Every ratio is formed inside ONE process (same interleaved cell): reference_ms / variant_ms.
- A cell's direct algorithms may be split across processes of one allocation (memory);
  each process repeats the BOO variants. "Strongest direct" for a variant is the algorithm
  giving it the LOWEST paired speedup over all processes of that allocation and cell.
- Across allocations we report min / median / max of the per-allocation paired ratios;
  20 timing repeats within an allocation are not independent replicates.
- Identical-work control: P64/FPT4 slab512 vs slab1024 at F <= 512 (both grid.z = 1).
"""
import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path

FINALISTS = {'boo_p64_fpt4_s512': 'P64/FPT4', 'boo_p128_fpt4_s512': 'P128/FPT4',
             'boo_p64_fpt2_s512': 'P64/FPT2 (anchor)', 'boo_p64_fpt1_s512': 'P64/FPT1'}
ANCHOR = 'boo_p64_fpt2_s512'
CONTROL = ('boo_p64_fpt4_s512', 'boo_p64_fpt4_s1024')


def main(confirm, out):
    confirm, out = Path(confirm), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in sorted(confirm.glob('alloc*/*/summary.csv')):
        alloc, proc = path.parts[-3], path.parts[-2]
        for r in csv.DictReader(open(path)):
            r['alloc'], r['process'] = alloc, proc
            rows.append(r)
    nodes = {}
    for path in sorted(confirm.glob('alloc*/*/metadata.json')):
        import json
        m = json.loads(path.read_text())
        nodes[(path.parts[-3], path.parts[-2])] = (m.get('node'), m.get('slurm_job_id'), m.get('status'))
    ms = lambda r: float(r['median_ms']) if r.get('median_ms') else None
    cell = defaultdict(lambda: defaultdict(list))  # (alloc, case, F, scope) -> process -> rows
    for r in rows:
        scope = r['scope']
        cell[(r['alloc'], r['case'], int(r['features']), scope)][r['process']].append(r)

    per_alloc = []
    controls = []
    coverage = []
    for (alloc, case, F, scope), procs in sorted(cell.items()):
        boo_suffix = '' if scope == 'full_call' else '_cpp'
        algs_measured, algs_missing = set(), set()
        for proc, group in procs.items():
            for r in group:
                if r['family'] == 'direct':
                    (algs_measured if ms(r) else algs_missing).add(
                        (r['direct_algorithm'], r['unsupported_reason'] if not ms(r) else ''))
        measured = sorted({a for a, _ in algs_measured})
        missing = sorted({a for a, _ in algs_missing} - set(measured))
        coverage.append(dict(alloc=alloc, case=case, features=F, scope=scope,
                             direct_measured=';'.join(measured), direct_missing=';'.join(missing),
                             processes=';'.join(sorted(procs))))
        for name, label in FINALISTS.items():
            cfg = name + boo_suffix
            best = None  # (ratio, alg, process)
            vs_anchor, variant_ms, iqr = None, None, None
            for proc, group in procs.items():
                g = {r['config']: r for r in group}
                v = g.get(cfg)
                if not v or not ms(v):
                    continue
                if vs_anchor is None and g.get(ANCHOR + boo_suffix) and ms(g[ANCHOR + boo_suffix]):
                    vs_anchor = ms(g[ANCHOR + boo_suffix]) / ms(v)
                    variant_ms = ms(v)
                    iqr = (float(v['p25_ms']), float(v['p75_ms']))
                for r in group:
                    if r['family'] == 'direct' and r['scope'] == scope and ms(r):
                        ratio = ms(r) / ms(v)
                        if best is None or ratio < best[0]:
                            best = (ratio, r['direct_algorithm'], proc, ms(r))
            if variant_ms is None:
                continue
            per_alloc.append(dict(alloc=alloc, case=case, features=F, scope=scope, variant=label,
                                  median_ms=round(variant_ms, 4), p25_ms=iqr[0], p75_ms=iqr[1],
                                  vs_anchor=round(vs_anchor, 4) if vs_anchor else None,
                                  vs_strongest_direct=round(best[0], 4) if best else None,
                                  strongest_direct=best[1] if best else None,
                                  strongest_direct_ms=round(best[3], 4) if best else None,
                                  strongest_direct_process=best[2] if best else None,
                                  direct_measured=';'.join(measured), direct_missing=';'.join(missing)))
        for proc, group in procs.items():
            g = {r['config']: r for r in group}
            a, b = (g.get(c + boo_suffix) for c in CONTROL)
            if a and b and ms(a) and ms(b) and F <= 512:
                controls.append(dict(alloc=alloc, case=case, features=F, scope=scope, process=proc,
                                     s512_ms=ms(a), s1024_ms=ms(b), ratio_s512_over_s1024=round(ms(a) / ms(b), 4)))

    def write(name, data):
        if not data:
            return
        with open(out / name, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(data[0]))
            w.writeheader()
            w.writerows(data)
    write('confirm_per_allocation.csv', per_alloc)
    write('identical_work_control.csv', controls)
    write('direct_coverage.csv', coverage)
    agg = defaultdict(list)
    for r in per_alloc:
        agg[(r['case'], r['features'], r['scope'], r['variant'])].append(r)
    summary = []
    for (case, F, scope, variant), g in sorted(agg.items()):
        va = [r['vs_anchor'] for r in g if r['vs_anchor']]
        vd = [r['vs_strongest_direct'] for r in g if r['vs_strongest_direct']]
        summary.append(dict(case=case, features=F, scope=scope, variant=variant, allocations=len(g),
                            median_ms=';'.join(f"{r['median_ms']:.3f}" for r in g),
                            vs_anchor_min=min(va) if va else None,
                            vs_anchor_median=statistics.median(va) if va else None,
                            vs_anchor_max=max(va) if va else None,
                            vs_direct_min=min(vd) if vd else None,
                            vs_direct_median=statistics.median(vd) if vd else None,
                            vs_direct_max=max(vd) if vd else None,
                            strongest_direct=';'.join(str(r['strongest_direct']) for r in g),
                            direct_missing=';'.join(sorted({m for r in g for m in r['direct_missing'].split(';') if m}))))
    write('confirm_summary.csv', summary)
    write('allocations.csv', [dict(alloc=a, process=p, node=n, job=j, status=s)
                              for (a, p), (n, j, s) in sorted(nodes.items())])
    print(f'{len(per_alloc)} per-allocation rows, {len(summary)} summary rows, '
          f'{len(controls)} control rows -> {out}')


if __name__ == '__main__':
    main(*sys.argv[1:])
