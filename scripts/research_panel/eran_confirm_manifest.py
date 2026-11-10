"""Freeze the focused confirmation panel after the screening results exist.

For each predeclared input, choose one nonidentity clustering method by its
fastest F512/P64 screening BOO time across k and FPT. Confirm identity and that
method at every requested F and k, on three independent allocations. This
selection is deliberately based on screening and must be reported as such.
"""
import argparse
import csv
import json
from pathlib import Path

REAL_FOCUS = ('Boeing/pwtk', 'Williams/webbase-1M', 'SNAP/roadNet-CA')
SYNTH_FOCUS = ('fem2d', 'fem2d_permuted', 'diag_b1.0', 'diag_dd0.05')
SIZES = (262144, 1048576)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    args = parser.parse_args()
    with (args.root / 'aggregate' / 'summary.csv').open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    tasks = []
    decisions = []
    inputs = [dict(kind='real', matrix=name) for name in REAL_FOCUS]
    inputs += [dict(kind='generated', profile=profile, size=size)
               for size in SIZES for profile in SYNTH_FOCUS]
    for item in inputs:
        matrix = item['matrix'] if item['kind'] == 'real' else (
            f"generated/{item['profile']}_n{item['size']}")
        candidates = [r for r in rows if r['matrix'] == matrix and
                      r['cluster'] in ('hem', 'graclus', 'metis') and
                      r['algorithm'] == 'boo' and r['scope'] == 'full_call' and
                      r['correctness'] == 'passed' and r['median_ms']]
        best = None
        for features in ('512', '256', '128'):
            at_f = [r for r in candidates if r['features'] == features]
            if at_f:
                best = min(at_f, key=lambda r: float(r['median_ms']))
                break
        method = best['cluster'] if best else None
        decisions.append(dict(matrix=matrix, selected_method=method,
                              screening_label=best['labeling'] if best else None,
                              screening_features=best['features'] if best else None,
                              screening_fpt=best['features_per_thread'] if best else None,
                              screening_ms=best['median_ms'] if best else None,
                              task_index=best['task_index'] if best else None,
                              reason=None if best else 'no feasible nonidentity P64 screen'))
        tasks.extend([dict(item, method=m) for m in ('identity', method) if m])
    manifest = dict(selection='fastest nonidentity method at F512 in screening; '
                              'fallback to F256 then F128 only if no F512 candidate; '
                              'fixed for every F, k and allocation',
                    allocations=3, features=[128, 256, 512], groups=[32, 64, 128],
                    fpt=[1, 2, 4], p=64, slab=512, threads=256, warmup=5,
                    repeats=20, seed=0, tasks=tasks, decisions=decisions)
    out = args.root / 'confirmation_manifest.json'
    if out.exists():
        raise SystemExit(f'refusing to overwrite frozen selection: {out}')
    out.write_text(json.dumps(manifest, indent=2) + '\n')
    print(out)


if __name__ == '__main__':
    main()
