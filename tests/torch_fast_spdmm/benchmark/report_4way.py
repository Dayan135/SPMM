"""Summarize bench_4way.csv: per-cell table, geomeans, and confound flags.

Usage::  python -m benchmark.report_4way bench_4way.csv
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from math import exp, log

KERNELS = ('boo', 'block', 'nnzflat_shared', 'nnzflat_onthefly')


def geomean(values):
    values = [v for v in values if v and v > 0]
    return exp(sum(log(v) for v in values) / len(values)) if values else float('nan')


def main() -> None:
    path = sys.argv[1] if len(sys.argv) > 1 else 'bench_4way.csv'
    rows = list(csv.DictReader(open(path)))
    if not rows:
        raise SystemExit('no rows')

    for r in rows:
        for k in KERNELS:
            r[f'ms_{k}'] = float(r[f'ms_{k}'])
            r[f'spread_{k}'] = float(r[f'spread_{k}'])
        r['size'] = int(r['size'])
        r['features'] = int(r['features'])
        r['pbr_cf'] = float(r['pbr_compressed_fraction'])
        r['boo_cf'] = float(r['boo_compressed_fraction'])

    print('=' * 118)
    print('Per-cell wall time (ms, median of interleaved rounds).  '
          'CLEAN = PBR compressed_fraction >= 0.98')
    print('=' * 118)
    hdr = (f"{'matrix':22s} {'N':>7s} {'f':>4s} " +
           ' '.join(f'{k:>16s}' for k in KERNELS) +
           f" {'boo/block':>10s}  flag")
    print(hdr)
    for r in sorted(rows, key=lambda r: (r['size'], r['features'], r['matrix'])):
        ratio = r['ms_block'] / r['ms_boo']
        clean = r['pbr_cf'] >= 0.98
        flag = '' if clean else f"CSR-fallback pbr_cf={r['pbr_cf']:.2f}"
        cells = ' '.join(f"{r[f'ms_{k}']:>16.4f}" for k in KERNELS)
        star = '*' if r['best_kernel'] == 'boo' else ' '
        print(f"{r['matrix']:22s} {r['size']:>7d} {r['features']:>4d} {cells} "
              f"{ratio:>9.2f}x{star} {flag}")

    clean_rows = [r for r in rows if r['pbr_cf'] >= 0.98]
    print()
    print('=' * 118)
    print(f'Geomean speedup of BOO over each PBR kernel  '
          f'(>1 = BOO faster).  clean cells only: {len(clean_rows)}/{len(rows)}')
    print('=' * 118)

    def block(title, subset, keyfn):
        if not subset:
            return
        print(f'\n{title}')
        groups = defaultdict(list)
        for r in subset:
            groups[keyfn(r)].append(r)
        print(f"  {'group':28s} {'n':>4s} " +
              ' '.join(f'{"vs " + k:>18s}' for k in KERNELS[1:]))
        for key in sorted(groups):
            g = groups[key]
            cols = ' '.join(
                f"{geomean([r[f'ms_{k}'] / r['ms_boo'] for r in g]):>18.3f}"
                for k in KERNELS[1:])
            print(f'  {str(key):28s} {len(g):>4d} {cols}')

    block('by matrix:', clean_rows, lambda r: r['matrix'])
    block('by size:', clean_rows, lambda r: r['size'])
    block('by features:', clean_rows, lambda r: r['features'])

    if clean_rows:
        print(f"\n  {'OVERALL':28s} {len(clean_rows):>4d} " + ' '.join(
            f"{geomean([r[f'ms_{k}'] / r['ms_boo'] for r in clean_rows]):>18.3f}"
            for k in KERNELS[1:]))

    wins = [r for r in clean_rows if r['best_kernel'] == 'boo']
    print(f'\nBOO is fastest in {len(wins)}/{len(clean_rows)} clean cells.')

    # Run-to-run spread, as a sanity bound on how much of any effect is noise.
    print('\nMax within-cell spread (max-min across repeats), as % of median:')
    for k in KERNELS:
        worst = max(rows, key=lambda r: r[f'spread_{k}'] / max(r[f'ms_{k}'], 1e-9))
        pct = 100 * worst[f'spread_{k}'] / worst[f'ms_{k}']
        print(f'  {k:20s} {pct:6.1f}%   ({worst["matrix"]} N={worst["size"]} '
              f'f={worst["features"]})')


if __name__ == '__main__':
    main()
