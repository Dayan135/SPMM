"""Combine the hem and identity BOO passes and report BOO best-of-two vs PBR.

PBR gets a swept config in every published comparison in this repo (best-of-grid
over block size / min_nnz / smem tier). Pinning BOO to a single clustering method
is therefore not apples-to-apples: it charges BOO for a bad choice on matrices
whose natural order is already good. This takes BOO's better of {hem, identity}
per cell, which is the same courtesy.

Usage::  python -m benchmark.report_4way_combined hem.csv identity.csv
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from math import exp, log

PBR_KERNELS = ('block', 'nnzflat_shared', 'nnzflat_onthefly')


def geomean(values):
    values = [v for v in values if v and v > 0]
    return exp(sum(log(v) for v in values) / len(values)) if values else float('nan')


def load(path):
    out = {}
    for r in csv.DictReader(open(path)):
        key = (r['matrix'], int(r['size']), int(r['features']))
        out[key] = r
    return out


def main() -> None:
    hem = load(sys.argv[1])
    ident = load(sys.argv[2])
    keys = sorted(set(hem) & set(ident), key=lambda k: (k[1], k[2], k[0]))

    merged = []
    for key in keys:
        h, i = hem[key], ident[key]
        pbr_cf = float(h['pbr_compressed_fraction'])
        boo_hem, boo_id = float(h['ms_boo']), float(i['ms_boo'])
        best_ms = min(boo_hem, boo_id)
        merged.append({
            'matrix': key[0], 'size': key[1], 'features': key[2],
            'boo_hem': boo_hem, 'boo_identity': boo_id,
            'boo_best': best_ms,
            'boo_choice': 'hem' if boo_hem <= boo_id else 'identity',
            'pbr_cf': pbr_cf,
            **{k: float(h[f'ms_{k}']) for k in PBR_KERNELS},
        })

    clean = [m for m in merged if m['pbr_cf'] >= 0.98]

    print('=' * 120)
    print('BOO best-of{hem, identity} vs PBR kernels (ms).  * = BOO fastest overall')
    print('=' * 120)
    print(f"{'matrix':22s} {'N':>7s} {'f':>4s} {'boo_hem':>9s} {'boo_id':>9s} "
          f"{'boo_best':>9s} {'pick':>9s} " + ' '.join(f'{k:>17s}' for k in PBR_KERNELS)
          + f" {'boo/block':>10s}")
    for m in merged:
        ratio = m['block'] / m['boo_best']
        star = '*' if m['boo_best'] < min(m[k] for k in PBR_KERNELS) else ' '
        flag = '' if m['pbr_cf'] >= 0.98 else f"  CSR-fallback pbr_cf={m['pbr_cf']:.2f}"
        print(f"{m['matrix']:22s} {m['size']:>7d} {m['features']:>4d} "
              f"{m['boo_hem']:>9.4f} {m['boo_identity']:>9.4f} {m['boo_best']:>9.4f} "
              f"{m['boo_choice']:>9s} "
              + ' '.join(f"{m[k]:>17.4f}" for k in PBR_KERNELS)
              + f" {ratio:>9.2f}x{star}{flag}")

    print()
    print('=' * 120)
    print(f'Geomean speedup of BOO best-of-two over each PBR kernel  '
          f'(>1 = BOO faster).  clean cells: {len(clean)}/{len(merged)}')
    print('=' * 120)

    def block_report(title, keyfn):
        print(f'\n{title}')
        groups = defaultdict(list)
        for m in clean:
            groups[keyfn(m)].append(m)
        print(f"  {'group':26s} {'n':>4s} {'pick':>10s} "
              + ' '.join(f'{"vs " + k:>18s}' for k in PBR_KERNELS))
        for key in sorted(groups):
            g = groups[key]
            picks = [m['boo_choice'] for m in g]
            pick = 'hem' if picks.count('hem') > len(picks) / 2 else 'identity'
            cols = ' '.join(
                f"{geomean([m[k] / m['boo_best'] for m in g]):>18.3f}" for k in PBR_KERNELS)
            print(f'  {str(key):26s} {len(g):>4d} {pick:>10s} {cols}')

    block_report('by matrix:', lambda m: m['matrix'])
    block_report('by size:', lambda m: m['size'])

    print(f"\n  {'OVERALL':26s} {len(clean):>4d} {'':>10s} " + ' '.join(
        f"{geomean([m[k] / m['boo_best'] for m in clean]):>18.3f}" for k in PBR_KERNELS))

    wins = [m for m in clean if m['boo_best'] < min(m[k] for k in PBR_KERNELS)]
    print(f'\nBOO best-of-two is fastest in {len(wins)}/{len(clean)} clean cells.')

    # How much did the clustering choice matter?
    print('\nClustering choice (how often each wins, and by how much):')
    hem_wins = [m for m in clean if m['boo_choice'] == 'hem']
    print(f'  hem chosen in      {len(hem_wins):>3d}/{len(clean)} cells; '
          f'geomean identity/hem = '
          f'{geomean([m["boo_identity"] / m["boo_hem"] for m in hem_wins]):.3f}')
    id_wins = [m for m in clean if m['boo_choice'] == 'identity']
    if id_wins:
        print(f'  identity chosen in {len(id_wins):>3d}/{len(clean)} cells; '
              f'geomean hem/identity = '
              f'{geomean([m["boo_hem"] / m["boo_identity"] for m in id_wins]):.3f}')


if __name__ == '__main__':
    main()
