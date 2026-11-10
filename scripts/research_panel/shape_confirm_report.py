"""Report the block-shape confirmation against the claim rules frozen in RUN/manifests/confirm.json.

  python scripts/research_panel/shape_confirm_report.py --results RUN/results --audits RUN/launch_audit \
      --manifest RUN/manifests/confirm.json --out ANALYSIS/confirm

Speedup is time(reference) / time(BOO winner); above 1 means the BOO winner is faster.
Per (allocation, input, F, pair), where a pair is a frozen (winner, comparator) selection (F256 has
the F512-selected primary pair and the F128-selected secondary pair):
  shape        comparator (tuned 32x32) / winner, in the _orig cell
  direct       fastest original-order direct_<alg>_api / winner, in the _orig cell
  perm_steady  fastest reordered steady_api (resident X', Y') / winner, over the _perm cells
               (each against the winner timed in the same cell; the strongest reordering is kept)
  perm_gather  the same for gather_api (X gather and Y restore every call, original coordinates)
  torch_csr    torch.sparse.mm / winner
Verdicts use the minimum over allocations: > 1.05 win, 0.95-1.05 parity, < 0.95 loss. Allocations
must be on distinct GPUs; the GPU count is reported. Every table is produced twice: all accepted
cells, and with the post-hoc control-band filter (a (allocation, input, F) group is dropped when a
_dup or _pad0 control in its _orig cell is outside [0.95, 1.05]).
"""
import argparse
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'tests' / 'torch_fast_spdmm'), str(ROOT), str(ROOT / 'scripts' / 'research_panel')]

from benchmark import nnzflat_common as nc  # noqa: E402
import shape_report as sr  # noqa: E402

BASELINE_FAMILIES = ('direct', 'direct_perm', 'torch_csr')
WIN, PARITY = 1.05, 0.95


def accept(d, audits, out):
    nsys = sr.nsys_verified(audits, 'confirm') if audits else set()
    m = d[d.status == 'measured'].copy()
    m['launch_ok'] = ((m.launch_verified == 'verified') | m.arm.isin(nsys) | m.family.isin(BASELINE_FAMILIES))
    t = m.groupby('alloc').agg(gpus=('gpu_uuid', 'nunique'), measured=('arm', 'size'),
                               unverified=('launch_ok', lambda s: int((~s).sum())))
    t['not_measured'] = d[d.status != 'measured'].groupby('alloc').size()
    print(f'\n== confirm audit\n{t.fillna(0).astype(int).to_string()}')
    problems = d[~d.status.isin(['measured'])]
    if len(problems):
        problems.to_csv(out / 'confirm_not_measured.csv', index=False)
        print(problems.groupby(['status', 'reason']).size().to_string())
    m = sr.arm_tag(m[m.launch_ok])
    m['group'] = m.cell.str.replace(r'_(orig|perm\d+x\d+)$', '', regex=True)
    m['part'] = m.cell.str.extract(r'_(orig|perm\d+x\d+)$')[0]
    return m


def control_drop(m, out):
    key = ['alloc', 'cell', 'k', 'base']
    boo = m[m.k.notna() & (m.part == 'orig')]
    base = boo[boo.suffix == ''].set_index(key).median_ms
    c = boo[boo.suffix.isin(['dup', 'pad0'])].join(base.rename('base_ms'), on=key)
    c['ratio'] = c.median_ms / c.base_ms
    c.to_csv(out / 'confirm_controls.csv', index=False)
    print('\n== controls, arm / base time')
    print(c.groupby('suffix').ratio.describe()[['count', 'min', '50%', 'max']].round(4).to_string())
    bad = c[(c.ratio < 0.95) | (c.ratio > 1.05)]
    if len(bad):
        print('groups dropped by the control band:')
        print(bad[['alloc', 'group', 'gpu_uuid', 'tag', 'ratio']].round(3).to_string(index=False))
    return set(zip(bad.alloc, bad.group))


def pairs(manifest):
    """(input, F) -> [(pair_name, winner_shape, winner_arm, comparator_arm)], incl. 1M and F256."""
    sel = {(s['input'], s['features']): s for s in manifest['selections']}
    big_of = nc.SCALE_INPUTS
    out = {}
    for (inp, f), s in sel.items():
        p = (f'F{f}-selected', s['winner_shape'], s['winner_arm'], s['comparator_arm'])
        out.setdefault((inp, f), []).append(p)
        out.setdefault((inp, 256), []).append(p)
        if inp in big_of:
            out.setdefault((big_of[inp], f), []).append(p)
    for v in out.values():
        v.sort(key=lambda p: p[0] != 'F512-selected')
    return out


def ratios(m, manifest, dropped):
    rows = []
    roles = {c['cell'].rsplit('_', 1)[0]: c['role'] for t in manifest['tasks'] for c in t['cells']}
    for (alloc, inp, f, group), g in m.groupby(['alloc', 'input', 'features', 'group']):
        for i, (pname, wshape, warm, carm) in enumerate(pairs(manifest).get((inp, f), [])):
            def boo_ms(part, shape, arm):
                r = g[(g.part == part) & (g.k == shape) & (g.tag == arm)].median_ms
                return float(r.iloc[0]) if len(r) else float('nan')
            w = boo_ms('orig', wshape, warm)
            row = dict(alloc=alloc, input=inp, features=f, role=roles.get(group, ''), pair=pname,
                       primary=i == 0, winner=f'{wshape}:{warm}', comparator=f'32x32:{carm}',
                       gpu_uuid=g.gpu_uuid.iloc[0], dropped=(alloc, group) in dropped,
                       winner_ms=w, shape=boo_ms('orig', '32x32', carm) / w)
            o = g[g.part == 'orig']
            d = o[(o.family == 'direct') & o.tag.str.endswith('_api')]
            if len(d):
                best = d.loc[d.median_ms.idxmin()]
                row.update(direct=best.median_ms / w, direct_alg=best.direct_algorithm)
            t = o[o.family == 'torch_csr'].median_ms
            if len(t):
                row['torch_csr'] = float(t.iloc[0]) / w
            for kind in ('steady', 'gather'):
                best = None
                for part, pg in g[g.part.fillna('').str.startswith('perm')].groupby('part'):
                    pw = boo_ms(part, wshape, warm)
                    b = pg[(pg.family == 'direct_perm') & pg.tag.str.endswith(f'_{kind}_api')]
                    if len(b) and pw == pw:
                        r = b.median_ms.min() / pw
                        if best is None or r < best[0]:
                            best = (r, f"{part}:{b.loc[b.median_ms.idxmin()].direct_algorithm}")
                if best:
                    row[f'perm_{kind}'], row[f'perm_{kind}_by'] = best
            rows.append(row)
    return pd.DataFrame(rows)


def verdict(x):
    return 'win' if x > WIN else ('parity' if x >= PARITY else 'loss')


def summarize(r, label, out):
    cols = [c for c in ('shape', 'direct', 'perm_steady', 'perm_gather', 'torch_csr') if c in r]
    g = r.groupby(['input', 'features', 'role', 'pair', 'primary', 'winner'])
    raw = g[cols].agg(['min', 'median', 'count'])
    t = raw.round(3)
    t.columns = [f'{a}_{b}' for a, b in t.columns]
    t['allocs'] = g.alloc.nunique()
    t['gpus'] = g.gpu_uuid.nunique()
    for c in cols:  # verdict on the unrounded minimum; '*' marks fewer than 3 allocations
        t[f'{c}_verdict'] = [('' if x != x else verdict(x) + ('*' if n < 3 else ''))
                             for x, n in zip(raw[(c, 'min')], raw[(c, 'count')])]
    t = t.reset_index()
    t.to_csv(out / f'confirm_{label}.csv', index=False)
    show = ['input', 'features', 'role', 'pair', 'winner', 'allocs', 'gpus'] + \
        [f'{c}_{s}' for c in cols for s in ('min', 'verdict')]
    print(f'\n== confirmation ({label}); speedup of the BOO winner, minimum over allocations')
    print(t[t.primary][show].to_string(index=False))
    sec = t[~t.primary]
    if len(sec):
        print('\nsecondary F256 pairs (F128-selected):')
        print(sec[show].to_string(index=False))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--results', type=Path, required=True)
    ap.add_argument('--audits', type=Path, default=None)
    ap.add_argument('--manifest', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    pd.set_option('display.width', 260, 'display.max_columns', 40, 'display.max_rows', 300)
    manifest = json.loads(a.manifest.read_text())
    d = sr.load(a.results, 'confirm')
    m = accept(d, a.audits, a.out)
    dropped = control_drop(m, a.out)
    r = ratios(m, manifest, dropped)
    r.to_csv(a.out / 'confirm_ratios.csv', index=False)
    summarize(r, 'all', a.out)
    summarize(r[~r.dropped], 'filtered', a.out)


if __name__ == '__main__':
    main()
