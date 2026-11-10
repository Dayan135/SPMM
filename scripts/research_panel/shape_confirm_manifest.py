"""Freeze the block-shape confirmation (stage 7) from screen data only. No GPU; never rewrites.

  python scripts/research_panel/shape_confirm_manifest.py --results RAW/results --audits RAW/launch_audit \
      --relative GATE/gate_relative.csv --run RUN --freeze-out ANALYSIS/confirm_freeze

Selection, per (input, F in {128, 512}), on the screen and kp allocations after the report's launch
audit and control-band filter (P16-P128, plain arms only):
  per allocation, speedup = best 32x32 time in that allocation / arm time;
  score = minimum over allocations; arms seen on fewer than two GPUs are not eligible;
  winner     = the highest-scoring arm of any shape other than 32x32;
  comparator = the highest-scoring 32x32 arm (the tuned baseline).
The same selection without the control-band filter is written next to it; any difference is listed.

Confirmation cells (fixed here, before any confirmation timing):
  F128, F512   the selection made at that F (re-measured: screened_feature)
  F256         held out: the union of the F128 and F512 selections. The primary F256 claim is
               the F512-selected pair; the F128-selected pair is secondary.
  1M           held out (generated inputs only): each F selection at its own F.
Each cell is split by baseline family, every part re-timing the BOO arms as its in-cell anchor:
  <cell>_orig          BOO winner (+_dup), comparator (+_dup, +_pad0), torch CSR, 4 direct algorithms
  <cell>_perm<kr>x<kc> BOO winner and comparator, plus the 4 direct algorithms on A reordered by
                       that object's labels (resident steady state and per-call gather/restore).
                       Only for inputs whose frozen labeling is not identity.
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

INPUTS = ('fem2d_permuted', 'roadnet', 'fem2d', 'diag_dd0.05', 'diag_b1.0', 'pwtk', 'webbase')
NEGATIVE_CONTROLS = ('webbase',)
DIRECT = ['DEFAULT', 'CSR_ALG1', 'CSR_ALG2', 'CSR_ALG3']
SELECT_F = (128, 512)
MARGIN = 1.05


def scores(m):
    d = m[m.suffix == ''].rename(columns={'k': 'shape'})
    ref = d[d['shape'] == '32x32'].groupby(['alloc', 'input', 'features']).median_ms.min().rename('ref_ms')
    d = d.join(ref, on=['alloc', 'input', 'features'])
    d['speedup'] = d.ref_ms / d.median_ms
    return (d.groupby(['input', 'features', 'shape', 'base'])
            .agg(score=('speedup', 'min'), median=('speedup', 'median'), allocs=('alloc', 'nunique'),
                 gpus=('gpu_uuid', 'nunique'), p=('p', 'first'), threads=('threads', 'first'),
                 fpt=('fpt', 'first'))
            .reset_index())


def select(s):
    s = s[s.gpus >= 2]
    out = []
    for (inp, f), g in s.groupby(['input', 'features']):
        win = g[g['shape'] != '32x32'].sort_values('score', ascending=False).iloc[0]
        cmp_ = g[g['shape'] == '32x32'].sort_values('score', ascending=False).iloc[0]
        out.append(dict(input=inp, features=int(f), winner_shape=win['shape'], winner_arm=win.base,
                        winner_score=round(win.score, 4), winner_median=round(win['median'], 4),
                        comparator_arm=cmp_.base, comparator_score=round(cmp_.score, 4),
                        allocs=int(win.allocs), gpus=int(win.gpus)))
    return pd.DataFrame(out)


def gate_lookup(relative):
    g = pd.read_csv(relative)
    return {(r.matrix, r.method, r.shape): r for r in g.itertuples()}


def obj(gate, name, shape, arms):
    inp = nc.INPUTS[name]
    r = gate[(inp.key, inp.method, shape)]
    kr, kc = map(int, shape.split('x'))
    return dict(input=name, method=inp.method, kr=kr, kc=kc, row_label_sha256=r.row_label_sha256,
                col_label_sha256=r.col_label_sha256, boo_digest=r.boo_digest, gate_passes=bool(r.passes),
                arms=arms)


def merge_objects(objs):
    merged = {}
    for o in objs:
        key = (o['input'], o['kr'], o['kc'])
        if key in merged:
            merged[key]['arms'] += [a for a in o['arms'] if a not in merged[key]['arms']]
        else:
            merged[key] = dict(o, arms=list(o['arms']))
    return list(merged.values())


def cells_for(gate, name, f, picks, role):
    """picks: selection rows to time in this cell (one, or two for held-out F256)."""
    full, anchors = [], []
    for p in picks:
        w, c = p['winner_arm'], p['comparator_arm']
        full += [obj(gate, name, p['winner_shape'], [w, f'{w}_dup']),
                 obj(gate, name, '32x32', [c, f'{c}_dup', f'{c}_pad0'])]
        anchors += [obj(gate, name, p['winner_shape'], [w]), obj(gate, name, '32x32', [c])]
    full, anchors = merge_objects(full), merge_objects(anchors)
    stem = f'confirm_{name}_F{f}'
    cells = [dict(cell=f'{stem}_orig', features=f, role=role, objects=full,
                  baselines={name: dict(torch_csr=True, direct=DIRECT)})]
    if nc.INPUTS[name].method != 'identity':
        for o in anchors:
            spec = {k: o[k] for k in ('input', 'method', 'kr', 'kc', 'row_label_sha256', 'col_label_sha256')}
            cells.append(dict(cell=f"{stem}_perm{o['kr']}x{o['kc']}", features=f, role=role, objects=anchors,
                              reordered=[dict(spec, direct=DIRECT)]))
    return cells


def build(sel, gate):
    pick = {(r.input, r.features): r._asdict() for r in sel.itertuples()}
    tasks = []
    for name in INPUTS:
        cells = []
        for f in (128, 256, 512):
            if f == 256:
                cells += cells_for(gate, name, f, [pick[(name, 128)], pick[(name, 512)]], 'heldout_feature')
            else:
                cells += cells_for(gate, name, f, [pick[(name, f)]], 'screened_feature')
        tasks.append(dict(task=len(tasks), inputs=[name], cells=cells))
    for name in INPUTS:
        big = nc.SCALE_INPUTS.get(name)
        if big is None:
            continue
        cells = []
        for f in SELECT_F:
            cells += cells_for(gate, big, f, [pick[(name, f)]], 'heldout_scale')
        tasks.append(dict(task=len(tasks), inputs=[big], cells=cells))
    return tasks


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--results', type=Path, required=True)
    ap.add_argument('--audits', type=Path, required=True)
    ap.add_argument('--relative', type=Path, required=True)
    ap.add_argument('--run', type=Path, required=True, help='manifest goes to RUN/manifests/confirm.json')
    ap.add_argument('--freeze-out', type=Path, required=True)
    a = ap.parse_args()
    a.freeze_out.mkdir(parents=True, exist_ok=True)
    pd.set_option('display.width', 220, 'display.max_columns', 30)
    scr = pd.concat([sr.load(a.results, 'screen'), sr.load(a.results, 'kp')], ignore_index=True)
    accepted = sr.accept(scr, a.audits, 'screen', a.freeze_out)
    sel = select(scores(sr.valid_cells(accepted, 'screen', a.freeze_out)))
    raw = select(scores(accepted))
    sel = sel[sel.input.isin(INPUTS)].reset_index(drop=True)
    raw = raw[raw.input.isin(INPUTS)].reset_index(drop=True)
    sel['role'] = ['negative_control' if i in NEGATIVE_CONTROLS else
                   ('finalist' if s > MARGIN else 'borderline') for i, s in zip(sel.input, sel.winner_score)]
    sel.to_csv(a.freeze_out / 'selection.csv', index=False)
    raw.to_csv(a.freeze_out / 'selection_unfiltered.csv', index=False)
    cols = ['input', 'features', 'winner_shape', 'winner_arm', 'comparator_arm']
    diff = sel[cols].merge(raw[cols], on=['input', 'features'], suffixes=('', '_unfiltered'))
    diff = diff[(diff.winner_shape != diff.winner_shape_unfiltered) | (diff.winner_arm != diff.winner_arm_unfiltered)
                | (diff.comparator_arm != diff.comparator_arm_unfiltered)]
    print('\n== frozen selection (filtered screen)')
    print(sel.to_string(index=False))
    print('\n== selections that change without the control-band filter')
    print(diff.to_string(index=False) if len(diff) else 'none')

    tasks = build(sel, gate_lookup(a.relative))
    arms = sorted({x for t in tasks for c in t['cells'] for o in c['objects'] for x in o['arms']})
    manifest = dict(
        stage='confirm', warmup=5, repeats=20, order_seed=61500, allocations=3,
        order_mode='independent seeded shuffle per repeat; seed offset per allocation',
        allocation_rule='three allocations on disjoint node sets, so every input is timed on three distinct GPUs',
        selection_rule=__doc__.split('Selection,', 1)[1].split('Confirmation cells', 1)[0].strip(),
        claim_rules=dict(
            shape=('winner beats comparator by more than 5% in every allocation (three distinct GPUs), '
                   'separately for re-measured F128/F512, held-out F256 (primary: F512-selected pair) '
                   'and held-out 1M'),
            cusparse=('BOO winner against the fastest same-cell direct algorithm, full-call scope: '
                      'original-order direct_<alg>_api; reordered steady_api (resident X\', Y\'); '
                      'reordered gather_api (original coordinates). A win needs the minimum over '
                      'allocations above 1.05; 0.95-1.05 is parity'),
            sensitivity='every comparison reported with and without the post-hoc control-band filter'),
        fixed=dict(dtype='float32', batch=1, seed=nc.SEED, slab=nc.SLAB, nz_order='rowmajor',
                   launch_mode='auto', csr_mode=0, threshold=nc.THRESHOLD, explicit_zeros='keep',
                   small_block_policy='remainder'),
        selections=sel.to_dict('records'), requested_arm_names=arms, tasks=tasks,
        gate_relative_sha256=nc.sha256_file(a.relative))
    out = a.run / 'manifests' / 'confirm.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and json.loads(out.read_text()) != manifest:
        raise SystemExit(f'{out} exists with different content; manifests are frozen')
    out.write_text(json.dumps(manifest, indent=1) + '\n')
    print()
    for t in tasks:
        for c in t['cells']:
            n_boo = sum(len(o['arms']) for o in c['objects'])
            base = 'orig' if 'baselines' in c else 'reordered'
            print(f"task {t['task']:2d} {c['cell']:44s} {c['role']:16s} {n_boo} BOO arms + {base}")
    print(f'-> {out}')


if __name__ == '__main__':
    main()
