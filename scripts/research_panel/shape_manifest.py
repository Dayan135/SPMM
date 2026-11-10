"""Freeze block-shape timing manifests from the CPU gate. No GPU; manifests are never rewritten.

  python scripts/research_panel/shape_manifest.py --run RUN --relative ANALYSIS/gate_relative.csv --stage screen

Stages:
  sanity     two inputs, F128, few arms: does the rectangular and padded path run and verify?
  screen     frozen labeling per input, F128 and F512, P{32,64,128} x T{128,256,512} x FPT{1,2,4}.
             Objects: square anchors, 64x32 and 128x32 on every input (so a gate failure can be
             checked against time), and every other rectangular shape the gate passed.
  occupancy  the same k32 work with the arena padded to fewer CTAs/SM. No shape change.
  kp         the screen objects with P16 added (P16-P128 in the same cells), for the k by P matrix.

Each object carries the digests the gate recorded; bench_shape refuses a rebuild that differs.
"""
import argparse
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'tests' / 'torch_fast_spdmm'), str(ROOT)]

from benchmark import nnzflat_common as nc  # noqa: E402

ANCHORS = ('32x32', '64x64', '128x128')
ALWAYS = ('64x32', '128x32')
PS, TS, FS = (32, 64, 128), (128, 256, 512), (1, 2, 4)
GRID = [f'rs_p{p}_t{t}_f{f}' for p in PS for t in TS for f in FS]
KP_GRID = [f'rs_p{p}_t{t}_f{f}' for p in (16,) + PS for t in TS for f in FS]
CONTROLS = ('rs_p64_t256_f2_dup', 'rs_p64_t256_f2_pad0')
OCC_T256 = [f'rs_p64_t256_f{f}{s}' for f in (1, 2, 4) for s in ('', '_pad0', '_cta4', '_cta3', '_cta2', '_cta1')]
OCC_T512 = [f'rs_p64_t512_f{f}{s}' for f in (2, 4) for s in ('', '_pad0', '_cta2', '_cta1')]
SANITY_INPUTS = ('pwtk', 'fem2d_permuted')
SANITY_SHAPES = ('32x32', '128x32')
SANITY_ARMS = ('rs_p64_t256_f2', 'rs_p64_t256_f2_dup', 'rs_p64_t256_f2_pad0', 'rs_p64_t256_f2_cta3',
               'rs_p64_t512_f4', 'rs_p64_t512_f4_cta1')


def gate_rows(relative):
    frozen = {nc.INPUTS[n].key: (n, nc.INPUTS[n].method) for n in nc.STAGE2_INPUTS}
    d = pd.read_csv(relative)
    d = d[d.apply(lambda r: frozen.get(r.matrix, (None, None))[1] == r.method, axis=1)].copy()
    d['input'] = d.matrix.map(lambda m: frozen[m][0])
    return d


def make_object(row, arms):
    return dict(input=row['input'], method=row['method'], kr=int(row['kr']), kc=int(row['kc']),
                row_label_sha256=row['row_label_sha256'], col_label_sha256=row['col_label_sha256'],
                boo_digest=row['boo_digest'], gate_passes=bool(row['passes']), arms=list(arms))


def objects_for(rows, shapes, arms_of):
    rows = rows.set_index('shape', drop=False)
    return [make_object(rows.loc[s], arms_of(s)) for s in shapes if s in rows.index]


def screen_shapes(rows):
    picked = rows[rows.passes & (rows.kr > rows.kc)]['shape']
    passing = sorted((s for s in picked if s not in ANCHORS + ALWAYS),
                     key=lambda s: tuple(map(int, s.split('x'))))
    return list(ANCHORS) + list(ALWAYS) + passing


def build(stage, relative):
    d = gate_rows(relative)
    tasks = []
    if stage == 'sanity':
        cells = []
        for name in SANITY_INPUTS:
            objs = objects_for(d[d.input == name], SANITY_SHAPES, lambda _: SANITY_ARMS)
            cells.append(dict(cell=f'sanity_{name}_F128', features=128, objects=objs))
        tasks.append(dict(task=0, inputs=list(SANITY_INPUTS), cells=cells))
        return dict(stage=stage, warmup=2, repeats=5, order_seed=61100, requested_arm_names=[], tasks=tasks)
    for name in nc.STAGE2_INPUTS:
        rows = d[d.input == name]
        cells = []
        for f in nc.SCREEN_F:
            if stage in ('screen', 'kp'):
                grid = GRID if stage == 'screen' else KP_GRID
                arms_of = lambda s, grid=grid: grid + list(CONTROLS) if s == '32x32' else grid
                objs = objects_for(rows, screen_shapes(rows), arms_of)
            else:
                objs = objects_for(rows, ['32x32'], lambda _: OCC_T256 + OCC_T512 + ['rs_p64_t256_f2_dup'])
            cells.append(dict(cell=f'{stage}_{name}_F{f}', features=f, objects=objs))
        tasks.append(dict(task=len(tasks), inputs=[name], cells=cells))
    return dict(stage=stage, warmup=5, repeats=20,
                order_seed={'screen': 61200, 'occupancy': 61300, 'kp': 61400}[stage],
                order_mode='independent seeded shuffle per repeat',
                fixed=dict(dtype='float32', batch=1, seed=nc.SEED, slab=nc.SLAB, nz_order='rowmajor',
                           launch_mode='auto', csr_mode=0, threshold=nc.THRESHOLD, explicit_zeros='keep',
                           small_block_policy='remainder'),
                requested_arm_names=KP_GRID if stage == 'kp' else GRID, tasks=tasks)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run', type=Path, required=True)
    ap.add_argument('--relative', type=Path, required=True)
    ap.add_argument('--stage', choices=['sanity', 'screen', 'occupancy', 'kp'], required=True)
    a = ap.parse_args()
    manifest = build(a.stage, a.relative)
    manifest['gate_relative_sha256'] = nc.sha256_file(a.relative)
    out = a.run / 'manifests' / f'{a.stage}.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and json.loads(out.read_text()) != manifest:
        raise SystemExit(f'{out} exists with different content; manifests are frozen')
    out.write_text(json.dumps(manifest, indent=1) + '\n')
    for task in manifest['tasks']:
        for cell in task['cells']:
            shapes = [f"{o['kr']}x{o['kc']}" for o in cell['objects']]
            arms = sum(len(o['arms']) for o in cell['objects'])
            print(f"{cell['cell']}: {arms} arms, shapes {' '.join(shapes)}")
    print(f'-> {out}')


if __name__ == '__main__':
    main()
