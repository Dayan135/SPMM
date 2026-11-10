"""Freeze the Stage 1 / Stage 2 nnzflat timing manifests from completed CPU preparation.

  python scripts/research_panel/nnzflat_manifest.py --root RUN --stage 1
  python scripts/research_panel/nnzflat_manifest.py --root RUN --stage 2

Requested nnzflat configurations (the audited counts):
  Stage 1: 2 patterns x k{16,32,64,128} x P5 x T3 x FPT2 x F{128,512} = 480
  Stage 2: 8 inputs   x k{32,64,128}    x P5 x T3 x FPT2 x F{128,512} = 1,440
Every BOO object additionally carries, outside those counts: rowseg at the same
P/T/FPT (30 arms, including the P64/T256/FPT1/2 anchors), the original nnzflat
FPT1 reference at P8/T256 and P64/T256 (identical-work code control), and a
duplicate nf_p64_t256_f1 arm (pure measurement-noise control).

A cell is one F; its objects are every (input, k) of the task, so k and (Stage 1)
balanced/skewed comparisons are interleaved in the same shuffled run.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'tests' / 'torch_fast_spdmm'), str(ROOT)]

from benchmark import nnzflat_common as nc  # noqa: E402

REQUESTED = [nc.arm_name('nnzflat', p, t, f) for p in nc.PS for t in nc.THREADS for f in nc.FPTS]
ROWSEG = [nc.arm_name('rowseg', p, t, f) for p in nc.PS for t in nc.THREADS for f in nc.FPTS]
CONTROLS = [nc.arm_name('nnzflat', 8, 256, 1, 'reference'),
            nc.arm_name('nnzflat', 64, 256, 1, 'reference'),
            nc.arm_name('nnzflat', 64, 256, 1, suffix='_dup')]
ARMS = REQUESTED + ROWSEG + CONTROLS


def label_hashes(root, inputs):
    out = {}
    for name in inputs:
        prep = json.loads((nc.art(root) / 'prep' / f'{name}.json').read_text())
        out[name] = {str(k): v for k, v in prep['label_sha256'].items()}
    return out


def build(root, stage):
    if stage == 1:
        inputs, ks = nc.STAGE1_INPUTS, nc.STAGE1_K
        groups = [list(inputs)]
    else:
        inputs, ks = nc.STAGE2_INPUTS, nc.STAGE2_K
        groups = [[name] for name in inputs]
    tasks, requested = [], []
    for ti, group in enumerate(groups):
        cells = []
        for f in nc.SCREEN_F:
            objects = [dict(input=name, k=k, arms=ARMS) for k in ks for name in group]
            cells.append(dict(cell=f"s{stage}_{'+'.join(group)}_F{f}", features=f, objects=objects))
            requested += [dict(stage=stage, task=ti, input=name, k=k, features=f, arm=a)
                          for name in group for k in ks for a in REQUESTED]
        tasks.append(dict(task=ti, inputs=group, cells=cells))
    expected = {1: 480, 2: 1440}[stage]
    if len(requested) != expected:
        raise AssertionError(f'stage {stage}: {len(requested)} requested, expected {expected}')
    return dict(stage=stage, warmup=5, repeats=20, order_seed=14100 + stage,
                order_mode='independent seeded shuffle per repeat', features=list(nc.SCREEN_F),
                ks=list(ks), inputs=list(inputs),
                methods={n: nc.INPUTS[n].method for n in inputs},
                fixed=dict(dtype='float32', batch=1, seed=nc.SEED, slab=nc.SLAB, nz_order='rowmajor',
                           launch_mode='auto', csr_mode=0, threshold=nc.THRESHOLD,
                           explicit_zeros='keep', small_block_policy='remainder'),
                requested_arm_names=REQUESTED, rowseg_arm_names=ROWSEG, control_arm_names=CONTROLS,
                requested_count=len(requested), requested=requested,
                label_hashes=label_hashes(root, inputs), tasks=tasks)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--stage', type=int, choices=[1, 2], required=True)
    a = ap.parse_args()
    manifest = build(a.root, a.stage)
    out = a.root / 'manifests' / f'stage{a.stage}.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        old = json.loads(out.read_text())
        if old != manifest:
            raise SystemExit(f'{out} exists with different content; manifests are frozen')
    out.write_text(json.dumps(manifest, indent=1) + '\n')
    print(f"stage {a.stage}: {len(manifest['tasks'])} tasks, {manifest['requested_count']} requested "
          f"nnzflat configurations -> {out}")


if __name__ == '__main__':
    main()
