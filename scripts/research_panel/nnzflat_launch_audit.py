"""Nsight Systems launch audit for timed nnzflat cells (independent of Kineto).

  nsys profile --capture-range=cudaProfilerApi --export=sqlite -o T \
      python nnzflat_launch_audit.py run --root RUN --manifest M --task 3 --cell 1 --out DIR
  python nnzflat_launch_audit.py check --out DIR --sqlite T.sqlite

`run` rebuilds the cell exactly as bench_nnzflat does (same artifacts, digests,
launch accounting and boo_matmul calls), skips arms the launcher reports
infeasible, and launches every remaining BOO arm once inside the capture range,
in manifest order. `check` requires the captured BOO main-kernel sequence (CSR
remainder and memset kernels ignored) to equal the expected specializations.
"""
import argparse
import json
from pathlib import Path
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'tests' / 'torch_fast_spdmm'), str(ROOT / 'torch_fast_spdmm'), str(ROOT)]

from benchmark import bench_nnzflat as bn  # noqa: E402
from benchmark import nnzflat_common as nc  # noqa: E402
from benchmark import real_matrices as rm  # noqa: E402


def run(a):
    import torch
    import torch_fast_spdmm as spmm
    import torch_fast_spdmm_core as core
    manifest = json.loads(a.manifest.read_text())
    cell = manifest['tasks'][a.task]['cells'][a.cell]
    limit = torch.cuda.get_device_properties(0).shared_memory_per_block_optin
    expected, arms, inputs = [], [], {}
    with torch.inference_mode():
        for obj in cell['objects']:
            name = obj['input']
            if name not in inputs:
                inputs[name] = bn.load_input_ctx(torch, a.root, a.cache, name, cell['features'])
            ctx = bn.prepare_object(a.root, inputs[name], obj['k'], manifest['label_hashes'])
            if ctx['alias_of']:
                continue
            made, _ = bn.boo_arms(torch, core, spmm, ctx, obj['arms'], limit, set())
            arms += made
        for arm in arms:  # untimed warm call outside the capture range
            arm.run()
        torch.cuda.synchronize()
        torch.cuda.profiler.start()
        for arm in arms:
            y = arm.run()
            torch.cuda.synchronize()
            del y
            expected.append(dict(arm=arm.name, kernel=nc.expected_kernel(arm.verify)))
        torch.cuda.profiler.stop()
        nc.reset_settings(core)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / 'expected.json').write_text(json.dumps(dict(cell=cell['cell'], task=a.task,
                                                         manifest=str(a.manifest), arms=expected),
                                                    indent=1) + '\n')
    print(f"{cell['cell']}: {len(expected)} arms launched for audit")


def check(a):
    exp = json.loads((a.out / 'expected.json').read_text())
    con = sqlite3.connect(a.sqlite)
    q = ('SELECT s.value FROM CUPTI_ACTIVITY_KIND_KERNEL k '
         'JOIN StringIds s ON s.id = COALESCE(k.demangledName, k.shortName) ORDER BY k.start')
    got = [c for c in (nc.classify(n) for (n,) in con.execute(q)) if c]
    want = [e['kernel'] for e in exp['arms']]
    rows = [dict(arm=e['arm'], expected=e['kernel'], launched=g, status='verified-nsys' if g == e['kernel']
                 else 'MISMATCH') for e, g in zip(exp['arms'], got)]
    ok = got == want
    result = dict(cell=exp['cell'], task=exp['task'], expected=len(want), captured=len(got), match=ok,
                  arms=rows)
    (a.out / 'audit.json').write_text(json.dumps(result, indent=1) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'arms'}))
    if not ok:
        raise SystemExit('launch audit FAILED')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('mode', choices=['run', 'check'])
    ap.add_argument('--root', type=Path)
    ap.add_argument('--manifest', type=Path)
    ap.add_argument('--task', type=int)
    ap.add_argument('--cell', type=int)
    ap.add_argument('--cache', type=Path, default=rm.default_cache())
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--sqlite', type=Path)
    a = ap.parse_args()
    run(a) if a.mode == 'run' else check(a)


if __name__ == '__main__':
    main()
