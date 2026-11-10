"""Profile one arm of a block-shape confirmation cell (run under ncu or nsys, GPU node).

  ncu --profile-from-start off --set full -o OUT \
      python shape_profile.py --artifact-root A --run RUN --source-labels L --manifest M \
          --cell confirm_fem2d_permuted_F512_orig --arm 'fem2d_permuted@k256x64:rs_p64_t512_f4' --out DIR

Rebuilds the cell exactly as bench_shape does (same labels, digests, BOO objects, direct and reordered
cuSPARSE plans), but builds only the object or baseline family that owns --arm. The arm is called
once untimed and checked against the float64 reference outside the capture range, then called
--calls times inside cudaProfilerStart/Stop. Writes DIR/profile.json with the arm's launch and
structure fields. Profiled durations are diagnostic; the confirmation timings decide.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'tests' / 'torch_fast_spdmm'), str(ROOT / 'torch_fast_spdmm'), str(ROOT),
                str(ROOT / 'scripts' / 'research_panel')]

from benchmark import bench_nnzflat as bn  # noqa: E402
from benchmark import bench_shape as bs  # noqa: E402
from benchmark import nnzflat_common as nc  # noqa: E402
from benchmark import real_matrices as rm  # noqa: E402


def find_cell(manifest, name):
    for t in manifest['tasks']:
        for c in t['cells']:
            if c['cell'] == name:
                return t['task'], c
    raise SystemExit(f'cell {name} not in manifest')


def build_arm(torch, core, spmm, cell, arm_name, a):
    limit = torch.cuda.get_device_properties(0).shared_memory_per_block_optin
    label_dirs = [a.source_labels, a.run / 'labels']
    who, kind = arm_name.split(':', 1)
    name = who.split('@')[0]
    ictx = bn.load_input_ctx(torch, a.artifact_root, a.cache, name, cell['features'])
    if '@k' in who:
        shape = who.split('@k')[1]
        objs = [o for o in cell['objects'] if f"{o['kr']}x{o['kc']}" == shape and kind in o['arms']]
        if not objs:
            raise SystemExit(f'{arm_name}: no object in {cell["cell"]} carries it')
        ctx = bs.prepare_object(ictx, objs[0], label_dirs)
        arms, rows = bs.object_arms(torch, core, spmm, ctx, [kind], limit, set())
        extra = {k: ctx[k] for k in ('boo_digest', 'kr', 'kc') if k in ctx}
    elif kind.startswith('perm'):
        tag = kind.split('_direct_')[0]
        specs = [s for s in cell.get('reordered', []) if f"perm{s['kr']}x{s['kc']}" == tag]
        arms, rows = bs.reordered_arms(torch, core, ictx, specs[0], label_dirs)
        extra = {}
    else:
        arms, rows = bn.baseline_arms(torch, core, ictx, cell['baselines'][name])
        extra = {}
    hit = [x for x in arms if x.name == arm_name]
    if not hit:
        reasons = [r.get('reason') for r in rows if r.get('arm') == arm_name]
        raise SystemExit(f'{arm_name}: not built ({reasons or "unknown name"})')
    return ictx, hit[0], extra


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--artifact-root', type=Path, required=True)
    ap.add_argument('--run', type=Path, required=True)
    ap.add_argument('--source-labels', type=Path, required=True)
    ap.add_argument('--manifest', type=Path, required=True)
    ap.add_argument('--cell', required=True)
    ap.add_argument('--arm', required=True)
    ap.add_argument('--calls', type=int, default=1)
    ap.add_argument('--cache', type=Path, default=rm.default_cache())
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    import torch
    import torch_fast_spdmm as spmm
    import torch_fast_spdmm_core as core
    manifest = json.loads(a.manifest.read_text())
    task, cell = find_cell(manifest, a.cell)
    with torch.inference_mode():
        ictx, arm, extra = build_arm(torch, core, spmm, cell, a.arm, a)
        y = arm.run()
        torch.cuda.synchronize()
        check = arm.check(y) if arm.check else None
        del y
        if check is not None and check.get('bound_violations', 0):
            raise SystemExit(f'{a.arm}: correctness failed {check}')
        torch.cuda.profiler.start()
        for _ in range(a.calls):
            y = arm.run()
            torch.cuda.synchronize()
            del y
        torch.cuda.profiler.stop()
        nc.reset_settings(core)
    a.out.mkdir(parents=True, exist_ok=True)
    out = {k: v for k, v in arm.row.items() if isinstance(v, (int, float, str, bool)) or v is None}
    out.update(extra)
    out.update(cell=a.cell, task=task, arm=a.arm, calls=a.calls, features=cell['features'],
               expected_kernel=nc.expected_kernel(arm.verify) if arm.verify else None, check=check)
    (a.out / 'profile.json').write_text(json.dumps(out, indent=1, default=str) + '\n')
    print(f'{a.arm}: profiled {a.calls} call(s)')


if __name__ == '__main__':
    main()
