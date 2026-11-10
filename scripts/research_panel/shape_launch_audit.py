"""Nsight Systems launch audit for block-shape cells (independent of the Kineto check in bench_shape).

  nsys profile --capture-range=cudaProfilerApi --export=sqlite -o T \
      python shape_launch_audit.py run --artifact-root A --run RUN --source-labels L \
          --manifest M --task 3 --cell 1 --out DIR
  python shape_launch_audit.py check --out DIR --sqlite T.sqlite

`run` rebuilds the cell exactly as bench_shape does (same labels, digests, launch accounting, padded
calls), skips arms the launcher reports infeasible, and launches every remaining arm once inside the
capture range, in manifest order. `check` is nnzflat_launch_audit.check: the captured BOO main-kernel
sequence must equal the expected specializations. Kernel selection depends on the manifest and device
type, not on timing, so one audit per cell covers every allocation of it.
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
import nnzflat_launch_audit as nla  # noqa: E402


def run(a):
    import torch
    import torch_fast_spdmm as spmm
    import torch_fast_spdmm_core as core
    manifest = json.loads(a.manifest.read_text())
    cell = manifest['tasks'][a.task]['cells'][a.cell]
    limit = torch.cuda.get_device_properties(0).shared_memory_per_block_optin
    label_dirs = [a.source_labels, a.run / 'labels']
    expected, arms, inputs = [], [], {}
    with torch.inference_mode():
        for obj in cell['objects']:
            name = obj['input']
            if name not in inputs:
                inputs[name] = bn.load_input_ctx(torch, a.artifact_root, a.cache, name, cell['features'])
            ctx = bs.prepare_object(inputs[name], obj, label_dirs)
            made, _ = bs.object_arms(torch, core, spmm, ctx, obj['arms'], limit, set())
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
    (a.out / 'expected.json').write_text(json.dumps(dict(cell=cell['cell'], task=a.task, manifest=str(a.manifest),
                                                         arms=expected), indent=1) + '\n')
    print(f"{cell['cell']}: {len(expected)} arms launched for audit")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('mode', choices=['run', 'check'])
    ap.add_argument('--artifact-root', type=Path)
    ap.add_argument('--run', type=Path)
    ap.add_argument('--source-labels', type=Path)
    ap.add_argument('--manifest', type=Path)
    ap.add_argument('--task', type=int)
    ap.add_argument('--cell', type=int)
    ap.add_argument('--cache', type=Path, default=rm.default_cache())
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--sqlite', type=Path)
    a = ap.parse_args()
    run(a) if a.mode == 'run' else nla.check(a)


if __name__ == '__main__':
    main()
