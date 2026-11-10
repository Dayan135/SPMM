"""Profile one BOO arm of the nnzflat experiment on saved artifacts (no CPU preprocessing).

  python -m benchmark.profile_nnzflat --root RUN --input syn_skewed --k 64 --features 512 \
      --arm nf_p8_t256_f2 --out DIR [--calls 1]

Validates against the saved float64 reference, warms up, then brackets --calls
complete calls with cudaProfilerStart/Stop (run under `ncu --profile-from-start off`
or `nsys --capture-range=cudaProfilerApi`). profile.json records the launcher's
report, the expected kernel name, the BOO structure and the Step-B work model.
Profiler durations diagnose; bench_nnzflat's unprofiled timings decide performance.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from benchmark import bench_nnzflat as bn
from benchmark import nnzflat_common as nc
from benchmark import real_matrices as rm


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--input', required=True, choices=sorted(nc.INPUTS))
    ap.add_argument('--k', type=int, required=True)
    ap.add_argument('--features', type=int, required=True)
    ap.add_argument('--arm', required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--cache', type=Path, default=rm.default_cache())
    ap.add_argument('--warmup', type=int, default=5)
    ap.add_argument('--calls', type=int, default=1)
    a = ap.parse_args(argv)
    import torch
    import torch_fast_spdmm as spmm
    import torch_fast_spdmm_core as core
    a.out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((a.root / 'manifests' / 'stage2.json').read_text())
    hashes = dict(manifest['label_hashes'])
    hashes.update(json.loads((a.root / 'manifests' / 'stage1.json').read_text())['label_hashes'])
    limit = torch.cuda.get_device_properties(0).shared_memory_per_block_optin
    with torch.inference_mode():
        ictx = bn.load_input_ctx(torch, a.root, a.cache, a.input, a.features)
        ctx = bn.prepare_object(a.root, ictx, a.k, hashes)
        if ctx['alias_of']:
            raise SystemExit(f'{a.input} k{a.k} is an alias of {ctx["alias_of"]}')
        arms, unsupported = bn.boo_arms(torch, core, spmm, ctx, [a.arm], limit, set())
        if unsupported:
            raise SystemExit(f'unsupported: {unsupported[0]["reason"]}')
        arm = arms[0]
        y = arm.run()
        torch.cuda.synchronize()
        check = arm.check(y)
        del y
        if check['bound_violations']:
            raise AssertionError(f'validation failed: {check}')
        status, found = nc.verify_call(torch, arm.run, arm.verify)
        for _ in range(a.warmup):
            arm.run()
        torch.cuda.synchronize()
        arm.prepare()
        torch.cuda.profiler.start()
        for i in range(a.calls):
            torch.cuda.nvtx.range_push(f'{a.input}@k{a.k}:{a.arm}:{i}')
            y = arm.run()
            torch.cuda.synchronize()
            torch.cuda.nvtx.range_pop()
            del y
        torch.cuda.profiler.stop()
        nc.reset_settings(core)
    row = {k: v for k, v in arm.row.items() if not callable(v)}
    details = dict(row, **check, correctness='passed', launch_verified=status, launched=found,
                   expected_kernel=nc.expected_kernel(arm.verify), input=a.input, k=a.k,
                   features=a.features, profiled_calls=a.calls, warmup=a.warmup,
                   matrix_sha256=ctx['matrix_sha256'], label_sha256=ctx['label_sha256'],
                   boo_digest=ctx['boo_digest'], x_sha256=ctx['x_sha256'], structure=ctx['structure'])
    rm._write_json(a.out / 'profile.json', details)
    print(f'PROFILE_NNZFLAT_DONE {a.out}', flush=True)


if __name__ == '__main__':
    main()
