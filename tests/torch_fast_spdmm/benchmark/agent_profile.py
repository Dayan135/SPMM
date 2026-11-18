"""Profile one validated SpMM call between CUDA profiler start/stop markers."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from benchmark import bench_nnzflat as bn
from benchmark import universal_suite as suite
from optimizer.agent_tools.bench import parse_kernel


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--kernel', required=True)
    ap.add_argument('--mat', required=True)
    ap.add_argument('--F', type=int, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--commit', required=True)
    args = ap.parse_args(argv)
    import torch
    import torch_fast_spdmm as spmm
    import torch_fast_spdmm_core as core
    build = Path(os.environ['PYTHONPATH'].split(':')[0]).resolve()
    assert Path(spmm.__file__).resolve().is_relative_to(build)
    assert Path(core.__file__).resolve().is_relative_to(build)
    spec = parse_kernel(args.kernel)
    root = Path(os.environ['AGENT_ROOT'])/'suite'
    manifest = json.loads((root/'manifest.json').read_text())
    csr, _ = suite.load_matrix(root, args.mat, os.environ['SPMM_MATRIX_CACHE'])
    x_cpu = suite.make_x(csr.shape[1], args.F)
    reference = bn.GpuReference(torch, root/'refs'/f'{args.mat}_F{args.F}',
                                manifest['matrices'][args.mat]['csr_hash'],
                                suite.x_hash(x_cpu), args.F)
    x = torch.from_numpy(x_cpu).cuda()
    if spec['fmt'] == 'pbr':
        sparse = spmm.csr_to_pbr(csr, **spec['params']).to('cuda')
        fn = lambda: spmm.pbr_matmul(sparse, x, kernel=spec['variant'])
    else:
        p = spec['params']
        sparse = spmm.csr_to_boo(csr, cluster=p['cluster'], target_nr=p['target_nr'],
                                  target_nc=p['target_nc']).to('cuda')
        rowseg = spec['variant'] != 'nnzflat'
        launch = 'legacy' if spec['variant'] == 'legacy' else p['launch_mode']
        fn = lambda: spmm.boo_matmul(sparse, x, threads=p['threads'], p=p['p'],
                                     row_seg=rowseg, launch_mode=launch,
                                     features_per_thread=p['features_per_thread'],
                                     nnzflat_impl=p['impl'] if not rowseg else None)
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    checked = reference.check(fn())
    if checked['bound_violations']:
        raise RuntimeError(f'profile output failed reference: {checked}')
    torch.cuda.synchronize()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.with_suffix('.case.json').write_text(json.dumps({
        'mat': args.mat, 'F': args.F, 'fmt': spec['fmt'],
        'kernel': spec['kernel'], 'commit': args.commit}, sort_keys=True))
    torch.cuda.cudart().cudaProfilerStart()
    fn()
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()


if __name__ == '__main__':
    main()
