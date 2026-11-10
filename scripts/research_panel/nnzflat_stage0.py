"""Stage 0 audits for the nnzflat FPT template (GPU node, after the build).

  python nnzflat_stage0.py launch --out DIR     # attributes/occupancy of every specialization
  python nnzflat_stage0.py sequence --out DIR   # run every specialization once, in a fixed
                                                # order (wrap in nsys; see `audit`)
  python nnzflat_stage0.py audit --out DIR --sqlite T.sqlite   # nsys kernel names == sequence
  python nnzflat_stage0.py sass --out DIR --sass all.sass      # instruction counts per kernel,
                                                # reference vs template FPT1 diff, FPT2 loads/atomics
"""
import argparse
import collections
import csv
import json
from pathlib import Path
import re
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'tests' / 'torch_fast_spdmm'), str(ROOT / 'torch_fast_spdmm'), str(ROOT)]

from benchmark import nnzflat_common as nc  # noqa: E402

VARIANTS = [('nnzflat', 'reference', 1), ('nnzflat', 'template', 1), ('nnzflat', 'template', 2),
            ('rowseg', '', 1), ('rowseg', '', 2)]


def all_arms():
    for kernel, impl, fpt in VARIANTS:
        for t in nc.THREADS:
            for p in nc.PS:
                yield dict(kernel=kernel, impl=impl, p=p, threads=t, fpt=fpt, duplicate=False)


def example():
    import numpy as np
    import scipy.sparse as sp
    import torch_fast_spdmm as spmm
    rng = np.random.default_rng(0)
    mat = sp.random(512, 512, density=0.05, format='csr', random_state=rng, dtype=np.float32)
    boo = spmm.csr_to_boo(mat, row_labels=np.arange(512) // 32, col_labels=np.arange(512) // 32,
                          small_block_policy='remainder', small_block_threshold=4, nz_order='rowmajor')
    return mat, boo


def launch(out):
    import torch
    import torch_fast_spdmm_core as core
    _, boo = example()
    limit = torch.cuda.get_device_properties(0).shared_memory_per_block_optin
    rows = []
    for arm in all_arms():
        info, reason = nc.launch_info(core, boo, 512, arm, limit)
        row = dict(info or {})
        row.update(kernel=arm['kernel'], impl=arm['impl'], p=arm['p'], threads=arm['threads'],
                   fpt=arm['fpt'], expected=nc.expected_kernel(arm), reason=reason or '')
        rows.append(row)
    fields = sorted({k for r in rows for k in r})
    with (out / 'launch_attributes.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    spills = [r for r in rows if r.get('local_bytes_per_thread')]
    print(f'{len(rows)} specializations; {len(spills)} with local memory; device opt-in {limit} B')
    for r in rows:
        print(f"  {r['expected']:62s} regs={r.get('registers_per_thread')} "
              f"local={r.get('local_bytes_per_thread')} ctas/SM={r.get('max_active_blocks_per_sm')}")


def sequence(out):
    import numpy as np
    import torch
    import torch_fast_spdmm as spmm
    import torch_fast_spdmm_core as core
    mat, boo = example()
    boo.to('cuda')
    x = torch.randn(512, 200, generator=torch.Generator().manual_seed(1)).cuda()
    expected = torch.from_numpy(mat.toarray()).double() @ x.cpu().double()
    order = []
    torch.cuda.synchronize()
    torch.cuda.profiler.start()
    for arm in all_arms():
        kwargs = dict(threads=arm['threads'], p=arm['p'], launch_mode='auto',
                      features_per_thread=arm['fpt'], row_seg=arm['kernel'] == 'rowseg')
        if arm['kernel'] == 'nnzflat':
            kwargs['nnzflat_impl'] = arm['impl']
        y = spmm.boo_matmul(boo, x, **kwargs)
        torch.cuda.synchronize()
        torch.testing.assert_close(y.cpu().double(), expected, rtol=3e-5, atol=3e-5)
        order.append(nc.expected_kernel(arm))
    torch.cuda.profiler.stop()
    nc.reset_settings(core)
    (out / 'expected_sequence.json').write_text(json.dumps(order, indent=1) + '\n')
    print(f'{len(order)} specializations launched and numerically checked')


def audit(out, sqlite):
    order = json.loads((out / 'expected_sequence.json').read_text())
    con = sqlite3.connect(sqlite)
    q = ('SELECT k.start, s.value FROM CUPTI_ACTIVITY_KIND_KERNEL k '
         'JOIN StringIds s ON s.id = COALESCE(k.demangledName, k.shortName) ORDER BY k.start')
    names = [n for _, n in con.execute(q)]
    boo = [nc.classify(n) for n in names if nc.classify(n)]
    ok = boo == order
    result = dict(expected=len(order), captured_boo_kernels=len(boo), all_kernels=len(names),
                  match=ok, mismatches=[dict(i=i, expected=e, launched=g)
                                        for i, (e, g) in enumerate(zip(order, boo)) if e != g][:20])
    (out / 'nsys_kernel_audit.json').write_text(json.dumps(result, indent=1) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'mismatches'}))
    if not ok:
        raise SystemExit('nsys kernel-name audit FAILED')


FUNC_RE = re.compile(r'Function : (\S+)')
OP_RE = re.compile(r'/\*[0-9a-f]{4,}\*/\s+(?:@!?U?P\w+\s+)?([A-Z][A-Z0-9_.]+)')


def sass(out, path):
    import subprocess
    funcs, cur = {}, None
    for line in Path(path).read_text().splitlines():
        m = FUNC_RE.search(line)
        if m:
            cur = m.group(1)
            funcs[cur] = []
            continue
        if cur:
            m = OP_RE.search(line)
            if m:
                funcs[cur].append(m.group(1))
    demangled = {}
    names = list(funcs)
    if names:
        res = subprocess.run(['c++filt'], input='\n'.join(names), capture_output=True, text=True)
        demangled = dict(zip(names, res.stdout.splitlines()))
    report = {}
    keep = ('LDS', 'LDS.64', 'LDS.U16', 'LDS.U.16', 'ATOMS', 'RED', 'ATOM', 'STS', 'BAR', 'FMUL',
            'FFMA', 'FADD', 'LDG', 'DMUL', 'DADD')
    for mangled, ops in funcs.items():
        name = demangled.get(mangled, mangled)
        if 'nnzflat' not in name:
            continue
        c = collections.Counter(ops)
        prefix = collections.Counter()
        for op, n in c.items():
            for k in keep:
                if op == k or op.startswith(k + '.'):
                    prefix[op] += n
        report[name] = dict(instructions=len(ops), selected=dict(sorted(prefix.items())),
                            ops=ops)
    # Reference vs template FPT1: identical instruction sequence expected.
    diffs = {}
    for name, r in report.items():
        m = re.search(r'boo_spmm_nnzflat_kernel<(\w+), (\d+), (\d+)>', name)
        if not m:
            continue
        twin = [n for n in report if f'boo_spmm_nnzflat_fpt_kernel<{m.group(1)}, {m.group(2)}, {m.group(3)}, 1>' in n]
        if twin:
            a, b = r['ops'], report[twin[0]]['ops']
            diffs[f'{m.group(1)}_T{m.group(2)}_PL{m.group(3)}'] = dict(
                reference_instructions=len(a), template_instructions=len(b), identical_opcodes=a == b,
                opcode_count_delta={k: v for k, v in (collections.Counter(b) - collections.Counter(a)).items()},
                opcode_count_missing={k: v for k, v in (collections.Counter(a) - collections.Counter(b)).items()})
    summary = {n: {k: v for k, v in r.items() if k != 'ops'} for n, r in report.items()}
    (out / 'sass_summary.json').write_text(json.dumps(dict(kernels=summary, fpt1_vs_reference=diffs),
                                                     indent=1, sort_keys=True) + '\n')
    same = sum(d['identical_opcodes'] for d in diffs.values())
    print(f'{len(summary)} nnzflat kernels; template FPT1 opcode-identical to reference in {same}/{len(diffs)}')
    for n, r in sorted(summary.items()):
        if 'float' in n and ', 256, 6' in n:
            print(f"  {n[:90]}: {r['selected']}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('mode', choices=['launch', 'sequence', 'audit', 'sass'])
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--sqlite', type=Path)
    ap.add_argument('--sass', type=Path)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    if a.mode == 'launch':
        launch(a.out)
    elif a.mode == 'sequence':
        sequence(a.out)
    elif a.mode == 'audit':
        audit(a.out, a.sqlite)
    else:
        sass(a.out, a.sass)


if __name__ == '__main__':
    main()
