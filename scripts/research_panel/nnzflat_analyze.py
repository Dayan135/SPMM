"""Audit, freeze and report for the nnzflat FPT1/2 experiment.

  python nnzflat_analyze.py --root RUN audit    # coverage of the 480 + 1,440 requests; exit 1 if
                                                # any request is missing or failed
  python nnzflat_analyze.py --root RUN freeze   # Stage 3 manifest from Stage 2 screening (frozen)
  python nnzflat_analyze.py --root RUN report   # tables and figures (see nnzflat_report.py)

Every requested configuration ends in exactly one state: measured, alias (identical
saved labels), unsupported (launcher/feasibility reason), failed (pending repair) or
missing. Baselines and controls are audited separately from the requested counts.
"""
import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'tests' / 'torch_fast_spdmm'), str(ROOT)]

from benchmark import nnzflat_common as nc  # noqa: E402

DIRECT = ['DEFAULT', 'CSR_ALG2', 'CSR_ALG1', 'CSR_ALG3']


def read_rows(paths):
    rows = []
    for p in paths:
        with open(p) as f:
            rows += list(csv.DictReader(f))
    return rows


def stage_rows(run, stage, alloc=None):
    base = run / 'results' / f'stage{stage}'
    if alloc:
        base = base / alloc
    return read_rows(sorted(base.glob('task_*/summary.csv')))


def arm_key(row):
    """(input, k, features, arm-name) for BOO arms."""
    who, name = row['arm'].split(':', 1)
    inp, k = who.split('@k')
    return inp, int(k), int(row['features']), name


def audit(run):
    out = run / 'analysis'
    out.mkdir(exist_ok=True)
    result, ok = {}, True
    for stage in (1, 2):
        manifest = json.loads((run / 'manifests' / f'stage{stage}.json').read_text())
        rows = [r for r in stage_rows(run, stage) if '@k' in r['arm']]
        by = {}
        for r in rows:
            key = arm_key(r)
            if key in by:
                raise SystemExit(f'duplicate result for {key}')
            by[key] = r
        coverage, counts = [], {}
        for req in manifest['requested']:
            key = (req['input'], req['k'], req['features'], req['arm'])
            r = by.get(key)
            status = r['status'] if r else 'missing'
            counts[status] = counts.get(status, 0) + 1
            coverage.append(dict(req, status=status, reason=(r or {}).get('reason', ''),
                                 alias_of=(r or {}).get('alias_of', ''),
                                 median_ms=(r or {}).get('median_ms', ''),
                                 arena_bytes=(r or {}).get('arena_bytes', ''),
                                 launch_verified=(r or {}).get('launch_verified', '')))
        extra = {}
        for key, r in by.items():
            if key[3] in manifest['requested_arm_names']:
                continue
            fam = 'rowseg' if key[3].startswith('rs_') else 'control'
            extra.setdefault(fam, {}).setdefault(r['status'], 0)
            extra[fam][r['status']] += 1
        expected_extra = len(manifest['requested']) // len(manifest['requested_arm_names'])
        with (out / f'coverage_stage{stage}.csv').open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(coverage[0]))
            w.writeheader()
            w.writerows(coverage)
        verified = sum(1 for c in coverage if c['status'] == 'measured' and c['launch_verified'] == 'verified')
        result[f'stage{stage}'] = dict(requested=len(coverage), statuses=counts,
                                       measured_launch_verified=verified,
                                       baselines_and_controls=extra,
                                       objects=expected_extra)
        if counts.get('missing') or counts.get('failed') or sum(
                v.get('failed', 0) for v in extra.values()):
            ok = False
    result['complete'] = ok
    (out / 'coverage_summary.json').write_text(json.dumps(result, indent=1) + '\n')
    print(json.dumps(result, indent=1))
    if not ok:
        raise SystemExit('coverage audit: missing or failed cells (see analysis/coverage_*.csv)')


def _ms(r):
    return float(r['median_ms'])


def freeze(run):
    """Deterministic finalist rule over the Stage 2 screen (README: Stage 3)."""
    out_path = run / 'manifests' / 'stage3.json'
    s2 = json.loads((run / 'manifests' / 'stage2.json').read_text())
    rows = [r for r in stage_rows(run, 2) if '@k' in r['arm'] and r['status'] == 'measured']
    table = {}
    for r in rows:
        inp, k, f, name = arm_key(r)
        table.setdefault((inp, f), {})[(k, name)] = r
    selections, tasks = {}, []
    for inp in nc.STAGE2_INPUTS:
        per_f = {}
        for f in nc.SCREEN_F:
            t = table.get((inp, f), {})
            nf = {kn: r for kn, r in t.items() if r['kernel'] == 'nnzflat' and r['impl'] == 'template'
                  and r['duplicate'] == 'False'}
            rs = {kn: r for kn, r in t.items() if r['kernel'] == 'rowseg'}
            sel = dict(why={})
            arms = {}

            def add(k, name, why):
                arms.setdefault(k, [])
                if name not in arms[k]:
                    arms[k].append(name)
                sel['why'].setdefault(f'k{k}:{name}', []).append(why)
            if nf:
                (bk, bname), best = min(nf.items(), key=lambda kv: _ms(kv[1]))
                b = nc.parse_arm(bname)
                add(bk, bname, 'best nnzflat')
                other = nc.arm_name('nnzflat', b['p'], b['threads'], 3 - b['fpt'])
                if (bk, other) in nf:
                    add(bk, other, 'FPT counterpart')
                alts = {kn: r for kn, r in nf.items()
                        if kn[0] != bk or nc.parse_arm(kn[1])['p'] != b['p']}
                if alts:
                    (ak, aname), _ = min(alts.items(), key=lambda kv: _ms(kv[1]))
                    add(ak, aname, 'best alternative with different P or k')
                gains = []
                for (k, name), r in nf.items():
                    a = nc.parse_arm(name)
                    if a['threads'] != 512:
                        continue
                    twin = (k, nc.arm_name('nnzflat', a['p'], 256, a['fpt']))
                    if twin in nf and _ms(nf[twin]) > _ms(r):
                        gains.append((_ms(nf[twin]) / _ms(r), k, twin[1], name))
                if gains:
                    g, k, n256, n512 = max(gains)
                    add(k, n256, f'T256 side of largest T512 gain ({g:.3f}x)')
                    add(k, n512, f'T512 side of largest T512 gain ({g:.3f}x)')
                same = nc.arm_name('rowseg', b['p'], b['threads'], b['fpt'])
                if (bk, same) in rs:
                    add(bk, same, 'rowseg at the candidate setting')
                add(bk, nc.arm_name('nnzflat', b['p'], b['threads'], 1, 'reference'),
                    'identical-work control (reference)')
                add(bk, nc.arm_name('nnzflat', b['p'], b['threads'], 1), 'identical-work control (template)')
                add(bk, bname + '_dup', 'duplicate-arm noise control')
                anchor_k = bk
                sel['best'] = dict(k=bk, arm=bname, median_ms=_ms(best))
            else:
                anchor_k = min(nc.STAGE2_K)
                sel['best'] = None
                sel['note'] = 'no measured nnzflat configuration in the screen'
            for fpt in (1, 2):
                name = nc.arm_name('rowseg', 64, 256, fpt)
                if (anchor_k, name) in rs or not nf:
                    add(anchor_k, name, 'rowseg P64/T256 anchor')
            if rs:
                (tk, tname), tr = min(rs.items(), key=lambda kv: _ms(kv[1]))
                add(tk, tname, 'best tuned rowseg')
                sel['best_rowseg'] = dict(k=tk, arm=tname, median_ms=_ms(tr))
            sel['arms'] = {str(k): v for k, v in sorted(arms.items())}
            per_f[f] = (arms, sel)
            selections[f'{inp}_F{f}'] = sel
        cells = []
        for f in (128, 256, 512):
            if f == 256:
                arms = {}
                for g in nc.SCREEN_F:
                    for k, names in per_f[g][0].items():
                        arms.setdefault(k, [])
                        arms[k] += [n for n in names if n not in arms[k]]
            else:
                arms = per_f[f][0]
            role = 'heldout_feature' if f == 256 else 'screened_feature'
            cells.append(dict(cell=f's3_{inp}_F{f}', features=f, role=role,
                              objects=[dict(input=inp, k=k, arms=names) for k, names in sorted(arms.items())],
                              baselines={inp: dict(torch_csr=True, direct=DIRECT)}))
        if inp in nc.SCALE_INPUTS:
            big = nc.SCALE_INPUTS[inp]
            for f in nc.SCREEN_F:
                arms = per_f[f][0]
                cells.append(dict(cell=f's3_{big}_F{f}', features=f, role='heldout_scale',
                                  objects=[dict(input=big, k=k, arms=names) for k, names in sorted(arms.items())],
                                  baselines={big: dict(torch_csr=True, direct=DIRECT)}))
        tasks.append(dict(task=len(tasks), inputs=[inp], cells=cells))
    inputs = list(nc.STAGE2_INPUTS) + [nc.SCALE_INPUTS[i] for i in nc.STAGE2_INPUTS if i in nc.SCALE_INPUTS]
    lh = {}
    for name in inputs:
        prep = json.loads((nc.art(run) / 'prep' / f'{name}.json').read_text())
        lh[name] = {str(k): v for k, v in prep['label_sha256'].items()}
    n_arms = sum(len(o['arms']) for t in tasks for c in t['cells'] for o in c['objects'])
    manifest = dict(stage=3, warmup=5, repeats=20, order_seed=14103, allocations=3,
                    order_mode='independent seeded shuffle per repeat; seed offset per allocation',
                    rule=('per input/F: best feasible nnzflat (template, non-duplicate) by screen median; '
                          'FPT counterpart; best alternative with different P or k; largest T512-over-T256 '
                          'gain pair; rowseg at the candidate setting; rowseg P64/T256 FPT1/2 anchors at '
                          'the candidate k; best tuned rowseg; reference/template FPT1 identical-work '
                          'control; duplicate candidate arm. F256 uses the union of F128/F512 selections; '
                          'N1048576 uses each F selection. Direct cuSPARSE (4 algorithms) and torch CSR '
                          'in every cell.'),
                    requested_arm_names=s2['requested_arm_names'], label_hashes=lh,
                    selections=selections, boo_arm_count=n_arms, tasks=tasks,
                    fixed=s2['fixed'], screen_manifest_sha256=nc.sha256_file(run / 'manifests' / 'stage2.json'))
    if out_path.exists():
        old = json.loads(out_path.read_text())
        if old != manifest:
            raise SystemExit(f'{out_path} already frozen with different content')
    out_path.write_text(json.dumps(manifest, indent=1) + '\n')
    print(f"stage 3 frozen: {len(tasks)} tasks, {sum(len(t['cells']) for t in tasks)} cells, "
          f"{n_arms} BOO arms per allocation -> {out_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('mode', choices=['audit', 'freeze', 'report'])
    a = ap.parse_args()
    if a.mode == 'audit':
        audit(a.root)
    elif a.mode == 'freeze':
        freeze(a.root)
    else:
        from nnzflat_report import report
        report(a.root)


if __name__ == '__main__':
    main()
