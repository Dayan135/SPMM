"""Tables and figures for the nnzflat FPT1/2 experiment (CPU; reads RUN/results).

Paired ratios use the per-repeat samples of two arms of the same cell (same
allocation, A, X, shuffled repeat): ratio_r = t_B,r / t_A,r; the reported value is
the median over repeats with its IQR. ratio > 1 means A is faster.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'tests' / 'torch_fast_spdmm'), str(ROOT)]

from benchmark import nnzflat_common as nc  # noqa: E402

BLUE, ORANGE, AQUA = '#2a78d6', '#eb6834', '#1baf7a'
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e3df'
SEQ = ['#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5', '#256abf', '#184f95', '#0d366b']
DIV = ['#256abf', '#6da7ec', '#b7d3f6', '#f0efec', '#f4b9b8', '#e66767', '#c23a39']
T_COLORS = {128: BLUE, 256: ORANGE, 512: AQUA}


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------

def _read(paths):
    frames = [pd.read_csv(p, low_memory=False) for p in paths]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load(run, stage, alloc=None):
    base = Path(run) / 'results' / f'stage{stage}'
    if alloc:
        base = base / alloc
    s = _read(sorted(base.glob('task_*/summary.csv')))
    m = _read(sorted(base.glob('task_*/samples.csv')))
    for df in (s, m):
        if df.empty:
            continue
        who = df['arm'].str.split(':', n=1)
        df['obj'] = who.str[0]
        df['name'] = who.str[1]
        df['inp'] = df['obj'].str.split('@').str[0]
        if 'k' not in df or df['k'].isna().all():
            df['k'] = df['obj'].str.extract(r'@k(\d+)')[0]
        df['k'] = pd.to_numeric(df['k'], errors='coerce')
    return s, m


def boo(s, status='measured'):
    d = s[s['obj'].str.contains('@k', na=False)].copy()
    if status:
        d = d[d['status'] == status]
    parsed = d['name'].map(nc.parse_arm)
    for key in ('kernel', 'impl', 'p', 'threads', 'fpt', 'duplicate'):
        d['a_' + key] = parsed.map(lambda a, key=key: a[key])
    d['median_ms'] = d['median_ms'].astype(float)
    return d


def wide(samples):
    """repeat x (cell, arm) time table for paired ratios."""
    return samples.pivot_table(index=['cell', 'repeat'], columns='arm', values='time_ms')


def paired(w, cell, a, b):
    """median, p25, p75 of t_b / t_a over repeats of one cell (a faster when > 1)."""
    try:
        x = w.loc[cell]
        r = (x[b] / x[a]).dropna().to_numpy()
    except KeyError:
        return None
    if not len(r):
        return None
    return dict(ratio=float(np.median(r)), p25=float(np.percentile(r, 25)),
                p75=float(np.percentile(r, 75)), n=int(len(r)))


def gmean(x):
    x = np.asarray([v for v in x if v and v > 0 and math.isfinite(v)])
    return float(np.exp(np.log(x).mean())) if len(x) else float('nan')


def summarize(ratios, label):
    r = np.asarray([v for v in ratios if v is not None and math.isfinite(v)])
    if not len(r):
        return dict(comparison=label, n=0)
    return dict(comparison=label, n=int(len(r)), geomean=gmean(r), median=float(np.median(r)),
                min=float(r.min()), max=float(r.max()), frac_gt_1_05=float((r > 1.05).mean()),
                frac_lt_0_95=float((r < 0.95).mean()))


# ---------------------------------------------------------------------------
# screening analyses (Stage 1 and 2)
# ---------------------------------------------------------------------------

def controls(s, m):
    w = wide(m)
    rows = []
    for (cell, obj), g in boo(s).groupby(['cell', 'obj']):
        names = set(g['name'])
        for p in (8, 64):
            ref, tpl = f'nfref_p{p}_t256_f1', f'nf_p{p}_t256_f1'
            if {ref, tpl} <= names:
                pr = paired(w, cell, f'{obj}:{tpl}', f'{obj}:{ref}')
                rows.append(dict(cell=cell, obj=obj, control=f'reference/template FPT1 P{p}T256', **pr))
        if {'nf_p64_t256_f1', 'nf_p64_t256_f1_dup'} <= names:
            pr = paired(w, cell, f'{obj}:nf_p64_t256_f1', f'{obj}:nf_p64_t256_f1_dup')
            rows.append(dict(cell=cell, obj=obj, control='duplicate arm P64T256 FPT1', **pr))
    return pd.DataFrame(rows)


def pair_table(s, m, vary, a_val, b_val, kernel='nnzflat'):
    """Paired t_b/t_a for arms equal except `vary` (a_* column) a_val vs b_val."""
    d = boo(s)
    d = d[(d['a_kernel'] == kernel) & (~d['a_duplicate']) & (d['a_impl'] != 'reference')]
    w = wide(m)
    keys = [c for c in ('a_p', 'a_threads', 'a_fpt') if c != vary]
    rows = []
    for (cell, obj, *rest), g in d.groupby(['cell', 'obj'] + keys):
        ga, gb = g[g[vary] == a_val], g[g[vary] == b_val]
        if len(ga) == 1 and len(gb) == 1:
            pr = paired(w, cell, ga['arm'].iloc[0], gb['arm'].iloc[0])
            rows.append(dict(cell=cell, obj=obj, inp=g['inp'].iloc[0], k=g['k'].iloc[0],
                             features=g['features'].iloc[0], **dict(zip(keys, rest)),
                             a=ga['name'].iloc[0], b=gb['name'].iloc[0], **pr))
    return pd.DataFrame(rows)


def nf_vs_rs(s, m):
    d = boo(s)
    d = d[~d['a_duplicate'] & (d['a_impl'] != 'reference')]
    w = wide(m)
    rows = []
    for (cell, obj, p, t, f), g in d.groupby(['cell', 'obj', 'a_p', 'a_threads', 'a_fpt']):
        nf, rs = g[g['a_kernel'] == 'nnzflat'], g[g['a_kernel'] == 'rowseg']
        if len(nf) == 1 and len(rs) == 1:
            pr = paired(w, cell, nf['arm'].iloc[0], rs['arm'].iloc[0])
            rows.append(dict(cell=cell, obj=obj, inp=g['inp'].iloc[0], k=g['k'].iloc[0],
                             features=g['features'].iloc[0], p=p, threads=t, fpt=f,
                             nf_ms=nf['median_ms'].iloc[0], rs_ms=rs['median_ms'].iloc[0], **pr))
    return pd.DataFrame(rows)


def best_by(s, keys):
    d = boo(s)
    d = d[~d['a_duplicate'] & (d['a_impl'] != 'reference')]
    out = []
    for key, g in d.groupby(keys + ['a_kernel']):
        b = g.loc[g['median_ms'].idxmin()]
        out.append(dict(zip(keys + ['kernel'], key if isinstance(key, tuple) else (key,)),
                        arm=b['name'], median_ms=b['median_ms'], arena_bytes=b.get('arena_bytes'),
                        num_blocks=b.get('num_blocks'), nnz_per_block_p50=b.get('nnz_per_block_p50'),
                        max_nnz_blk=b.get('max_nnz_blk'), max_active_blocks_per_sm=b.get('max_active_blocks_per_sm'),
                        remainder_fraction=b.get('remainder_fraction')))
    return pd.DataFrame(out)


def skew_table(s, m):
    """Stage 1: t(skewed)/t(balanced) per identical arm (same cell, same repeat)."""
    d = boo(s)
    d = d[~d['a_duplicate'] & (d['a_impl'] != 'reference')]
    w = wide(m)
    rows = []
    for (cell, k, name), g in d.groupby(['cell', 'k', 'name']):
        bal, skw = g[g['inp'] == 'syn_balanced'], g[g['inp'] == 'syn_skewed']
        if len(bal) == 1 and len(skw) == 1:
            pr = paired(w, cell, bal['arm'].iloc[0], skw['arm'].iloc[0])
            a = nc.parse_arm(name)
            rows.append(dict(cell=cell, k=k, features=g['features'].iloc[0], arm=name, kernel=a['kernel'],
                             p=a['p'], threads=a['threads'], fpt=a['fpt'],
                             balanced_ms=bal['median_ms'].iloc[0], skewed_ms=skw['median_ms'].iloc[0],
                             skew_penalty=pr['ratio'], p25=pr['p25'], p75=pr['p75']))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------

def _plt():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 8, 'axes.edgecolor': INK2, 'axes.labelcolor': INK,
                         'xtick.color': INK2, 'ytick.color': INK2, 'axes.titlesize': 9,
                         'axes.spines.top': False, 'axes.spines.right': False})
    return plt


def heatmaps(s_all, inp, features, ks, path, title):
    """P x T heatmap per (k, FPT): median ms relative to the best nnzflat of the panel."""
    plt = _plt()
    from matplotlib.colors import LinearSegmentedColormap, LogNorm
    cmap = LinearSegmentedColormap.from_list('seq', SEQ)
    d = s_all[(s_all['inp'] == inp) & (s_all['features'] == features)
              & s_all['name'].str.match(r'^nf_p\d+_t\d+_f\d$')].copy()
    meas = d[d['status'] == 'measured']
    if meas.empty:
        return False
    best = meas['median_ms'].astype(float).min()
    fig, axes = plt.subplots(len(ks), 2, figsize=(6.4, 1.9 * len(ks) + 0.6), squeeze=False)
    norm = LogNorm(vmin=1.0, vmax=max(2.0, float(meas['median_ms'].astype(float).max() / best)))
    for i, k in enumerate(ks):
        for j, fpt in enumerate(nc.FPTS):
            ax = axes[i][j]
            grid = np.full((len(nc.THREADS), len(nc.PS)), np.nan)
            text = [[''] * len(nc.PS) for _ in nc.THREADS]
            for _, r in d[d['k'] == k].iterrows():
                a = nc.parse_arm(r['name'])
                if a['fpt'] != fpt:
                    continue
                ti, pi = nc.THREADS.index(a['threads']), nc.PS.index(a['p'])
                if r['status'] == 'measured':
                    grid[ti, pi] = float(r['median_ms']) / best
                    text[ti][pi] = f"{grid[ti, pi]:.2f}"
                else:
                    text[ti][pi] = {'unsupported': 'n/a', 'alias': 'alias', 'failed': 'FAIL'}.get(r['status'], '?')
            ax.imshow(grid, cmap=cmap, norm=norm, aspect='auto')
            for ti in range(len(nc.THREADS)):
                for pi in range(len(nc.PS)):
                    v = grid[ti, pi]
                    color = 'white' if (math.isfinite(v) and v > norm.vmax ** 0.55) else INK
                    ax.text(pi, ti, text[ti][pi], ha='center', va='center', fontsize=7,
                            color=color if math.isfinite(v) else INK2)
            ax.set_xticks(range(len(nc.PS)), [f'P{p}' for p in nc.PS])
            ax.set_yticks(range(len(nc.THREADS)), [f'T{t}' for t in nc.THREADS])
            ax.set_title(f'k{k}, FPT{fpt}', color=INK)
            for sp in ax.spines.values():
                sp.set_visible(False)
    fig.suptitle(f'{title}\ncell value = median complete-call time / best nnzflat ({best:.3f} ms); '
                 'n/a = infeasible', fontsize=8, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    fig.savefig(Path(path).with_suffix('.pdf'))
    plt.close(fig)
    return True


def fpt_plot(fpt_pairs, path, title):
    if fpt_pairs.empty:
        return
    plt = _plt()
    inputs = list(dict.fromkeys(fpt_pairs['inp']))
    cols = min(4, len(inputs))
    rows = -(-len(inputs) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(2.2 * cols + 0.6, 2.0 * rows + 0.8), squeeze=False,
                             sharey=True)
    for ax in axes.flat[len(inputs):]:
        ax.set_visible(False)
    for ax, inp in zip(axes.flat, inputs):
        g = fpt_pairs[fpt_pairs['inp'] == inp]
        for t in nc.THREADS:
            h = g[g['a_threads'] == t]
            x = np.log2(h['a_p'].astype(float)) + (nc.THREADS.index(t) - 1) * 0.12
            ax.scatter(x, h['ratio'], s=10, color=T_COLORS[t], label=f'T{t}', linewidths=0)
        ax.axhline(1.0, color=INK2, lw=0.8)
        ax.set_xticks(np.log2(nc.PS), [str(p) for p in nc.PS])
        ax.set_title(inp, color=INK)
        ax.grid(axis='y', color=GRID, lw=0.6)
        ax.set_xlabel('P')
    axes[0][0].set_ylabel('t(FPT1) / t(FPT2)  (>1: FPT2 faster)')
    axes[0][0].legend(frameon=False, fontsize=7)
    fig.suptitle(title, fontsize=9, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    fig.savefig(Path(path).with_suffix('.pdf'))
    plt.close(fig)


def k_curves(best, path, title, ks):
    if best.empty:
        return
    plt = _plt()
    inputs = list(dict.fromkeys(best['inp']))
    cols = min(4, len(inputs))
    rows = -(-len(inputs) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(2.6 * cols + 0.6, 2.3 * rows + 0.8), squeeze=False)
    for ax in axes.flat[len(inputs):]:
        ax.set_visible(False)
    for ax, inp in zip(axes.flat, inputs):
        g = best[best['inp'] == inp]
        for kernel, color in (('nnzflat', BLUE), ('rowseg', ORANGE)):
            h = g[g['kernel'] == kernel].sort_values('k')
            ax.plot(h['k'], h['median_ms'], marker='o', ms=4, lw=2, color=color, label=kernel)
            if kernel == 'nnzflat':
                for _, r in h.iterrows():
                    ax.annotate(f"{int(r['num_blocks'])} blk\n{int(r['arena_bytes']) // 1024} KiB",
                                (r['k'], r['median_ms']), textcoords='offset points', xytext=(3, 4),
                                fontsize=5.5, color=INK2)
        ax.set_xscale('log', base=2)
        ax.set_xticks(ks, [str(k) for k in ks])
        ax.set_yscale('log')
        ax.set_title(inp, color=INK)
        ax.grid(color=GRID, lw=0.6)
        ax.set_xlabel('k (max group size)')
    axes[0][0].set_ylabel('best median ms (log)')
    axes[0][0].legend(frameon=False, fontsize=7)
    fig.suptitle(title + '\nannotation: nnzflat best arm block count and arena', fontsize=8, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    fig.savefig(Path(path).with_suffix('.pdf'))
    plt.close(fig)


def comparison_plot(table, path, title):
    """Stage 3: best nnzflat, best rowseg, best direct per input/F (median over allocations)."""
    plt = _plt()
    t = table.copy()
    t['label'] = t['input'] + ' F' + t['features'].astype(str)
    t = t.sort_values(['input', 'features'])
    y = np.arange(len(t))
    fig, ax = plt.subplots(figsize=(6.4, 0.32 * len(t) + 1.2))
    for col, color, name, dy in (('nnzflat_ms', BLUE, 'best nnzflat', -0.2),
                                 ('rowseg_ms', ORANGE, 'best rowseg', 0.0),
                                 ('direct_ms', AQUA, 'best direct cuSPARSE', 0.2)):
        ax.scatter(t[col], y + dy, s=14, color=color, label=name, linewidths=0, zorder=3)
    ax.set_yticks(y, t['label'])
    ax.set_xscale('log')
    ax.invert_yaxis()
    ax.grid(axis='x', color=GRID, lw=0.6)
    ax.set_xlabel('median complete-call ms over three allocations (log)')
    ax.legend(frameon=False, fontsize=7, loc='lower right')
    ax.set_title(title, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    fig.savefig(Path(path).with_suffix('.pdf'))
    plt.close(fig)


# ---------------------------------------------------------------------------
# Stage 3
# ---------------------------------------------------------------------------

def confirmation(run, out):
    manifest = json.loads((Path(run) / 'manifests' / 'stage3.json').read_text())
    rows, totals = [], []
    ctrl_rows = []
    sources = [(3, a, a, '') for a in (1, 2, 3)] + [('3_supp', a, f'supp{a}', '_supp') for a in (1, 2, 3)]
    for stage, a_dir, a, cell_suffix in sources:
        if not (Path(run) / 'results' / f'stage{stage}' / f'alloc{a_dir}').exists():
            continue
        s, m = load(run, stage, f'alloc{a_dir}')
        if s.empty:
            continue
        if cell_suffix:  # supplementary re-measurement: its own cell, never pooled with the originals
            s['cell'] = s['cell'] + cell_suffix
            m['cell'] = m['cell'] + cell_suffix
        w = wide(m)
        for cell, g in s.groupby('cell'):
            meas = g[g['status'] == 'measured']
            inp = g['inp'].iloc[0]
            features = int(g['features'].iloc[0])
            role = 'heldout_scale' if inp.endswith('_1m') else ('heldout_feature' if features == 256
                                                                else 'screened_feature')
            base_inp = inp[:-3] if inp.endswith('_1m') else inp
            sel_f = features if features != 256 else None
            role = role + ('_supplementary' if cell.endswith('_supp') else '')
            b = meas[meas['obj'].str.contains('@k', na=False)].copy()
            if b.empty:
                continue
            b['parsed'] = b['name'].map(nc.parse_arm)
            direct = meas[meas['name'].str.match(r'^direct_\w+_api$', na=False)]
            dbest = direct.loc[direct['median_ms'].astype(float).idxmin()] if len(direct) else None
            sels = ([manifest['selections'].get(f'{base_inp}_F{sel_f}')] if sel_f else
                    [manifest['selections'].get(f'{base_inp}_F{f}') for f in nc.SCREEN_F])
            for sel in sels:
                if not sel or not sel.get('best'):
                    continue
                cand = f"{inp}@k{sel['best']['k']}:{sel['best']['arm']}"
                for arm_obj, why in [(f"{inp}@{key}", whys) for key, whys in sel['why'].items()]:
                    if arm_obj == cand or arm_obj.endswith('_dup'):
                        continue
                    pr = paired(w, cell, cand, arm_obj)
                    if pr:
                        rows.append(dict(allocation=a, cell=cell, role=role, input=inp, features=features,
                                         candidate=cand, other=arm_obj, why='; '.join(why), **pr))
                if dbest is not None:
                    pr = paired(w, cell, cand, dbest['arm'])
                    if pr:
                        rows.append(dict(allocation=a, cell=cell, role=role, input=inp, features=features,
                                         candidate=cand, other=f'{inp}:best_direct_api',
                                         other_actual=dbest['arm'], why='best direct (api)', **pr))
                rsb = sel.get('best_rowseg')
                if rsb:
                    pr = paired(w, cell, cand, f"{inp}@k{rsb['k']}:{rsb['arm']}")
                    if pr:
                        rows.append(dict(allocation=a, cell=cell, role=role, input=inp, features=features,
                                         candidate=cand, other=f'{inp}:screen_tuned_rowseg',
                                         other_actual=f"k{rsb['k']}:{rsb['arm']}",
                                         why='screen-tuned rowseg', **pr))
                dup = cand + '_dup'
                pr = paired(w, cell, cand, dup)
                if pr:
                    ctrl_rows.append(dict(allocation=a, cell=cell, control='duplicate', **pr))
                pk = nc.parse_arm(sel['best']['arm'])
                ref = f"{inp}@k{sel['best']['k']}:nfref_p{pk['p']}_t{pk['threads']}_f1"
                tpl = f"{inp}@k{sel['best']['k']}:nf_p{pk['p']}_t{pk['threads']}_f1"
                pr = paired(w, cell, tpl, ref)
                if pr:
                    ctrl_rows.append(dict(allocation=a, cell=cell, control='reference/template FPT1', **pr))
            nfm = b[b['parsed'].map(lambda x: x['kernel'] == 'nnzflat' and x['impl'] == 'template'
                                    and not x['duplicate'])]
            rsm = b[b['parsed'].map(lambda x: x['kernel'] == 'rowseg')]
            totals.append(dict(allocation=a, cell=cell, role=role, input=inp, features=features,
                               nnzflat_ms=nfm['median_ms'].astype(float).min() if len(nfm) else np.nan,
                               nnzflat_arm=nfm.loc[nfm['median_ms'].astype(float).idxmin(), 'arm'] if len(nfm) else '',
                               rowseg_ms=rsm['median_ms'].astype(float).min() if len(rsm) else np.nan,
                               rowseg_arm=rsm.loc[rsm['median_ms'].astype(float).idxmin(), 'arm'] if len(rsm) else '',
                               direct_ms=float(dbest['median_ms']) if dbest is not None else np.nan,
                               direct_arm=dbest['arm'] if dbest is not None else '',
                               unsupported=int((g['status'] == 'unsupported').sum()),
                               failed=int((g['status'] == 'failed').sum())))
    pairs = pd.DataFrame(rows)
    ctrl = pd.DataFrame(ctrl_rows)
    tot = pd.DataFrame(totals)
    if pairs.empty:
        return pairs, ctrl, tot, pd.DataFrame()
    lo = float(ctrl['ratio'].min()) if len(ctrl) else 1.0
    hi = float(ctrl['ratio'].max()) if len(ctrl) else 1.0
    verdicts = []
    for (cell, cand, other), g in pairs.groupby(['cell', 'candidate', 'other']):
        r = g['ratio'].to_numpy()
        n = len(r)
        if n < 3:
            v = 'incomplete (<3 allocations)'
        elif (r > hi).all() and (r > 1).all():
            v = 'confirmed: candidate faster'
        elif (r < lo).all() and (r < 1).all():
            v = 'confirmed: candidate slower' + (' (>5% regression)' if (r < 1 / 1.05).all() else '')
        else:
            v = 'unresolved'
        verdicts.append(dict(cell=cell, candidate=cand, other=other, why=g['why'].iloc[0],
                             role=g['role'].iloc[0], allocations=n, ratio_min=float(r.min()),
                             ratio_median=float(np.median(r)), ratio_max=float(r.max()),
                             control_lo=lo, control_hi=hi, verdict=v))
    return pairs, ctrl, tot, pd.DataFrame(verdicts)


def setup_costs(run):
    """Clustering (saved labels) and BOO build times from the CPU preparation artifacts.

    Clustering times come from the October 13 label files (their own hosts); BOO
    build times were measured once per object on the preparation CPU node.
    """
    rows = []
    for p in sorted((Path(run) / 'artifacts' / 'prep').glob('*.json')):
        prep = json.loads(p.read_text())
        for k, lab in prep['labels'].items():
            b = prep['boo'].get(k, {})
            rows.append(dict(input=prep['input'], method=prep['method'], k=int(k),
                             cluster_ms=lab.get('cluster_ms'), cluster_node=lab.get('cluster_node'),
                             boo_build_ms=b.get('boo_build_ms'), num_blocks=b.get('num_blocks'),
                             max_nnz_blk=b.get('max_nnz_blk'), prep_node=prep['host']['node']))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# entry
# ---------------------------------------------------------------------------

def report(run, out=None):
    run = Path(run)
    out = Path(out or run / 'analysis')
    (out / 'figures').mkdir(parents=True, exist_ok=True)
    summary = {}
    for stage, ks, inputs in ((1, nc.STAGE1_K, nc.STAGE1_INPUTS), (2, nc.STAGE2_K, nc.STAGE2_INPUTS)):
        s, m = load(run, stage)
        if s.empty:
            continue
        s.to_csv(out / f'stage{stage}_summary.csv', index=False)
        c = controls(s, m)
        c.to_csv(out / f'stage{stage}_controls.csv', index=False)
        thr = pd.concat([pair_table(s, m, 'a_threads', 256, 128).assign(comparison='T128/T256'),
                         pair_table(s, m, 'a_threads', 512, 256).assign(comparison='T256/T512')])
        thr.to_csv(out / f'stage{stage}_threads.csv', index=False)
        fpt = pair_table(s, m, 'a_fpt', 2, 1)
        fpt.to_csv(out / f'stage{stage}_fpt.csv', index=False)
        pp = pd.concat([pair_table(s, m, 'a_p', p, 64).assign(comparison=f'P64/P{p}') for p in (8, 16, 32)]
                       + [pair_table(s, m, 'a_p', 128, 64).assign(comparison='P64/P128')])
        pp.to_csv(out / f'stage{stage}_p.csv', index=False)
        nr = nf_vs_rs(s, m)
        nr.to_csv(out / f'stage{stage}_nnzflat_vs_rowseg.csv', index=False)
        best = best_by(s, ['inp', 'features', 'k'])
        best.to_csv(out / f'stage{stage}_best_by_k.csv', index=False)
        tuned = best_by(s, ['inp', 'features'])
        tuned.to_csv(out / f'stage{stage}_best_tuned.csv', index=False)
        st = dict(controls=[summarize(c[c['control'] == name]['ratio'], name)
                            for name in sorted(set(c['control']))] if len(c) else [],
                  threads=[summarize(thr[thr['comparison'] == cmp]['ratio'], cmp)
                           for cmp in ('T128/T256', 'T256/T512')],
                  threads_by_p={f"{cmp} P{p}": summarize(g['ratio'], cmp)
                                for (cmp, p), g in thr.groupby(['comparison', 'a_p'])},
                  fpt=summarize(fpt['ratio'], 't(FPT1)/t(FPT2)'),
                  fpt_by_p={f'P{p}': summarize(g['ratio'], f'P{p}') for p, g in fpt.groupby('a_p')},
                  small_p=[summarize(g['ratio'], cmp) for cmp, g in pp.groupby('comparison')],
                  rowseg_same_setting=summarize(nr['ratio'], 't(rowseg)/t(nnzflat) same P/T/FPT/k'),
                  counts=s['status'].value_counts().to_dict())
        if stage == 1:
            sk = skew_table(s, m)
            sk.to_csv(out / 'stage1_skew.csv', index=False)
            st['skew_penalty'] = {kern: summarize(g['skew_penalty'], kern) for kern, g in sk.groupby('kernel')}
            both = sk.pivot_table(index=['cell', 'k', 'p', 'threads', 'fpt'], columns='kernel',
                                  values='skew_penalty').dropna()
            if len(both):
                st['skew_penalty_rowseg_over_nnzflat'] = summarize(both['rowseg'] / both['nnzflat'],
                                                                   'rowseg penalty / nnzflat penalty')
            g = nr.copy()
            g['pattern'] = g['inp']
            st['rowseg_same_setting_by_pattern'] = {pat: summarize(h['ratio'], pat)
                                                    for pat, h in g.groupby('pattern')}
        summary[f'stage{stage}'] = st
        for inp in inputs:
            for f in nc.SCREEN_F:
                heatmaps(s, inp, f, ks, out / 'figures' / f'heatmap_s{stage}_{inp}_F{f}.png',
                         f'Stage {stage}: {inp}, F{f}, nnzflat (T x P per k and FPT)')
        if len(fpt):
            for f in nc.SCREEN_F:
                fpt_plot(fpt[fpt['features'] == f], out / 'figures' / f'fpt_pairs_s{stage}_F{f}.png',
                         f'Stage {stage}, F{f}: paired FPT2 vs FPT1 (all k)')
        for f in nc.SCREEN_F:
            k_curves(best[best['features'] == f], out / 'figures' / f'k_curves_s{stage}_F{f}.png',
                     f'Stage {stage}, F{f}: best nnzflat and best rowseg per k', list(ks))
    if (run / 'artifacts' / 'prep').exists():
        setup_costs(run).to_csv(out / 'setup_costs.csv', index=False)
    if (run / 'manifests' / 'stage3.json').exists() and (run / 'results' / 'stage3').exists():
        pairs, ctrl, tot, verdicts = confirmation(run, out)
        pairs.to_csv(out / 'stage3_pairs.csv', index=False)
        ctrl.to_csv(out / 'stage3_controls.csv', index=False)
        tot.to_csv(out / 'stage3_best.csv', index=False)
        verdicts.to_csv(out / 'stage3_verdicts.csv', index=False)
        if len(tot):
            med = tot.groupby(['input', 'features'], as_index=False)[['nnzflat_ms', 'rowseg_ms', 'direct_ms']].median()
            med.to_csv(out / 'stage3_best_median.csv', index=False)
            comparison_plot(med, out / 'figures' / 'stage3_nnzflat_rowseg_direct.png',
                            'Confirmation: nnzflat vs rowseg vs direct cuSPARSE')
        summary['stage3'] = dict(controls=summarize(ctrl['ratio'], 'all controls') if len(ctrl) else {},
                                 verdict_counts=verdicts['verdict'].value_counts().to_dict() if len(verdicts) else {})
    (out / 'report_summary.json').write_text(json.dumps(summary, indent=1, default=float) + '\n')
    print(json.dumps(summary, indent=1, default=float)[:6000])


if __name__ == '__main__':
    report(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
