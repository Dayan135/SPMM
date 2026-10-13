"""Tables and plots for bench_tiles result directories (unprofiled timings only).

  python report_tiles.py screen  <out> <bench dir> [...]   P x FPT heatmaps + cell table
  python report_tiles.py slab    <out> <bench dir> [...]   slab-width plots + table
  python report_tiles.py confirm <out> <bench dir> [...]   per-allocation finalist summary

All ratios are paired within one (allocation, case, F) cell and one scope:
ratio = reference median / variant median, so >1 means the variant is faster.
'best direct' is the fastest validated original-order cuSPARSE algorithm measured
in that same cell and scope; the number of algorithms measured is reported.
"""
import csv
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ANCHOR = (64, 2, 512)
FPT1 = (64, 1, 512)
BLUE, ORANGE, AQUA, YELLOW = '#2a78d6', '#eb6834', '#1baf7a', '#eda100'
INK, MUTED, GRID = '#1a1a19', '#6b6a64', '#e4e3df'


def load(dirs):
    rows, samples = [], []
    for d in dirs:
        d = Path(d)
        rows += list(csv.DictReader(open(d / 'summary.csv')))
        if (d / 'samples.csv').exists():
            samples += list(csv.DictReader(open(d / 'samples.csv')))
    return rows, samples


def key(r):
    return (r['allocation'], r['case'], int(r['features']))


def cfg(r):
    return (int(r['p']), int(r['features_per_thread']), int(r['feat_block']))


def cells(rows):
    out = defaultdict(list)
    for r in rows:
        out[key(r)].append(r)
    return out


def ms(r):
    return float(r['median_ms']) if r.get('median_ms') else None


def summarize(group, scope):
    """Per-cell references at one scope ('full_call' | 'cpp_only')."""
    boo_family = 'boo' if scope == 'full_call' else 'boo_cpp'
    boo = {cfg(r): r for r in group if r['family'] == boo_family}
    direct = [r for r in group if r['family'] == 'direct' and r['scope'] == scope and ms(r)]
    best_direct = min(direct, key=ms) if direct else None
    torch_row = next((r for r in group if r['config'] == 'torch_csr' and ms(r)), None)
    return boo, best_direct, len({r['direct_algorithm'] for r in direct}), torch_row


def cell_table(rows, out, scope_list=('full_call', 'cpp_only')):
    lines = []
    fields = ['allocation', 'case', 'role', 'features', 'scope', 'config', 'p', 'effective_p',
              'features_per_thread', 'feat_block', 'supported', 'unsupported_reason', 'median_ms',
              'p25_ms', 'p75_ms', 'ratio_vs_anchor', 'ratio_vs_fpt1_p64', 'ratio_vs_best_direct',
              'best_direct', 'direct_algorithms_measured', 'ratio_vs_torch', 'arena_bytes', 'grid_z',
              'registers_per_thread', 'local_bytes_per_thread', 'max_active_blocks_per_sm',
              'launch_verified']
    for k, group in sorted(cells(rows).items()):
        for scope in scope_list:
            boo, best_direct, n_direct, torch_row = summarize(group, scope)
            anchor, fpt1 = boo.get(ANCHOR), boo.get(FPT1)
            for c, r in sorted(boo.items()):
                v = ms(r)
                lines.append(dict(
                    {f: r.get(f) for f in fields if f in r}, scope=scope,
                    ratio_vs_anchor=ms(anchor) / v if v and anchor and ms(anchor) else None,
                    ratio_vs_fpt1_p64=ms(fpt1) / v if v and fpt1 and ms(fpt1) else None,
                    ratio_vs_best_direct=ms(best_direct) / v if v and best_direct else None,
                    best_direct=best_direct['direct_algorithm'] if best_direct else None,
                    direct_algorithms_measured=n_direct,
                    ratio_vs_torch=(ms(torch_row) / v if v and torch_row and scope == 'full_call'
                                    else None)))
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(lines)
    return lines


def _plt():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
    plt.rcParams.update({'font.size': 8, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK,
                         'xtick.color': MUTED, 'ytick.color': MUTED, 'text.color': INK,
                         'axes.titlesize': 9, 'figure.facecolor': '#fcfcfb',
                         'axes.facecolor': '#fcfcfb', 'savefig.facecolor': '#fcfcfb'})
    cmap = LinearSegmentedColormap.from_list('div', ['#b3261e', '#ec835a', '#f0efec', '#86b6ef', '#1c5cab'])
    return plt, cmap, TwoSlopeNorm


def heatmaps(lines, out, scope='full_call', allocation=None):
    plt, cmap, Norm = _plt()
    sel = [l for l in lines if l['scope'] == scope and int(l['feat_block']) == 512
           and (allocation is None or l['allocation'] == allocation)]
    cases = sorted({l['case'] for l in sel}, key=lambda c: [l['case'] for l in sel].index(c))
    feats = sorted({int(l['features']) for l in sel})
    ps, fpts = [16, 32, 64, 128], [1, 2, 4]
    fig, axes = plt.subplots(len(cases), len(feats), figsize=(3.3 * len(feats), 2.5 * len(cases)),
                             squeeze=False)
    norm = Norm(vcenter=1.0, vmin=0.5, vmax=1.5)
    for i, case in enumerate(cases):
        for j, F in enumerate(feats):
            ax = axes[i][j]
            grid = [[math.nan] * len(fpts) for _ in ps]
            for l in sel:
                if l['case'] == case and int(l['features']) == F and l['ratio_vs_anchor']:
                    grid[ps.index(int(l['p']))][fpts.index(int(l['features_per_thread']))] = \
                        float(l['ratio_vs_anchor'])
            ax.imshow(grid, cmap=cmap, norm=norm, aspect='auto')
            for l in sel:
                if l['case'] != case or int(l['features']) != F:
                    continue
                y, x = ps.index(int(l['p'])), fpts.index(int(l['features_per_thread']))
                if not l['median_ms']:
                    ax.text(x, y, 'n/a', ha='center', va='center', color=INK, fontsize=6)
                    continue
                spill = int(l['local_bytes_per_thread'] or 0)
                text = (f"{float(l['ratio_vs_anchor']):.2f}x\n{float(l['median_ms']):.2f} ms\n"
                        f"{int(l['arena_bytes']) / 1024:.1f}K {l['max_active_blocks_per_sm']}cta "
                        f"{l['registers_per_thread']}r" + (f' L{spill}' if spill else ''))
                ax.text(x, y, text, ha='center', va='center', color=INK, fontsize=5.5)
                if (int(l['p']), int(l['features_per_thread'])) == ANCHOR[:2]:
                    ax.add_patch(plt.Rectangle((x - .5, y - .5), 1, 1, fill=False, ec=INK, lw=1.2))
            ax.set_xticks(range(len(fpts)), [f'FPT{f}' for f in fpts])
            ax.set_yticks(range(len(ps)), [f'P{p}' for p in ps])
            ax.set_title(f'{case}  F={F}')
    fig.suptitle(f'Speedup vs P64/FPT2/slab512 (boxed), {scope}; text: ms, arena, CTAs/SM, regs/thread',
                 fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(out, dpi=150)
    fig.savefig(str(out).replace('.png', '.pdf'))
    plt.close(fig)


def slab_plot(lines, out, scope='full_call'):
    plt, _, _ = _plt()
    sel = [l for l in lines if l['scope'] == scope and l['median_ms']]
    cases = list(dict.fromkeys(l['case'] for l in sel))
    feats = sorted({int(l['features']) for l in sel})
    colors = [BLUE, ORANGE, AQUA, YELLOW]
    fig, axes = plt.subplots(len(cases), len(feats), figsize=(3.0 * len(feats), 2.3 * len(cases)),
                             squeeze=False, sharex=True, sharey=True)
    for i, case in enumerate(cases):
        for j, F in enumerate(feats):
            ax = axes[i][j]
            pf = sorted({(int(l['p']), int(l['features_per_thread'])) for l in sel
                         if l['case'] == case and int(l['features']) == F})
            for n, (p, f) in enumerate(pf[:4]):
                pts = sorted((int(l['feat_block']), float(l['median_ms'])) for l in sel
                             if l['case'] == case and int(l['features']) == F
                             and (int(l['p']), int(l['features_per_thread'])) == (p, f))
                base = dict(pts).get(512)
                if not base:
                    continue
                ax.plot([s for s, _ in pts], [base / v for _, v in pts], marker='o', ms=4, lw=2,
                        color=colors[n], label=f'P{p}/FPT{f}')
            ax.axhline(1.0, color=MUTED, lw=0.8)
            ax.set_xscale('log', base=2)
            ax.set_xticks([128, 256, 512, 1024], ['128', '256', '512', '1024'])
            ax.grid(color=GRID, lw=0.5)
            ax.set_title(f'{case}  F={F}')
            if j == 0:
                ax.set_ylabel('speedup vs slab512')
            if i == len(cases) - 1:
                ax.set_xlabel('slab width')
            if i == 0 and j == len(feats) - 1:
                ax.legend(frameon=False, fontsize=7)
    fig.suptitle('Slab width at fixed P/FPT: speedup vs slab512 (F128: all widths run one identical slab)',
                 fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(out, dpi=150)
    fig.savefig(str(out).replace('.png', '.pdf'))
    plt.close(fig)


def confirm(lines, samples, out_csv, finalists):
    """Finalist vs anchor vs best direct per allocation, plus cross-allocation spread."""
    per = []
    for l in lines:
        if not l['median_ms'] or (int(l['p']), int(l['features_per_thread']), int(l['feat_block'])) \
                not in finalists:
            continue
        per.append(l)
    by = defaultdict(list)
    for l in per:
        by[(l['case'], int(l['features']), l['scope'], l['config'])].append(l)
    rows = []
    for (case, F, scope, config), group in sorted(by.items()):
        va = [float(g['ratio_vs_anchor']) for g in group if g['ratio_vs_anchor']]
        vd = [float(g['ratio_vs_best_direct']) for g in group if g['ratio_vs_best_direct']]
        rows.append(dict(case=case, role=group[0].get('role'), features=F, scope=scope, config=config,
                         allocations=len(group),
                         median_ms_per_allocation=';'.join(f"{float(g['median_ms']):.4f}" for g in group),
                         vs_anchor_min=min(va) if va else None, vs_anchor_median=statistics.median(va) if va else None,
                         vs_anchor_max=max(va) if va else None,
                         vs_best_direct_min=min(vd) if vd else None,
                         vs_best_direct_median=statistics.median(vd) if vd else None,
                         vs_best_direct_max=max(vd) if vd else None,
                         best_direct_per_allocation=';'.join(str(g['best_direct']) for g in group),
                         direct_algorithms_measured=';'.join(str(g['direct_algorithms_measured']) for g in group)))
    with open(out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ['case'])
        w.writeheader()
        w.writerows(rows)
    return rows


def main():
    mode, out, dirs = sys.argv[1], Path(sys.argv[2]), sys.argv[3:]
    out.mkdir(parents=True, exist_ok=True)
    rows, samples = load(dirs)
    lines = cell_table(rows, out / f'{mode}_cells.csv')
    if mode == 'screen':
        for alloc in sorted({l['allocation'] for l in lines}):
            heatmaps(lines, out / f'screen_heatmap_full_call_{alloc}.png', 'full_call', alloc)
            heatmaps(lines, out / f'screen_heatmap_cpp_{alloc}.png', 'cpp_only', alloc)
    elif mode == 'slab':
        slab_plot(lines, out / 'slab_full_call.png')
    elif mode == 'confirm':
        finalists = {tuple(int(v) for v in s.split(':')) for s in
                     (Path(out) / 'finalists.txt').read_text().split()}
        confirm(lines, samples, out / 'confirm_summary.csv', finalists | {ANCHOR})
    print(f'{len(lines)} rows -> {out}')


if __name__ == '__main__':
    main()
