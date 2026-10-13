"""Confirmation figure: per-allocation paired speedup vs strongest same-cell cuSPARSE (F512)."""
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BLUE, ORANGE, AQUA, YELLOW, INK, MUTED, GRID = '#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#1a1a19', '#6b6a64', '#e4e3df'
VARIANTS = [('P64/FPT2 (anchor)', ORANGE, 'o'), ('P64/FPT4', BLUE, 's'), ('P128/FPT4', AQUA, '^')]
ORDER = ['diag_b1.0', 'diag_b1.0_1m', 'fem2d_permuted', 'fem2d_permuted_1m', 'pwtk_hem', 'pwtk',
         'fem2d', 'fem2d_1m', 'diag_dd0.05', 'roadnet', 'webbase']
LABEL = {'diag_dd0.05': 'diag_dd0.05 (held out)', 'diag_b1.0_1m': 'diag_b1.0 · 1M',
         'fem2d_1m': 'fem2d · 1M', 'fem2d_permuted_1m': 'fem2d_permuted · 1M', 'pwtk_hem': 'pwtk (HEM k32)',
         'pwtk': 'pwtk (identity k32)', 'roadnet': 'roadNet (METIS k64)', 'webbase': 'webbase (identity k32)'}


def main(per_alloc, out):
    rows = [r for r in csv.DictReader(open(per_alloc))
            if r['features'] == '512' and r['scope'] == 'full_call' and r['vs_strongest_direct']]
    plt.rcParams.update({'font.size': 8, 'axes.edgecolor': MUTED, 'xtick.color': MUTED, 'ytick.color': INK,
                         'figure.facecolor': '#fcfcfb', 'axes.facecolor': '#fcfcfb', 'savefig.facecolor': '#fcfcfb'})
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for i, case in enumerate(ORDER):
        for k, (variant, color, marker) in enumerate(VARIANTS):
            vals = [float(r['vs_strongest_direct']) for r in rows if r['case'] == case and r['variant'] == variant]
            y = i + (k - 1) * 0.22
            ax.scatter(vals, [y] * len(vals), s=22, color=color, marker=marker, edgecolors='#fcfcfb',
                       linewidths=0.8, zorder=3, label=variant if i == 0 else None)
    ax.axvline(1.0, color=INK, lw=1)
    ax.set_xscale('log', base=2)
    ticks = [0.4, 0.5, 0.67, 1, 1.5, 2, 3]
    ax.set_xticks(ticks, [f'{t:g}×' for t in ticks])
    ax.set_yticks(range(len(ORDER)), [LABEL.get(c, c) for c in ORDER])
    ax.invert_yaxis()
    ax.grid(axis='x', color=GRID, lw=0.6)
    ax.set_xlabel('complete-call speedup vs strongest same-cell cuSPARSE (>1: BOO faster); one marker per allocation')
    ax.set_title('F=512 confirmation, 3 independent RTX 3090 allocations, randomized order', fontsize=9)
    ax.legend(frameon=False, loc='lower right')
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    fig.savefig(str(out).replace('.png', '.pdf'))


if __name__ == '__main__':
    main(*sys.argv[1:])
