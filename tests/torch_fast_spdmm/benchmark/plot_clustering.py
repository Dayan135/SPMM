"""Plot the clustering-backend sweep (identity / HEM / METIS / Graclus) from bench_clustering CSVs.

Headline metric is SpMM (matmul) time; preprocessing only appears in the break-even figure.

Usage (from tests/torch_fast_spdmm/, local ``fem`` env)::

    conda run -n fem python -m benchmark.plot_clustering \
        --results benchmark/results/clustering_20261006/results \
        --outdir ../../experiments/2026-10-06_clustering/plots/initial

    # all four sizes (16k, 64k from the first sweep; 256k, 1M from the offline-clustered sweep)
    conda run -n fem python -m benchmark.plot_clustering \
        --results benchmark/results/clustering_20261006/results benchmark/results/clustering_20261006_xl \
        --outdir ../../experiments/2026-10-06_clustering/plots/xl
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

BACKENDS = ('identity', 'hem', 'metis', 'graclus')
LABEL = {'identity': 'identity', 'hem': 'HEM', 'metis': 'METIS', 'graclus': 'Graclus (greedy)'}
# Fixed categorical order (validated: scripts/validate_palette.js, light surface). Two slots are
# below 3:1 contrast, so every series also carries a distinct marker and a legend.
COLOR = {'identity': '#2a78d6', 'hem': '#eb6834', 'metis': '#1baf7a', 'graclus': '#eda100'}
MARKER = {'identity': 'o', 'hem': 's', 'metis': '^', 'graclus': 'D'}
SURFACE, INK, INK2, GRID, BASE = '#fcfcfb', '#1f1e1c', '#52514e', '#e6e5e0', '#8a8984'
PROFILES = ('fem2d', 'fem2d_permuted', 'diag_b0.3', 'diag_b1.0')
TARGETS = (8, 16, 32, 64, 128, 256)
SHORT = {'fem2d': 'fem2d', 'fem2d_permuted': 'fem2d perm', 'diag_b0.3': 'diag 0.3', 'diag_b1.0': 'diag 1.0'}

plt.rcParams.update({
    'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
    'axes.edgecolor': GRID, 'axes.labelcolor': INK2, 'xtick.color': INK2, 'ytick.color': INK2,
    'text.color': INK, 'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6,
    'axes.spines.top': False, 'axes.spines.right': False, 'font.size': 9,
    'axes.titlesize': 10, 'legend.frameon': False, 'lines.linewidth': 1.5, 'lines.markersize': 6,
})


def sweep_dirs(results: list[Path]) -> list[Path]:
    """Each --results entry is a sweep dir (has summary.csv) or a parent of sweep dirs; smoke runs skipped."""
    dirs = []
    for r in results:
        if (r / 'summary.csv').exists():
            dirs.append(r)
        else:
            dirs += sorted(p.parent for p in r.glob('*/summary.csv') if p.parent.name != 'smoke')
    if not dirs:
        raise SystemExit(f'no summary.csv under {results}')
    return dirs


def load(results: list[Path]) -> pd.DataFrame:
    df = pd.concat([pd.read_csv(d / 'summary.csv') for d in sweep_dirs(results)], ignore_index=True)
    df['N'] = df['requested_size'].astype(int)
    df['features'] = df['features'].astype(int)
    if df.duplicated(['N', 'profile', 'features', 'config']).any():
        raise SystemExit('duplicate (N, profile, F, config) rows across --results dirs')
    return df


def load_samples(results: list[Path]) -> pd.DataFrame:
    return pd.concat([pd.read_csv(d / 'samples.csv') for d in sweep_dirs(results)], ignore_index=True)


def size_tag(n: int) -> str:
    return f'{n // 2**20}M' if n % 2**20 == 0 else f'{n // 1024}k'


def size_panels(Ns, ylabel):
    """One panel per N, at most two per row; ylabel on the left column. Returns fig, [(ax, N)]."""
    ncols = min(2, len(Ns))
    nrows = -(-len(Ns) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.5 * ncols, 3.8 * nrows), squeeze=False,
                             sharey=True)
    for ax in axes.flat[len(Ns):]:
        ax.set_visible(False)
    for row in axes:
        row[0].set_ylabel(ylabel)
    return fig, list(zip(axes.flat, Ns))


def boo(df):
    return df[df.algorithm == 'boo']


def cusparse_ms(df):
    c = df[df.algorithm == 'cusparse_csr']
    return c.set_index(['N', 'profile', 'features'])['median_ms']


def legend_handles(include_cusparse=True, include_unsup=False):
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color=COLOR[b], marker=MARKER[b], label=LABEL[b]) for b in BACKENDS]
    if include_cusparse:
        h.append(Line2D([], [], color=BASE, ls='--', label='cuSPARSE'))
    if include_unsup:
        h.append(Line2D([], [], color=INK2, marker='x', ls='', label='unsupported (smem)'))
    return h


def fig_time_vs_target(df, N, out):
    """A1: median matmul time vs target group size, one panel per profile x F."""
    d, cs = boo(df[df.N == N]), cusparse_ms(df)
    feats = sorted(d.features.unique())
    fig, axes = plt.subplots(len(feats), len(PROFILES), figsize=(13, 3.1 * len(feats)), squeeze=False)
    for i, F in enumerate(feats):
        for j, prof in enumerate(PROFILES):
            ax = axes[i, j]
            cell = d[(d.profile == prof) & (d.features == F)]
            for b in BACKENDS:
                s = cell[cell.cluster == b].sort_values('k')
                ok = s[s.supported == True]  # noqa: E712
                ax.fill_between(ok.k, ok.p25_ms, ok.p75_ms, color=COLOR[b], alpha=0.15, lw=0)
                ax.plot(ok.k, ok.median_ms, color=COLOR[b], marker=MARKER[b], label=LABEL[b])
                bad = s[s.supported == False]  # noqa: E712
                if len(bad):
                    top = cell.median_ms.max()
                    ax.plot(bad.k, [top * 1.3] * len(bad), 'x', color=COLOR[b], ms=8, mew=2)
            ax.axhline(cs[(N, prof, F)], color=BASE, ls='--', lw=1.2)
            ax.set_xscale('log', base=2)
            ax.set_yscale('log')
            ax.set_xticks(TARGETS, [str(t) for t in TARGETS])
            ax.set_title(f'{prof}  ·  F={F}')
            if i == len(feats) - 1:
                ax.set_xlabel('target group size')
            if j == 0:
                ax.set_ylabel('median matmul time (ms, log)')
    fig.legend(handles=legend_handles(include_unsup=True), loc='upper center', ncol=6,
               bbox_to_anchor=(0.5, 1.0))
    fig.suptitle(f'Matmul time vs target group size  —  N={N:,}  (band = p25–p75)', y=1.04)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)


def best_per_backend(df):
    """Best supported config (min median_ms over k) per N x profile x F x backend."""
    d = boo(df)
    d = d[d.supported == True]  # noqa: E712
    idx = d.groupby(['N', 'profile', 'features', 'cluster'])['median_ms'].idxmin()
    best = d.loc[idx].copy()
    cs = cusparse_ms(df)
    best['cusparse_ms'] = [cs[(r.N, r.profile, r.features)] for r in best.itertuples()]
    best['speedup'] = best.cusparse_ms / best.median_ms
    return best


def fig_best_speedup(best, out):
    """A2: best-k speedup vs cuSPARSE, grouped bars per profile x F, one panel per N."""
    fig, panels = size_panels(sorted(best.N.unique()), 'speedup vs cuSPARSE at best target (×)')
    for ax, N in panels:
        b = best[best.N == N]
        cells = [(p, F) for p in PROFILES for F in sorted(b.features.unique())]
        x = np.arange(len(cells))
        w = 0.8 / len(BACKENDS)
        for i, be in enumerate(BACKENDS):
            vals = [b[(b.profile == p) & (b.features == F) & (b.cluster == be)].speedup.squeeze()
                    for p, F in cells]
            vals = [v if np.isscalar(v) else np.nan for v in vals]
            ax.bar(x + (i - 1.5) * w, vals, w * 0.9, color=COLOR[be], label=LABEL[be])
        ax.axhline(1.0, color=BASE, ls='--', lw=1.2)
        ax.set_xticks(x, [f'{SHORT[p]}\nF={F}' for p, F in cells], fontsize=7.5)
        ax.set_title(f'N={N:,}')
        ax.grid(axis='x', visible=False)
    fig.legend(handles=legend_handles(include_cusparse=False), loc='upper center', ncol=4,
               bbox_to_anchor=(0.5, 1.04))
    fig.suptitle('Best-config matmul speedup over cuSPARSE (dashed = parity)', y=1.1)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)


def fig_heatmap(df, out):
    """A3: time relative to best config in each column (1.0 = best), backend x target rows."""
    d = boo(df)
    d = d[d.supported == True].copy()  # noqa: E712
    d['col'] = d.profile.map(SHORT) + '\nF=' + d.features.astype(str) + '\nN=' + d.N.map(size_tag)
    d['row'] = d.cluster.map(LABEL) + '  k=' + d.k.astype(int).astype(str)
    d['rel'] = d.median_ms / d.groupby('col').median_ms.transform('min')
    rows = [f'{LABEL[b]}  k={t}' for b in BACKENDS for t in TARGETS]
    key = d.drop_duplicates('col').set_index('col')
    cols = sorted(key.index, key=lambda c: (key.N[c], PROFILES.index(key.profile[c]), key.features[c]))
    m = d.pivot_table(index='row', columns='col', values='rel').reindex(index=rows, columns=cols)
    fig, ax = plt.subplots(figsize=(1.0 * len(cols) + 2.5, 0.32 * len(rows) + 1.5))
    from matplotlib.colors import LinearSegmentedColormap, LogNorm
    cmap = LinearSegmentedColormap.from_list('seq', ['#cde2fb', '#3987e5', '#0d366b'])
    cmap.set_bad('#eeeeea')
    im = ax.imshow(m.values, cmap=cmap, norm=LogNorm(1, 4, clip=True), aspect='auto')
    for (r, c), v in np.ndenumerate(m.values):
        if np.isnan(v):
            ax.text(c, r, 'n/a', ha='center', va='center', fontsize=6, color=INK2)
        elif v < 1.10:  # selective labels: only near-best cells
            ax.text(c, r, f'{v:.2f}', ha='center', va='center', fontsize=6.5,
                    color=INK if v < 1.6 else 'white', fontweight='bold' if v == 1 else None)
    ax.set_xticks(range(len(cols)), cols, fontsize=6.5)
    ax.set_yticks(range(len(rows)), rows, fontsize=7)
    for y in range(len(TARGETS), len(rows), len(TARGETS)):
        ax.axhline(y - 0.5, color=SURFACE, lw=2)
    sizes = [key.N[c] for c in cols]
    for x in range(1, len(cols)):
        if sizes[x] != sizes[x - 1]:
            ax.axvline(x - 0.5, color=SURFACE, lw=4)
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, shrink=0.6)
    cb.set_label('time / best time in column (log, saturates at 4×)')
    ax.set_title('Matmul time relative to the best config per (profile, F, N); labels only ≤1.10×')
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)


def fig_group_sizes(df, N, out):
    """B4: requested target vs actual mean/max row- and col-group sizes (clustering is F-independent)."""
    d = boo(df[(df.N == N)])
    d = d[d.features == d.features.min()]
    fig, axes = plt.subplots(2, len(PROFILES), figsize=(13, 6), squeeze=False, sharex=True)
    for j, prof in enumerate(PROFILES):
        for i, (kind, lab) in enumerate((('row', 'rows'), ('col', 'cols'))):
            ax = axes[i, j]
            ax.plot(TARGETS, TARGETS, color=BASE, ls=':', lw=1, label='y = target')
            for b in BACKENDS:
                s = d[(d.profile == prof) & (d.cluster == b)].sort_values('k')
                ax.plot(s.k, s[f'{kind}_group_mean'], color=COLOR[b], marker=MARKER[b])
                ax.plot(s.k, s[f'{kind}_group_max'], color=COLOR[b], ls='--', lw=1, alpha=0.7)
            ax.set_xscale('log', base=2)
            ax.set_yscale('log', base=2)
            ax.set_xticks(TARGETS, [str(t) for t in TARGETS])
            if i == 0:
                ax.set_title(prof)
            else:
                ax.set_xlabel('target group size')
            if j == 0:
                ax.set_ylabel(f'actual group size ({lab})')
    from matplotlib.lines import Line2D
    h = legend_handles(include_cusparse=False) + [
        Line2D([], [], color=INK2, label='solid = mean'),
        Line2D([], [], color=INK2, ls='--', lw=1, label='dashed = max'),
        Line2D([], [], color=BASE, ls=':', label='y = target')]
    fig.legend(handles=h, loc='upper center', ncol=7, bbox_to_anchor=(0.5, 1.03))
    fig.suptitle(f'Requested vs actual group size — N={N:,}', y=1.07)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)


def fig_tile(df, N, out):
    """B6: feature tile p selected by the smem fit (64 > 32 > 16 > 8), with time overlaid as label."""
    d = boo(df[df.N == N])
    feats = sorted(d.features.unique())
    fig, axes = plt.subplots(len(feats), len(PROFILES), figsize=(13, 2.6 * len(feats)), squeeze=False,
                             sharey=True)
    offs = {b: (i - 1.5) * 0.09 for i, b in enumerate(BACKENDS)}
    for i, F in enumerate(feats):
        for j, prof in enumerate(PROFILES):
            ax = axes[i, j]
            cell = d[(d.profile == prof) & (d.features == F)]
            for b in BACKENDS:
                s = cell[cell.cluster == b].sort_values('k')
                p = s.p.where(s.supported == True, 4)  # noqa: E712  4 => unsupported row
                ax.plot(np.log2(s.k) + offs[b], np.log2(p), color=COLOR[b], marker=MARKER[b], lw=1)
            ax.set_xticks(np.log2(TARGETS), [str(t) for t in TARGETS])
            ax.set_yticks([2, 3, 4, 5, 6], ['unsup.', '8', '16', '32', '64'])
            ax.set_title(f'{prof}  ·  F={F}')
            if i == len(feats) - 1:
                ax.set_xlabel('target group size')
            if j == 0:
                ax.set_ylabel('feature tile p')
    fig.legend(handles=legend_handles(include_cusparse=False), loc='upper center', ncol=4,
               bbox_to_anchor=(0.5, 1.04))
    fig.suptitle(f'Feature tile chosen by shared-memory fit — N={N:,}', y=1.09)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)


def fig_structure(df, N, out):
    """B5: structural metrics vs matmul time (normalised to cuSPARSE so profiles share an axis)."""
    d = boo(df[df.N == N])
    d = d[d.supported == True].copy()  # noqa: E712
    cs = cusparse_ms(df)
    d['rel'] = [r.median_ms / cs[(r.N, r.profile, r.features)] for r in d.itertuples()]
    d['remainder_frac'] = d.remainder_nnz / d.nnz
    d['x_amp'] = d.x_loads / d.N
    metrics = [('remainder_frac', 'remainder nnz / nnz'), ('max_nnz_blk', 'max nnz per block (log)'),
               ('num_blocks', 'number of blocks (log)'), ('x_amp', 'X-row loads / N (log)'),
               ('arena_bytes', 'smem arena bytes (log)')]
    fig, axes = plt.subplots(1, len(metrics), figsize=(3.0 * len(metrics), 3.2), sharey=True)
    for ax, (col, lab) in zip(axes, metrics):
        for b in BACKENDS:
            s = d[d.cluster == b]
            ax.scatter(s[col], s.rel, s=18, color=COLOR[b], marker=MARKER[b], edgecolors=SURFACE,
                       linewidths=0.6, alpha=0.85)
        if col in ('max_nnz_blk', 'num_blocks', 'x_amp', 'arena_bytes'):
            ax.set_xscale('log')
        ax.set_xlabel(lab)
        ax.axhline(1, color=BASE, ls='--', lw=1)
    axes[0].set_yscale('log')
    axes[0].set_ylabel('time / cuSPARSE time (log)')
    fig.legend(handles=legend_handles(include_cusparse=False), loc='upper center', ncol=4,
               bbox_to_anchor=(0.5, 1.06))
    fig.suptitle(f'What predicts matmul time? (all profiles, F, targets) — N={N:,}', y=1.12)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)


def fig_scaling(best, out):
    """C7: best-k speedup at N=16k vs N=65k, one slope per backend per (profile, F)."""
    Ns = sorted(best.N.unique())
    if len(Ns) < 2:
        return
    feats = sorted(best.features.unique())
    fig, axes = plt.subplots(1, len(PROFILES), figsize=(13, 3.2), sharey=True)
    for ax, prof in zip(axes, PROFILES):
        for b in BACKENDS:
            for F, ls in zip(feats, ('-', '--')):
                s = best[(best.profile == prof) & (best.cluster == b) & (best.features == F)].sort_values('N')
                ax.plot(range(len(s)), s.speedup, color=COLOR[b], marker=MARKER[b], ls=ls)
        ax.axhline(1, color=BASE, ls=':', lw=1)
        ax.set_xticks(range(len(Ns)), [size_tag(n) for n in Ns])
        ax.set_xlim(-0.3, len(Ns) - 0.7)
        ax.set_title(prof)
        ax.set_xlabel('N')
    axes[0].set_ylabel('best speedup vs cuSPARSE (×)')
    from matplotlib.lines import Line2D
    h = legend_handles(include_cusparse=False) + [
        Line2D([], [], color=INK2, ls=ls, label=f'F={F}') for F, ls in zip(feats, ('-', '--'))]
    fig.legend(handles=h, loc='upper center', ncol=6, bbox_to_anchor=(0.5, 1.07))
    fig.suptitle('Does the ranking hold as N grows?', y=1.13)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)


def fig_distributions(df, samples, N, out, top=3):
    """C8: per-sample timing strips for the top configs (plus cuSPARSE) per profile x F."""
    d = boo(df[df.N == N])
    d = d[d.supported == True]  # noqa: E712
    s = samples[samples.requested_size == N]
    feats = sorted(d.features.unique())
    fig, axes = plt.subplots(len(feats), len(PROFILES), figsize=(13, 3.0 * len(feats)), squeeze=False)
    rng = np.random.default_rng(0)
    for i, F in enumerate(feats):
        for j, prof in enumerate(PROFILES):
            ax = axes[i, j]
            cell = d[(d.profile == prof) & (d.features == F)].nsmallest(top, 'median_ms')
            configs = list(cell.config) + ['cusparse_csr']
            for x, cfg in enumerate(configs):
                t = s[(s.profile == prof) & (s.features == F) & (s.config == cfg)].time_ms.values
                be = cfg.split('_k')[0]
                col = COLOR.get(be, BASE)
                ax.scatter(x + rng.uniform(-0.18, 0.18, len(t)), t, s=5, color=col, alpha=0.35,
                           marker=MARKER.get(be, 'o'), linewidths=0)
                ax.hlines(np.median(t), x - 0.3, x + 0.3, color=INK, lw=1.5)
            ax.set_xticks(range(len(configs)),
                          [c.replace('cusparse_csr', 'cuSPARSE').replace('_k', '\nk=') for c in configs],
                          fontsize=7)
            ax.set_yscale('log')
            ax.set_title(f'{prof}  ·  F={F}')
            ax.grid(axis='x', visible=False)
            if j == 0:
                ax.set_ylabel('per-sample time (ms, log)')
    fig.suptitle(f'Timing distributions, top-{top} configs + cuSPARSE (bar = median) — N={N:,}', y=1.01)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)


def fig_breakeven(best, out):
    """D9: matmul calls needed to amortise preprocessing vs cuSPARSE (best-k config per backend)."""
    b = best.copy()
    b['prep_ms'] = b.cluster_ms.fillna(0) + b.build_ms.fillna(0)
    b['saving_ms'] = b.cusparse_ms - b.median_ms
    b['calls'] = np.where(b.saving_ms > 0, b.prep_ms / b.saving_ms, np.nan)
    fig, panels = size_panels(sorted(b.N.unique()), 'matmuls to break even vs cuSPARSE (log)')
    for ax, N in panels:
        bb = b[b.N == N]
        cells = [(p, F) for p in PROFILES for F in sorted(bb.features.unique())]
        x = np.arange(len(cells))
        w = 0.8 / len(BACKENDS)
        for i, be in enumerate(BACKENDS):
            for xi, (p, F) in zip(x, cells):
                r = bb[(bb.profile == p) & (bb.features == F) & (bb.cluster == be)]
                xpos = xi + (i - 1.5) * w
                if len(r) and np.isfinite(r.calls.iloc[0]):
                    ax.bar(xpos, r.calls.iloc[0], w * 0.9, color=COLOR[be])
                else:
                    ax.text(xpos, 0.01, '∞', ha='center', va='bottom', color=COLOR[be], fontsize=9,
                            transform=ax.get_xaxis_transform())
        ax.set_yscale('log')
        ax.set_xticks(x, [f'{SHORT[p]}\nF={F}' for p, F in cells], fontsize=7.5)
        ax.set_title(f'N={N:,}')
        ax.grid(axis='x', visible=False)
    fig.legend(handles=legend_handles(include_cusparse=False), loc='upper center', ncol=4,
               bbox_to_anchor=(0.5, 1.04))
    fig.suptitle('Preprocessing amortisation (cluster + build) / (cuSPARSE − BOO per call);'
                 ' ∞ = never faster', y=1.1)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)


def write_tables(df, best, outdir):
    cols = ['N', 'profile', 'features', 'cluster', 'k', 'p', 'median_ms', 'cusparse_ms', 'speedup',
            'row_group_mean', 'cluster_ms', 'build_ms']
    t = best[cols].sort_values(['N', 'profile', 'features', 'median_ms'])
    t.to_csv(outdir / 'best_per_backend.csv', index=False)
    with open(outdir / 'best_per_backend.md', 'w') as f:
        f.write('| ' + ' | '.join(cols) + ' |\n|' + '---|' * len(cols) + '\n')
        for r in t.itertuples(index=False):
            f.write('| ' + ' | '.join(f'{v:.3f}' if isinstance(v, float) else str(v) for v in r)
                    + ' |\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', type=Path, nargs='+',
                    default=[Path('benchmark/results/clustering_20261006/results')],
                    help='sweep dirs (with summary.csv) or parents of sweep dirs')
    ap.add_argument('--outdir', type=Path, default=Path('../../experiments/2026-10-06_clustering/plots/initial'))
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    df = load(args.results)
    samples = load_samples(args.results)
    best = best_per_backend(df)
    o = args.outdir
    for N in sorted(df.N.unique()):
        tag = size_tag(N)
        fig_time_vs_target(df, N, o / f'A1_time_vs_target_{tag}.png')
        fig_group_sizes(df, N, o / f'B4_group_sizes_{tag}.png')
        fig_structure(df, N, o / f'B5_structure_vs_time_{tag}.png')
        fig_tile(df, N, o / f'B6_tile_p_{tag}.png')
        fig_distributions(df, samples, N, o / f'C8_distributions_{tag}.png')
    fig_best_speedup(best, o / 'A2_best_speedup.png')
    fig_heatmap(df, o / 'A3_relative_heatmap.png')
    fig_scaling(best, o / 'C7_scaling.png')
    fig_breakeven(best, o / 'D9_breakeven.png')
    write_tables(df, best, o)
    print(f'wrote {len(list(o.glob("*.png")))} figures + tables to {o}')


if __name__ == '__main__':
    main()
