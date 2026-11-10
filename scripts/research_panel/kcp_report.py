"""Report for the k_c x P trade-off experiment (experiments/2026-11-10_kc_p_tradeoff), per its frozen plan.

  python scripts/research_panel/kcp_report.py --raw ANALYSIS/raw_gpu --freeze ANALYSIS/freeze --out ANALYSIS/report

Every ratio is computed inside one allocation and one interleaved cell, then summarized as min / median /
max over allocations (distinct GPUs). Claims need every allocation beyond 1.05 (or below 0.95); the
[0.95, 1.05] band is descriptive only. Every table is written unfiltered and with the control filter
(an (allocation, cell) is dropped when any _dup or _pad0 control leaves [0.95, 1.05]).

  Q1 model.csv / matched.csv predictions against launch_info (arena bytes, CTAs/SM), exact integers.
  Q2 native P curves per (input, F, shape, T): time / fastest P of that shape and T.
  Q3 occupancy-matched P curves at T256 (matched.csv arms): time / fastest matched P, plus each padded
     arm / its own _pad0 (the cost of padding itself).
  Q4 k_c sweeps at k_r = 128 and 256: speedup over the fastest 32x32 arm in the cell, best over P and T,
     and at every fixed P, with the model's X-staged and Y-commit counts relative to 32x32.
"""
import argparse
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'tests' / 'torch_fast_spdmm'), str(ROOT), str(ROOT / 'scripts' / 'research_panel')]
import shape_report as sr  # noqa: E402

INK, INK2, GRID, SURFACE = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'
BLUE, ORANGE = '#2a78d6', '#eb6834'          # categorical slots 1-2
DIVERGING = LinearSegmentedColormap.from_list('slow_fast', ['#e34948', '#f0efec', '#2a78d6'])
PS = (8, 16, 32, 64, 128)


def spread(g, col):
    return pd.Series({f'{col}_min': g[col].min(), f'{col}_median': g[col].median(), f'{col}_max': g[col].max(),
                      'allocs': g.alloc.nunique(), 'gpus': g.gpu_uuid.nunique()})


def verdict(lo, hi):
    return 'faster' if lo > 1.05 else ('slower' if hi < 0.95 else 'band')


def q1(m, model, matched, out):
    key = ['input', 'k', 'p', 'threads']
    mod = model.rename(columns={'shape': 'k'})[key + ['arena', 'ctas', 'regs', 'binding']]
    nat = m[m.suffix.isin(['', 'dup', 'pad0'])].merge(mod, on=key, how='left')
    nat['pred_arena'], nat['pred_cta'] = nat.arena, nat.ctas
    pad = m[m.suffix.str.startswith('cta')].merge(
        matched.rename(columns={'shape': 'k', 'arm': 'tag', 'arena': 'pred_arena', 'target_cta': 'pred_cta'})
        [['cell', 'k', 'tag', 'pred_arena', 'pred_cta']], on=['cell', 'k', 'tag'], how='left')
    j = pd.concat([nat, pad], ignore_index=True)
    j['arena_ok'] = j.arena_bytes == j.pred_arena
    j['cta_ok'] = j.max_active_blocks_per_sm == j.pred_cta
    bad = j[~(j.arena_ok & j.cta_ok)]
    bad.to_csv(out / 'q1_discrepancies.csv', index=False)
    regs = j.groupby(['threads', 'fpt', 'p']).registers_per_thread.agg(['min', 'max']).reset_index()
    print(f'\n== Q1 model vs launch_info: {len(j)} arms, arena exact {int(j.arena_ok.sum())}, '
          f'CTAs/SM exact {int(j.cta_ok.sum())}, discrepancies {len(bad)}; '
          f'local bytes > 0: {int((j.local_bytes_per_thread > 0).sum())}')
    if len(bad):
        print(bad[['alloc', 'cell', 'k', 'tag', 'arena_bytes', 'pred_arena', 'max_active_blocks_per_sm',
                   'pred_cta']].head(30).to_string(index=False))
    print('registers per thread by (T, FPT, P): ' + ', '.join(
        f'{r.threads}/{r.fpt}/{r.p}:{r["min"]}' + (f'-{r["max"]}' if r['max'] != r['min'] else '')
        for _, r in regs.iterrows()))
    return j


def p_curves(m, model, out, label):
    nat = m[m.suffix == ''].copy()
    g = ['alloc', 'input', 'features', 'k', 'threads']
    nat['vs_best_p'] = nat.median_ms / nat.groupby(g).median_ms.transform('min')
    nat['best_p_here'] = nat.groupby(g).median_ms.transform('idxmin') == nat.index
    s = (nat.groupby(['input', 'features', 'k', 'threads', 'p'])
         .apply(lambda x: pd.concat([spread(x, 'vs_best_p'),
                                     pd.Series({'cta': x.max_active_blocks_per_sm.iloc[0],
                                                'best_in_allocs': int(x.best_p_here.sum())})]))
         .reset_index())
    pmax = model[model.feasible].groupby(['input', 'shape', 'threads']).p.max().rename('p_max_launchable')
    s = s.join(pmax, on=['input', 'k', 'threads'])
    s.to_csv(out / f'q2_native_p_{label}.csv', index=False)
    best = s[s.best_in_allocs == s.allocs]
    best = best.assign(best_is_pmax=best.p == best.p_max_launchable)
    if label == 'all':
        print(f'\n== Q2 native P curves: {s.groupby(["input", "features", "k", "threads"]).ngroups} '
              f'(input, F, shape, T) groups; groups with the same best P in every allocation: {len(best)}, '
              f'of which best P = largest launchable P: {int(best.best_is_pmax.sum())}')
        print('best P distribution (consistent groups): ' + str(best.p.value_counts().sort_index().to_dict()))
    return s, nat


def q3(m, matched, out, label):
    keys = ['cell', 'k', 'tag']
    mm = matched.rename(columns={'shape': 'k', 'arm': 'tag', 'target_cta': 'matched_cta',
                                 'pad_nnz': 'matched_pad'})[keys + ['matched_cta', 'matched_pad']]
    x = m.merge(mm, on=keys)
    g = ['alloc', 'cell', 'input', 'features', 'k']
    x['vs_best_matched'] = x.median_ms / x.groupby(g).median_ms.transform('min')
    pad0 = m[m.suffix == 'pad0'].set_index(['alloc', 'cell', 'k', 'base']).median_ms.rename('pad0_ms')
    x = x.join(pad0, on=['alloc', 'cell', 'k', 'base'])
    x['padding_cost'] = x.median_ms / x.pad0_ms
    s = (x.groupby(['input', 'features', 'k', 'p'])
         .apply(lambda y: pd.concat([spread(y, 'vs_best_matched'), spread(y, 'padding_cost').drop(['allocs', 'gpus']),
                                     pd.Series({'target_cta': y.matched_cta.iloc[0], 'padded': y.tag.iloc[0]})]))
         .reset_index())
    s.to_csv(out / f'q3_matched_p_{label}.csv', index=False)
    if label == 'all':
        low = s[s.p.isin([8, 16])]
        print(f'\n== Q3 occupancy-matched (T256): {s.groupby(["input", "features", "k"]).ngroups} groups. '
              f'P8/P16 slower than the best matched P by > 5% in every allocation: '
              f'{int((low.vs_best_matched_min > 1.05).sum())} of {len(low)} (P, group) rows')
        best = s.loc[s.groupby(['input', 'features', 'k']).vs_best_matched_median.idxmin()]
        print('best matched P (median over allocations): ' + str(best.p.value_counts().sort_index().to_dict()))
    return s


def q4(m, model, out, label):
    nat = m[m.suffix == ''].copy()
    ref = nat[nat.k == '32x32'].groupby(['alloc', 'cell']).median_ms.min().rename('ref_ms')
    nat = nat.join(ref, on=['alloc', 'cell'])
    nat['speedup'] = nat.ref_ms / nat.median_ms
    shape_model = model.drop_duplicates(['input', 'shape'])[['input', 'shape', 'kr', 'kc', 'x_staged', 'y_commits']]
    base = shape_model[shape_model['shape'] == '32x32'].set_index('input')
    shape_model = shape_model.assign(
        x_staged_rel=shape_model.x_staged / shape_model.input.map(base.x_staged),
        y_commits_rel=shape_model.y_commits / shape_model.input.map(base.y_commits))
    nat = nat.drop(columns=[c for c in ('kr', 'kc') if c in nat]).merge(
        shape_model.rename(columns={'shape': 'k'}), on=['input', 'k'])
    best = nat.loc[nat.groupby(['alloc', 'cell', 'k']).speedup.idxmax()]
    s = (best.groupby(['input', 'features', 'k', 'kr', 'kc'])
         .apply(lambda x: pd.concat([spread(x, 'speedup'),
                                     pd.Series({'best_arm': '/'.join(sorted(set(x.tag))),
                                                'x_staged_rel': x.x_staged_rel.iloc[0],
                                                'y_commits_rel': x.y_commits_rel.iloc[0]})]))
         .reset_index())
    s['verdict'] = [verdict(lo, hi) for lo, hi in zip(s.speedup_min, s.speedup_max)]
    s.to_csv(out / f'q4_kc_best_{label}.csv', index=False)
    fixed = (nat.groupby(['input', 'features', 'k', 'kr', 'kc', 'p', 'threads'])
             .apply(lambda x: pd.concat([spread(x, 'speedup'),
                                         pd.Series({'cta': x.max_active_blocks_per_sm.iloc[0]})]))
             .reset_index())
    fixed.to_csv(out / f'q4_kc_fixed_p_{label}.csv', index=False)
    if label == 'all':
        show = s[s.kr.isin([128, 256]) | (s.k == '32x32')]
        print('\n== Q4 speedup over the fastest 32x32 arm, best P and T per shape (min / median over allocations)')
        print(show[['input', 'features', 'k', 'speedup_min', 'speedup_median', 'verdict', 'best_arm',
                    'x_staged_rel', 'y_commits_rel']].round(3).to_string(index=False))
    return s, fixed


def plot_grid(fixed, out):
    """Heatmap per (input, F): k_c rows x P columns at k_r = 128, T256; value = median speedup vs 32x32."""
    f = fixed[(fixed.kr == 128) & (fixed.threads == 256)]
    groups = sorted(f.groupby(['input', 'features']).groups)
    kcs = sorted(f.kc.unique())
    ncol = 4
    nrow = -(-len(groups) // ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.6 * ncol, 3.0 * nrow + 0.9), facecolor=SURFACE, squeeze=False)
    norm = TwoSlopeNorm(vmin=0.4, vcenter=1.0, vmax=1.4)
    for ax, (inp, feat) in zip(axes.flat, groups):
        g = f[(f.input == inp) & (f.features == feat)].set_index(['kc', 'p'])
        ax.set_facecolor(SURFACE)
        for i, kc in enumerate(kcs):
            for j, p in enumerate(PS):
                if (kc, p) in g.index:
                    r = g.loc[(kc, p)]
                    v = r.speedup_median
                    ax.add_patch(plt.Rectangle((j, i), 0.96, 0.92, color=DIVERGING(norm(min(max(v, 0.4), 1.4)))))
                    ink = SURFACE if abs(v - 1) > 0.3 else INK
                    ax.text(j + 0.48, i + 0.56, f'{v:.2f}', ha='center', va='center', fontsize=8, color=ink,
                            weight='bold' if r.speedup_min > 1.05 else 'normal')
                    ax.text(j + 0.48, i + 0.24, f'{r.cta:.0f} CTA', ha='center', va='center', fontsize=6.5, color=ink)
                else:
                    ax.add_patch(plt.Rectangle((j, i), 0.96, 0.92, facecolor=SURFACE, edgecolor=GRID, hatch='///', lw=0))
        ax.set_xlim(0, len(PS)); ax.set_ylim(0, len(kcs))
        ax.set_xticks([j + 0.48 for j in range(len(PS))], [f'P{p}' for p in PS], fontsize=8)
        ax.set_yticks([i + 0.46 for i in range(len(kcs))], [f'k_c {k}' for k in kcs], fontsize=8)
        ax.tick_params(length=0, colors=INK2)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_title(f'{inp}, F{feat}', fontsize=9.5, color=INK, loc='left')
    for ax in list(axes.flat)[len(groups):]:
        ax.set_visible(False)
    fig.suptitle('k_r = 128, T = 256: speedup over the fastest 32x32 arm (median of 2 GPUs)', x=0.01, ha='left',
                 fontsize=12, weight='bold', color=INK)
    fig.text(0.01, 1 - 0.55 / (3.0 * nrow + 0.9), 'Blue faster, red slower, gray = parity. Bold: faster on both GPUs '
             'by > 5%. Hatched: cannot launch (shared-memory cap or 16-bit index). Small text: CTAs/SM.',
             fontsize=8.5, color=INK2)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.85 / (3.0 * nrow + 0.9)))
    fig.savefig(out / 'kc_p_grid.png', dpi=160, facecolor=SURFACE)


def plot_matched(native, matched_s, out, shape='128x64'):
    """Native vs occupancy-matched P curves at one shape, T256, per (input, F)."""
    nat = native[(native.k == shape) & (native.threads == 256)]
    mat = matched_s[matched_s.k == shape]
    groups = sorted(set(map(tuple, mat[['input', 'features']].values)))
    ncol = 4
    nrow = -(-len(groups) // ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.6 * ncol, 2.9 * nrow + 0.9), facecolor=SURFACE,
                             sharey=True, squeeze=False)
    for ax, (inp, feat) in zip(axes.flat, groups):
        ax.set_facecolor(SURFACE)
        a = nat[(nat.input == inp) & (nat.features == feat)].sort_values('p')
        b = mat[(mat.input == inp) & (mat.features == feat)].sort_values('p')
        xa = [PS.index(p) for p in a.p]
        xb = [PS.index(p) for p in b.p]
        ax.axhline(1.0, color=INK2, lw=1, ls=(0, (4, 3)))
        ax.plot(xa, a.vs_best_p_median, color=BLUE, lw=2, marker='o', ms=6, mec=SURFACE, mew=1.5)
        ax.plot(xb, b.vs_best_matched_median, color=ORANGE, lw=2, marker='s', ms=6, mec=SURFACE, mew=1.5)
        for x, r in zip(xa, a.itertuples()):
            ax.annotate(f'{r.cta:.0f}', (x, r.vs_best_p_median), textcoords='offset points', xytext=(0, 6),
                        ha='center', fontsize=7.5, color=BLUE)
        if len(b):
            ax.text(0.98, 0.95, f'matched: {b.target_cta.iloc[0]:.0f} CTAs/SM', transform=ax.transAxes, ha='right',
                    va='top', fontsize=8, color=INK2)
        ax.set_xticks(range(len(PS)), [f'P{p}' for p in PS], fontsize=8)
        ax.set_title(f'{inp}, F{feat}', fontsize=9.5, color=INK, loc='left')
        ax.grid(axis='y', color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
    for ax in list(axes.flat)[len(groups):]:
        ax.set_visible(False)
    for ax in axes[:, 0]:
        ax.set_ylabel('time / fastest P (median)')
    fig.suptitle(f'Block {shape}, T = 256: is low P slow because of occupancy?', x=0.01, ha='left', fontsize=12,
                 weight='bold', color=INK)
    fig.text(0.01, 1 - 0.55 / (2.9 * nrow + 0.9), 'Blue: native occupancy (numbers = CTAs/SM). Orange: every P padded '
             'to the same CTAs/SM. If orange stays high at low P, the cost is per-pass work, not occupancy.',
             fontsize=8.5, color=INK2)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.85 / (2.9 * nrow + 0.9)))
    fig.savefig(out / 'p_matched_occupancy.png', dpi=160, facecolor=SURFACE)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--raw', type=Path, required=True)
    ap.add_argument('--freeze', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    pd.set_option('display.width', 250, 'display.max_columns', 40, 'display.max_rows', 400)
    model = pd.read_csv(a.freeze / 'model.csv')
    matched = pd.read_csv(a.freeze / 'matched.csv')
    d = sr.load(a.raw / 'results', 'kcp')
    acc = sr.arm_tag(sr.accept(d, a.raw / 'launch_audit', 'kcp', a.out))
    filt = sr.valid_cells(acc, 'kcp', a.out)
    q1(acc, model, matched, a.out)
    for label, m in (('all', acc), ('filtered', filt)):
        native, nat_rows = p_curves(m, model, a.out, label)
        mat = q3(m, matched, a.out, label)
        best, fixed = q4(m, model, a.out, label)
        if label == 'all':
            plot_grid(fixed, a.out)
            plot_matched(native, mat, a.out)
    print(f'\nfiltered: {filt.groupby(["alloc", "cell"]).ngroups} of {acc.groupby(["alloc", "cell"]).ngroups} '
          f'(allocation, cell) kept -> {a.out}')


if __name__ == '__main__':
    main()
