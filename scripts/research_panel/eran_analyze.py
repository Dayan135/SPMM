"""Audit and plot the requested Eran FPT grid from a completed aggregate CSV.

This script is safe to rerun while jobs are active: absent cells stay explicitly
missing. Figures and claims must be treated as final only after the status audit.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import median


FPT = (1, 2, 4)
METHODS = ('identity', 'hem', 'graclus', 'metis')


def read_csv(path):
    with path.open(newline='') as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows, fields):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def input_key(task):
    return task['matrix'] if task['kind'] == 'real' else (
        f"generated/{task['profile']}_n{task['size']}")


def boo_name(fpt):
    return 'boo' if fpt == 1 else f'boo_fpt{fpt}'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path, help='run root containing sweep.json and aggregate/')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((args.root / 'sweep.json').read_text())
    rows = read_csv(args.root / 'aggregate' / 'summary.csv')
    status = json.loads((args.root / 'aggregate' / 'status.json').read_text())
    by_cell = defaultdict(list)
    for row in rows:
        by_cell[(int(row['task_index']), row['matrix'], int(row['features']),
                 row['labeling'])].append(row)

    coverage = []
    analytic = []
    for index, task in enumerate(manifest['tasks']):
        matrix = input_key(task)
        method = task['method']
        for k in manifest['groups']:
            for features in manifest['features']:
                labeling = f'{method}_k{k}'
                entries = by_cell[(index, matrix, features, labeling)]
                duplicate = next((r['duplicate_of'] for r in entries
                                  if r['duplicate_of']), '')
                if duplicate:
                    original = by_cell[(index, matrix, features, duplicate)]
                else:
                    original = entries
                for fpt in manifest['fpt']:
                    name = boo_name(fpt)
                    item = next((r for r in original if r['config'] == name), None)
                    state = ('duplicate' if item and duplicate else
                             'measured' if item and item['supported'] == 'True' and
                             item['correctness'] == 'passed' and item['median_ms'] else
                             'unsupported' if item and item['supported'] == 'False' else
                             'missing')
                    reason = (f'identical labels to {duplicate}' if state == 'duplicate' else
                              item['unsupported_reason'] if state == 'unsupported' else
                              status[index].get('error') or status[index]['status']
                              if state == 'missing' else '')
                    coverage.append(dict(task_index=index, matrix=matrix, method=method,
                                         k=k, features=features, fpt=fpt, status=state,
                                         alias_of=duplicate, reason=reason,
                                         label_sha256=item['label_sha256'] if item else '',
                                         median_ms=item['median_ms'] if item and
                                         state in ('measured', 'duplicate') else ''))
                    if item and state in ('measured', 'duplicate'):
                        analytic.append(dict(item, requested_k=k, requested_labeling=labeling,
                                             alias=bool(duplicate), time=float(item['median_ms'])))
    assert len(coverage) == manifest['expected_boo_full_call_combinations']
    write_csv(args.out / 'coverage.csv', coverage, list(coverage[0]))
    counts = {state: sum(r['status'] == state for r in coverage)
              for state in ('measured', 'duplicate', 'unsupported', 'missing')}

    grouped = defaultdict(dict)
    for row in analytic:
        key = (row['matrix'], int(row['features']), row['cluster'], int(row['requested_k']))
        grouped[key][int(row['features_per_thread'])] = row
    gains = []
    for key, variants in grouped.items():
        if 1 not in variants:
            continue
        for fpt in (2, 4):
            if fpt not in variants:
                continue
            base, other = variants[1], variants[fpt]
            gains.append(dict(matrix=key[0], features=key[1], method=key[2],
                              k=key[3], fpt=fpt, fpt1_ms=base['time'],
                              other_ms=other['time'], gain=base['time']/other['time'],
                              alias=other['alias'], task_index=other['task_index']))
    write_csv(args.out / 'paired_fpt.csv', gains,
              list(gains[0]) if gains else ['matrix', 'features', 'method', 'k', 'fpt', 'gain'])

    direct = defaultdict(list)
    for row in rows:
        if (row['algorithm'] == 'cusparse_direct' and row['scope'] == 'full_call'
                and row['config'].startswith('direct_') and
                not row['config'].startswith('direct_perm_') and row['median_ms']):
            direct[(int(row['task_index']), row['matrix'], int(row['features']),
                    row['labeling'])].append(row)
    winners = []
    for matrix, features in sorted({(r['matrix'], int(r['features'])) for r in analytic}):
        candidates = [r for r in analytic if r['matrix'] == matrix and
                      int(r['features']) == features and not r['alias']]
        if not candidates:
            continue
        best = min(candidates, key=lambda r: r['time'])
        dkey = (int(best['task_index']), matrix, features, best['labeling'])
        baselines = direct[dkey]
        chosen_direct = min(baselines, key=lambda r: float(r['median_ms'])) if baselines else None
        direct_ms = float(chosen_direct['median_ms']) if chosen_direct else None
        boo_setup = sum(float(best.get(field) or 0) for field in
                        ('cluster_ms', 'boo_build_ms', 'boo_staging_ms'))
        direct_setup = float(chosen_direct.get('plan_build_ms') or 0) if chosen_direct else 0
        incremental_setup = boo_setup - direct_setup
        break_even = (math.ceil(incremental_setup / (direct_ms - best['time']))
                      if direct_ms and direct_ms > best['time'] and
                      incremental_setup > 0 else '')
        winners.append(dict(matrix=matrix, features=features, method=best['cluster'],
                            k=best['requested_k'], fpt=best['features_per_thread'],
                            boo_ms=best['time'], direct_ms=direct_ms or '',
                            speedup=direct_ms/best['time'] if direct_ms else '',
                            direct_algorithms=len({r['direct_algorithm'] for r in baselines}),
                            task_index=best['task_index'], label_sha256=best['label_sha256'],
                            boo_setup_ms_approx=boo_setup,
                            direct_plan_ms=direct_setup,
                            incremental_setup_ms_approx=incremental_setup,
                            break_even_calls_approx=break_even))
    write_csv(args.out / 'screened_winners.csv', winners,
              list(winners[0]) if winners else ['matrix', 'features', 'method', 'k', 'fpt'])

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import TwoSlopeNorm

    plt.rcParams.update({'font.size': 9, 'axes.spines.top': False,
                         'axes.spines.right': False, 'savefig.dpi': 180})
    def save(fig, name):
        fig.savefig(args.out / f'{name}.png', bbox_inches='tight')
        fig.savefig(args.out / f'{name}.pdf', bbox_inches='tight')
        plt.close(fig)

    # Paired FPT effects preserve method/k/F, excluding aliased labels from counts.
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, real in zip(axes, (True, False)):
        labels, data = [], []
        for features in manifest['features']:
            for fpt in (2, 4):
                values = [r['gain'] for r in gains if r['features'] == features and
                          r['fpt'] == fpt and not r['alias'] and
                          r['matrix'].startswith('generated/') != real]
                if values:
                    labels.append(f'F{features}\nFPT{fpt}')
                    data.append(values)
        if data:
            ax.boxplot(data, tick_labels=labels, showfliers=True)
        ax.axhline(1, color='black', linestyle='--', linewidth=1)
        ax.set_yscale('log', base=2)
        ax.set_title('Real matrices' if real else 'Synthetic matrices')
        ax.set_ylabel('FPT1 time / variant time')
    fig.tight_layout()
    save(fig, 'paired_fpt')

    # Same-cell direct baseline of each post-hoc screening winner.
    for family in ('real', 'synthetic'):
        family_rows = [r for r in winners if r['matrix'].startswith('generated/') ==
                       (family == 'synthetic')]
        matrices = sorted({r['matrix'] for r in family_rows})
        fs = manifest['features']
        if not matrices:
            continue
        heat = np.full((len(matrices), len(fs)), np.nan)
        labels = {}
        for i, matrix in enumerate(matrices):
            for j, features in enumerate(fs):
                row = next((r for r in family_rows if r['matrix'] == matrix and
                            r['features'] == features), None)
                if row and row['speedup']:
                    heat[i, j] = math.log2(row['speedup'])
                    labels[(i, j)] = f"{row['speedup']:.2f}" + (
                        '*' if row['direct_algorithms'] < 4 else '')
        fig, ax = plt.subplots(figsize=(6.5, max(3, len(matrices) * .35)))
        scale = max(1., float(np.nanmax(np.abs(heat)))) if np.isfinite(heat).any() else 1.
        im = ax.imshow(heat, cmap='RdYlGn', norm=TwoSlopeNorm(vmin=-scale,
                       vcenter=0, vmax=scale), aspect='auto')
        ax.set_xticks(range(len(fs)), fs)
        ax.set_yticks(range(len(matrices)), [m.replace('generated/', '') for m in matrices])
        ax.set_xlabel('Dense feature width F')
        for (i, j), value in labels.items():
            ax.text(j, i, value, ha='center', va='center', fontsize=8)
        ax.set_title('Best screened BOO / same-cell direct cuSPARSE\n'
                     'Speedup above 1 favors BOO; * fewer than four direct algorithms')
        fig.colorbar(im, ax=ax, label='log₂(direct time / BOO time)')
        save(fig, f'speedup_{family}')

    # At fixed FPT4, show all methods and requested targets at each feature width.
    for family in ('real', 'synthetic'):
        matrices = sorted({r['matrix'] for r in analytic if
                           r['matrix'].startswith('generated/') ==
                           (family == 'synthetic')})
        if not matrices:
            continue
        for features in manifest['features']:
            for tuned in (False, True):
                if tuned and features != 512:
                    continue
                columns = 2
                fig, axes = plt.subplots(math.ceil(len(matrices)/columns), columns,
                    figsize=(11, max(4, math.ceil(len(matrices)/columns)*2.6)),
                    squeeze=False)
                for ax, matrix in zip(axes.flat, matrices):
                    for method in METHODS:
                        selected = [r for r in analytic if r['matrix'] == matrix and
                                    int(r['features']) == features and
                                    (tuned or int(r['features_per_thread']) == 4) and
                                    r['cluster'] == method]
                        by_k = {}
                        for row in selected:
                            old = by_k.get(row['requested_k'])
                            if old is None or row['time'] < old['time']:
                                by_k[row['requested_k']] = row
                        ks = sorted(by_k)
                        if ks:
                            ax.plot(ks, [by_k[k]['time'] for k in ks], marker='o',
                                    label=method)
                            for k in ks:
                                if by_k[k]['alias']:
                                    ax.plot(k, by_k[k]['time'], marker='o',
                                            color='white', markeredgecolor='black',
                                            markersize=4)
                    ax.set_title(matrix.replace('generated/', ''), fontsize=9)
                    ax.set_xticks(manifest['groups'])
                    ax.set_ylabel('Complete-call ms')
                    ax.grid(alpha=.2)
                for ax in list(axes.flat)[len(matrices):]:
                    ax.axis('off')
                handles, legends = axes.flat[0].get_legend_handles_labels()
                if handles:
                    fig.legend(handles, legends, loc='upper center', ncol=4)
                tuning = 'best screened FPT per k' if tuned else 'fixed FPT4'
                fig.suptitle(f'F{features} / P64 / {tuning}: runtime by target k; '
                             'hollow markers are aliases')
                fig.tight_layout(rect=(0, 0, 1, .97))
                suffix = '_tuned' if tuned else ''
                save(fig, f'k_curves_{family}_f{features}{suffix}')

    scaling = []
    for row in winners:
        if not row['matrix'].endswith('_n262144') or not row['speedup']:
            continue
        other_name = row['matrix'].replace('_n262144', '_n1048576')
        other = next((r for r in winners if r['matrix'] == other_name and
                      r['features'] == row['features'] and r['speedup']), None)
        if other:
            scaling.append(dict(profile=row['matrix'].replace('generated/', '').
                                replace('_n262144', ''), features=row['features'],
                                boo_growth=other['boo_ms']/row['boo_ms'],
                                direct_growth=other['direct_ms']/row['direct_ms']))
    write_csv(args.out / 'scaling.csv', scaling,
              list(scaling[0]) if scaling else ['profile', 'features', 'boo_growth', 'direct_growth'])
    f512 = [r for r in scaling if r['features'] == 512]
    if f512:
        fig, ax = plt.subplots(figsize=(10, 4))
        positions = np.arange(len(f512))
        ax.bar(positions-.18, [r['boo_growth'] for r in f512], .36, label='BOO')
        ax.bar(positions+.18, [r['direct_growth'] for r in f512], .36, label='Direct')
        ax.axhline(4, color='black', linestyle='--', linewidth=1)
        ax.set_xticks(positions, [r['profile'] for r in f512], rotation=30,
                      ha='right')
        ax.set_ylabel('N1M / N262k runtime')
        ax.set_title('F512: each size selects its fastest screened configuration')
        ax.legend()
        fig.tight_layout()
        save(fig, 'synthetic_scaling')

    lines = ['# Eran FPT screening audit', '',
             f"Completed shards: {sum(r['status']=='complete' for r in status)}/{len(status)}.",
             f"Requested BOO cells: {len(coverage)}; " +
             ', '.join(f'{key} {value}' for key, value in counts.items()) + '.', '',
             'Ratios use complete-call timings and same-cell direct baselines. '
             'Winners are selected after screening; method rankings across shards '
             'are not paired and need independent confirmation. The winner table '
             'contains an approximate incremental setup break-even where BOO is '
             'faster per call. It adds saved-label clustering, BOO construction '
             'and staging and subtracts direct plan creation; CPU hosts differ, '
             'and data-transfer, selection and value-update costs are not fully '
             'accounted for.', '',
             '## Figures', '',
             *[f'- [{name}]({name}.pdf)' for name in
               ('paired_fpt', 'speedup_real', 'speedup_synthetic',
                *[f'k_curves_{family}_f{f}' for family in ('real', 'synthetic')
                  for f in manifest['features']],
                'k_curves_real_f512_tuned', 'k_curves_synthetic_f512_tuned',
                'synthetic_scaling')
               if (args.out / f'{name}.pdf').exists()], '',
             'See coverage.csv for every requested cell and its measured, '
             'duplicate, unsupported or missing status. The raw samples and '
             'allocation logs must be archived separately.']
    (args.out / 'screening_report.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines[:4]))


if __name__ == '__main__':
    main()
