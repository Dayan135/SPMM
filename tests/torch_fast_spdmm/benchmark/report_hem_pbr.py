"""Generate an offline Markdown report from bench_hem_pbr output (stdlib only)."""
import argparse
import csv
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    rows = list(csv.DictReader((args.directory / 'summary.csv').open()))
    metadata = json.loads((args.directory / 'metadata.json').read_text())
    cells = sorted({(int(r['requested_size']), int(r['features']), int(r['seed'])) for r in rows})
    text = ['# HEM ordering → existing PBR: symmetric FEM experiment', '',
            f"GPU: {metadata['gpu']}; job {metadata['slurm_job_id']}; status: {metadata['status']}.",
            f"Warmups: {metadata['arguments']['warmup']}; interleaved repeats: "
            f"{metadata['arguments']['repeats']}. Times below are median milliseconds.", '',
            'All methods multiply the same FEM operator and dense input, expressed in the appropriate '
            'coordinates. Every output is scattered back to the original order and checked against '
            'float64 SciPy SpMM (rtol=atol=1e-4). Random permutation shuffles four-DOF mesh nodes, '
            'matching `fem2d_permuted`; it does not shuffle individual DOFs.', '',
            'HEM is computed once per target on the randomly permuted matrix, with feasible-first '
            'matching and equal row/column labels and caps. Stable label sorting gives `p[new]=old`, '
            '`P[i,p[i]]=1`, and `A_prime=A_permuted[p][:,p]`. The identical labels construct BOO. '
            'For a reordered multiplication, `X_prime=X_permuted[p]` and `Y_permuted[p]=Y_prime`. '
            'Group boundaries are not padded to PBR tile boundaries.', '',
            'The timed region is the complete resident-input matmul call, including output allocation/'
            'zeroing and stream joins. HEM, sparse reordering, format construction, staging and dense '
            'input/output permutations are excluded. This is steady-state execution in the chosen '
            'coordinates, not end-to-end latency for a one-off multiplication.', '',
            '## Fixed parameters: PBR 8×8 / min_nnz=2, HEM target=32', '',
            '| N | features | seed | original PBR | random PBR | random BOO/HEM | HEM→PBR | random PBR / HEM→PBR | BOO / HEM→PBR |',
            '|---:|---:|---:|---:|---:|---:|---:|---:|---:|']

    def select(cell, case, *, fixed=False, kernel=None):
        candidates = [r for r in rows if (int(r['requested_size']), int(r['features']), int(r['seed'])) == cell
                      and r['case'] == case and (kernel is None or r['kernel'] == kernel)]
        if fixed:
            candidates = [r for r in candidates if (r['kernel'] != 'block' or
                          (int(r['block']) == 8 and int(r['min_nnz']) == 2))
                          and (int(r['k']) == 0 or int(r['k']) == 32)]
        return min(candidates, key=lambda r: float(r['median_ms'])) if candidates else None

    for cell in cells:
        records = [select(cell, case, fixed=True, kernel=kernel) for case, kernel in
                   [('original_pbr', 'block'), ('permuted_pbr', 'block'),
                    ('permuted_boo', 'rowseg'), ('hem_reordered_pbr', 'block')]]
        if not all(records):
            continue
        a, b, c, d = [float(r['median_ms']) for r in records]
        text.append(f'| {cell[0]} | {cell[1]} | {cell[2]} | {a:.4f} | {b:.4f} | {c:.4f} | {d:.4f} | {b/d:.2f}× | {c/d:.2f}× |')
    text += ['', 'Ratios above 1 favor HEM→PBR.', '', '## Best measured configurations (post-hoc tuning)', '',
             'All PBR orders receive the same block/minimum sweep. Reordered PBR additionally sweeps '
             'HEM targets; BOO sweeps the same targets. These minima are tuning results, not held-out '
             'configuration selections. The full grid and IQRs remain in `summary.csv`.', '',
             '| N | f | seed | original PBR | random PBR | random BOO/HEM | HEM→PBR | winning reordered config |',
             '|---:|---:|---:|---:|---:|---:|---:|:---|']
    for cell in cells:
        records = [select(cell, case, kernel=kernel) for case, kernel in
                   [('original_pbr', 'block'), ('permuted_pbr', 'block'),
                    ('permuted_boo', 'rowseg'), ('hem_reordered_pbr', 'block')]]
        values = ' | '.join(f"{float(r['median_ms']):.4f}" for r in records)
        text.append(f'| {cell[0]} | {cell[1]} | {cell[2]} | {values} | {records[-1]["config"]} |')
    text += ['', '## Structure at fixed PBR parameters, all HEM targets', '',
             '| N | seed | case | k | blocks | compressed nnz % | CSR remainder nnz | build ms | HEM ms | sparse reorder ms |',
             '|---:|---:|:---|---:|---:|---:|---:|---:|---:|---:|']
    seen = set()
    for r in rows:
        key = (r['requested_size'], r['seed'], r['config'])
        if key in seen or r['kernel'] == 'shared_atomic' or r['case'] == 'hem_reordered_boo':
            continue
        if r['kernel'] == 'block' and (r['block'], r['min_nnz']) != ('8', '2'):
            continue
        seen.add(key)
        text.append(f"| {r['requested_size']} | {r['seed']} | {r['case']} | {r['k']} | {r['num_blocks']} | "
                    f"{100*float(r['compressed_fraction']):.2f} | {r['remainder_nnz']} | "
                    f"{float(r['build_ms']):.1f} | {float(r['hem_ms']):.1f} | {float(r['sparse_reorder_ms']):.1f} |")
    text += ['', '## Controls using exactly the same HEM groups', '',
             'Shared-atomic / rowseg holds the BOO object fixed and changes only the existing kernel '
             'selector. Random / reordered BOO holds group membership and the rowseg kernel fixed '
             'and changes coordinate locality. Ratios above 1 favor rowseg and reordered coordinates, respectively.', '',
             '| N | f | seed | HEM k | shared-atomic / rowseg | random / reordered BOO |',
             '|---:|---:|---:|---:|---:|---:|']
    for cell in cells:
        for k in metadata['arguments']['hem_k']:
            matched = [r for r in rows if (int(r['requested_size']), int(r['features']), int(r['seed'])) == cell
                       and int(r['k']) == k]
            def timing(case, kernel):
                return float(next(r['median_ms'] for r in matched if r['case'] == case and r['kernel'] == kernel))
            rowseg = timing('permuted_boo', 'rowseg')
            text.append(f'| {cell[0]} | {cell[1]} | {cell[2]} | {k} | '
                        f'{timing("permuted_boo", "shared_atomic")/rowseg:.2f}× | '
                        f'{rowseg/timing("hem_reordered_boo", "rowseg"):.2f}× |')
    text += ['', '## What the comparisons establish', '',
             '- Original versus random PBR measures sensitivity to node order with the PBR settings held fixed.',
             '- Random versus HEM-reordered PBR measures the benefit of this HEM-derived order using the unchanged PBR representation and kernel.',
             '- BOO/HEM versus HEM→PBR compares two execution stacks after sharing the clustering input. Their block partitions, metadata, remainder policy and kernels still differ; this is not a pure format-only speed comparison.',
             '- The BOO kernel control isolates the existing shared-atomic versus rowseg implementation effect on identical data. It does not isolate every difference between the PBR and BOO kernels.',
             '- HEM groups are variable sized; stable concatenation need not recover the original mesh order or align groups with fixed PBR tiles. Failure to accelerate PBR alone does not prove the grouping is poor.',
             '- Raw samples, all configurations, group/permutation arrays, preprocessing costs, source/extension hashes and environment metadata are retained beside this report.', '']
    output = args.directory / 'comparison.md'
    output.write_text('\n'.join(text))
    print(output)


if __name__ == '__main__':
    main()
