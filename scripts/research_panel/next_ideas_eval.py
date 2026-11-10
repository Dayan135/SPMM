"""Evaluate four next ideas from per-block BOO statistics (next_ideas_blocks.py output).

  python scripts/research_panel/next_ideas_eval.py --blocks DIR --profiles KCP_PROFILE_SUMMARY.csv --out OUT

Everything uses the occupancy model that matched 4,344 timed arms exactly:
  arena = 6*max_nnz + 8*max_nr + 4*max_nc + 4*P*max_nc + 20
  alloc = roundup_128(arena + 1024);  CTAs/SM = min(102400//alloc, 1536//T, 65536//(regs*T), 16)
Labels in the output: computed (exact from structures + model), fit (regression on measured ncu
counters), estimated (model extrapolation).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

PS = (8, 16, 32, 64, 128)
TS = (256, 512)
F = 512
FPT = {'diag_dd0.05': 2}            # FPT4 elsewhere (frozen per input in the k_c x P experiment)
REGS = 40


def arena(nnz, nr, nc, p):
    return 6 * nnz + 8 * nr + 4 * nc + 4 * p * nc + 20


def ctas(ar, t, regs=REGS):
    if ar > 101376:
        return 0
    alloc = -(-(ar + 1024) // 128) * 128
    reg_warp = -(-(regs * 32) // 256) * 256
    return min(102400 // alloc, 1536 // t, 65536 // (reg_warp * (t // 32)), 16)


def p_best(nnz, nr, nc, t, min_threads):
    """Largest P whose CTAs/SM * T >= min_threads (0 if none)."""
    ok = [p for p in PS if ctas(arena(nnz, nr, nc, p), t) * t >= min_threads]
    return max(ok) if ok else 0


def load(blocks):
    out = {}
    for f in sorted(blocks.glob('*__*.npz')):
        name, shape = f.stem.split('__')
        out[(name, shape)] = dict(np.load(f))
    return out


# ---------------------------------------------------------------- idea 1: active-column compaction
def idea1(S):
    rows = []
    for (name, shape), b in S.items():
        nnz, nr, nc, act = b['nnz'].max(), b['n_r'].max(), b['n_c'].max(), b['active_cols'].max()
        r = dict(input=name, shape=shape, blocks=len(b['nnz']), max_nnz=int(nnz), max_nr=int(nr), max_nc=int(nc),
                 max_active=int(act), p99_active=int(np.percentile(b['active_cols'], 99)),
                 staged_ratio=b['active_cols'].sum() / b['n_c'].sum(),
                 x_sectors_now_M=b['n_c'].sum() * F * 4 / 32 / 1e6,
                 x_sectors_compact_M=b['active_cols'].sum() * F * 4 / 32 / 1e6)
        for t in TS:
            for p in PS:
                r[f'cta_t{t}_p{p}_now'] = ctas(arena(nnz, nr, nc, p), t)
                r[f'cta_t{t}_p{p}_cmp'] = ctas(arena(nnz, nr, act, p), t)
            r[f'pfull_t{t}_now'] = p_best(nnz, nr, nc, t, 1536)
            r[f'pfull_t{t}_cmp'] = p_best(nnz, nr, act, t, 1536)
            r[f'p67_t{t}_now'] = p_best(nnz, nr, nc, t, 1024)
            r[f'p67_t{t}_cmp'] = p_best(nnz, nr, act, t, 1024)
        rows.append(r)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- idea 2: size-class launches
def class_p(b, idx, compact):
    nc_col = b['active_cols'] if compact else b['n_c']
    nnz, nr, nc = b['nnz'][idx].max(), b['n_r'][idx].max(), nc_col[idx].max()
    return max(max(p_best(nnz, nr, nc, t, 1536) for t in TS), 0), arena(nnz, nr, nc, 64)


def idea2(S):
    rows = []
    for (name, shape), b in S.items():
        for compact in (False, True):
            nc_col = b['active_cols'] if compact else b['n_c']
            need = arena(b['nnz'], b['n_r'], nc_col, 64)          # per-block requirement at P64
            order = np.argsort(need, kind='stable')
            tot = b['nnz'].sum()
            p1, a1 = class_p(b, order, compact)
            base = dict(input=name, shape=shape, compact=compact, classes=1, cut_q='',
                        p_nnz_weighted=p1, nnz_at_p64plus=float(p1 >= 64), launches=1,
                        detail=f'P{p1} (arena@P64 {a1})')
            rows.append(base)
            for q in (0.5, 0.75, 0.9, 0.95, 0.99):
                k = int(len(order) * q)
                lo, hi = order[:k], order[k:]
                if len(lo) == 0 or len(hi) == 0:
                    continue
                pl, al = class_p(b, lo, compact)
                ph, ah = class_p(b, hi, compact)
                wl, wh = b['nnz'][lo].sum() / tot, b['nnz'][hi].sum() / tot
                rows.append(dict(input=name, shape=shape, compact=compact, classes=2, cut_q=q,
                                 p_nnz_weighted=wl * pl + wh * ph,
                                 nnz_at_p64plus=wl * (pl >= 64) + wh * (ph >= 64), launches=2,
                                 detail=f'small {len(lo)} blk {wl:.0%} nnz P{pl}; large {len(hi)} blk {wh:.0%} nnz P{ph}'))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- idea 3: row-length-aware mapping
def chunk_stats(b, lanes, mode):
    """Per pass: warp-iterations (work-item setup) and loop trips, for the Step B mapping.

    mode: 'now' (rows in block order, empty rows included), 'sorted' (rows sorted by length
    within block), 'sorted_skip' (sorted, empty rows not given work items)."""
    L, ptr, n_r = b['row_len'], b['row_ptr'], b['n_r']
    if lanes >= 32:                      # a row spans lanes/32 whole warps: no mixing
        rows = L if mode != 'sorted_skip' else L[L > 0]
        nrows = len(rows) if mode == 'sorted_skip' else n_r.sum()
        return dict(warp_iters=nrows * lanes // 32, trips=int(rows.sum()) * lanes // 32,
                    lane_eff=1.0, rows_per_warp=1)
    g = 32 // lanes
    blk = np.repeat(np.arange(len(n_r)), n_r)
    pos = np.arange(len(L)) - ptr[blk]
    if mode == 'now':
        key_len = L
        rank = pos
    else:
        o = np.lexsort((-L, blk))                       # by block, then length descending
        key_len = L[o]
        rank = np.arange(len(L)) - ptr[blk[o]]
        blk = blk[o]
        if mode == 'sorted_skip':
            keep = key_len > 0
            key_len, blk = key_len[keep], blk[keep]
            first = np.r_[0, np.flatnonzero(np.diff(blk)) + 1]
            rank = np.arange(len(key_len)) - np.repeat(first, np.diff(np.r_[first, len(key_len)]))
    chunk = blk * 65536 + rank // g
    uc, inv = np.unique(chunk, return_inverse=True)
    mx = np.zeros(len(uc), np.int64)
    np.maximum.at(mx, inv, key_len)
    trips = int(mx.sum())
    useful = int(key_len.sum())
    return dict(warp_iters=len(uc), trips=trips, lane_eff=useful / (trips * g) if trips else 1.0,
                rows_per_warp=g)


def idea3(S, prof):
    rows = []
    for (name, shape), b in S.items():
        fpt = FPT.get(name, 4)
        for p in PS:
            if fpt > p:
                continue
            lanes = p // fpt
            for mode in ('now', 'sorted', 'sorted_skip'):
                c = chunk_stats(b, lanes, mode)
                rows.append(dict(input=name, shape=shape, p=p, mode=mode, passes=F // p, **c,
                                 empty_row_frac=float((b['row_len'] == 0).mean())))
    d = pd.DataFrame(rows)
    # fit inst = c0 + c1 * passes*warp_iters + c2 * passes*trips on measured native profiles
    fits = []
    for name, label in (('fem2d_permuted', 'fem2d_permuted_r128c64_rs_p{p}_t256_f4'),
                        ('diag_dd0.05', 'diag_dd0.05_r128c64_rs_p{p}_t256_f2')):
        m = d[(d.input == name) & (d['shape'] == '128x64') & (d['mode'] == 'now')].set_index('p')
        y, X = [], []
        for p in PS:
            r = prof[prof.label == label.format(p=p)]
            if len(r) and p in m.index:
                y.append(float(r.warp_inst_executed.iloc[0]))
                X.append([1.0, m.loc[p, 'passes'] * m.loc[p, 'warp_iters'], m.loc[p, 'passes'] * m.loc[p, 'trips']])
        if len(y) >= 4:
            X, y = np.array(X), np.array(y)
            coef, *_ = np.linalg.lstsq(X, y, rcond=None)
            pred = X @ coef
            fits.append(dict(input=name, shape='128x64', c0_M=coef[0] / 1e6, per_warp_iter=coef[1], per_trip=coef[2],
                             max_rel_err=float(np.max(np.abs(pred - y) / y)), points=len(y)))
    fits = pd.DataFrame(fits)
    return d, fits


def predict3(d, fits):
    out = []
    for f in fits.itertuples():
        g = d[(d.input == f.input) & (d['shape'] == '128x64')]
        for p in PS:
            gp = g[g.p == p].set_index('mode')
            if 'now' not in gp.index:
                continue
            inst = {m: f.c0_M * 1e6 + f.per_warp_iter * gp.loc[m, 'passes'] * gp.loc[m, 'warp_iters']
                    + f.per_trip * gp.loc[m, 'passes'] * gp.loc[m, 'trips'] for m in gp.index}
            out.append(dict(input=f.input, shape='128x64', p=p, inst_now_M=inst['now'] / 1e6,
                            sorted_rel=inst['sorted'] / inst['now'], sorted_skip_rel=inst['sorted_skip'] / inst['now']))
    return pd.DataFrame(out)


# ---------------------------------------------------------------- idea 4: A in registers across passes
def idea4(S):
    rows = []
    for (name, shape), b in S.items():
        fpt = FPT.get(name, 4)
        for t in TS:
            for p in (32, 64):
                lanes = p // fpt
                L, ptr, n_r = b['row_len'], b['row_ptr'], b['n_r']
                # per thread: rows of its work items i = tx + j*T, row = i // lanes
                held_max = np.zeros(len(n_r), np.int64)
                items = n_r * lanes
                for bi in np.flatnonzero(items > 0):
                    i = np.arange(items[bi])
                    lens = L[ptr[bi] + i // lanes]
                    held = np.bincount(i % t, weights=lens, minlength=t)
                    held_max[bi] = held.max()
                nnz = b['nnz']
                r = dict(input=name, shape=shape, threads=t, p=p, items_per_thread=float(np.ceil(items / t).mean()),
                         held_max_p50=float(np.percentile(held_max, 50)), held_max_p99=float(np.percentile(held_max, 99)),
                         held_max_max=int(held_max.max()))
                for extra in (4, 8, 16, 32):
                    cap = int(extra / 1.5)            # 32-bit value + 16-bit offset per held nonzero
                    fit = held_max <= cap
                    r[f'nnz_frac_fit_r{extra}'] = float(nnz[fit].sum() / nnz.sum())
                    r[f'cta_t{t}_r{extra}'] = ctas(0, t, REGS + extra)
                r['cta_now'] = ctas(0, t, REGS)
                rows.append(r)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--blocks', type=Path, required=True)
    ap.add_argument('--profiles', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    pd.set_option('display.width', 250, 'display.max_columns', 60, 'display.max_rows', 300)
    S = load(a.blocks)
    prof = pd.read_csv(a.profiles)
    i1 = idea1(S); i1.to_csv(a.out / 'idea1_active_columns.csv', index=False)
    i2 = idea2(S); i2.to_csv(a.out / 'idea2_size_classes.csv', index=False)
    i3, fits = idea3(S, prof); i3.to_csv(a.out / 'idea3_mapping.csv', index=False)
    fits.to_csv(a.out / 'idea3_instruction_fit.csv', index=False)
    p3 = predict3(i3, fits); p3.to_csv(a.out / 'idea3_predicted.csv', index=False)
    i4 = idea4(S); i4.to_csv(a.out / 'idea4_registers.csv', index=False)
    print('== idea 1 (computed)')
    print(i1[['input', 'shape', 'max_nc', 'max_active', 'p99_active', 'staged_ratio', 'x_sectors_now_M',
              'x_sectors_compact_M', 'pfull_t256_now', 'pfull_t256_cmp', 'pfull_t512_now', 'pfull_t512_cmp',
              'p67_t256_now', 'p67_t256_cmp']].round(3).to_string(index=False))
    print('\n== idea 2 (computed): best cut per (input, shape, compact)')
    best = i2.loc[i2.groupby(['input', 'shape', 'compact']).p_nnz_weighted.idxmax()]
    one = i2[i2.classes == 1].set_index(['input', 'shape', 'compact']).p_nnz_weighted.rename('p_one_class')
    print(best.join(one, on=['input', 'shape', 'compact'])[['input', 'shape', 'compact', 'p_one_class', 'cut_q',
                                                            'p_nnz_weighted', 'nnz_at_p64plus', 'detail']].to_string(index=False))
    print('\n== idea 3 instruction fit (fit)')
    print(fits.round(4).to_string(index=False))
    print('\n== idea 3 predicted instructions vs now (estimated)')
    print(p3.round(3).to_string(index=False))
    print('\n== idea 3 lane efficiency (computed)')
    piv = i3.pivot_table(index=['input', 'shape', 'p'], columns='mode', values='lane_eff').round(3)
    print(piv.to_string())
    print('\n== idea 4 (computed fit fractions; CTA cost from model)')
    print(i4.round(3).to_string(index=False))
    (a.out / 'done.json').write_text(json.dumps(dict(shapes=len(S)), indent=1))


if __name__ == '__main__':
    main()
