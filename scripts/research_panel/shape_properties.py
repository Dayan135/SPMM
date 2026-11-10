"""CPU-only structural analysis. Run: conda run -n fem python scripts/research_panel/shape_properties.py

Reads saved CSVs only; never imports the CUDA extension, Torch, or cluster code.
Outputs are immutable: reruns verify existing bytes and refuse changed outputs.
Use --out for a separate reproduction directory. Predictions are not GPU results.
"""
from pathlib import Path
import argparse
import hashlib
import json
import warnings
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

DEFINITIONS = {
 'read_fraction': 'x_read / x_staged: fraction of staged column slots used by at least one stored entry',
 'reuse_staged': 'compressed_nnz / x_staged: stored-entry MAC opportunities per staged X scalar',
 'reuse_used': 'compressed_nnz / x_read: stored-entry MAC opportunities per used X scalar',
 'reuse_gain': 'reuse_staged(shape) / reuse_staged(32x32), same matrix and method',
 'x_staged_rel': 'x_staged(shape) / x_staged(32x32), same matrix and method',
 'y_commits_rel': 'active block-row count(shape) / active block-row count(32x32)',
 'output_fanout': 'y_commits / rows: active block-row incidences per global row, including empty global rows in denominator',
 'imbalance': 'balance_gain_t256_p64_f2 = sum busiest row-residue workload / sum ceil(block_nnz/8); >=1 is worse',
 'busiest_f2_rel': 'sum busiest row-residue workload(shape) / same at 32x32; T256/P64/FPT2 fixed',
 'capacity_p64': 'best_threads_sm_p64 / 1536; theoretical best over T128/256/512, not achieved occupancy',
 'cta_p64_t256': 'threads_sm_p64_t256 / 256: estimated CTAs/SM at fixed T256',
 'arena_p64_rel': 'arena_p64(shape) / arena_p64(32x32)',
 'max_p99': 'max_nnz_blk / nz_p99; outlier pressure from the largest block',
 'remainder_fraction': 'remainder_nnz / total_nnz',
 'remainder_delta': 'remainder_fraction(shape) - remainder_fraction(32x32)',
}
# Small fixed candidate set: single properties plus four mechanistic pairs.
# No thresholds searched, no model hyperparameter tuning, no matrix identifiers.
SINGLES = ['x_staged_rel', 'read_fraction', 'reuse_staged', 'reuse_gain',
           'y_commits_rel', 'busiest_f2_rel', 'imbalance', 'capacity_p64',
           'max_p99', 'remainder_fraction', 'remainder_delta']
CANDIDATES = {'null': [], 'gate': ['passes']}
CANDIDATES.update({s: [s] for s in SINGLES})
for b in ['busiest_f2_rel', 'capacity_p64', 'read_fraction', 'y_commits_rel']:
    CANDIDATES['x+' + b] = ['x_staged_rel', b]
LINEAR = {'capacity_p64', 'read_fraction', 'remainder_fraction', 'remainder_delta', 'passes'}

def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def weights(d, group='input'):
    w = 1 / d.groupby(group)[group].transform('size').to_numpy(float)
    return w / w.sum()

def transform(d, cols):
    a = np.column_stack([d[c].to_numpy(float) if c in LINEAR else np.log(d[c].to_numpy(float)) for c in cols] +
                        [(d.features.to_numpy() == 512).astype(float)])
    assert np.isfinite(a).all(), 'Undefined properties: do not silently impute'
    return a

def fit(d, cols):
    a, w, y = transform(d, cols), weights(d), np.log(d.best_med.to_numpy())
    mu = w @ a
    sd = np.sqrt(w @ (a-mu)**2)
    sd[sd < 1e-10] = 1
    z = np.column_stack([np.ones(len(d)), (a-mu)/sd])
    # Fixed mild shrinkage; weighted objective totals one, penalty 0.02.
    penalty = np.eye(z.shape[1]) * .02
    penalty[0, 0] = 0
    beta = np.linalg.solve(z.T @ (w[:, None]*z) + penalty, z.T @ (w*y))
    return dict(cols=cols, mu=mu.tolist(), sd=sd.tolist(), beta=beta.tolist())

def predict(model, d):
    a = transform(d, model['cols'])
    return np.column_stack([np.ones(len(d)), (a-model['mu'])/model['sd']]) @ model['beta']

def cv(d, cols, group='input'):
    out = np.empty(len(d))
    for g in sorted(d[group].unique()):
        test = (d[group] == g).to_numpy()
        out[test] = predict(fit(d.loc[~test], cols), d.loc[test])
    return out

def scores(d, pred):
    y, w = np.log(d.best_med.to_numpy()), weights(d)
    yes, guessed = y > np.log(1.05), pred > np.log(1.05)
    precision = float(w[yes & guessed].sum()/w[guessed].sum()) if guessed.any() else np.nan
    recall = float(w[yes & guessed].sum()/w[yes].sum()) if yes.any() else np.nan
    specificity = float(w[~yes & ~guessed].sum()/w[~yes].sum()) if (~yes).any() else np.nan
    return dict(log_mae=float(w @ abs(pred-y)), log_rmse=float(np.sqrt(w @ (pred-y)**2)),
                accuracy=float(w @ (yes == guessed)), balanced_accuracy=(recall+specificity)/2, precision=precision, recall=recall,
                predicted_gains=int(guessed.sum()), actual_gains=int(yes.sum()))

def rank(a, b):
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    return float(spearmanr(a, b).statistic)

def markdown(d):
    # No optional tabulate dependency.
    def s(v):
        return f'{v:.4f}' if isinstance(v, (float, np.floating)) else str(v)
    return '| ' + ' | '.join(d.columns) + ' |\n|' + '|'.join(['---']*len(d.columns)) + '|\n' + '\n'.join(
        '| ' + ' | '.join(s(v) for v in row) + ' |' for row in d.itertuples(index=False, name=None))

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[2])
    ap.add_argument('--out', type=Path)
    args = ap.parse_args()
    root = args.repo.resolve()
    analysis = root / 'experiments/2026-11-06_block_shape/analysis'
    out = args.out or analysis / 'properties'
    sources = [analysis/'gate/gate_relative.csv', analysis/'gpu/screen_table.csv', analysis/'gpu/screen_best.csv',
               root/'scripts/research_panel/shape_gate.py', root/'scripts/research_panel/shape_gate_report.py',
               root/'scripts/research_panel/shape_report.py']
    g, screen, best = [pd.read_csv(p) for p in sources[:3]]
    assert not g.duplicated(['matrix','method','shape']).any()
    g['reuse_staged'] = g.compressed_nnz/g.x_staged
    g['reuse_used'] = g.compressed_nnz/g.x_read
    base = g[g['shape']=='32x32'].set_index(['matrix','method']).reuse_staged
    g = g.join(base.rename('base_reuse'), on=['matrix','method'])
    g['reuse_gain'] = g.reuse_staged/g.base_reuse
    g['output_fanout'] = g.y_commits/g.rows
    g['imbalance'] = g.balance_gain_t256_p64_f2
    g['capacity_p64'] = g.best_threads_sm_p64/1536
    g['cta_p64_t256'] = g.threads_sm_p64_t256/256
    g['max_p99'] = g.max_nnz_blk/g.nz_p99
    mapping = best[['input','matrix','method']].drop_duplicates()
    assert not mapping.input.duplicated().any()
    d = screen[screen['shape']!='32x32'].drop(columns=['x_staged_rel','y_commits_rel','passes']).merge(mapping, on='input', validate='many_to_one')
    d = d.merge(g, on=['matrix','method','shape'], validate='many_to_one')
    assert len(d) == len(screen[screen['shape']!='32x32']) and d.input.nunique()==8
    # Reconstruct the published aggregate to ensure no accidental change of target.
    reconstructed = best.groupby(['input','features','shape']).speedup.median()
    check = d.set_index(['input','features','shape']).best_med - reconstructed
    assert check.dropna().abs().max() <= .00051
    d['family_group'] = d.input.replace({'fem2d_permuted':'fem2d', 'diag_skewed':'diagonal',
                                        'diag_b1.0':'diagonal', 'diag_dd0.05':'diagonal'})
    corrs = []
    for c in DEFINITIONS:
        vals = [rank(t[c],t.best_med) for _,t in d.groupby('input')]
        corrs.append(dict(property=c, pooled_spearman=rank(d[c], d.best_med),
                          median_within_input_spearman=float(np.nanmedian(vals)),
                          min_leave_one_input_out_rho=min(rank(t[c], t.best_med) for n in d.input.unique() if len(t:=d[d.input!=n])),
                          max_leave_one_input_out_rho=max(rank(t[c], t.best_med) for n in d.input.unique() if len(t:=d[d.input!=n]))))
    correlations = pd.DataFrame(corrs)
    predictions, rows = {}, []
    for name, cols in CANDIDATES.items():
        predictions[name] = cv(d, cols)
        rows.append(dict(model=name, properties=','.join(cols), **scores(d, predictions[name])))
    metrics = pd.DataFrame(rows).sort_values(['log_mae','model'])
    regression_selected = metrics.iloc[0].model
    selected = metrics.sort_values(['balanced_accuracy','model'],ascending=[False,True]).iloc[0].model
    # Nested selection: each outer input is unseen by feature/model selection too.
    nested = np.empty(len(d)); nested_gain = np.empty(len(d)); folds=[]
    for name in sorted(d.input.unique()):
        mask = (d.input==name).to_numpy(); train=d.loc[~mask]
        inner={c:scores(train,cv(train,cols)) for c,cols in CANDIDATES.items()}
        chosen = min(CANDIDATES, key=lambda c:(inner[c]['log_mae'],c))
        gain_choice = min(CANDIDATES, key=lambda c:(-inner[c]['balanced_accuracy'],c))
        nested[mask] = predict(fit(train,CANDIDATES[chosen]),d.loc[mask])
        nested_gain[mask] = predict(fit(train,CANDIDATES[gain_choice]),d.loc[mask])
        folds.append(dict(input=name, selected_model=chosen, gain_selected_model=gain_choice, **scores(d.loc[mask],nested_gain[mask])))
    metrics = pd.concat([metrics, pd.DataFrame([dict(model='nested_selection', properties='inner LOIO selection', **scores(d,nested))])],ignore_index=True)
    metrics = pd.concat([metrics, pd.DataFrame([dict(model='nested_gain_selection', properties='inner LOIO balanced-accuracy selection', **scores(d,nested_gain))])],ignore_index=True)
    # Additional correlated-family holdout sensitivity, using fixed candidates, no new tuning.
    family_metrics = pd.DataFrame([dict(model=n, **scores(d,cv(d,c,group='family_group'))) for n,c in CANDIDATES.items()])
    gate = d.groupby('passes').agg(rows=('best_med','size'), median_speedup=('best_med','median'),
                                 wins=('best_med',lambda s:int((s>1.05).sum())))
    gate['win_fraction']=gate.wins/gate.rows
    gate=gate.reset_index()
    oof=d[['input','matrix','method','features','shape','best_med','best_min','passes']].copy()
    for n,p in predictions.items(): oof[n]=np.exp(p)
    oof['nested_selection']=np.exp(nested)
    oof['nested_gain_selection']=np.exp(nested_gain)
    model=fit(d,CANDIDATES[selected])
    beta=np.array(model['beta']); sd=np.array(model['sd']); mu=np.array(model['mu'])
    raw_beta=beta[1:]/sd
    equation=dict(intercept=float(beta[0]-mu@raw_beta), coefficients=dict(zip(model['cols']+['F512'],raw_beta.tolist())),
                  transforms={c:'identity' if c in LINEAR else 'ln' for c in model['cols']},
                  output='ln(predicted speedup); F512 is 1 at F512, otherwise 0')
    # Freeze every untimed matrix/method/tall-shape prediction; no timing-based method choice.
    untimed=g[~g.matrix.isin(mapping.matrix) & (g.kr>g.kc)].copy()
    q=pd.concat([untimed.assign(features=f) for f in (128,512)],ignore_index=True)
    q['predicted_speedup']=np.exp(predict(model,q))
    q['predicted_gain_gt5pct']=q.predicted_speedup>1.05
    # A deterministic descriptive sensitivity band, not a statistical prediction interval.
    resid=np.log(d.best_med.to_numpy())-nested_gain
    radius=float(pd.DataFrame({'input':d.input,'error':abs(resid)}).groupby('input').error.quantile(.9).max())
    q['sensitivity_low']=q.predicted_speedup*np.exp(-radius)
    q['sensitivity_high']=q.predicted_speedup*np.exp(radius)
    q['extrapolated_properties']=q.apply(lambda r: ','.join(c for c in SINGLES if r[c]<d[c].min() or r[c]>d[c].max()),axis=1)
    q['unseen_method']=~q.method.isin(mapping.method)
    q['scale_sibling_in_training']=q.matrix.str.replace('_n1048576','_n262144',regex=False).isin(mapping.matrix)
    # Descriptive distinction: forecasts beyond any measured marginal range stay explicit.
    q['prediction_status']=np.where(q.predicted_gain_gt5pct,'gain candidate','no >5% gain predicted')
    q['primary_gate_prediction']=np.where(q.passes,'gain candidate','no gain claim')
    q=q.sort_values(['matrix','method','features','shape'])
    cols=['matrix','method','features','shape','predicted_speedup','predicted_gain_gt5pct','prediction_status',
          'sensitivity_low','sensitivity_high','extrapolated_properties','unseen_method','scale_sibling_in_training','passes','primary_gate_prediction']+list(DEFINITIONS)
    cols=list(dict.fromkeys(cols))
    q=q[cols]
    # One best tall shape per method and F. This selects structural predictions only.
    recommended=q.loc[q.groupby(['matrix','method','features']).predicted_speedup.idxmax()].sort_values(['matrix','method','features'])
    top=recommended.loc[recommended.groupby(['matrix','features']).predicted_speedup.idxmax()].sort_values(['matrix','features'])
    # Review table: same held-out measurements and decision threshold for learned rules and gate.
    comparison=metrics[metrics.model.isin(['null','gate',selected,regression_selected,'nested_selection','nested_gain_selection'])].copy()
    gate_fixed=scores(d,np.where(d.passes,np.log(1.06),np.log(.99)))
    gate_fixed['log_mae']=np.nan; gate_fixed['log_rmse']=np.nan
    comparison=pd.concat([comparison,pd.DataFrame([dict(model='original_gate_boolean',properties='published pass rule',**gate_fixed)])],ignore_index=True)
    for i,r in comparison.iterrows():
        pred = (np.where(d.passes,np.log(1.06),np.log(.99)) if r.model=='original_gate_boolean'
                else nested if r.model=='nested_selection' else nested_gain if r.model=='nested_gain_selection'
                else predictions[r.model])
        yes=pred>np.log(1.05)
        comparison.loc[i,'called_gain_actual_median']=d.loc[yes,'best_med'].median()
        comparison.loc[i,'called_no_gain_actual_median']=d.loc[~yes,'best_med'].median()
        comparison.loc[i,'pooled_x_staged_spearman']=rank(d.x_staged_rel,d.best_med)
    files={}
    def csv(name,frame): files[name]=frame.to_csv(index=False,float_format='%.9g').encode()
    csv('joined_measured.csv',d); csv('property_definitions.csv',pd.DataFrame(DEFINITIONS.items(),columns=['property','definition']))
    csv('correlations.csv',correlations); csv('gate_separation.csv',gate); csv('model_loio.csv',metrics)
    csv('model_family_holdout.csv',family_metrics); csv('loio_predictions.csv',oof); csv('nested_folds.csv',pd.DataFrame(folds))
    csv('gate_vs_properties.csv',comparison); csv('frozen_predictions.csv',q)
    csv('frozen_best_by_method.csv',recommended); csv('frozen_matrix_shortlist.csv',top)
    provenance=dict(schema=1,inputs={str(p.relative_to(root)):digest(p) for p in sources}, script_sha256=digest(Path(__file__)),
                    training_rows=len(d),training_inputs=sorted(d.input.unique()),
                    selected_model=selected,regression_selected_model=regression_selected,selection_metric='input-weighted balanced accuracy at speedup >1.05',model=model,equation=equation,candidates=CANDIDATES,
                    primary_prediction_rule='original published gate; learned model is a frozen exploratory challenger, not a validated replacement',
                    threshold=1.05,shrinkage=.02,nested_sensitivity_log_radius=radius,
                    forecast_scope='untimed shapes only as of this input snapshot; 1M siblings are scale checks, not independent families',
                    frozen_predictions_sha256=hashlib.sha256(files['frozen_predictions.csv']).hexdigest())
    files['FREEZE.json']=(json.dumps(provenance,indent=2,sort_keys=True)+'\n').encode()
    r=metrics.set_index('model'); improvement=1-r.loc['nested_selection','log_mae']/r.loc['gate','log_mae']
    web=oof[oof.input=='webbase']
    doc=f'''# Structural predictors of block-shape gain

CPU-only analysis of saved files. Run `conda run -n fem python scripts/research_panel/shape_properties.py`.
Only this script and this output directory are authored. No CUDA, Torch, SSH or cluster execution.
Outputs are write-once: an identical rerun verifies bytes; changed sources require `--out` to a new directory.
`FREEZE.json` records source/script hashes, the exact fitted equation and prediction hash.

## Properties and target

{markdown(pd.DataFrame(DEFINITIONS.items(),columns=['Property','Exact computation']))}

All counts concern retained BOO blocks, exclude CSR work, count stored entries (including explicit zeros), and are per feature.
`y_commits` is a potential update count: cancellation/zero sums can suppress actual atomics. Read fraction measures utilization, not overlap itself;
reuse_used and reuse_staged capture the multiplicity of nonzeros sharing X. Neither is measured DRAM traffic.
Arena capacity is an SM86 model ignoring register limits. The fixed T256/P64/FPT2 imbalance proxy does not track each tuned winner's mapping.
Zero-denominator properties remain undefined; the training/prediction path asserts finite model inputs rather than imputing them.

Join: screen_best maps input to one matrix/method; screen_table joins gate_relative on matrix/method/shape.
32x32 identity-ratio anchors are excluded. There are {len(d)} rows, 8 inputs and two F values; all allocations are aggregated, never treated as independent data points.
Target is published best_med (median across allocation-specific best-vs-best ratios, rounded to 3 decimals), checked against screen_best within rounding.
These are filtered, post-hoc tuned screening times, not confirmation results. The untimed set is defined by matrix key absent from this shape screen,
not absence of historical square-block timings. Neither F256 nor new confirmation timing is read.

## Gate benchmark and explanations

{markdown(gate)}

Pooled Spearman(speedup, x_staged_rel) = {rank(d.x_staged_rel,d.best_med):.3f}. This reproduces the supplied benchmark.
All correlations, their within-input medians and leave-one-input-out ranges are in correlations.csv; no row-independent p-values are reported.

## Models and validation

Each candidate is a log-speedup linear model with one or two structural properties, plus a shared F512 indicator.
Positive ratio/count properties are log-transformed; bounded fractions, remainder_delta and the gate indicator remain linear.
Each input has equal total fit/score weight. Centering/scaling use training folds only. Fixed ridge penalty 0.02 on standardized coefficients,
intercept unpenalized; no penalty tuning, threshold search, trees or matrix-name features. Gain threshold is fixed at predicted speedup >1.05.
Candidates are the 11 single properties and four X-staging pairs listed in FREEZE.json, plus null/F-only and original-gate/F baselines.
The gate regression learns effect size in training folds; original_gate_boolean preserves the published pass/fail decision.

Every outer fold leaves out all shapes/F/allocations of one input. model_loio.csv compares fixed candidates; its best ranking is exploratory
because the winner was selected on those scores. Nested selection repeats candidate selection using inner LOIO among the seven remaining inputs;
its outer predictions are the fairer estimate of this whole selection procedure. model_family_holdout.csv additionally holds natural/permuted FEM together
and all three diagonal inputs together; these sensitivity groups leave only five units, so estimates remain fragile.

{markdown(comparison)}

Lowest fixed-candidate LOIO error: **{regression_selected}**. Nested MAE selection changes gate log-MAE by **{improvement*100:.1f}% reduction**.
Best gain detector among the fixed candidates: **{selected}**, chosen by input-weighted balanced accuracy at the fixed >5% threshold.
Both selection objectives are reported because low prediction error need not imply useful gain detection. The classification objective was added after
the initial regression comparison exposed that distinction; this is exploratory, not a preregistered model competition.
Nested gain selection repeats balanced-accuracy selection inside each training fold. It does not remove analyst-choice or model-class selection bias.
The **published gate remains the primary rule**: its balanced accuracy exceeds the fixed two-property challenger, and the nested search is near chance.
The two-property challenger improves precision by sacrificing recall; it has not demonstrated an overall improvement over the gate.
Its full-fit equation is `log(speedup) = {equation['intercept']:.6f}` plus the transformed terms in FREEZE.json.
For the selected X/busiest pair, a 20% reduction in staging with other predictors fixed corresponds to about 6.7% higher predicted speedup;
a 10% reduction in busiest work corresponds to about 6.1%. These are fitted associations, not isolated timing effects.
Natural and permuted FEM have nearly identical structural predictors but materially different shape gains; these counts omit global address/cache locality.
Webbase's max/p99 ratio is approximately 6-12, versus about 1-2.5 elsewhere. That distinguishes it descriptively, but there is no second such input
from which a held-out predictor can learn the regime. Do not fit a special webbase exception and call it validation.
Errors are in natural-log units; exp(log-MAE) is an interpretable multiplicative error factor, not a confidence interval.
Fixed selected-model LOIO webbase forecasts call {int((web[selected]>1.05).sum())}/{len(web)} shapes/F gains;
nested gain selection calls {int((web.nested_gain_selection>1.05).sum())}/{len(web)}; observed gains are {int((web.best_med>1.05).sum())}/{len(web)}.
Per-input errors and selected models are in nested_folds.csv. A model's ability to rank the common FEM shapes is not evidence of generalizing to a power-law graph.

## Frozen out-of-sample forecasts

Exploratory challenger `{selected}` is refit to the 98 rows; FREEZE.json contains a directly evaluable equation.
Every row also freezes primary_gate_prediction. A primary non-pass is abstention/no gain claim, not a forecast of a slowdown.
frozen_predictions.csv fixes all {len(q)} available tall-shape/method/F predictions on {q.matrix.nunique()} untimed matrix keys.
frozen_best_by_method.csv selects a shape separately for each method/F; frozen_matrix_shortlist.csv additionally selects a method by predicted performance.
No method was selected using future timings. Predictions compare against the SAME method's tuned 32x32, not the fastest 32x32 across methods or cuSPARSE.
Methods and shapes were not all observed in training (especially Graclus); extrapolated marginal properties and unseen methods are flagged.
The multiplicative sensitivity range uses the largest input-wise 90th percentile absolute nested residual; it is descriptive and NOT calibrated coverage.
Selection of the largest forecast adds optimism. Treat these as frozen falsifiable test candidates, not established expected speedups.

{markdown(top[['matrix','features','method','shape','predicted_speedup','primary_gate_prediction']])}

## Limits and next evaluation

Only eight inputs, with related FEM/diagonal families and one labeling per input; method effects are confounded with input effects.
Gate-based selection determined much of which tall shapes were timed, so the training sample is not an unbiased shape sweep.
Two F measurements share the same structure; equal-input weighting and grouped folds reduce pseudoreplication but do not create more independent matrices.
These regressions describe best-vs-best tuning, not causal attribution. X staging, useful reuse, output fanout and imbalance can move together.
The pre-existing control exclusions are inherited; raw excluded observations are not reconstructed from aggregate tables.
A 1M sibling is a held-out scale, not a new family; noisy diagonal density changes nnz scaling with N.
For an honest future test, measure the frozen shape and frozen-method 32x32 comparator with a predeclared launch-selection protocol;
score >5% labels and log errors without reselecting shapes/methods from test timings. This forecast freezes shapes, not GPU P/T/FPT settings.
Failure to beat the gate on grouped validation is a negative result, not a reason to expand the feature search on eight inputs.
'''
    files['PROPERTIES.md']=doc.encode()
    out.mkdir(parents=True,exist_ok=True)
    for name,content in files.items():
        p=out/name
        if p.exists() and p.read_bytes()!=content:
            raise SystemExit(f'Refusing to overwrite changed frozen artifact: {p}; choose --out NEW_DIRECTORY')
    for name,content in files.items():
        p=out/name
        if not p.exists(): p.write_bytes(content)
    print(markdown(comparison)); print('Selected:',selected, json.dumps(equation)); print(markdown(pd.DataFrame(folds)))
    print(markdown(top[['matrix','features','method','shape','predicted_speedup','prediction_status']]))
    print('Output:',out)

if __name__=='__main__':
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning)
        main()
