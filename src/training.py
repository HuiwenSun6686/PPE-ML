import os, json, time, hashlib, warnings, shutil
from pathlib import Path
import numpy as np, pandas as pd, joblib
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_curve, precision_recall_curve
from threadpoolctl import threadpool_limits
from concurrent.futures import ProcessPoolExecutor, as_completed
import matplotlib.pyplot as plt
from data_features import load_features, SEED, figsave, sha
from paths import step, ROOT
from io_utils import csv, excel, json_out, copy_inputs
from estimators import MODELS, MODEL_NAMES, build_pipeline as oldbuild, SCORING, refit_rule, pick_index, evaluate
from feature_utils import SafeKBest
FEATURE_LABELS = {'Physchem': 'Physicochemical', 'PubChem': 'PubChem', 'Morgan': 'Morgan', 'MACCS': 'MACCS', 'RDKit': 'RDKit path', 'Targets': 'Target_mechanism', 'PubChem_Targets': 'PubChem + Targets', 'Morgan_Targets': 'Morgan + Targets', 'MACCS_Targets': 'MACCS + Targets', 'RDKit_Targets': 'RDKit + Targets', 'Morgan_MACCS_RDKit_Targets': 'Morgan + MACCS + RDKit + Targets'}

class RareTargets(BaseEstimator, TransformerMixin):

    def __init__(self, min_count=1):
        self.min_count = min_count

    def fit(self, X, y=None):
        self.mask_ = np.asarray(X).sum(axis=0) >= self.min_count
        if not self.mask_.any():
            raise ValueError('No mechanism features remain after frequency filtering')
        return self

    def transform(self, X):
        return np.asarray(X)[:, self.mask_]

    def get_support(self):
        return self.mask_

def build(feature, model, seed, columns, min_target=1):
    base, grid = oldbuild('physchem', model, seed)
    if model == 'MLP':
        base.named_steps['model'].set_params(max_iter=10000, max_fun=200000)
    scaler = StandardScaler() if model in ['KNN', 'MLP'] or (feature == 'Physchem' and model == 'LR') else 'passthrough'
    if feature == 'Physchem':
        pre = Pipeline([('impute', SimpleImputer(strategy='median', keep_empty_features=True)), ('variance', VarianceThreshold())])
    elif feature == 'Targets':
        pre = Pipeline([('rare', RareTargets(min_target)), ('variance', VarianceThreshold())])
    else:
        nf = sum((c.startswith(('PUB', 'MORGAN', 'MACCS', 'RDK')) for c in columns))
        fp = Pipeline([('variance', VarianceThreshold()), ('select', SafeKBest())])
        grid['pre__fingerprint__select__k'] = [16, 32]
        transformers = [('fingerprint', fp, list(range(nf)))]
        if nf < len(columns):
            transformers.append(('targets', Pipeline([('rare', RareTargets(min_target)), ('variance', VarianceThreshold())]), list(range(nf, len(columns)))))
        pre = ColumnTransformer(transformers, sparse_threshold=0, verbose_feature_names_out=False)
    return (Pipeline([('pre', pre), ('scale', scaler), ('model', base.named_steps['model'])]), grid)

def selected_columns(pipe, columns):
    pre = pipe.named_steps['pre']
    columns = np.array(columns, dtype=object)
    if isinstance(pre, ColumnTransformer):
        out = []
        for name, p, idx in pre.transformers_:
            z = columns[idx]
            if 'rare' in p.named_steps:
                z = z[p.named_steps['rare'].get_support()]
            z = z[p.named_steps['variance'].get_support()]
            if 'select' in p.named_steps:
                z = z[p.named_steps['select'].get_support()]
            out.extend(z.tolist())
        return out
    z = columns
    if 'rare' in pre.named_steps:
        z = z[pre.named_steps['rare'].get_support()]
    return z[pre.named_steps['variance'].get_support()].tolist()

def fit_search(feature, model, seed, columns, X, y, splits, min_target=1):
    pipe, grid = build(feature, model, seed, columns, min_target)
    search = GridSearchCV(pipe, grid, scoring=SCORING, refit=refit_rule, cv=splits, n_jobs=1, error_score='raise', return_train_score=False)
    with threadpool_limits(1):
        search.fit(X, y)
    return search

def worker(task):
    feature, model, plan, signature = task
    names, q, y, X, cols = load_features()
    d = step(4) / 'results'
    tag = f'{feature}__{model}__' + ('full' if plan is None else f"r{plan['repeat']}f{plan['fold']}")
    dest = d / 'tuning_checkpoints' / f'{tag}.json'
    if dest.exists():
        old = json.loads(dest.read_text())
        assert old['signature'] == signature, "Inputs or code changed; remove this stage's old results before rerunning"
        return (tag, False)
    if plan is None:
        tr = np.arange(len(y))
        va = None
        full = json.loads((step(2) / 'results/split_plan.json').read_text())
        splits = full['full']
        seed = SEED + 900000
    else:
        tr = np.array(plan['tr'])
        va = np.array(plan['va'])
        splits = plan['inner']
        seed = SEED + (plan['repeat'] - 1) * 10000 + plan['fold']
    splits = [(np.array(a), np.array(b)) for a, b in splits]
    fit = fit_search(feature, model, seed, cols[feature], X[feature][tr], y[tr], splits)
    best = int(fit.best_index_)
    r = fit.cv_results_
    est = fit.best_estimator_
    result = {'signature': signature, 'feature': feature, 'model': model, 'repeat': 0 if plan is None else plan['repeat'], 'fold': 0 if plan is None else plan['fold'], 'best_params': fit.best_params_, 'inner_AUROC': float(r['mean_test_AUROC'][best]), 'inner_AP': float(r['mean_test_AP'][best]), 'inner_SD': float(r['std_test_AUROC'][best]), 'selected_fit_convergence': int(getattr(est.named_steps['model'], 'convergence_warning_count_', 0)), 'selected_grid_convergence': float(r['mean_test_convergence'][best]), 'all_grid_convergence': float(sum(r['mean_test_convergence'])), 'selected_columns': selected_columns(est, cols[feature]), 'grid': [{'params': r['params'][i], 'AUROC': float(r['mean_test_AUROC'][i]), 'AP': float(r['mean_test_AP'][i]), 'SD_AUROC': float(r['std_test_AUROC'][i]), 'mean_convergence_warnings': float(r['mean_test_convergence'][i])} for i in range(len(r['params']))]}
    result['train_metrics'] = evaluate(y[tr], est.predict_proba(X[feature][tr])[:, 1])
    if va is not None:
        p = est.predict_proba(X[feature][va])[:, 1]
        result['validation_metrics'] = evaluate(y[va], p)
        result['OOF'] = [{'active_name': names[i], 'ML_label': int(y[i]), 'probability': float(z), 'repeat': plan['repeat'], 'fold': plan['fold'], 'feature': feature, 'model': model, 'drug_class': q.drug_class.iloc[i]} for i, z in zip(va, p)]
    else:
        joblib.dump(est, d / 'development_models' / f'{feature}__{model}.joblib')
    tmp = dest.with_suffix('.tmp')
    json_out(tmp, result)
    tmp.replace(dest)
    return (tag, True)

def train_models(workers=None):
    d = step(4)
    out = d / 'results'
    (out / 'tuning_checkpoints').mkdir(exist_ok=True)
    (out / 'development_models').mkdir(exist_ok=True)
    copy_inputs([(step(2) / 'results/split_plan.json', 'split_plan.json')], d / 'inputs')
    for p in (step(1) / 'results').glob('*_matrix.csv'):
        copy_inputs([(p, p.name)], d / 'inputs')
    copy_inputs([(step(1) / 'results/drug_labels.csv', 'drug_labels.csv')], d / 'inputs')
    names, q, y, X, cols = load_features()
    plan = json.loads((step(2) / 'results/split_plan.json').read_text())
    candidates = [(f, m) for f in X for m in MODELS]
    sig = hashlib.sha256(''.join((sha(p) for p in [Path(__file__), ROOT / 'src/estimators.py', ROOT / 'src/model_extensions.py', step(2) / 'results/split_plan.json', *[p for p in sorted((step(1) / 'results').glob('*_matrix.csv'))]])).encode()).hexdigest()
    json_out(out / 'selection_rules.json', {'n_candidates': len(candidates), 'features': list(X), 'algorithms': list(MODELS), 'candidate_order': candidates, 'selection': 'Use inner AUROC, then AP and prespecified order; select the final candidate by separate development CV, never outer performance', 'outer': '5 folds x 3 repeats; structural groups do not cross training/validation', 'feature_selection': 'Training-only chi2 fingerprint selection (k=16/32), target frequency/variance filtering; no chi2 for continuous descriptors', 'decision_threshold': 0.5, 'calibration': 'Original scores are assessed without fitting calibration on the evaluation predictions', 'signature': sig})
    tasks = [(f, m, p, sig) for f, m in candidates for p in plan['outer']] + [(f, m, None, sig) for f, m in candidates]
    workers = workers or int(os.environ.get('PPE_WORKERS', '6'))
    t0 = time.time()
    done = 0
    print(f'Step 4 started: {len(candidates)} candidates; {len(tasks)} tuning tasks; {workers} worker processes.', flush=True)
    if workers == 1:
        for task in tasks:
            tag, new = worker(task)
            done += 1
            if done % 20 == 0:
                print(f'{done}/{len(tasks)} completed; elapsed {(time.time() - t0) / 60:.1f} minutes', flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futs = [pool.submit(worker, t) for t in tasks]
            for fut in as_completed(futs):
                tag, new = fut.result()
                done += 1
                if done % 20 == 0:
                    print(f'{done}/{len(tasks)} completed; elapsed {(time.time() - t0) / 60:.1f} minutes', flush=True)
    results = [json.loads(p.read_text()) for p in (out / 'tuning_checkpoints').glob('*.json')]
    assert len(results) == len(tasks)
    outer = [r for r in results if r['repeat']]
    full = [next((r for r in results if r['repeat'] == 0 and r['feature'] == f and (r['model'] == m))) for f, m in candidates]
    best = pick_index([r['inner_AUROC'] for r in full], [r['inner_AP'] for r in full])
    chosen = full[best]
    json_out(out / 'primary_model.json', {**chosen, 'candidate_count': len(candidates), 'final_selection_uses_outer_scores': False, 'selection_source': 'Prespecified 3-fold x 3-repeat development CV; not independent validation performance', 'model_path': f"development_models/{chosen['feature']}__{chosen['model']}.joblib"})
    all_oof = pd.DataFrame([row for r in outer for row in r['OOF']])
    csv(out / 'all_candidate_oof_predictions.csv', all_oof)
    repeated = []
    summary = []
    for f, m in candidates:
        rows = all_oof[(all_oof.feature == f) & (all_oof.model == m)]
        assert len(rows) == 3 * len(y)
        for rep, g in rows.groupby('repeat'):
            assert len(g) == len(y) and g.active_name.nunique() == len(y)
            repeated.append({'feature': f, 'model': m, 'repeat': rep, **evaluate(g.ML_label, g.probability)})
    rm = pd.DataFrame(repeated)
    csv(out / 'candidate_metrics_by_repeat.csv', rm)
    for f, m in candidates:
        z = rm[(rm.feature == f) & (rm.model == m)]
        fr = next((r for r in full if r['feature'] == f and r['model'] == m))
        record = {'feature_set_label': FEATURE_LABELS[f], 'feature': f, 'model': m, 'algorithm_label': MODEL_NAMES[m], 'primary_model': int(f == chosen['feature'] and m == chosen['model']), 'development_CV_AUROC': fr['inner_AUROC'], 'development_CV_AP': fr['inner_AP'], 'development_best_parameters': json.dumps(fr['best_params'], ensure_ascii=False)}
        for metric in ['AUROC', 'AP', 'accuracy', 'F1', 'Brier', 'sensitivity', 'specificity', 'balanced_accuracy']:
            record[metric + '_mean'] = z[metric].mean()
            record[metric + '_repeat_SD'] = z[metric].std(ddof=1)
        record['outer_mean_training_accuracy'] = np.mean([r['train_metrics']['accuracy'] for r in outer if r['feature'] == f and r['model'] == m])
        summary.append(record)
    summary = pd.DataFrame(summary)
    csv(out / 'candidate_performance.csv', summary)
    select_oof = []
    selection = []
    for p in plan['outer']:
        subset = [next((r for r in outer if r['feature'] == f and r['model'] == m and (r['repeat'] == p['repeat']) and (r['fold'] == p['fold']))) for f, m in candidates]
        k = pick_index([r['inner_AUROC'] for r in subset], [r['inner_AP'] for r in subset])
        r = subset[k]
        selection.append({'repeat': p['repeat'], 'fold': p['fold'], 'selected_feature': r['feature'], 'selected_model': r['model'], 'inner_AUROC': r['inner_AUROC'], 'params': json.dumps(r['best_params'], ensure_ascii=False), **r['validation_metrics']})
        select_oof.extend(r['OOF'])
    so = pd.DataFrame(select_oof)
    sm = pd.DataFrame([{'repeat': rep, **evaluate(g.ML_label, g.probability)} for rep, g in so.groupby('repeat')])
    csv(out / 'nested_selection_oof_predictions.csv', so)
    csv(out / 'nested_selection_metrics.csv', sm)
    csv(out / 'outer_model_selection.csv', selection)
    main = all_oof[(all_oof.feature == chosen['feature']) & (all_oof.model == chosen['model'])].sort_values(['repeat', 'active_name'])
    csv(out / 'primary_model_oof_predictions.csv', main)
    wrong = main.assign(predicted_label=lambda z: (z.probability >= 0.5).astype(int))
    wrong['classification'] = np.where(wrong.ML_label == wrong['predicted_label'], 'correct', np.where(wrong.ML_label == 1, 'false_negative', 'false_positive'))
    csv(out / 'primary_model_classifications.csv', wrong)
    errors = []
    for n, g in wrong.groupby('active_name'):
        errors.append({'active_name': n, 'ML_label': int(g.ML_label.iloc[0]), 'drug_class': g.drug_class.iloc[0], 'mean_oof_score': g.probability.mean(), 'oof_score_SD': g.probability.std(ddof=1), 'minimum_oof_score': g.probability.min(), 'maximum_oof_score': g.probability.max(), 'false_negative_count': int((g['classification'] == 'false_negative').sum()), 'false_positive_count': int((g['classification'] == 'false_positive').sum()), 'correct_count': int((g['classification'] == 'correct').sum())})
    errors = pd.DataFrame(errors)
    csv(out / 'drug_prediction_stability.csv', errors)
    params = [{'feature': r['feature'], 'model': r['model'], 'repeat': r['repeat'], 'fold': r['fold'], 'inner_AUROC': r['inner_AUROC'], 'inner_AP': r['inner_AP'], 'parameters': json.dumps(r['best_params'], ensure_ascii=False), 'selected_feature_count': len(r['selected_columns']), 'selected_grid_mean_convergence_warnings': r['selected_grid_convergence'], 'refit_convergence_warnings': r['selected_fit_convergence'], 'training_accuracy': r['train_metrics']['accuracy'], 'validation_accuracy': r.get('validation_metrics', {}).get('accuracy', np.nan)} for r in results]
    csv(out / 'selected_parameters.csv', params)
    grid = [{'feature': r['feature'], 'model': r['model'], 'repeat': r['repeat'], 'fold': r['fold'], 'parameters': json.dumps(g['params'], ensure_ascii=False), **{k: v for k, v in g.items() if k != 'params'}} for r in results for g in r['grid']]
    csv(out / 'grid_search_results.csv', grid)
    excel(out / 'model_performance.xlsx', {'candidate_comparison': summary, 'nested_selection_validation': sm, 'primary_model_by_repeat': rm[(rm.feature == chosen['feature']) & (rm.model == chosen['model'])], 'primary_model_errors': errors, 'parameters_convergence': params, 'outer_fold_selections': selection})
    plots(out, summary, rm, all_oof, so, chosen)
    z = rm[(rm.feature == chosen['feature']) & (rm.model == chosen['model'])]
    print('Step 4 completed; final candidate ', chosen['feature'], chosen['model'], 'development CV ', chosen['inner_AUROC'], flush=True)

def plots(out, summary, rm, oof, so, chosen):
    from matplotlib.colors import LinearSegmentedColormap
    p = out / 'figures'
    p.mkdir(exist_ok=True)
    features = list(FEATURE_LABELS)
    cmap = LinearSegmentedColormap.from_list('model', ['#e6f3ff', '#008BFB', '#ab397e', '#FF0051'])
    for metric in ['AUROC', 'F1']:
        arr = np.array([[summary.query('feature==@f and model==@m')[metric + '_mean'].iloc[0] for m in MODELS] for f in features])
        fig, ax = plt.subplots(figsize=(10.5, 7))
        im = ax.imshow(arr, cmap=cmap, vmin=0.3, vmax=1)
        ax.set_xticks(range(7), MODELS)
        ax.set_yticks(range(len(features)), [f.replace('_', ' + ') for f in features], fontsize=9)
        for i in range(len(features)):
            for j in range(7):
                ax.text(j, i, f'{arr[i, j]:.3f}', ha='center', va='center', color='white' if arr[i, j] > 0.72 else '#177652', fontsize=10)
        ax.set_title(f'77 candidates: repeated grouped OOF {metric}')
        fig.colorbar(im, ax=ax, label=metric)
        fig.tight_layout()
        figsave(fig, p / f'77_candidates_{metric}heatmap')
    colors = ['#008BFB', '#FF0051', '#7852a0', '#38a3a5', '#e0882e', '#3f6580', '#c87828']
    for f in features:
        fig, ax = plt.subplots(figsize=(7.2, 6))
        for m, c in zip(MODELS, colors):
            g = oof.query('feature==@f and model==@m and repeat==1')
            fp, tp, _ = roc_curve(g.ML_label, g.probability)
            z = rm.query('feature==@f and model==@m')
            ax.plot(fp, tp, color=c, lw=1.7, label=f'{m}: {z.AUROC.mean():.3f} ± {z.AUROC.std():.3f}')
        ax.plot([0, 1], [0, 1], color='#aaaaaa', ls='--')
        ax.set(xlim=(0, 1), ylim=(0, 1.02), xlabel='False-positive rate', ylabel='True-positive rate', title=f"{f.replace('_', ' + ')}: seven algorithms")
        ax.legend(title='Mean ± SD AUROC (3 repeats)\nCurves: pre-fixed repeat 1', fontsize=8, loc='lower right')
        fig.tight_layout()
        figsave(fig, p / (f + 'algorithm_ROC'))
    for title, z in [('primary_model', oof.query('feature==@chosen_feature and model==@chosen_model', local_dict={'chosen_feature': chosen['feature'], 'chosen_model': chosen['model']})), ('nested_selection', so)]:
        for kind in ['ROC', 'PR']:
            fig, ax = plt.subplots(figsize=(6.8, 5.5))
            for rep, g in z.groupby('repeat'):
                met = evaluate(g.ML_label, g.probability)
                if kind == 'ROC':
                    a, b, _ = roc_curve(g.ML_label, g.probability)
                    label = f"Repeat {rep}: AUROC {met['AUROC']:.3f}"
                else:
                    b, a, _ = precision_recall_curve(g.ML_label, g.probability)
                    label = f"Repeat {rep}: AP {met['AP']:.3f}"
                ax.plot(a, b, label=label, color=colors[rep - 1])
            if kind == 'ROC':
                ax.plot([0, 1], [0, 1], ls='--', color='#aaaaaa')
                ax.set(xlabel='False-positive rate', ylabel='True-positive rate')
            else:
                ax.axhline(29 / 82, ls='--', color='#aaaaaa')
                ax.set(xlabel='Recall', ylabel='Precision')
            ax.set(xlim=(0, 1), ylim=(0, 1.02), title=('Final candidate' if title == 'primary_model' else 'Nested selection workflow') + ' ' + kind)
            ax.legend(fontsize=9)
            fig.tight_layout()
            figsave(fig, p / (title + 'three_repeats' + kind))
