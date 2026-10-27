from pathlib import Path
import json
import numpy as np, pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.base import clone
from sklearn.metrics import log_loss, confusion_matrix
from threadpoolctl import threadpool_limits
import matplotlib.pyplot as plt
from data_features import load_features, SEED, figsave
from training import fit_search
from feature_utils import make_splits, connected_groups, tanimoto_matrix
from estimators import evaluate
from io_utils import csv, excel, json_out, copy_inputs
from paths import ROOT, step

def calibration_metrics(frame):
    records = []
    bins = []
    for (scheme, rep), g in frame.groupby(['scheme', 'repeat']):
        y = g.ML_label.to_numpy(int)
        p = g.probability.to_numpy(float)
        row = {'scheme': scheme, 'repeat': rep, **evaluate(y, p), 'LogLoss': log_loss(y, np.clip(p, 1e-15, 1 - 1e-15), labels=[0, 1])}
        for nb in [3, 4, 5]:
            ids = np.searchsorted(np.linspace(0, 1, nb + 1)[1:-1], p, side='left')
            ece = 0
            for b in range(nb):
                mask = ids == b
                if mask.any():
                    mean = p[mask].mean()
                    freq = y[mask].mean()
                    ece += mask.mean() * abs(mean - freq)
                    bins.append({'scheme': scheme, 'repeat': rep, 'n_bins': nb, 'bin': b + 1, 'n_drugs': int(mask.sum()), 'mean_score': mean, 'observed_positive_fraction': freq})
            row[f'ECE_{nb}_bins'] = ece
        records.append(row)
    return (pd.DataFrame(records), pd.DataFrame(bins))

def evaluate_primary_model():
    d = step(5)
    out = d / 'results'
    (out / 'figures').mkdir(exist_ok=True)
    names, q, y, X, cols = load_features()
    chosen = json.loads((step(4) / 'results/primary_model.json').read_text())
    f = chosen['feature']
    m = chosen['model']
    plans = json.loads((step(2) / 'results/split_plan.json').read_text())['outer']
    raw = pd.read_csv(step(4) / 'results/primary_model_oof_predictions.csv').assign(scheme='uncalibrated_fixed_model')
    nested = pd.read_csv(step(4) / 'results/nested_selection_oof_predictions.csv').assign(scheme='nested_77_candidate_selection')
    outputs = [raw, nested]
    record = []
    copy_inputs([(step(4) / 'results/primary_model.json', 'primary_model.json'), (step(4) / 'results/primary_model_oof_predictions.csv', 'primary_model_oof_predictions.csv'), (step(2) / 'results/split_plan.json', 'split_plan.json')], d / 'inputs')
    schemes = ['training_fold_sigmoid', 'exclude_infigratinib', 'Murcko_scaffold_grouping']
    if 'Targets' in f:
        schemes.append('minimum_3_training_target_carriers')
    for scheme in schemes:
        mask = np.array([n != 'infigratinib' for n in names]) if scheme == 'exclude_infigratinib' else np.ones(len(y), dtype=bool)
        nn = np.array(names)[mask]
        yy = y[mask]
        xx = X[f][mask]
        gg = connected_groups(tanimoto_matrix(X['Morgan'][mask]), 0.85)
        if scheme == 'Murcko_scaffold_grouping':
            sc = pd.read_csv(step(3) / 'results/drug_scaffolds.csv').set_index('active_name').loc[nn, 'Murcko_SMILES'].fillna('')
            gg = pd.factorize(sc)[0]
        pp = plans
        if scheme in ['exclude_infigratinib', 'Murcko_scaffold_grouping']:
            pp = []
            for rep in [1, 2, 3]:
                outer, _ = make_splits(yy, gg, 5, SEED + 500000 + rep * 10000, True)
                for fold, (tr, va) in enumerate(outer, 1):
                    inner, _ = make_splits(yy[tr], gg[tr], 3, SEED + 500000 + rep * 10000 + fold * 100, True)
                    pp.append({'repeat': rep, 'fold': fold, 'tr': tr, 'va': va, 'inner': inner})
        rows = []
        for p in pp:
            tr = np.array(p['tr'])
            va = np.array(p['va'])
            inner = [(np.array(a), np.array(b)) for a, b in p['inner']]
            fit = fit_search(f, m, SEED + (p['repeat'] - 1) * 10000 + p['fold'], cols[f], xx[tr], yy[tr], inner, min_target=3 if scheme == 'minimum_3_training_target_carriers' else 1)
            est = fit
            if scheme == 'training_fold_sigmoid':
                est = CalibratedClassifierCV(clone(fit.best_estimator_), method='sigmoid', cv=inner, ensemble=True, n_jobs=1)
                with threadpool_limits(1):
                    est.fit(xx[tr], yy[tr])
            pred = est.predict_proba(xx[va])[:, 1]
            rows.extend(({'scheme': scheme, 'repeat': p['repeat'], 'fold': p['fold'], 'active_name': nn[i], 'ML_label': int(yy[i]), 'probability': float(z)} for i, z in zip(va, pred)))
            record.append({'scheme': scheme, 'repeat': p['repeat'], 'fold': p['fold'], 'n_training_drugs': len(tr), 'n_validation_drugs': len(va), 'group_overlap': int(bool(set(gg[tr]) & set(gg[va]))), 'best_parameters': json.dumps(fit.best_params_, ensure_ascii=False), 'refit_convergence_warnings': int(getattr(fit.best_estimator_.named_steps['model'], 'convergence_warning_count_', 0))})
        outputs.append(pd.DataFrame(rows))
        print('Step 5 completed: ', scheme, flush=True)
    frame = pd.concat(outputs, ignore_index=True)
    metrics, bins = calibration_metrics(frame)
    csv(out / 'calibration_sensitivity_predictions.csv', frame)
    csv(out / 'calibration_sensitivity_metrics.csv', metrics)
    csv(out / 'calibration_bins.csv', bins)
    csv(out / 'sensitivity_settings.csv', record)
    summary = metrics.groupby('scheme').agg({c: ['mean', 'std'] for c in ['AUROC', 'AP', 'accuracy', 'F1', 'sensitivity', 'specificity', 'Brier', 'LogLoss', 'ECE_5_bins']})
    summary.columns = ['_'.join(c) for c in summary.columns]
    summary = summary.reset_index()
    csv(out / 'calibration_sensitivity_summary.csv', summary)
    excel(out / 'calibration_sensitivity.xlsx', {'metric_summary': summary, 'metrics_by_repeat': metrics, 'calibration_bins': bins, 'drug_predictions': frame, 'split_tuning_settings': record})
    colors = ['#FF0051', '#008BFB', '#8556a3']
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4))
    for rep, ax in zip([1, 2, 3], axes):
        for scheme, color in zip(['uncalibrated_fixed_model', 'training_fold_sigmoid', 'nested_77_candidate_selection'], colors):
            z = bins[(bins['scheme'] == scheme) & (bins['repeat'] == rep) & (bins['n_bins'] == 5)]
            ax.plot(z['mean_score'], z['observed_positive_fraction'], 'o-', color=color, label={'uncalibrated_fixed_model': 'Fixed candidate', 'training_fold_sigmoid': 'Train-only sigmoid', 'nested_77_candidate_selection': 'Nested selection'}[scheme])
        ax.plot([0, 1], [0, 1], '--', color='#aaa')
        ax.set(xlim=(0, 1), ylim=(0, 1), xlabel='Mean OOF score', ylabel='Observed FAERS-positive fraction', title=f'Repeat {rep}')
        ax.legend(fontsize=8)
    fig.tight_layout()
    figsave(fig, out / 'figures/calibration_curves')
    raw['predicted_label'] = (raw.probability >= 0.5).astype(int)
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
    for rep, ax in zip([1, 2, 3], axes):
        z = raw[raw['repeat'] == rep]
        cm = confusion_matrix(z.ML_label, z['predicted_label'], labels=[0, 1])
        ax.imshow(cm, cmap='Blues')
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha='center', va='center', color='white' if cm[i, j] > cm.max() / 2 else '#244969', fontsize=16)
        ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=['Ref. negative', 'Positive'], yticklabels=['Ref. negative', 'Positive'], xlabel='Predicted label', ylabel='FAERS label', title=f'Repeat {rep}')
    fig.tight_layout()
    figsave(fig, out / 'figures/primary_model_confusion_matrices')
    json_out(out / 'validation_scope.json', {'fixed_candidate': f + '__' + m, 'reselect_77_after_exclusions': False, 'sensitivity_scope': 'Sensitivity analyses retune the fixed candidate without model reselection; altered samples/splits are not paired effect tests', 'calibration': 'Sigmoid calibration is fitted within outer training data only', 'limitations': 'Drug-level internal OOF validation only; scores describe FAERS labels, not clinical incidence'})
if __name__ == '__main__':
    evaluate_primary_model()
