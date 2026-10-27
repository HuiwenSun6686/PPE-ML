import os, json, hashlib, warnings
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np, pandas as pd, joblib, shap
from shap.utils._legacy import DenseData
from scipy.stats import spearmanr
from sklearn.metrics import log_loss, brier_score_loss, confusion_matrix
from threadpoolctl import threadpool_limits
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator, MACCSkeys, Draw
from rdkit.Chem.Scaffolds import MurckoScaffold
import matplotlib.pyplot as plt
from data_features import load_features, SEED, figsave, read, fingerprints
from training import build, selected_columns, fit_search, evaluate, FEATURE_LABELS
from feature_utils import make_splits, tanimoto_matrix, connected_groups
from paths import step, ROOT
from io_utils import csv, excel, json_out, copy_inputs
NSAMPLES = 1024

def shap_worker(plan):
    names, q, y, X, cols = load_features()
    main = json.loads((step(4) / 'results/primary_model.json').read_text())
    f, m = (main['feature'], main['model'])
    r, fold = (plan['repeat'], plan['fold'])
    out = step(5) / 'results'
    cache = out / 'shap_checkpoints' / f'r{r}f{fold}.json'
    signature = hashlib.sha256(json.dumps({'training_signature': main['signature'], 'plan': plan, 'nsamples': NSAMPLES, 'method_version': '82-grouped-probability-2seeds-v1'}, sort_keys=True).encode()).hexdigest()
    if cache.exists():
        assert json.loads(cache.read_text())['signature'] == signature, 'SHAP inputs or settings changed; remove old SHAP checkpoints before rerunning'
        return (r, fold, True)
    tr, va = (np.array(plan['tr']), np.array(plan['va']))
    seed = SEED + (r - 1) * 10000 + fold
    chosen = json.loads((step(4) / 'results/tuning_checkpoints' / f'{f}__{m}__r{r}f{fold}.json').read_text())
    pipe, _ = build(f, m, seed, cols[f])
    pipe.set_params(**chosen['best_params'])
    with threadpool_limits(1):
        pipe.fit(X[f][tr], y[tr])
    p = pipe.predict_proba(X[f][va])[:, 1]
    expected = {z['active_name']: z['probability'] for z in chosen['OOF']}
    assert max((abs(p[j] - expected[names[i]]) for j, i in enumerate(va))) < 1e-10
    joblib.dump(pipe, out / 'outer_models' / f'primary_model_repeat{r}_fold{fold}.joblib')
    columns = selected_columns(pipe, cols[f])
    background = pipe.named_steps['pre'].transform(X[f][tr])
    test = pipe.named_steps['pre'].transform(X[f][va])
    bg = np.asarray(background, float)
    test = np.asarray(test, float)
    groups = []
    patterns = {}
    for j in range(bg.shape[1]):
        pattern = np.ascontiguousarray(bg[:, j]).tobytes()
        if pattern not in patterns:
            patterns[pattern] = len(groups)
            groups.append([])
        groups[patterns[pattern]].append(j)
    defs = []
    gnames = []
    for g in groups:
        members = [columns[j] for j in g]
        gid = 'G' + hashlib.sha256('|'.join(sorted(members)).encode()).hexdigest()[:12]
        gnames.append(gid)
        defs.append({'group_id': gid, 'members': members, 'n_members': len(g), 'binary': bool(np.isin(bg[:, g], [0, 1]).all())})

    def predict(z):
        scaler = pipe.named_steps['scale']
        a = z if isinstance(scaler, str) else scaler.transform(z)
        return pipe.named_steps['model'].predict_proba(a)[:, 1]
    values = []
    bases = []
    with threadpool_limits(1):
        for s in [SEED + 700000 + r * 100 + fold, SEED + 800000 + r * 100 + fold]:
            np.random.seed(s)
            explainer = shap.KernelExplainer(predict, DenseData(bg, gnames, groups), link='identity')
            sv = np.asarray(explainer.shap_values(test, nsamples=NSAMPLES, l1_reg=0, silent=True))
            values.append(sv)
            bases.append(float(explainer.expected_value))
    assert abs(bases[0] - bases[1]) < 1e-10
    v = np.mean(values, axis=0)
    error = float(np.max(abs(bases[0] + v.sum(axis=1) - p)))
    assert error < 1e-07, error
    rows = []
    for j, i in enumerate(va):
        for k, (gd, g) in enumerate(zip(defs, groups)):
            rows.append({'active_name': names[i], 'ML_label': int(y[i]), 'repeat': r, 'fold': fold, 'group_id': gd['group_id'], 'members': '; '.join(gd['members']), 'n_members': gd['n_members'], 'binary': gd['binary'], 'feature_value': float(test[j, g[0]]), 'SHAP': float(v[j, k]), 'SHAP_seed1': float(values[0][j, k]), 'SHAP_seed2': float(values[1][j, k]), 'base_value': bases[0], 'probability': float(p[j])})
    result = {'signature': signature, 'repeat': r, 'fold': fold, 'definitions': defs, 'rows': rows, 'additivity_max_error': error, 'n_transformed_features': len(columns), 'n_groups': len(groups), 'max_seed_difference': float(np.max(abs(values[0] - values[1]))), 'mean_seed_difference': float(np.mean(abs(values[0] - values[1]))), 'selected_fit_convergence': int(getattr(pipe.named_steps['model'], 'convergence_warning_count_', 0))}
    json_out(cache, result)
    return (r, fold, False)

def calculate_shap():
    import re
    d = step(5)
    out = d / 'results'
    for sub in ['outer_models', 'shap_checkpoints', 'figures']:
        (out / sub).mkdir(exist_ok=True)
    copy_inputs([(step(4) / 'results' / name, name) for name in ['primary_model.json', 'primary_model_oof_predictions.csv', 'drug_prediction_stability.csv']], d / 'inputs')
    plan = json.loads((step(2) / 'results/split_plan.json').read_text())
    with ProcessPoolExecutor(max_workers=int(os.environ.get('PPE_SHAP_WORKERS', '4'))) as pool:
        for fut in as_completed([pool.submit(shap_worker, p) for p in plan['outer']]):
            print('Step 5 OOF SHAP completed: ', fut.result(), flush=True)
    checks = [json.loads(p.read_text()) for p in sorted((out / 'shap_checkpoints').glob('*.json'))]
    assert len(checks) == 15
    names, lab, y, XX, cm = load_features()
    chosen = json.loads((step(4) / 'results/primary_model.json').read_text())
    f = chosen['feature']
    ni = {n: i for i, n in enumerate(names)}
    ci = {c: i for i, c in enumerate(cm[f])}
    for c in checks:
        for rr in c['rows']:
            values = XX[f][ni[rr['active_name']], [ci[a] for a in rr['members'].split('; ')]]
            rr['feature_value'] = float(values.mean())
            rr['group_color_definition'] = 'Mean presence of binary group members'
    rows = pd.DataFrame([r for c in checks for r in c['rows']])
    csv(out / 'oof_shap_values.csv', rows)
    qc = pd.DataFrame([{k: v for k, v in c.items() if k not in ['rows', 'definitions']} for c in checks])
    csv(out / 'shap_computation_summary.csv', qc)
    importance = []
    for gid, g in rows.groupby('group_id'):
        directions = []
        for _, z in g.groupby(['repeat', 'fold']):
            v = z[z.feature_value > 0]
            if len(v):
                directions.append(float(v.SHAP.mean()))
        consistency = max(sum((a > 0 for a in directions)), sum((a < 0 for a in directions))) / len(directions) if directions else 0
        importance.append({'group_id': gid, 'feature_members': g.members.iloc[0], 'n_members': int(g.n_members.iloc[0]), 'binary': bool(g.binary.iloc[0]), 'n_outer_models': g[['repeat', 'fold']].drop_duplicates().shape[0], 'n_covered_drugs': g.active_name.nunique(), 'n_positive_value_drugs': g[g.feature_value > 0].active_name.nunique(), 'mean_absolute_SHAP_retained': g.SHAP.abs().mean(), 'n_directional_folds': len(directions), 'direction_consistency': consistency, 'dominant_direction': ('increase_score' if sum(directions) > 0 else 'decrease_score') if directions else 'undetermined'})
    imp = pd.DataFrame(importance).sort_values('mean_absolute_SHAP_retained', ascending=False)
    rank = []
    for (r, f), g in rows.groupby(['repeat', 'fold']):
        for gid in g.groupby('group_id').SHAP.apply(lambda v: v.abs().mean()).sort_values(ascending=False).head(10).index:
            rank.append({'group_id': gid, 'repeat': r, 'fold': f})
    rk = pd.DataFrame(rank).groupby('group_id').size()
    imp['top10_count'] = imp.group_id.map(rk).fillna(0).astype(int)
    imp['meets_stability_criteria'] = (imp['n_outer_models'] >= 12) & (imp['top10_count'] >= 5) & (imp['n_positive_value_drugs'] >= 3) & (imp['n_directional_folds'] >= 3) & (imp['direction_consistency'] >= 0.8)
    text = (d / 'inputs/pubchem_fingerprint_definition.txt').read_text()
    spec = {}
    for line in text.splitlines():
        mt = re.match('^\\s*(\\d{1,3})\\s+(.+?)\\s*$', line)
        if mt and int(mt.group(1)) <= 880:
            spec[f'PUB{int(mt.group(1)):03d}'] = mt.group(2)
    assert len(spec) == 881, (len(spec), 'Incomplete official fingerprint bit definitions')
    csv(out / 'pubchem_bit_definitions.csv', [{'feature_name': c, 'official_definition': v, 'position_origin': 'zero_based', 'source': 'PubChem Substructure Fingerprint V1.3 (frozen specification)'} for c, v in spec.items()])
    definitions = read(1, 'target_feature_dictionary.csv').set_index('feature_name')
    explain = []
    for _, r in imp.iterrows():
        members = r['feature_members'].split('; ')
        desc = []
        for c in members:
            if c in spec:
                desc.append(c + ': ' + spec[c])
            elif c in definitions.index:
                desc.append(c + ': ' + str(definitions.loc[c].to_dict()))
        explain.append({**r.to_dict(), 'feature_definitions': ' | '.join(desc), 'interpretation_scope': 'Joint feature-group association; member effects cannot be separated', 'mechanistic_evidence_scope': 'Target annotations are model inputs, not independent mechanistic validation', 'causal_scope': 'SHAP explains model scores and does not establish causal effects on PPE'})
    csv(out / 'shap_importance_stability.csv', imp)
    csv(out / 'shap_feature_interpretation.csv', explain)
    excel(out / 'shap_summary.xlsx', {'importance_stability': imp, 'shap_computation_summary': qc, 'feature_interpretation': explain, 'pubchem_bit_definitions': pd.DataFrame([{'feature_name': c, 'official_definition': v} for c, v in spec.items()])})
    assert rows.groupby(['repeat', 'active_name']).ngroups == 82 * 3
    exported = pd.read_csv(out / 'oof_shap_values.csv')
    assert exported.groupby(['repeat', 'active_name']).ngroups == 82 * 3
    hold = exported.groupby(['repeat', 'active_name']).agg(base=('base_value', 'first'), total=('SHAP', 'sum'), score=('probability', 'first'))
    assert max(abs(hold.base + hold.total - hold.score)) < 1e-07
    print('Step 5 SHAP completed: ', len(rows), ' rows; stable groups ', int(imp['meets_stability_criteria'].sum()), flush=True)
if __name__ == '__main__':
    calculate_shap()
