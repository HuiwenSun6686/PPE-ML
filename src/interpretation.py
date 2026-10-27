import json, os, warnings, hashlib
from pathlib import Path
import numpy as np, pandas as pd, joblib, shap
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
from rdkit import Chem
from data_features import load_features, read, figsave, fingerprints, descriptors, PHYS, FP_NAMES
from training import FEATURE_LABELS
from feature_utils import tanimoto_matrix
from paths import step
from io_utils import csv, excel, json_out, copy_inputs
BLUE = '#008BFB'
PINK = '#FF0051'
CMAP = LinearSegmentedColormap.from_list('shap_color', [BLUE, PINK])

def short_label(members, maxlen=48):
    a = members.split('; ')
    s = ' / '.join(a[:3])
    if len(a) > 3:
        s += f' (+{len(a) - 3})'
    return s if len(s) <= maxlen else s[:maxlen - 3] + '...'

def plot_shap():
    dependencies = [step(5) / 'results/oof_shap_values.csv', step(5) / 'results/shap_importance_stability.csv', step(4) / 'results/drug_prediction_stability.csv', step(4) / 'results/primary_model.json', Path(__file__)]
    signature = hashlib.sha256(''.join((hashlib.sha256(a.read_bytes()).hexdigest() for a in dependencies)).encode()).hexdigest()
    cache = step(6) / 'results/shap_plot_cache.json'
    if cache.exists():
        old = json.loads(cache.read_text())
        if old['signature'] == signature and all(((step(6) / 'results' / f).exists() for f in old['output_files'])):
            print('Existing SHAP plots match current inputs and code; remove the plot-cache file to redraw.', flush=True)
            return
    d = step(6)
    out = d / 'results'
    p = out / 'figures'
    p.mkdir(exist_ok=True)
    for file in ['oof_shap_values.csv', 'shap_importance_stability.csv']:
        copy_inputs([(step(5) / 'results' / file, file)], d / 'inputs')
    rows = read(5, 'oof_shap_values.csv')
    imp = read(5, 'shap_importance_stability.csv')
    main = json.loads((step(4) / 'results/primary_model.json').read_text())
    errors = read(4, 'drug_prediction_stability.csv')
    avg = rows.groupby(['active_name', 'ML_label', 'group_id', 'members', 'binary'], as_index=False).agg(SHAP=('SHAP', 'mean'), feature_value=('feature_value', 'mean'), n_repeats=('repeat', 'nunique'))
    avg = avg[avg.n_repeats >= 2]
    csv(out / 'mean_shap_by_drug.csv', avg)
    eligible = imp[imp['n_outer_models'] >= 12].group_id.tolist()
    eligible += imp[~imp.group_id.isin(eligible)].sort_values(['n_outer_models', 'mean_absolute_SHAP_retained'], ascending=False).group_id.tolist()
    eligible = [g for g in eligible if g in set(avg.group_id)][:15]
    ranking = avg[avg.group_id.isin(eligible)].groupby('group_id').SHAP.apply(lambda x: x.abs().mean()).sort_values(ascending=False)
    gids = ranking.index.tolist()
    texts = {r.group_id: short_label(r['feature_members']) for _, r in imp.iterrows()}
    fig, ax = plt.subplots(figsize=(10, max(4.5, 0.42 * len(gids) + 1.5)))
    idx = np.arange(len(gids))
    ax.barh(idx[::-1], ranking.values, color=BLUE)
    ax.set_yticks(idx[::-1], [texts[g] for g in gids], fontsize=9)
    ax.set(xlabel='Mean absolute cross-repeat OOF SHAP', title=f"{main['feature'].replace('_', ' + ')} + {main['model']}: global explanation")
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    figsave(fig, p / 'global_shap_importance')
    fig, ax = plt.subplots(figsize=(10, max(4.5, 0.42 * len(gids) + 1.5)))
    rng = np.random.default_rng(20261006)
    for j, gid in enumerate(gids):
        z = avg[avg.group_id == gid]
        v = z.feature_value.to_numpy(float)
        v = (v - v.min()) / (v.max() - v.min()) if v.max() > v.min() else np.repeat(0.5, len(v))
        ax.scatter(z.SHAP, len(gids) - 1 - j + rng.uniform(-0.2, 0.2, len(z)), c=v, cmap=CMAP, vmin=0, vmax=1, s=20, alpha=0.8, linewidths=0)
    ax.axvline(0, color='#aaa', lw=0.8)
    ax.set_yticks(idx[::-1], [texts[g] for g in gids], fontsize=9)
    ax.set(xlabel='Mean OOF SHAP across available repeats', title='Cross-repeat SHAP summary (each dot = one drug)')
    bar = fig.colorbar(ScalarMappable(norm=Normalize(0, 1), cmap=CMAP), ax=ax, pad=0.03)
    bar.set_ticks([0, 1], labels=['Low', 'High'])
    bar.set_label('Member presence (scaled within each group)')
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    figsave(fig, p / 'global_shap_beeswarm')
    chosen = ['ripretinib']
    for n in errors[(errors['false_negative_count'] == 3) | (errors['false_positive_count'] == 3)].active_name:
        if n not in chosen:
            chosen.append(n)
    for label, errorfield, sort in [(1, 'false_negative_count', 'mean_oof_score'), (0, 'false_positive_count', 'mean_oof_score')]:
        z = errors[errors.ML_label == label].sort_values([errorfield, sort], ascending=[False, label == 1])
        n = z.active_name.iloc[0]
        if n not in chosen:
            chosen.append(n)
    for label in [1, 0]:
        z = errors[(errors.ML_label == label) & (errors['correct_count'] == 3)].sort_values('mean_oof_score', ascending=label == 0)
        if len(z) and z.active_name.iloc[0] not in chosen:
            chosen.append(z.active_name.iloc[0])
    local = []
    for n in chosen:
        for rep, g in rows[rows.active_name == n].groupby('repeat'):
            g = g.sort_values('SHAP', key=lambda x: x.abs(), ascending=False)
            base = float(g.base_value.iloc[0])
            pred = float(g.probability.iloc[0])
            yy = int(g.ML_label.iloc[0])
            vals = g.SHAP.to_numpy()
            labels = [short_label(a, 38) for a in g.members]
            counts = len(vals)
            top = min(10, counts)
            v = list(vals[:top])
            lab = labels[:top]
            if counts > top:
                v.append(float(vals[top:].sum()))
                lab.append(f'Other {counts - top} feature groups')
            order = np.argsort(np.abs(v))[::-1]
            v = np.array(v)[order]
            lab = list(np.array(lab)[order])
            current = base
            fig, ax = plt.subplots(figsize=(10.5, max(4, 0.44 * len(v) + 2)))
            for j, (delta, text) in enumerate(zip(v, lab)):
                ypos = len(v) - 1 - j
                ax.barh(ypos, abs(delta), left=min(current, current + delta), height=0.62, color=PINK if delta >= 0 else BLUE)
                ax.text(current + delta / 2, ypos, f'{delta:+.3f}', va='center', ha='center', fontsize=8, color='white', fontweight='bold')
                current += delta
            assert abs(current - pred) < 1e-07
            ax.axvline(base, color='#777', ls='--', lw=1)
            ax.axvline(pred, color='#222', ls=':', lw=1)
            ax.set_yticks(range(len(v) - 1, -1, -1), lab, fontsize=8)
            ax.set(xlabel='FAERS-positive model score', title=f'{n} | OOF repeat {rep} | label={yy}\nbase {base:.3f} → score {pred:.3f}')
            ax.spines[['top', 'right']].set_visible(False)
            fig.tight_layout()
            figsave(fig, p / f'{n}_repeat{rep}_SHAP_waterfall')
            shap.force_plot(base, vals, features=g.feature_value.to_numpy(), feature_names=labels, matplotlib=True, show=False, figsize=(16, 4), plot_cmap=[PINK, BLUE], contribution_threshold=0.07)
            fig = plt.gcf()
            fig.text(0.02, 0.02, f'{n}: OOF repeat {rep}; label={yy}; score={pred:.4f}', fontsize=12, ha='left')
            figsave(fig, p / f'{n}_repeat{rep}_SHAP_force')
            local.append({'active_name': n, 'repeat': rep, 'fold': int(g.fold.iloc[0]), 'ML_label': yy, 'base': base, 'score': pred, 'sum_SHAP': float(vals.sum()), 'explanation_type': 'Repeat 1 local explanation; repeats 2 and 3 are separate explanations'})
    csv(out / 'local_shap_index.csv', local)
    display = imp.set_index('group_id').loc[gids].reset_index()
    csv(out / 'global_shap_groups.csv', display)
    excel(out / 'shap_plot_data.xlsx', {'global_plot_features': display, 'local_plot_index': local, 'mean_shap_by_drug': avg})
    files = [str(a.relative_to(out)) for a in out.rglob('*') if a.is_file() and 'temporary' not in a.name and ('inspect' not in a.name) and ('workbook_input' not in a.name) and ('table_preview' not in str(a)) and (a.name != cache.name)]
    json_out(cache, {'signature': signature, 'output_files': files, 'explained_drug_repeats': rows.groupby(['repeat', 'active_name']).ngroups})
    assert rows.groupby(['repeat', 'active_name']).ngroups == 246
    print('Step 6 completed; local examples ', chosen, flush=True)

def score_exploratory_drugs():
    d = step(7)
    out = d / 'results'
    p = out / 'figures'
    p.mkdir(exist_ok=True)
    names, q, y, X, cols = load_features()
    main = json.loads((step(4) / 'results/primary_model.json').read_text())
    f, m = (main['feature'], main['model'])
    model = joblib.load(step(4) / 'results' / main['model_path'])
    outer = [joblib.load(a) for a in sorted((step(5) / 'results/outer_models').glob('*.joblib'))]
    assert len(outer) == 15
    records = json.loads((d / 'inputs/exploratory_drugs.json').read_text())
    assert not set((r['active_name'] for r in records)) & set(names)
    tg = read(1, 'target_mechanism_matrix.csv').drop(columns=['active_name', 'ML_label']).columns.tolist()
    mat = []
    raw = []
    metadata = []
    sim = tanimoto_matrix(X['Morgan'])
    np.fill_diagonal(sim, 0)
    threshold = float(np.quantile(sim.max(axis=1), 0.05))
    train_m = X['Morgan']
    known = {c[5:] for c in tg if c.startswith('GENE_')}
    for r in records:
        mol = Chem.MolFromSmiles(r['SMILES'])
        assert mol is not None and Chem.MolToInchiKey(mol) == r['InChIKey']
        fp = fingerprints(mol)
        mapping = dict(zip(PHYS, descriptors(mol)))
        for k in fp:
            mapping.update(zip(FP_NAMES[k], fp[k]))
        payload = bytes.fromhex(r['fingerprint_hex'])
        assert int.from_bytes(payload[:4], 'big') == 881
        pub = np.unpackbits(np.frombuffer(payload[4:], dtype=np.uint8))[:881]
        mapping.update(zip([f'PUB{i:03d}' for i in range(881)], pub))
        mapping.update({c: int(c.startswith('GENE_') and c[5:] in r['accepted_targets']) for c in tg})
        if 'PSMB5' in r['accepted_targets']:
            mapping['COMPLEX_PROTEASOME'] = 1
        mat.append([mapping[c] for c in cols[f]])
        raw.append({'active_name': r['active_name'], **{c: mapping[c] for c in cols[f]}})
        inter = train_m.astype(np.int64) @ fp['Morgan'].astype(np.int64)
        similar = inter / (train_m.sum(axis=1) + fp['Morgan'].sum() - inter)
        near = int(np.argmax(similar))
        unseen = set(r['accepted_targets']) - known
        encoded = sum((mapping[c] for c in tg))
        metadata.append({'active_name': r['active_name'], 'PubChem_CID': r['PubChem_CID'], 'first_FDA_approval': r['FDA_first_US_approval'], 'targets': '; '.join(r['accepted_targets']), 'targets_outside_training_dictionary': '; '.join(sorted(unseen)), 'n_encoded_mechanisms': encoded, 'nearest_development_drug': names[near], 'maximum_Morgan_similarity': float(similar[near]), 'within_structural_domain': bool(similar[near] >= threshold), 'insufficient_target_encoding': bool('Targets' in f and encoded == 0), 'description': r['context_note'], 'source_url': r['FDA_label_url']})
    xx = np.array(mat, float)
    scores = model.predict_proba(xx)[:, 1]
    multi = np.array([a.predict_proba(xx)[:, 1] for a in outer])
    result = pd.DataFrame(metadata)
    result['full_development_score'] = scores
    result['outer_model_mean_score'] = multi.mean(axis=0)
    result['outer_model_minimum_score'] = multi.min(axis=0)
    result['outer_model_maximum_score'] = multi.max(axis=0)
    result['outer_model_score_SD'] = multi.std(axis=0, ddof=1)
    result['n_outer_models_above_0_5'] = (multi >= 0.5).sum(axis=0)
    result['interpretation'] = np.where(~result['within_structural_domain'] | result['insufficient_target_encoding'], 'Outside or insufficient applicability domain; classification is uncertain', 'Exploratory signal score without independently labeled external validation')
    result = result.sort_values('full_development_score', ascending=False)
    csv(out / 'exploratory_scores.csv', result)
    csv(out / 'exploratory_feature_matrix.csv', raw)
    excel(out / 'exploratory_scores.xlsx', {'exploratory_scores': result, 'applicability_domain': pd.DataFrame([{'item': 'structural_threshold', 'definition': threshold}, {'item': 'temporal_scope', 'definition': 'Frozen FDA 2024/2025 panel within the study period; not temporal external validation'}, {'item': 'interpretation_limit', 'definition': 'Unvalidated signal scores, not PPE incidence or independent validation accuracy'}, {'item': 'score_range', 'definition': 'Range across 15 outer models, not a 95% confidence interval'}])})
    fig, ax = plt.subplots(figsize=(9, 5.5))
    z = result.iloc[::-1]
    pos = np.arange(len(z))
    ax.hlines(pos, z['outer_model_minimum_score'], z['outer_model_maximum_score'], color='#aacde9', lw=3)
    ax.scatter(z['full_development_score'], pos, c=[PINK if a and (not b) else '#888888' for a, b in zip(z['within_structural_domain'], z['insufficient_target_encoding'])], s=45, label='Full-data final candidate')
    ax.axvline(0.5, color='#777', ls='--')
    ax.set_yticks(pos, z.active_name)
    ax.set(xlim=(-0.02, 1.02), xlabel='Exploratory FAERS-label model score', title=f"{f.replace('_', ' + ')} + {m}: unlabeled drug panel")
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(fontsize=8)
    fig.tight_layout()
    figsave(fig, p / 'exploratory_scores_and_variation')
    json_out(out / 'applicability_domain.json', {'panel_n': len(records), 'overlap_training_drugs': 0, 'Morgan_domain_threshold_LOO_5percentile': threshold, 'actual_primary_feature': f, 'actual_primary_model': m, 'independent_external_validation': False, 'temporal_validation': False, 'classification_accuracy_available': False})
    unstable = int(((result['outer_model_minimum_score'] < 0.5) & (result['outer_model_maximum_score'] >= 0.5)).sum())
    print('Step 7 completed.', flush=True)
