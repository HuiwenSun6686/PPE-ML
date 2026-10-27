from pathlib import Path
import json, hashlib, shutil, sys, platform
import pandas as pd, numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import rdMolDescriptors, Descriptors, Crippen, MACCSkeys, rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import Draw
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, trustworthiness
from sklearn.metrics import silhouette_score
from threadpoolctl import threadpool_limits
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from paths import step, ROOT
from io_utils import excel, csv, json_out, copy_inputs
import feature_utils as core
N = 82
POS = 29
NEG = 53
SEED = 20261007
PHYS = ['MolWt', 'MolLogP_RDKit', 'TPSA_RDKit', 'HBD', 'HBA', 'RotatableBonds', 'HeavyAtoms', 'BertzCT', 'FormalCharge', 'CovalentUnits', 'DefinedAtomStereo', 'DefinedBondStereo']
FP_NAMES = {'Morgan': [f'MORGAN{i:04d}' for i in range(2048)], 'MACCS': [f'MACCS{i:03d}' for i in range(1, 167)], 'RDKit': [f'RDK{i:04d}' for i in range(2048)]}
COLORS = {1: '#FF0051', 0: '#008BFB'}
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11, 'pdf.fonttype': 42, 'savefig.dpi': 300})

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def read(n, f, source=False):
    return pd.read_csv(step(n) / ('inputs' if source else 'results') / f)

def figsave(fig, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    png = path.with_suffix('.png')
    pdf = path.with_suffix('.pdf')
    tpng = png.with_name(png.stem + '_temporary.png')
    tpdf = pdf.with_name(pdf.stem + '_temporary.pdf')
    fig.savefig(tpng, dpi=300, bbox_inches='tight')
    fig.savefig(tpdf, bbox_inches='tight')
    plt.close(fig)
    from PIL import Image
    with Image.open(tpng) as im:
        im.verify()
    tpng.replace(png)
    tpdf.replace(pdf)

def label_data():
    q = read(1, 'drug_labels.csv')
    names = q.active_name.tolist()
    y = q.ML_label.to_numpy(int)
    assert len(names) == N and np.bincount(y).tolist() == [NEG, POS] and (len(set(names)) == N)
    return (names, q, y)

def descriptors(m):
    return [Descriptors.MolWt(m), np.nan if any((a.GetAtomicNum() == 78 for a in m.GetAtoms())) else Crippen.MolLogP(m), rdMolDescriptors.CalcTPSA(m), rdMolDescriptors.CalcNumHBD(m), rdMolDescriptors.CalcNumHBA(m), rdMolDescriptors.CalcNumRotatableBonds(m), m.GetNumHeavyAtoms(), Descriptors.BertzCT(m), Chem.GetFormalCharge(m), len(Chem.GetMolFrags(m)), len(Chem.FindMolChiralCenters(m, includeUnassigned=False)), sum((b.GetStereo() in [Chem.BondStereo.STEREOE, Chem.BondStereo.STEREOZ] for b in m.GetBonds()))]

def fingerprints(m):
    g = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048, includeChirality=True, useBondTypes=True)
    mg = np.zeros(2048, dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(g.GetFingerprint(m), mg)
    rd = np.zeros(2048, dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(Chem.RDKFingerprint(m, minPath=1, maxPath=7, fpSize=2048, nBitsPerHash=2, useHs=True, branchedPaths=True, useBondOrder=True), rd)
    mc = np.zeros(167, dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(MACCSkeys.GenMACCSKeys(m), mc)
    return {'Morgan': mg, 'MACCS': mc[1:], 'RDKit': rd}

def load_features():
    names, q, y = label_data()
    X = {}
    columns = {}
    for k, f in [('Physchem', 'physicochemical_matrix.csv'), ('PubChem', 'PubChem_fingerprint_matrix.csv'), ('Morgan', 'Morgan_fingerprint_matrix.csv'), ('MACCS', 'MACCS_fingerprint_matrix.csv'), ('RDKit', 'RDKit_path_fingerprint_matrix.csv'), ('Targets', 'target_mechanism_matrix.csv')]:
        v = read(1, f).set_index('active_name').loc[names].drop(columns='ML_label', errors='ignore')
        columns[k] = v.columns.tolist()
        X[k] = v.to_numpy(float)
    for k in ['PubChem_Targets', 'Morgan_Targets', 'MACCS_Targets', 'RDKit_Targets', 'Morgan_MACCS_RDKit_Targets']:
        parts = k.split('_')
        X[k] = np.hstack([X[p] for p in parts])
        columns[k] = sum([columns[p] for p in parts], [])
    return (names, q, y, X, columns)

def generate_features():
    d = step(1)
    copy_inputs([(ROOT / 'data/development_drugs.csv', 'development_drugs.csv'), (ROOT / 'data/target_evidence.csv', 'target_evidence.csv')], d / 'inputs')
    q = read(1, 'development_drugs.csv', True)
    names = q.active_name.tolist()
    y = q.ML_label.to_numpy(int)
    physical = []
    fps = {k: [] for k in FP_NAMES}
    for s in q['parent_smiles']:
        m = Chem.MolFromSmiles(s)
        physical.append(descriptors(m))
        fp = fingerprints(m)
        for k in fps:
            fps[k].append(fp[k])
    phys = pd.DataFrame(physical, columns=PHYS)
    phys.insert(0, 'active_name', names)
    csv(d / 'results/physicochemical_matrix.csv', phys)
    pub = np.array([np.unpackbits(np.frombuffer(bytes.fromhex(z)[4:], dtype=np.uint8))[:881] for z in q['pubchem_fingerprint_hex']])
    pf = pd.DataFrame(pub, columns=[f'PUB{i:03d}' for i in range(881)])
    pf.insert(0, 'active_name', names)
    csv(d / 'results/PubChem_fingerprint_matrix.csv', pf)
    for k in fps:
        frame = pd.DataFrame(np.array(fps[k]), columns=FP_NAMES[k])
        frame.insert(0, 'active_name', names)
        csv(d / 'results' / {'RDKit': 'RDKit_path_fingerprint_matrix.csv'}.get(k, k + '_fingerprint_matrix.csv'), frame)
    e = read(1, 'target_evidence.csv', True).fillna('')
    accepted = {'mechanism_target', 'direct_binding_target', 'complex_mediated_target', 'mechanism_target_family', 'pharmacological_complex', 'mechanism_nonprotein_object'}
    rows = []
    pairs = []
    for _, r in e.iterrows():
        gene = r['gene_symbol']
        level = r['annotation_level']
        key = 'GENE_' + gene if gene else 'FAMILY_' + r['target_name'] if level == 'protein_family' else 'COMPLEX_PROTEASOME' if level == 'protein_complex' else 'OBJECT_' + r['target_name'] if level == 'nonprotein_object' else ''
        ok = bool(key and r['evidence_role'] in accepted)
        rows.append({**r.to_dict(), 'feature_name': key, 'included': int(ok), 'note': '0 indicates no accepted annotation, not established inactivity'})
        if ok:
            pairs.append((r['drug'], key))
    pairs.append(('ixazomib', 'COMPLEX_PROTEASOME'))
    tc = sorted({k for n, k in pairs})
    t = pd.DataFrame(0, index=names, columns=tc, dtype=int)
    for n, k in pairs:
        t.loc[n, k] = 1

    def savemat(mat, file):
        csv(d / 'results' / file, pd.concat([pd.DataFrame({'active_name': names, 'ML_label': y}).set_index('active_name'), mat.rename_axis('active_name')], axis=1).rename_axis('active_name').reset_index())
    savemat(t, 'target_mechanism_matrix.csv')
    proteins = t[[c for c in tc if not c.startswith('OBJECT_')]]
    savemat(proteins, 'protein_family_target_matrix.csv')
    dictionary = [{'feature_name': c, 'level': c.split('_')[0], 'n_carrier_drugs': int(t[c].sum()), 'carrier_drugs': '; '.join(t.index[t[c] == 1]), 'definition': '1: accepted annotation; 0: no accepted annotation, not established inactivity'} for c in tc]
    coverage = [{'active_name': n, 'protein_family_complex_feature_count': int(proteins.loc[n].sum()), 'all_mechanism_feature_count': int(t.loc[n].sum()), 'no_encoded_protein_target': int(proteins.loc[n].sum() == 0)} for n in names]
    csv(d / 'results/target_encoding_records.csv', rows)
    csv(d / 'results/target_feature_dictionary.csv', dictionary)
    csv(d / 'results/target_feature_coverage.csv', coverage)
    labels = q[['active_name', 'ML_label', 'drug_class', 'PubChem_CID', 'InChIKey', 'parent_smiles', 'source_url']]
    csv(d / 'results/drug_labels.csv', labels)
    sim = core.tanimoto_matrix(np.array(fps['Morgan']))
    pairsim = [{'drug1': names[i], 'drug2': names[j], 'Morgan_Tanimoto': float(sim[i, j]), 'same_label': bool(y[i] == y[j])} for i in range(N) for j in range(i + 1, N)]
    csv(d / 'results/pairwise_structure_similarity.csv', pd.DataFrame(pairsim).sort_values('Morgan_Tanimoto', ascending=False))
    missing = [{'active_name': names[i], 'missing_descriptor': PHYS[j], 'reason': 'Crippen logP not used for platinum complexes; imputed within training folds'} for i, j in zip(*np.where(np.isnan(np.array(physical))))]
    csv(d / 'results/missing_values.csv', missing)
    qc = pd.DataFrame([{'matrix': k, 'n_drugs': N, 'n_features': v, 'missing_cells': len(missing) if k == 'Physicochemical' else 0} for k, v in [('Physicochemical', 12), ('PubChem', 881), ('Morgan', 2048), ('MACCS', 166), ('RDKit', 2048), ('Target_mechanism', len(tc)), ('Protein targets', proteins.shape[1])]])
    excel(d / 'results/feature_summary.xlsx', {'matrix_summary': qc, 'physicochemical_features': phys, 'label': labels, 'Target_mechanism': t.reset_index(names='active_name'), 'missing_values': missing, 'mechanism_coverage': coverage, 'feature_definitions': dictionary})
    definitions = [{'descriptor': c, 'definition': v, 'source': 'RDKit 2026.03.6; calculated from frozen structures'} for c, v in zip(PHYS, ['Molecular weight (g/mol)', 'RDKit Crippen logP', 'Topological polar surface area (angstrom squared)', 'Hydrogen bond donors', 'Hydrogen bond acceptors', 'Rotatable bonds', 'Heavy atoms', 'RDKit Bertz complexity', 'Formal charge', 'SMILES components', 'Defined atom stereocenters', 'Defined E/Z bonds'])]
    csv(d / 'results/physicochemical_definitions.csv', definitions)
    json_out(d / 'results/feature_metadata.json', {'n': N, 'n_target_features': len(tc), 'n_protein_features': proteins.shape[1], 'fingerprints': {k: len(v) for k, v in FP_NAMES.items()}, 'missing_physics': len(missing), 'label_or_FAERS_count_in_features': False, 'columns_selected_using_whole_dataset_labels': False})
    print('Step 1 completed: ', qc.to_dict('records'), flush=True)

def generate_splits():
    d = step(2)
    names, labels, y, X, cols = load_features()
    copy_inputs([(step(1) / 'results/drug_labels.csv', 'drug_labels.csv'), (step(1) / 'results/Morgan_fingerprint_matrix.csv', 'Morgan_fingerprint_matrix.csv')], d / 'inputs')
    sim = core.tanimoto_matrix(X['Morgan'])
    groups = core.connected_groups(sim, 0.85)
    outer = []
    inner = []
    final = []
    audit = []
    plans = []
    full = []
    for rep in range(3):
        os, oseed = core.make_splits(y, groups, 5, SEED + rep * 10000, True)
        for fold, (tr, va) in enumerate(os, 1):
            ins, iseed = core.make_splits(y[tr], groups[tr], 3, SEED + rep * 10000 + fold * 100, True)
            assert sim[np.ix_(tr, va)].max() < 0.85
            plans.append({'repeat': rep + 1, 'fold': fold, 'tr': tr.tolist(), 'va': va.tolist(), 'inner': [[a.tolist(), b.tolist()] for a, b in ins]})
            audit.append({'repeat': rep + 1, 'outer_fold': fold, 'n_training_drugs': len(tr), 'n_validation_drugs': len(va), 'n_validation_positive': int(y[va].sum()), 'n_validation_reference_negative': int((1 - y[va]).sum()), 'maximum_cross_split_similarity': float(sim[np.ix_(tr, va)].max()), 'outer_seed': oseed, 'inner_seed': iseed})
            for role, idx in [('train', tr), ('validation', va)]:
                outer.extend(({'scheme': 'group_t085', 'repeat': rep + 1, 'outer_fold': fold, 'role': role, 'active_name': names[i], 'ML_label': int(y[i]), 'group': int(groups[i])} for i in idx))
            for fi, (it, iv) in enumerate(ins, 1):
                assert sim[np.ix_(tr[it], tr[iv])].max() < 0.85
                for role, idx in [('train', tr[it]), ('validation', tr[iv])]:
                    inner.extend(({'scheme': 'group_t085', 'repeat': rep + 1, 'outer_fold': fold, 'inner_fold': fi, 'role': role, 'active_name': names[i], 'ML_label': int(y[i]), 'group': int(groups[i])} for i in idx))
        fs, fseed = core.make_splits(y, groups, 3, SEED + 600000 + rep * 10000, True)
        for fold, (tr, va) in enumerate(fs, 1):
            full.append([tr.tolist(), va.tolist()])
            for role, idx in [('train', tr), ('validation', va)]:
                final.extend(({'repeat': rep + 1, 'fold': fold, 'role': role, 'active_name': names[i], 'ML_label': int(y[i]), 'group': int(groups[i]), 'effective_seed': fseed} for i in idx))
    for rep in [1, 2, 3]:
        assert sorted([i for p in plans if p['repeat'] == rep for i in p['va']]) == list(range(N))
    for filename, df in [('outer_splits.csv', outer), ('inner_splits.csv', inner), ('development_tuning_splits.csv', final), ('split_summary.csv', audit)]:
        csv(d / 'results' / filename, df)
    csv(d / 'results/structure_groups.csv', pd.DataFrame({'active_name': names, 'ML_label': y, 'Morgan_similarity_group': groups}))
    json_out(d / 'results/split_plan.json', {'n': N, 'grouping': 'Morgan radius-2 2048-bit chiral fingerprints; connected groups at Tanimoto >= 0.85, without label-based grouping', 'threshold': 0.85, 'outer': plans, 'full': full, 'primary_chemical_space_seed': 20261006, 'group_count': len(set(groups)), 'reason': 'Morgan structural grouping was prespecified without selection by validation performance'})
    excel(d / 'results/split_summary.xlsx', {'15_fold_summary': audit, 'structure_group': pd.DataFrame({'active_name': names, 'label': y, 'structure_group': groups}), 'outer_splits': outer})
    print('Step 2 completed; structural groups ', len(set(groups)), flush=True)

def analyze_chemical_space():
    d = step(3)
    names, lab, y, X, columns = load_features()
    copy_inputs([(step(1) / 'results/drug_labels.csv', 'drug_labels.csv'), (step(2) / 'results/outer_splits.csv', 'outer_splits.csv')], d / 'inputs')
    from scaffold_utils import ring_systems
    records = []
    rings = []
    ringmols = {}
    for _, r in lab.iterrows():
        m = Chem.MolFromSmiles(r['parent_smiles'])
        sc = MurckoScaffold.GetScaffoldForMol(m)
        rr = ring_systems(m)
        records.append({'active_name': r.active_name, 'ML_label': int(r.ML_label), 'drug_class': r.drug_class, 'Murcko_SMILES': Chem.MolToSmiles(sc, isomericSmiles=False), 'generic_Murcko_SMILES': Chem.MolToSmiles(MurckoScaffold.MakeScaffoldGeneric(sc), isomericSmiles=False), 'n_ring_systems': len(rr), 'n_SSSR_rings': m.GetRingInfo().NumRings()})
        for j, (s, mol, na, nb) in enumerate(rr):
            ringmols[s] = mol
            rings.append({'active_name': r.active_name, 'ML_label': int(r.ML_label), 'ring_system_code': s, 'ring_system_index': j + 1})
    rec = pd.DataFrame(records)
    ring = pd.DataFrame(rings)

    def aggregate(q, c):
        return pd.DataFrame([{'structure_code': k, 'n_carrier_drugs': v.active_name.nunique(), 'n_positive_drugs': int(v.drop_duplicates('active_name').ML_label.sum()), 'n_reference_negative_drugs': int((v.drop_duplicates('active_name').ML_label == 0).sum()), 'carrier_drugs': '; '.join(v.active_name.unique())} for k, v in q.groupby(c)]).sort_values('n_carrier_drugs', ascending=False)
    scag = aggregate(rec, 'Murcko_SMILES')
    rag = aggregate(ring, 'ring_system_code')
    csv(d / 'results/drug_scaffolds.csv', rec)
    csv(d / 'results/drug_ring_systems.csv', ring)
    csv(d / 'results/scaffold_frequencies.csv', scag)
    csv(d / 'results/ring_system_frequencies.csv', rag)
    figdir = d / 'results/figures'
    figdir.mkdir(exist_ok=True)
    for start in range(0, len(scag), 12):
        z = scag.iloc[start:start + 12]
        im = Draw.MolsToGridImage([Chem.MolFromSmiles(s) for s in z['structure_code']], molsPerRow=3, subImgSize=(420, 300), legends=[f"n={r['n_carrier_drugs']}, positive={r['n_positive_drugs']}" for _, r in z.iterrows()])
        im.save(figdir / f'Murcko_scaffolds_page_{start // 12 + 1}.png')
    z = rag.head(18)
    Draw.MolsToGridImage([ringmols[s] for s in z['structure_code']], molsPerRow=3, subImgSize=(400, 280), legends=[f'n={n}' for n in z['n_carrier_drugs']]).save(figdir / 'top18_ring_systems.png')
    coords = []
    checks = []
    for key in ['PubChem', 'Morgan', 'MACCS', 'RDKit']:
        xx = X[key]
        sim = core.tanimoto_matrix(xx)
        dist = np.maximum(0, 1 - sim)
        np.fill_diagonal(dist, 0)
        sil = float(silhouette_score(dist, y, metric='precomputed'))
        runs = []
        pca = PCA(n_components=2, svd_solver='full')
        co = pca.fit_transform(xx)
        runs.append(('PCA', co, {'explained_variance_ratio_sum': float(pca.explained_variance_ratio_.sum()), 'original_space_label_silhouette': sil}))
        for perp, seed in [(10, 20261006), (5, 20261006), (20, 20261006), (10, 20261007)]:
            with threadpool_limits(1):
                ts = TSNE(n_components=2, perplexity=perp, metric='precomputed', init='random', learning_rate='auto', max_iter=1500, random_state=seed, method='exact')
                co = ts.fit_transform(dist)
            runs.append((f'tSNE_perplexity{perp}_seed{seed}', co, {'perplexity': perp, 'seed': seed, 'KL': float(ts.kl_divergence_), 'trustworthiness_k5': float(trustworthiness(dist, co, n_neighbors=5, metric='precomputed')), 'original_space_label_silhouette': sil}))
        for name, co, ck in runs:
            fig, ax = plt.subplots(figsize=(7.2, 5.5))
            for label, marker, title in [(1, 's', f'FAERS positive (n={POS})'), (0, 'o', f'Reference negative (n={NEG})')]:
                mask = y == label
                ax.scatter(co[mask, 0], co[mask, 1], c=COLORS[label], marker=marker, label=title, s=42, alpha=0.83, edgecolors='white', linewidths=0.4)
            ax.set(title=f"{key}: {name.split('_')[0]}", xlabel='Dimension 1', ylabel='Dimension 2')
            ax.legend(frameon=False, fontsize=9)
            ax.spines[['top', 'right']].set_visible(False)
            fig.tight_layout()
            figsave(fig, figdir / (key + '_' + name + 'chemical_space'))
            checks.append({'fingerprint': key, 'method': name, **ck})
            coords.extend(({'active_name': n, 'ML_label': int(l), 'fingerprint': key, 'method': name, 'x': float(c[0]), 'y': float(c[1])} for n, l, c in zip(names, y, co)))
    outer = read(2, 'outer_splits.csv')
    overlap = []
    for (rep, fold), q in outer.groupby(['repeat', 'outer_fold']):
        tr = set(q.query("role=='train'").active_name)
        va = set(q.query("role=='validation'").active_name)
        common = set(rec[rec.active_name.isin(tr)].Murcko_SMILES) - {''} & set(rec[rec.active_name.isin(va)].Murcko_SMILES) - {''}
        overlap.append({'repeat': rep, 'fold': fold, 'n_shared_scaffolds': len(common), 'validation_drugs_sharing_training_scaffolds': '; '.join(rec[rec.active_name.isin(va) & rec.Murcko_SMILES.isin(common)].active_name)})
    csv(d / 'results/chemical_space_coordinates.csv', coords)
    csv(d / 'results/chemical_space_settings.csv', checks)
    csv(d / 'results/scaffold_overlap.csv', overlap)
    excel(d / 'results/chemical_space_summary.xlsx', {'scaffold_frequencies': scag, 'ring_system_frequencies': rag, 'drug_scaffolds': rec, 'visualization_settings': checks, 'scaffold_overlap': overlap})
    print('Step 3 completed: scaffolds, rings and chemical space.', flush=True)
