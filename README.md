# PPE-ML

Interpretable machine learning for drug-level palmar–plantar erythrodysesthesia (PPE) signals in small-molecule targeted anticancer therapy.

## Scope

The development dataset comprises 82 drugs: 29 signal-positive and 53 reference-negative. The supplied labels are fixed inputs derived from the study's ROR, PRR, BCPNN and MGPS criteria. This repository covers feature generation, model selection, internal validation, model interpretation and exploratory scoring. FAERS report processing, signal calculation, network toxicology and molecular docking are outside its scope.

## Files

- `run_ml.py`: one entry point for the complete workflow.
- `src/`: analysis code and shared utilities.
- `data/`: the necessary drug-level inputs and PubChem fingerprint specification.
- `requirements.txt`: dependency versions recorded in the original analysis.
- `LICENSE`: MIT license for project code.

Results, figures, fitted models and checkpoints are **not** included. They are generated in `outputs/` when the analysis runs. Source-database links are retained in the data files.

## Installation

Use Python 3.12 in a virtual environment. From the repository root:

```bash
python -m venv .venv
```

Activate the environment using `.venv\Scripts\Activate.ps1` in Windows PowerShell or `source .venv/bin/activate` on macOS/Linux, then install dependencies:

```bash
python -m pip install -r requirements.txt
```

## Run

```bash
# List the workflow without executing analysis
python run_ml.py --list

# Run all seven steps
python run_ml.py

# Limit parallel worker processes
python run_ml.py --workers 2 --shap-workers 2
```

| Step | Analysis |
| --- | --- |
| 1 | Generate physicochemical descriptors, molecular fingerprints and target/mechanism features |
| 2 | Generate prespecified structural-group cross-validation splits |
| 3 | Characterize scaffolds, ring systems and chemical space |
| 4 | Compare 77 candidates and evaluate the fully nested model-selection procedure |
| 5 | Evaluate calibration and sensitivity; compute grouped out-of-fold SHAP |
| 6 | Generate global and local SHAP figures |
| 7 | Score 10 drugs outside the development set and characterize applicability |

Steps must run in order on a fresh checkout. To resume a contiguous range after its prerequisites exist:

```bash
python run_ml.py --from-step 4 --to-step 5
```

The first complete run trains models and computes SHAP from scratch; it can be computationally expensive. The original defaults are six training workers and four SHAP workers. Intermediate checkpoints support continuation of an interrupted run. Use a fresh checkout for a new analysis after changing inputs or scientific settings; do not mix incompatible checkpoints.

## Data

`data/development_drugs.csv` contains the fixed labels, parent SMILES, InChIKeys, PubChem CIDs, source URLs and frozen PubChem fingerprint payloads for the 82 development drugs. `ML_label=1` denotes signal-positive and `ML_label=0` denotes reference-negative. Reference-negative does not establish absence of clinical risk.

`data/target_evidence.csv` contains the drug, target name, standardized gene symbol, annotation level, evidence role and source URL. The feature generator retains the original evidence-role rules. A target-feature value of zero denotes no accepted annotation, not proven inactivity. Non-modeling narrative and administrative columns have been omitted.

`data/exploratory_drugs.json` contains the structural, fingerprint and target inputs for the 10-drug application. `data/pubchem_fingerprint_definition.txt` is the frozen official PubChem specification; bit numbering is zero-based.

Input labels, drug order, structures, fingerprint payloads, encoded target identities, random seeds, hyperparameter grids, validation design, classification threshold and SHAP settings are preserved from the supplied analysis. Names of files, modules, columns and metadata categories have been translated into English. The workflow is numbered consecutively from 1 to 7. No new label audit, training or end-to-end execution was performed during this reorganization; syntax, static dependencies and input preservation were checked.

## Interpretation

The original selected candidate is logistic regression using PubChem fingerprints and target/mechanism features. Development CV, fixed-candidate OOF validation and fully nested selection validation are distinct analyses. Exploratory scores for the 10-drug panel do not constitute independently labeled external validation. Scores reflect drug-level FAERS signal labels, not individual patient risk or PPE incidence. SHAP explains model associations and does not establish causal mechanisms.

## License

Project code is distributed under the MIT license. Third-party packages, the PubChem specification and source-database materials retain their respective terms.
