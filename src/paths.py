"""Project-relative input and output locations."""
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parent.parent
STAGES = {
    1: '01_features',
    2: '02_validation_splits',
    3: '03_chemical_space',
    4: '04_model_training',
    5: '05_validation_and_shap',
    6: '06_shap_figures',
    7: '07_exploratory_scoring',
}

def step(number):
    directory = ROOT / 'outputs' / STAGES[number]
    for name in ('inputs', 'results'):
        (directory / name).mkdir(parents=True, exist_ok=True)
    inputs = {
        5: 'pubchem_fingerprint_definition.txt',
        7: 'exploratory_drugs.json',
    }
    if number in inputs:
        source = ROOT / 'data' / inputs[number]
        target = directory / 'inputs' / source.name
        if not target.exists():
            shutil.copy2(source, target)
    return directory
