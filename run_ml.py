"""Command-line entry point for the PPE machine-learning analysis."""
from pathlib import Path
import argparse
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))

STAGES = {
    1: ('Feature generation', [('data_features', 'generate_features')]),
    2: ('Cross-validation splits', [('data_features', 'generate_splits')]),
    3: ('Scaffolds and chemical space', [('data_features', 'analyze_chemical_space')]),
    4: ('Model selection and nested validation', [('training', 'train_models'), ('model_plots', 'blue_heatmaps')]),
    5: ('Calibration, sensitivity and SHAP', [('validation', 'evaluate_primary_model'), ('shap_analysis', 'calculate_shap')]),
    6: ('SHAP visualization', [('interpretation', 'plot_shap')]),
    7: ('Exploratory drug scoring', [('interpretation', 'score_exploratory_drugs')]),
}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--from-step', type=int, choices=range(1, 8), default=1)
    parser.add_argument('--to-step', type=int, choices=range(1, 8), default=7)
    parser.add_argument('--workers', type=int, help='Number of model-training worker processes')
    parser.add_argument('--shap-workers', type=int, help='Number of SHAP worker processes')
    parser.add_argument('--list', action='store_true', help='List steps without running analysis')
    parser.add_argument('--execute-stage', type=int, choices=range(1, 8), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.list:
        for number, (title, _) in STAGES.items():
            print(f'{number}. {title}')
        return 0
    if args.execute_stage is not None:
        import importlib
        for module, function in STAGES[args.execute_stage][1]:
            getattr(importlib.import_module(module), function)()
        return 0
    if args.from_step > args.to_step:
        parser.error('--from-step must not exceed --to-step')
    if any(n is not None and n < 1 for n in (args.workers, args.shap_workers)):
        parser.error('Worker counts must be positive integers')
    env = os.environ.copy()
    env.update(PYTHONIOENCODING='utf-8', PYTHONUNBUFFERED='1')
    if args.workers is not None:
        env['PPE_WORKERS'] = str(args.workers)
    if args.shap_workers is not None:
        env['PPE_SHAP_WORKERS'] = str(args.shap_workers)
    from paths import step
    for number in range(args.from_step, args.to_step + 1):
        title = STAGES[number][0]
        log_path = step(number) / 'run.log'
        print(f'Step {number}/7: {title}', flush=True)
        with log_path.open('w', encoding='utf-8') as log:
            process = subprocess.Popen(
                [sys.executable, str(Path(__file__).resolve()), '--execute-stage', str(number)],
                cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding='utf-8', errors='replace',
            )
            for line in process.stdout:
                log.write(line)
                log.flush()
                print(line, end='', flush=True)
            code = process.wait()
        if code:
            print(f'Step {number} failed; see {log_path}', file=sys.stderr)
            return code
    print('Analysis completed. Results are in outputs/.', flush=True)
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
