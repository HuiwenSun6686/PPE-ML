import numpy as np, pandas as pd, matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from paths import step
from data_features import figsave
from training import FEATURE_LABELS
from estimators import MODELS

def blue_heatmaps():
    q = pd.read_csv(step(4) / 'results/candidate_performance.csv')
    features = list(FEATURE_LABELS)
    models = list(MODELS)
    cmap = LinearSegmentedColormap.from_list('Blues', ['#f0f7fc', '#bddbec', '#73add2', '#357cad', '#123e65'])
    for metric in ['AUROC', 'F1']:
        z = q.pivot(index='feature', columns='model', values=metric + '_mean').loc[features, models]
        fig, ax = plt.subplots(figsize=(10.5, 7.2))
        im = ax.imshow(z.to_numpy(), cmap=cmap, vmin=0.3, vmax=1, aspect='auto')
        for i, f in enumerate(features):
            for j, m in enumerate(models):
                ax.text(j, i, f'{z.loc[f, m]:.3f}', ha='center', va='center', fontsize=10, color='white' if z.loc[f, m] >= 0.74 else '#173b58')
        ax.set_xticks(range(7), models)
        ax.set_yticks(range(11), [f.replace('_', ' + ') for f in features])
        ax.set(title=f'77 candidates: mean OOF {metric} (3 repeats)', xlabel='Algorithm', ylabel='Feature input')
        fig.colorbar(im, ax=ax, label=f'Mean OOF {metric}', pad=0.02)
        fig.tight_layout()
        figsave(fig, step(4) / 'results/figures' / f'77_candidates_{metric}heatmap')
if __name__ == '__main__':
    blue_heatmaps()
