from __future__ import annotations
import argparse
import hashlib
import json
import logging
import platform
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, roc_curve, precision_recall_curve
from sklearn.model_selection import GridSearchCV, ParameterGrid
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from threadpoolctl import threadpool_limits
import lightgbm
from lightgbm import LGBMClassifier
import xgboost
from xgboost import XGBClassifier
import feature_utils as core
from model_extensions import AuditedMLPClassifier
MODELS = ('LR', 'RF', 'XGBoost', 'LightGBM', 'KNN', 'DT', 'MLP')
FEATURES = ('physchem', 'fingerprint')
METRICS = (*core.METRICS, 'precision')
MODEL_NAMES = {'LR': 'Logistic regression', 'RF': 'Random forest', 'XGBoost': 'XGBoost', 'LightGBM': 'LightGBM', 'KNN': 'K-nearest neighbors', 'DT': 'Decision tree', 'MLP': 'Multilayer perceptron', 'Dummy': 'baseline_model', 'Selected': 'inner_selection'}
FEATURE_NAMES = {'physchem': 'physicochemical_features', 'fingerprint': 'fingerprint', 'baseline': 'baseline', 'adaptive': 'outer_fold_selection'}
warnings.filterwarnings('ignore', message='X does not have valid feature names, but LGBMClassifier was fitted with feature names')

def convergence_scorer(estimator, X, y):
    return float(getattr(estimator.named_steps['model'], 'convergence_warning_count_', 0))
SCORING = {'AUROC': 'roc_auc', 'AP': 'average_precision', 'convergence': convergence_scorer}

def pick_index(auc, ap):
    auc, ap = (np.asarray(auc, float), np.asarray(ap, float))
    if not np.isfinite(auc).all() or not np.isfinite(ap).all():
        raise ValueError('A candidate has nonfinite inner scores and cannot be silently excluded.')
    eligible = np.flatnonzero(np.isclose(auc, auc.max(), rtol=0, atol=1e-12))
    ap_max = ap[eligible].max()
    return int(next((i for i in eligible if abs(ap[i] - ap_max) <= 1e-12)))

def refit_rule(results):
    return pick_index(results['mean_test_AUROC'], results['mean_test_AP'])

def build_pipeline(feature, model, seed, quick=False):
    scale = feature == 'physchem' and model == 'LR' or model in ('KNN', 'MLP')
    steps = [('impute', SimpleImputer(strategy='median', keep_empty_features=True)), ('variance', VarianceThreshold(0)), ('select', core.SafeKBest(16) if feature == 'fingerprint' else 'passthrough'), ('scale', StandardScaler() if scale else 'passthrough')]
    if model == 'LR':
        estimator = LogisticRegression(solver='liblinear', max_iter=10000, random_state=seed)
        grid = {'model__C': [0.1, 1.0, 10.0]}
    elif model == 'RF':
        estimator = RandomForestClassifier(n_estimators=30 if quick else 200, max_features='sqrt', n_jobs=1, random_state=seed)
        grid = {'model__max_depth': [3, None], 'model__min_samples_leaf': [2, 4]}
    elif model == 'XGBoost':
        estimator = XGBClassifier(n_estimators=20 if quick else 100, learning_rate=0.05, reg_lambda=5, objective='binary:logistic', eval_metric='logloss', tree_method='hist', device='cpu', n_jobs=1, random_state=seed)
        grid = {'model__max_depth': [1, 2], 'model__min_child_weight': [1, 3]}
    elif model == 'LightGBM':
        estimator = LGBMClassifier(n_estimators=20 if quick else 100, learning_rate=0.05, reg_lambda=2, max_depth=3, n_jobs=1, random_state=seed, deterministic=True, force_col_wise=True, verbosity=-1)
        grid = {'model__num_leaves': [3, 7], 'model__min_child_samples': [3, 5]}
    elif model == 'KNN':
        estimator = KNeighborsClassifier(metric='euclidean', n_jobs=1)
        grid = {'model__n_neighbors': [3, 5, 7], 'model__weights': ['uniform', 'distance']}
    elif model == 'DT':
        estimator = DecisionTreeClassifier(random_state=seed)
        grid = {'model__max_depth': [2, 3], 'model__min_samples_leaf': [2, 4]}
    elif model == 'MLP':
        estimator = AuditedMLPClassifier(solver='lbfgs', max_iter=3000, max_fun=50000, early_stopping=False, random_state=seed)
        grid = {'model__hidden_layer_sizes': [(8,), (16,)], 'model__alpha': [0.1, 1.0]}
    else:
        raise ValueError(model)
    if feature == 'fingerprint':
        grid['select__k'] = [16, 32]
    if quick:
        grid = {k: [v[0]] for k, v in grid.items()}
    return (Pipeline([*steps, ('model', estimator)]), grid)

def evaluate(y, scores):
    return {**core.evaluate(np.asarray(y), np.asarray(scores)), 'precision': float(precision_score(y, np.asarray(scores) >= 0.5, zero_division=0))}
