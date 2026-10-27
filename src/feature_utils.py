from __future__ import annotations
import argparse
import hashlib
import json
import logging
import platform
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, VarianceThreshold, chi2
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, balanced_accuracy_score, brier_score_loss, confusion_matrix, f1_score, precision_recall_curve, roc_auc_score, roc_curve
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
DEFAULT_INPUT = 'development_features.xlsx'
PHYS_FEATURES = ['MolecularWeight', 'XLogP', 'TPSA', 'HBD', 'HBA', 'RotatableBonds', 'HeavyAtoms', 'Complexity', 'FormalCharge', 'CovalentUnits', 'DefinedAtomStereo', 'DefinedBondStereo']
FP_FEATURES = [f'PCFP{i:04d}' for i in range(881)]
MODELS = ('LR', 'RF')
METRICS = ('AUROC', 'AP', 'balanced_accuracy', 'sensitivity', 'specificity', 'accuracy', 'F1', 'Brier')
DOCS = {'nested_cv': 'https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html', 'group_cv': 'https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html', 'leakage': 'https://scikit-learn.org/stable/common_pitfalls.html'}

class SafeKBest(BaseEstimator, TransformerMixin):

    def __init__(self, k=16):
        self.k = k

    def fit(self, X, y):
        self.n_features_in_ = X.shape[1]
        self.k_effective_ = min(int(self.k), self.n_features_in_)
        if self.k_effective_ < 1:
            raise ValueError('No fingerprint features remain in this training fold.')
        self.selector_ = SelectKBest(chi2, k=self.k_effective_).fit(X, y)
        return self

    def transform(self, X):
        return self.selector_.transform(X)

    def get_support(self):
        return self.selector_.get_support()

def save_csv(frame, path):
    pd.DataFrame(frame).to_csv(path, index=False, encoding='utf-8-sig')

def save_json(value, path):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str, allow_nan=False), encoding='utf-8')

def tanimoto_matrix(fingerprint):
    fp = np.asarray(fingerprint, dtype=np.int64)
    intersection = fp @ fp.T
    count = fp.sum(axis=1)
    union = count[:, None] + count[None, :] - intersection
    if (union <= 0).any():
        raise ValueError('All-zero fingerprint: Tanimoto similarity is undefined.')
    return intersection / union

def connected_groups(similarity, threshold):
    n = len(similarity)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for i in range(n):
        for j in range(i + 1, n):
            if similarity[i, j] >= threshold:
                parent[find(j)] = find(i)
    mapping = {}
    groups = []
    for i in range(n):
        root = find(i)
        if root not in mapping:
            mapping[root] = len(mapping)
        groups.append(mapping[root])
    return np.asarray(groups, dtype=int)

def make_splits(y, groups, folds, seed, grouped):
    if np.bincount(y, minlength=2).min() < folds:
        raise ValueError(f'Insufficient minority-class drugs for {folds} folds.')
    if grouped and len(np.unique(groups)) < folds:
        raise ValueError(f'Insufficient structural groups for {folds} folds.')
    for attempt in range(50):
        effective_seed = seed + attempt
        splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=effective_seed) if grouped else StratifiedKFold(n_splits=folds, shuffle=True, random_state=effective_seed)
        splits = list(splitter.split(np.zeros((len(y), 1)), y, groups if grouped else None))
        valid = all((len(np.unique(y[tr])) == len(np.unique(y[va])) == 2 and (not grouped or not set(groups[tr]) & set(groups[va])) for tr, va in splits))
        if valid:
            coverage = np.concatenate([va for _, va in splits])
            if not np.array_equal(np.sort(coverage), np.arange(len(y))):
                raise AssertionError('Each drug must occur in validation exactly once per repeat.')
            return (splits, effective_seed)
    raise ValueError('Unable to generate nonoverlapping group splits containing both labels.')

def evaluate(y, probability):
    if len(np.unique(y)) != 2 or not np.isfinite(probability).all():
        raise ValueError('Evaluation requires both labels and valid scores.')
    pred = (np.asarray(probability) >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {'n': len(y), 'n_positive': int(np.sum(y)), 'AUROC': float(roc_auc_score(y, probability)), 'AP': float(average_precision_score(y, probability)), 'balanced_accuracy': float(balanced_accuracy_score(y, pred)), 'sensitivity': float(tp / (tp + fn)), 'specificity': float(tn / (tn + fp)), 'accuracy': float(accuracy_score(y, pred)), 'F1': float(f1_score(y, pred, zero_division=0)), 'Brier': float(brier_score_loss(y, probability)), 'TN': int(tn), 'FP': int(fp), 'FN': int(fn), 'TP': int(tp)}
