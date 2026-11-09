"""
metrics_utils.py
================
Classification, calibration, and statistical-comparison utilities
matching Section 2.20-2.21 of the paper.
"""

from __future__ import annotations
import numpy as np
from scipy import stats
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, log_loss, matthews_corrcoef, confusion_matrix,
    brier_score_loss,
)


def specificity_score(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return tn / (tn + fp) if (tn + fp) > 0 else np.nan


def full_metrics(y_true, y_pred, y_proba) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "auc": roc_auc_score(y_true, y_proba) if len(set(y_true)) > 1 else np.nan,
        "specificity": specificity_score(y_true, y_pred),
        "log_loss": log_loss(y_true, y_proba, labels=[0, 1]),
        "mcc": matthews_corrcoef(y_true, y_pred),
    }


def expected_calibration_error(y_true, y_proba, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (y_proba >= lo) & (y_proba < hi if i < n_bins - 1 else y_proba <= hi)
        if mask.sum() == 0:
            continue
        conf = y_proba[mask].mean()
        acc = y_true[mask].mean()
        ece += (mask.sum() / len(y_proba)) * abs(acc - conf)
    return ece


def max_calibration_error(y_true, y_proba, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    mce = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (y_proba >= lo) & (y_proba < hi if i < n_bins - 1 else y_proba <= hi)
        if mask.sum() == 0:
            continue
        conf = y_proba[mask].mean()
        acc = y_true[mask].mean()
        mce = max(mce, abs(acc - conf))
    return mce


def calibration_metrics(y_true, y_proba) -> dict:
    return {
        "brier": brier_score_loss(y_true, y_proba),
        "ece": expected_calibration_error(np.asarray(y_true), np.asarray(y_proba)),
        "mce": max_calibration_error(np.asarray(y_true), np.asarray(y_proba)),
    }


def paired_significance(fold_scores_a: list[float], fold_scores_b: list[float]) -> dict:
    """Paired t-test and Wilcoxon signed-rank test across matched CV folds
    (Section 2.21). Falls back gracefully when folds are constant/identical."""
    a, b = np.asarray(fold_scores_a), np.asarray(fold_scores_b)
    diffs = a - b
    try:
        t_stat, t_p = stats.ttest_rel(a, b)
    except Exception:
        t_p = np.nan
    try:
        if np.allclose(diffs, 0):
            w_p = 1.0
        else:
            _, w_p = stats.wilcoxon(a, b)
    except Exception:
        w_p = np.nan
    pooled_sd = np.std(diffs, ddof=1) if len(diffs) > 1 else np.nan
    cohens_d = np.mean(diffs) / pooled_sd if pooled_sd not in (0, np.nan) else np.nan
    return {"mean_a": a.mean(), "mean_b": b.mean(), "t_p": t_p, "wilcoxon_p": w_p,
            "cohens_d": cohens_d}


def mean_ci(values: list[float], confidence=0.95) -> tuple[float, float, float]:
    a = np.asarray(values)
    m = a.mean()
    if len(a) > 1:
        sem = stats.sem(a)
        h = sem * stats.t.ppf((1 + confidence) / 2, len(a) - 1)
    else:
        h = 0.0
    return m, m - h, m + h
