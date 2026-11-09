"""
ielf_model.py
=============
The Interpretable Ensemble Learning Framework (IELF): a soft-voting
combination of an Explainable Boosting Machine (EBM) and XGBoost
(Section 2.8-2.14 of the paper).

    y_hat = alpha * y_hat_EBM + (1 - alpha) * y_hat_XGB

alpha is selected only inside the inner CV loop from the discrete grid
{0.0, 0.25, 0.5, 0.75, 1.0}, maximizing mean AUC, then frozen for scoring
on the corresponding outer test fold (no leakage).
"""

from __future__ import annotations
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from interpret.glassbox import ExplainableBoostingClassifier
from xgboost import XGBClassifier

ALPHA_GRID = [0.0, 0.25, 0.5, 0.75, 1.0]

EBM_PARAM_GRID = {
    "learning_rate": [0.01, 0.05, 0.10],
    "max_bins": [128, 256],
    "max_leaves": [3, 5, 7],
    "interactions": [0, 10],
}

XGB_PARAM_GRID = {
    "learning_rate": [0.01, 0.10, 0.20],
    "max_depth": [3, 5, 7],
    "n_estimators": [100, 200],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0],
    "reg_lambda": [0.01, 0.10, 1.00],
    "reg_alpha": [0.0, 0.1, 1.0],
}


class IELFEnsemble(BaseEstimator, ClassifierMixin):
    """Soft-voting ensemble of a fitted EBM and a fitted XGBoost model."""

    def __init__(self, ebm=None, xgb=None, alpha=0.5):
        self.ebm = ebm
        self.xgb = xgb
        self.alpha = alpha

    def fit(self, X, y):
        # EBM / XGBoost are expected to already be fitted by the caller
        # (fit here only if not already fitted, for standalone use).
        if self.ebm is None:
            self.ebm = ExplainableBoostingClassifier(random_state=42)
            self.ebm.fit(X, y)
        if self.xgb is None:
            self.xgb = XGBClassifier(
                random_state=42, eval_metric="logloss"
            )
            self.xgb.fit(X, y)
        return self

    def predict_proba(self, X):
        p_ebm = self.ebm.predict_proba(X)[:, 1]
        p_xgb = self.xgb.predict_proba(X)[:, 1]
        p1 = self.alpha * p_ebm + (1 - self.alpha) * p_xgb
        return np.column_stack([1 - p1, p1])

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)


def _grid_search(model_ctor, param_grid, X_tr, y_tr, X_val, y_val, n_iter=None, seed=42):
    """Lightweight grid search maximizing validation AUC.
    Uses random sampling of the grid (n_iter) if the full grid is large,
    for computational tractability, matching the paper's inner-loop search."""
    from sklearn.model_selection import ParameterGrid
    import random

    grid = list(ParameterGrid(param_grid))
    if n_iter is not None and len(grid) > n_iter:
        random.Random(seed).shuffle(grid)
        grid = grid[:n_iter]

    best_score, best_params, best_model = -np.inf, None, None
    for params in grid:
        model = model_ctor(**params)
        model.fit(X_tr, y_tr)
        proba = model.predict_proba(X_val)[:, 1]
        score = roc_auc_score(y_val, proba)
        if score > best_score:
            best_score, best_params, best_model = score, params, model
    return best_model, best_params, best_score


def ebm_ctor(**params):
    return ExplainableBoostingClassifier(random_state=42, **params)


def xgb_ctor(**params):
    return XGBClassifier(
        random_state=42, eval_metric="logloss", **params
    )


def inner_loop_fit(X_train, y_train, seed=42, n_iter_xgb=10, n_iter_ebm=8):
    """
    Performs the inner-loop hyperparameter search (Section 2.16-2.17) for
    EBM and XGBoost using an internal train/validation split, then selects
    the ensemble weight alpha on the same validation split (Section 2.14).

    Returns the tuned EBM, tuned XGBoost, and the selected alpha.
    """
    from sklearn.model_selection import train_test_split

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.25, stratify=y_train, random_state=seed
    )

    best_ebm, ebm_params, ebm_auc = _grid_search(
        ebm_ctor, EBM_PARAM_GRID, X_tr, y_tr, X_val, y_val, n_iter=n_iter_ebm, seed=seed
    )
    best_xgb, xgb_params, xgb_auc = _grid_search(
        xgb_ctor, XGB_PARAM_GRID, X_tr, y_tr, X_val, y_val, n_iter=n_iter_xgb, seed=seed
    )

    # refit best models on full inner-training set (X_train) before alpha search
    best_ebm = ebm_ctor(**ebm_params).fit(X_train, y_train)
    best_xgb = xgb_ctor(**xgb_params).fit(X_train, y_train)

    # select alpha on the held-out validation split (still inside inner loop)
    p_ebm_val = best_ebm.predict_proba(X_val)[:, 1]
    p_xgb_val = best_xgb.predict_proba(X_val)[:, 1]
    best_alpha, best_auc = 0.5, -np.inf
    for alpha in ALPHA_GRID:
        p = alpha * p_ebm_val + (1 - alpha) * p_xgb_val
        auc = roc_auc_score(y_val, p)
        if auc > best_auc:
            best_auc, best_alpha = auc, alpha

    return best_ebm, best_xgb, best_alpha, {
        "ebm_params": ebm_params, "xgb_params": xgb_params,
        "ebm_val_auc": ebm_auc, "xgb_val_auc": xgb_auc, "alpha_val_auc": best_auc,
    }
