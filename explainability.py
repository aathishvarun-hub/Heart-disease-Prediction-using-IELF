"""
explainability.py
==================
SHAP global/local explanations, LIME local explanations, and quantitative
explanation-stability metrics (Kendall's tau-b, Overlap@k) — Sections
2.11-2.13 of the paper.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import kendalltau
import shap


def shap_global_importance(xgb_model, X: pd.DataFrame) -> pd.Series:
    """Mean |SHAP value| per feature from a fitted XGBoost model."""
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X)
    importance = np.abs(shap_values).mean(axis=0)
    return pd.Series(importance, index=X.columns).sort_values(ascending=False)


def ebm_global_importance(ebm_model) -> pd.Series:
    """Global feature importance from an EBM's additive terms."""
    global_exp = ebm_model.explain_global()
    names = global_exp.data()["names"]
    scores = global_exp.data()["scores"]
    return pd.Series(scores, index=names).sort_values(ascending=False)


def overlap_at_k(ranking_a: pd.Series, ranking_b: pd.Series, k: int = 10) -> float:
    top_a = set(ranking_a.sort_values(ascending=False).index[:k])
    top_b = set(ranking_b.sort_values(ascending=False).index[:k])
    return len(top_a & top_b) / k


def kendall_tau_b(ranking_a: pd.Series, ranking_b: pd.Series) -> float:
    common = ranking_a.index.intersection(ranking_b.index)
    tau, _ = kendalltau(ranking_a.loc[common], ranking_b.loc[common])
    return tau


def stability_across_folds(fold_rankings: list[pd.Series], k: int = 10) -> dict:
    """Mean pairwise Kendall's tau and Overlap@k across all outer-fold
    feature-importance rankings for a single model (Section 2.13)."""
    taus, overlaps = [], []
    n = len(fold_rankings)
    for i in range(n):
        for j in range(i + 1, n):
            taus.append(kendall_tau_b(fold_rankings[i], fold_rankings[j]))
            overlaps.append(overlap_at_k(fold_rankings[i], fold_rankings[j], k))
    return {
        "kendall_tau_mean": float(np.nanmean(taus)) if taus else np.nan,
        "overlap_at_k_mean": float(np.nanmean(overlaps)) if overlaps else np.nan,
    }


def lime_explain_instance(ielf_model, X_train: pd.DataFrame, X_instance: pd.Series,
                           num_features: int = 8, num_samples: int = 5000):
    """Local LIME explanation for a single patient using the IELF ensemble's
    predict_proba as the black-box function (Section 2.12)."""
    from lime.lime_tabular import LimeTabularExplainer

    explainer = LimeTabularExplainer(
        training_data=X_train.values,
        feature_names=list(X_train.columns),
        class_names=["no_disease", "disease"],
        mode="classification",
        random_state=42,
    )
    exp = explainer.explain_instance(
        X_instance.values,
        ielf_model.predict_proba,
        num_features=num_features,
        num_samples=num_samples,
    )
    return exp  # exp.as_list() -> [(feature_rule, contribution), ...]
