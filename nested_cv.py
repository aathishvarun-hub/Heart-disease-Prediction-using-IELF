"""
nested_cv.py
============
5x5 nested cross-validation driver (Sections 2.15-2.17). Outer loop gives
unbiased generalization estimates; inner loop (inside ielf_model.inner_loop_fit)
performs hyperparameter tuning + ensemble-weight (alpha) selection.

SMOTE is applied ONLY within each outer-fold's training partition (never on
the outer test fold), preventing leakage, matching Section 2.5 / 2.15.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

from ielf_model import inner_loop_fit, IELFEnsemble, ebm_ctor, xgb_ctor
from metrics_utils import full_metrics
from explainability import shap_global_importance, ebm_global_importance, stability_across_folds

RANDOM_STATE = 42


def run_nested_cv(X: pd.DataFrame, y: pd.Series, dataset_name: str,
                   use_smote: bool, n_splits: int = 5, seed: int = RANDOM_STATE,
                   n_iter_xgb: int = 10, n_iter_ebm: int = 8):
    """
    Runs 5-fold outer CV. For each outer fold:
      1. Scale continuous features (fit on outer-train only)
      2. SMOTE the outer-train partition only (if use_smote=True)
      3. Run inner-loop tuning (hyperparams + alpha) -> ielf_model.inner_loop_fit
      4. Evaluate EBM, XGBoost, IELF on the untouched outer-test fold
      5. Record SHAP / EBM global feature-importance rankings for stability analysis

    Returns a dict with per-model fold metrics, mean+-SD summaries, and
    feature-ranking stability metrics.
    """
    outer_cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    results = {"EBM": [], "XGBoost": [], "IELF": []}
    shap_rankings, ebm_rankings = [], []
    fold_info = []

    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)

    for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X, y)):
        X_train, X_test = X.iloc[train_idx].copy(), X.iloc[test_idx].copy()
        y_train, y_test = y.iloc[train_idx].copy(), y.iloc[test_idx].copy()

        # 1. Scale (fit on outer-train only)
        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train),
                                       columns=X_train.columns, index=X_train.index)
        X_test_scaled = pd.DataFrame(scaler.transform(X_test),
                                      columns=X_test.columns, index=X_test.index)

        # 2. SMOTE the training partition only
        if use_smote:
            sm = SMOTE(random_state=seed)
            X_train_res, y_train_res = sm.fit_resample(X_train_scaled, y_train)
        else:
            X_train_res, y_train_res = X_train_scaled, y_train

        # 3. Inner-loop tuning: EBM, XGBoost, alpha
        best_ebm, best_xgb, best_alpha, tune_info = inner_loop_fit(
            X_train_res, y_train_res, seed=seed + fold_idx,
            n_iter_xgb=n_iter_xgb, n_iter_ebm=n_iter_ebm
        )
        ielf = IELFEnsemble(ebm=best_ebm, xgb=best_xgb, alpha=best_alpha)

        # 4. Evaluate on the untouched outer-test fold
        for name, model in [("EBM", best_ebm), ("XGBoost", best_xgb), ("IELF", ielf)]:
            proba = model.predict_proba(X_test_scaled)[:, 1]
            pred = (proba >= 0.5).astype(int)
            m = full_metrics(y_test.values, pred, proba)
            m["fold"] = fold_idx
            results[name].append(m)

        # 5. Feature-importance rankings for this fold (stability analysis)
        shap_rankings.append(shap_global_importance(best_xgb, X_test_scaled))
        ebm_rankings.append(ebm_global_importance(best_ebm))

        fold_info.append({"fold": fold_idx, "alpha": best_alpha, **tune_info})
        print(f"[{dataset_name}] fold {fold_idx+1}/{n_splits} done "
              f"(alpha={best_alpha}, IELF AUC={results['IELF'][-1]['auc']:.3f})")

    summary = {}
    for name, fold_results in results.items():
        df = pd.DataFrame(fold_results)
        summary[name] = {
            "fold_scores": df,
            "mean_std": df.drop(columns="fold").agg(["mean", "std"]),
        }

    stability = {
        "SHAP_XGBoost": stability_across_folds(shap_rankings, k=10),
        "EBM": stability_across_folds(ebm_rankings, k=10),
    }

    return {
        "results": results,
        "summary": summary,
        "stability": stability,
        "fold_info": fold_info,
        "shap_rankings": shap_rankings,
        "ebm_rankings": ebm_rankings,
    }
