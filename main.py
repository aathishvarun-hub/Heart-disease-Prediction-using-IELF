"""
main.py
=======
End-to-end runner reproducing the IELF experiments (paper: Adekoya et al.,
2025, "Ensemble learning approach with explainable AI for improved heart
disease prediction", Front. Pharmacol. 16:1654681).

Usage:
    python main.py --cleveland_path path/to/cleveland.csv \
                    --framingham_path path/to/framingham.csv

If paths are omitted or files are not found, synthetic placeholder data with
the same schema is used automatically (see data_utils.py) so the full
pipeline can still be run/tested end-to-end.

Outputs (written to ./outputs/):
  - <dataset>_cv_results.csv          per-fold metrics for EBM/XGBoost/IELF
  - <dataset>_cv_summary.csv          mean +- SD summary table
  - <dataset>_paired_tests.csv        paired t-test / Wilcoxon: IELF vs base models
  - <dataset>_stability.csv           Kendall's tau / Overlap@10
  - <dataset>_calibration.csv         Brier / ECE / MCE for tuned test-set models
  - <dataset>_shap_importance.png     bar chart of SHAP feature importance
  - <dataset>_ebm_importance.png      bar chart of EBM feature importance
"""

from __future__ import annotations
import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from data_utils import prepare_cleveland, prepare_framingham, stratified_split, RANDOM_STATE
from nested_cv import run_nested_cv
from ielf_model import inner_loop_fit, IELFEnsemble
from metrics_utils import full_metrics, calibration_metrics, paired_significance

OUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUT_DIR, exist_ok=True)


def plot_importance(ranking: pd.Series, title: str, out_path: str, top_n: int = 10):
    top = ranking.sort_values(ascending=True).tail(top_n)
    plt.figure(figsize=(7, 5))
    plt.barh(top.index, top.values, color="#3d7cb8")
    plt.xlabel("Mean |importance|")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def run_pipeline(dataset_name: str, X: pd.DataFrame, y: pd.Series, use_smote: bool,
                  fast: bool = False):
    tune_kwargs = {"n_iter_xgb": 3, "n_iter_ebm": 2} if fast else {}
    print(f"\n===== {dataset_name} =====")
    print(f"n={len(X)}, positives={y.sum()}, negatives={len(y) - y.sum()}, "
          f"imbalance ratio={ (len(y)-y.sum()) / max(y.sum(),1):.2f}:1")

    # ---- Held-out test-set performance (single split, tuned via inner CV) ----
    X_train, X_test, y_train, y_test = stratified_split(X, y)
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train_s = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns, index=X_train.index)
    X_test_s = pd.DataFrame(scaler.transform(X_test), columns=X.columns, index=X_test.index)

    if use_smote:
        from imblearn.over_sampling import SMOTE
        X_train_res, y_train_res = SMOTE(random_state=RANDOM_STATE).fit_resample(X_train_s, y_train)
    else:
        X_train_res, y_train_res = X_train_s, y_train

    best_ebm, best_xgb, best_alpha, tune_info = inner_loop_fit(X_train_res, y_train_res, **tune_kwargs)
    ielf = IELFEnsemble(ebm=best_ebm, xgb=best_xgb, alpha=best_alpha)

    test_rows = []
    calib_rows = []
    for name, model in [("EBM", best_ebm), ("XGBoost", best_xgb), ("IELF", ielf)]:
        proba = model.predict_proba(X_test_s)[:, 1]
        pred = (proba >= 0.5).astype(int)
        m = full_metrics(y_test.values, pred, proba)
        m["model"] = name
        test_rows.append(m)
        c = calibration_metrics(y_test.values, proba)
        c["model"] = name
        calib_rows.append(c)

    test_df = pd.DataFrame(test_rows).set_index("model")
    calib_df = pd.DataFrame(calib_rows).set_index("model")
    print("\nHeld-out test-set performance:\n", test_df.round(4))
    print("\nCalibration (Brier / ECE / MCE):\n", calib_df.round(4))

    test_df.to_csv(os.path.join(OUT_DIR, f"{dataset_name}_test_performance.csv"))
    calib_df.to_csv(os.path.join(OUT_DIR, f"{dataset_name}_calibration.csv"))

    # ---- Nested 5x5 CV ----
    n_splits = 2 if fast else 5
    cv_out = run_nested_cv(X, y, dataset_name, use_smote=use_smote, n_splits=n_splits, **tune_kwargs)

    all_fold_rows = []
    for name in ["EBM", "XGBoost", "IELF"]:
        df = cv_out["summary"][name]["fold_scores"].copy()
        df["model"] = name
        all_fold_rows.append(df)
    fold_df = pd.concat(all_fold_rows, ignore_index=True)
    fold_df.to_csv(os.path.join(OUT_DIR, f"{dataset_name}_cv_results.csv"), index=False)

    mean_std_rows = []
    for name in ["EBM", "XGBoost", "IELF"]:
        ms = cv_out["summary"][name]["mean_std"]
        row = ms.loc["mean"].add_suffix("_mean").to_dict()
        row.update(ms.loc["std"].add_suffix("_std").to_dict())
        row["model"] = name
        mean_std_rows.append(row)
    summary_df = pd.DataFrame(mean_std_rows).set_index("model")
    summary_df.to_csv(os.path.join(OUT_DIR, f"{dataset_name}_cv_summary.csv"))
    print("\n5-fold CV mean +- SD:\n", summary_df.round(4))

    # ---- Paired significance: IELF vs EBM, IELF vs XGBoost ----
    paired_rows = []
    ielf_folds = cv_out["summary"]["IELF"]["fold_scores"]
    for base in ["EBM", "XGBoost"]:
        base_folds = cv_out["summary"][base]["fold_scores"]
        for metric in ["accuracy", "recall", "f1", "auc", "log_loss", "mcc"]:
            stats_res = paired_significance(ielf_folds[metric].tolist(), base_folds[metric].tolist())
            stats_res["comparison"] = f"IELF vs. {base}"
            stats_res["metric"] = metric
            paired_rows.append(stats_res)
    paired_df = pd.DataFrame(paired_rows)
    paired_df.to_csv(os.path.join(OUT_DIR, f"{dataset_name}_paired_tests.csv"), index=False)
    print("\nPaired significance (IELF vs base models):\n",
          paired_df[["comparison", "metric", "mean_a", "mean_b", "t_p", "wilcoxon_p"]].round(4))

    # ---- Explanation stability (Kendall's tau, Overlap@10) ----
    stab_rows = []
    for model_name, key in [("XGBoost (SHAP)", "SHAP_XGBoost"), ("EBM", "EBM")]:
        s = cv_out["stability"][key]
        s["model"] = model_name
        stab_rows.append(s)
    stab_df = pd.DataFrame(stab_rows).set_index("model")
    stab_df.to_csv(os.path.join(OUT_DIR, f"{dataset_name}_stability.csv"))
    print("\nExplanation stability across folds:\n", stab_df.round(3))

    # ---- Global importance plots (from the held-out fit) ----
    from explainability import shap_global_importance, ebm_global_importance
    shap_imp = shap_global_importance(best_xgb, X_test_s)
    ebm_imp = ebm_global_importance(best_ebm)
    plot_importance(shap_imp, f"{dataset_name}: SHAP global feature importance (XGBoost)",
                     os.path.join(OUT_DIR, f"{dataset_name}_shap_importance.png"))
    plot_importance(ebm_imp, f"{dataset_name}: EBM global feature importance",
                     os.path.join(OUT_DIR, f"{dataset_name}_ebm_importance.png"))

    # ---- One representative LIME explanation ----
    from explainability import lime_explain_instance
    instance = X_test_s.iloc[0]
    lime_exp = lime_explain_instance(ielf, X_train_res, instance)
    lime_list = lime_exp.as_list()
    pd.DataFrame(lime_list, columns=["rule", "contribution"]).to_csv(
        os.path.join(OUT_DIR, f"{dataset_name}_lime_example.csv"), index=False)

    return {
        "test_performance": test_df,
        "calibration": calib_df,
        "cv_summary": summary_df,
        "paired_tests": paired_df,
        "stability": stab_df,
        "tuned_models": {"ebm": best_ebm, "xgb": best_xgb, "alpha": best_alpha},
    }


def main():
    parser = argparse.ArgumentParser(description="Run the IELF heart-disease prediction pipeline.")
    parser.add_argument("--cleveland_path", type=str, default=None)
    parser.add_argument("--framingham_path", type=str, default=None)
    parser.add_argument("--only", choices=["cleveland", "framingham", "both"], default="both")
    parser.add_argument("--fast", action="store_true",
                         help="Use a small hyperparameter grid + 2 outer folds for a quick smoke test.")
    args = parser.parse_args()

    kwargs = {}
    if args.fast:
        import nested_cv, ielf_model
        kwargs = dict(n_iter_xgb=3, n_iter_ebm=2)

    if args.only in ("cleveland", "both"):
        X_c, y_c = prepare_cleveland(args.cleveland_path)
        run_pipeline("cleveland", X_c, y_c, use_smote=False, fast=args.fast)

    if args.only in ("framingham", "both"):
        X_f, y_f = prepare_framingham(args.framingham_path)
        run_pipeline("framingham", X_f, y_f, use_smote=True, fast=args.fast)

    print(f"\nAll outputs written to: {OUT_DIR}")


if __name__ == "__main__":
    main()
