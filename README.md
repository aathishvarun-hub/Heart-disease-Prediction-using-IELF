# IELF — Interpretable Ensemble Learning Framework

Reproduction code for: Adekoya, Saeed, Ghaban & Qasem (2025), *"Ensemble learning
approach with explainable AI for improved heart disease prediction"*,
Frontiers in Pharmacology 16:1654681.

Implements the paper's **EBM + XGBoost soft-voting ensemble (IELF)** with SHAP/LIME
explainability, nested 5×5 cross-validation, calibration analysis, and quantitative
explanation-stability metrics (Kendall's τ, Overlap@10).

## Files
| File | Purpose |
|---|---|
| `data_utils.py` | Loading, imputation, harmonization, scaling, SMOTE, train/test split |
| `ielf_model.py` | EBM, XGBoost, soft-voting `IELFEnsemble`, inner-loop hyperparameter + α tuning |
| `metrics_utils.py` | Accuracy/Precision/Recall/F1/AUC/Specificity/LogLoss/MCC, Brier/ECE/MCE, paired t-test & Wilcoxon |
| `explainability.py` | SHAP global importance, EBM global importance, LIME local explanations, stability metrics |
| `nested_cv.py` | 5×5 nested CV driver (outer generalization estimate, inner tuning, leakage-safe SMOTE) |
| `main.py` | End-to-end runner producing all tables/plots into `outputs/` |

## Setup
```bash
pip install interpret xgboost shap lime imbalanced-learn scikit-learn pandas numpy matplotlib scipy
```

## Data
Download the two datasets referenced in the paper's Data Availability statement:
- Cleveland: https://www.kaggle.com/datasets/ritwikb3/heart-disease-cleveland
- Framingham: https://www.kaggle.com/datasets/noeyislearning/framingham-heart-study

If you don't pass a path, the code auto-generates schema-matched **synthetic
placeholder data** so the pipeline still runs end-to-end (useful for testing).

## Usage
```bash
# Full reproduction (both datasets, full hyperparameter grids, 5x5 nested CV)
python main.py --cleveland_path path/to/cleveland.csv --framingham_path path/to/framingham.csv

# Quick smoke test (small grids, 2 outer folds) — verifies the pipeline runs correctly
python main.py --only cleveland --fast
```

⚠️ **Compute note:** the full EBM hyperparameter grid (36 combinations, some with
pairwise interactions) plus XGBoost's grid, repeated across 5 outer folds × 2
datasets, is compute-intensive (the paper reports ~1,388s/EBM fit-cycle on
Framingham using a Colab T4). Use `--fast` first to confirm everything runs,
then launch the full run with more time/compute.

## Outputs (written to `outputs/`)
- `<dataset>_test_performance.csv` — held-out test-set metrics (Tables 6–7)
- `<dataset>_calibration.csv` — Brier / ECE / MCE (Figure 17)
- `<dataset>_cv_results.csv`, `<dataset>_cv_summary.csv` — per-fold and mean±SD nested-CV metrics (Table 8/9, Figures 10/12)
- `<dataset>_paired_tests.csv` — IELF vs. EBM / XGBoost paired t-test & Wilcoxon (Tables 8–9)
- `<dataset>_stability.csv` — Kendall's τ and Overlap@10 across folds (Table 3)
- `<dataset>_shap_importance.png`, `<dataset>_ebm_importance.png` — global feature importance (Figures 14–15)
- `<dataset>_lime_example.csv` — a representative local LIME explanation (Figure 16)

## Key design choices (matching the paper)
- **No leakage:** SMOTE and feature scaling are fit only on each outer-fold's
  training partition; hyperparameters and the ensemble weight α are selected only
  within the inner loop, then frozen before outer-fold scoring.
- **Ensemble weight α** is chosen from `{0.0, 0.25, 0.5, 0.75, 1.0}` by maximizing
  validation AUC (Section 2.14).
- **Explanation stability** is computed as mean pairwise Kendall's τ-b and
  Overlap@10 between the SHAP/EBM global rankings of every pair of outer folds
  (Section 2.13).
