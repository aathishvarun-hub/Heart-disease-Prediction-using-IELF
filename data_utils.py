"""
data_utils.py
=============
Loading and preprocessing utilities for the Cleveland Heart Disease and
Framingham Heart Study datasets, as described in Section 2 of the IELF paper.

Handles:
  - Missing-value imputation (median for continuous, mode for categorical)
  - Feature harmonization / encoding
  - Class-imbalance correction (SMOTE, Framingham only, train-fold only)
  - Standard scaling
  - Stratified 80/20 train/test partitioning

Expected input files (download manually, see paper's Data Availability section):
  Cleveland:   https://www.kaggle.com/datasets/ritwikb3/heart-disease-cleveland
  Framingham:  https://www.kaggle.com/datasets/noeyislearning/framingham-heart-study

If the raw CSVs are not found, synthetic placeholder data with the same
schema is generated so the pipeline can still be exercised end-to-end.
"""

from __future__ import annotations
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

CLEVELAND_COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target",
]

FRAMINGHAM_COLUMNS = [
    "male", "age", "education", "currentSmoker", "cigsPerDay", "BPMeds",
    "prevalentStroke", "prevalentHyp", "diabetes", "totChol", "sysBP",
    "diaBP", "BMI", "heartRate", "glucose", "TenYearCHD",
]


def _make_synthetic_cleveland(n=303, seed=RANDOM_STATE) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "age": rng.integers(29, 77, n),
        "sex": rng.integers(0, 2, n),
        "cp": rng.integers(0, 4, n),
        "trestbps": rng.integers(94, 200, n),
        "chol": rng.integers(126, 564, n),
        "fbs": rng.integers(0, 2, n),
        "restecg": rng.integers(0, 3, n),
        "thalach": rng.integers(71, 202, n),
        "exang": rng.integers(0, 2, n),
        "oldpeak": rng.uniform(0, 6.2, n).round(1),
        "slope": rng.integers(0, 3, n),
        "ca": rng.choice([0, 1, 2, 3, np.nan], n, p=[0.4, 0.25, 0.2, 0.1, 0.05]),
        "thal": rng.choice([3, 6, 7, np.nan], n, p=[0.55, 0.15, 0.25, 0.05]),
    })
    logit = (0.03 * df["age"] + 0.8 * df["cp"] + 0.02 * df["chol"]
             - 0.02 * df["thalach"] + rng.normal(0, 1, n) - 5)
    df["target"] = (logit > np.median(logit)).astype(int)
    return df


def _make_synthetic_framingham(n=4240, seed=RANDOM_STATE) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "male": rng.integers(0, 2, n),
        "age": rng.integers(32, 70, n),
        "education": rng.choice([1, 2, 3, 4, np.nan], n, p=[0.35, 0.25, 0.2, 0.15, 0.05]),
        "currentSmoker": rng.integers(0, 2, n),
        "cigsPerDay": np.where(rng.integers(0, 2, n) == 1, rng.integers(0, 40, n), 0).astype(float),
        "BPMeds": rng.choice([0, 1, np.nan], n, p=[0.92, 0.05, 0.03]),
        "prevalentStroke": rng.choice([0, 1], n, p=[0.995, 0.005]),
        "prevalentHyp": rng.integers(0, 2, n),
        "diabetes": rng.choice([0, 1], n, p=[0.97, 0.03]),
        "totChol": rng.normal(236, 44, n).round(0),
        "sysBP": rng.normal(132, 22, n).round(1),
        "diaBP": rng.normal(83, 12, n).round(1),
        "BMI": rng.normal(25.8, 4.1, n).round(2),
        "heartRate": rng.normal(76, 12, n).round(0),
        "glucose": rng.normal(81, 24, n).round(0),
    })
    # induce missingness matching Table 1 roughly
    for col, frac in [("glucose", 388/4240), ("education", 105/4240),
                       ("totChol", 50/4240), ("BPMeds", 53/4240),
                       ("cigsPerDay", 29/4240), ("BMI", 19/4240), ("heartRate", 1/4240)]:
        mask = rng.random(n) < frac
        df.loc[mask, col] = np.nan
    logit = (0.05 * df["age"] + 0.01 * df["sysBP"].fillna(132)
             + 0.8 * df["diabetes"] + 0.5 * df["prevalentHyp"]
             + rng.normal(0, 1.5, n) - 6.5)
    df["TenYearCHD"] = (logit > np.quantile(logit, 1 - 644/4240)).astype(int)
    return df


def load_cleveland(path: str | None = None) -> pd.DataFrame:
    if path and os.path.exists(path):
        df = pd.read_csv(path, header=None, names=CLEVELAND_COLUMNS, na_values="?")
        # UCI target is 0-4 severity -> binarize
        df["target"] = (df["target"].astype(float) > 0).astype(int)
    else:
        print("[data_utils] Cleveland CSV not found — using synthetic placeholder data.")
        df = _make_synthetic_cleveland()
    for col in ["ca", "thal"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def load_framingham(path: str | None = None) -> pd.DataFrame:
    if path and os.path.exists(path):
        df = pd.read_csv(path)
        df = df.rename(columns={c: c for c in df.columns})
    else:
        print("[data_utils] Framingham CSV not found — using synthetic placeholder data.")
        df = _make_synthetic_framingham()
    return df


def impute_missing(df: pd.DataFrame, categorical_cols: list[str]) -> pd.DataFrame:
    """Median impute continuous columns, mode impute categorical/binary columns."""
    df = df.copy()
    for col in df.columns:
        if df[col].isna().sum() == 0:
            continue
        if col in categorical_cols:
            df[col] = df[col].fillna(df[col].mode(dropna=True).iloc[0])
        else:
            df[col] = df[col].fillna(df[col].median())
    return df


def prepare_cleveland(path: str | None = None):
    df = load_cleveland(path)
    df = impute_missing(df, categorical_cols=["sex", "cp", "fbs", "restecg",
                                               "exang", "slope", "ca", "thal"])
    df = pd.get_dummies(df, columns=["cp", "restecg", "slope", "thal"], drop_first=False)
    X = df.drop(columns=["target"])
    y = df["target"].astype(int)
    return X, y


def prepare_framingham(path: str | None = None):
    df = load_framingham(path)
    cat_cols = ["male", "education", "currentSmoker", "BPMeds",
                "prevalentStroke", "prevalentHyp", "diabetes"]
    df = impute_missing(df, categorical_cols=cat_cols)
    df = df.rename(columns={"male": "sex"})
    df = pd.get_dummies(df, columns=["education"], drop_first=False)
    X = df.drop(columns=["TenYearCHD"])
    y = df["TenYearCHD"].astype(int)
    return X, y


def stratified_split(X, y, test_size=0.2, seed=RANDOM_STATE):
    return train_test_split(X, y, test_size=test_size, stratify=y,
                             random_state=seed)
