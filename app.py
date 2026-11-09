# ============================================================
# app.py
# ============================================================
# IELF Heart Disease Prediction Web Application
#
# Features:
#   - Patient Management
#   - Cleveland Heart Disease Prediction
#   - IELF / EBM / XGBoost
#   - SHAP Explanation
#   - LIME Explanation
#   - PDF Report Generation
#   - Experiment Status
#   - Health / Status APIs
#
# IMPORTANT:
#   This application is for academic/research demonstration.
#   It is NOT a medical diagnostic system.
# ============================================================

from __future__ import annotations

import base64
import json
import os
import re
import subprocess
import sys
import threading
from datetime import datetime
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from flask import (
    Flask,
    jsonify,
    redirect,
    render_template_string,
    request,
    send_file,
    url_for,
)

# ------------------------------------------------------------
# Optional libraries
# ------------------------------------------------------------

try:
    import shap
except Exception:
    shap = None

try:
    from lime.lime_tabular import LimeTabularExplainer
except Exception:
    LimeTabularExplainer = None

try:
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import (
        ParagraphStyle,
        getSampleStyleSheet,
    )
    from reportlab.lib.units import inch
    from reportlab.platypus import (
        Image as RLImage,
        PageBreak,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )

    REPORTLAB_AVAILABLE = True

except Exception:
    REPORTLAB_AVAILABLE = False


# ============================================================
# FLASK
# ============================================================

app = Flask(__name__)

app.config["SECRET_KEY"] = "ielf-heart-disease-project"


# ============================================================
# DIRECTORIES
# ============================================================

BASE_DIR = os.path.abspath(
    os.path.dirname(__file__)
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models",
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "outputs",
)

OUTPUT_MODEL_DIR = os.path.join(
    OUTPUT_DIR,
    "models",
)

DATA_DIR = os.path.join(
    BASE_DIR,
    "data",
)

REPORT_DIR = os.path.join(
    OUTPUT_DIR,
    "reports",
)

os.makedirs(
    MODEL_DIR,
    exist_ok=True,
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True,
)

os.makedirs(
    OUTPUT_MODEL_DIR,
    exist_ok=True,
)

os.makedirs(
    DATA_DIR,
    exist_ok=True,
)

os.makedirs(
    REPORT_DIR,
    exist_ok=True,
)


# ============================================================
# PATIENT STORAGE
# ============================================================

PATIENT_FILE = os.path.join(
    BASE_DIR,
    "patients.json",
)


def load_patients() -> List[Dict[str, Any]]:
    """Load patient records."""

    if not os.path.exists(PATIENT_FILE):
        return []

    try:
        with open(
            PATIENT_FILE,
            "r",
            encoding="utf-8",
        ) as file:

            data = json.load(file)

        if isinstance(data, list):
            return data

        return []

    except Exception as exc:

        print(
            "Patient loading error:",
            exc,
        )

        return []


def save_patients(
    patients: List[Dict[str, Any]],
) -> None:
    """Save patient records."""

    temporary_file = (
        PATIENT_FILE + ".tmp"
    )

    try:

        with open(
            temporary_file,
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                patients,
                file,
                indent=4,
                ensure_ascii=False,
            )

        os.replace(
            temporary_file,
            PATIENT_FILE,
        )

    except Exception as exc:

        print(
            "Patient saving error:",
            exc,
        )

        try:
            if os.path.exists(
                temporary_file
            ):
                os.remove(
                    temporary_file
                )
        except Exception:
            pass


def generate_patient_id(
    patients: List[Dict[str, Any]],
) -> str:
    """Generate P001, P002, ..."""

    numbers = []

    for patient in patients:

        value = str(
            patient.get(
                "patient_id",
                "",
            )
        )

        match = re.search(
            r"(\d+)$",
            value,
        )

        if match:
            numbers.append(
                int(match.group(1))
            )

    next_number = (
        max(numbers) + 1
        if numbers
        else 1
    )

    return f"P{next_number:03d}"


def find_patient(
    patient_id: str,
) -> Optional[Dict[str, Any]]:

    for patient in load_patients():

        if str(
            patient.get("patient_id")
        ) == str(patient_id):

            return patient

    return None


# ============================================================
# CLEVELAND FEATURES
# ============================================================

CLEVELAND_FEATURES = [
    "age",
    "sex",
    "trestbps",
    "chol",
    "fbs",
    "thalach",
    "exang",
    "oldpeak",
    "ca",
    "cp_0",
    "cp_1",
    "cp_2",
    "cp_3",
    "restecg_0",
    "restecg_1",
    "restecg_2",
    "slope_0",
    "slope_1",
    "slope_2",
    "thal_3.0",
    "thal_6.0",
    "thal_7.0",
]


READABLE_FEATURES = {
    "age": "Age",
    "sex": "Gender",
    "trestbps": "Resting Blood Pressure",
    "chol": "Cholesterol",
    "fbs": "Fasting Blood Sugar",
    "thalach": "Maximum Heart Rate",
    "exang": "Exercise Induced Angina",
    "oldpeak": "ST Depression (Oldpeak)",
    "ca": "Major Vessels (CA)",
    "cp_0": "Chest Pain Type 0",
    "cp_1": "Chest Pain Type 1",
    "cp_2": "Chest Pain Type 2",
    "cp_3": "Chest Pain Type 3",
    "restecg_0": "Resting ECG Type 0",
    "restecg_1": "Resting ECG Type 1",
    "restecg_2": "Resting ECG Type 2",
    "slope_0": "Slope Type 0",
    "slope_1": "Slope Type 1",
    "slope_2": "Slope Type 2",
    "thal_3.0": "Thalassemia Normal",
    "thal_6.0": "Thalassemia Type 6",
    "thal_7.0": "Thalassemia Type 7",
}


def readable_feature_name(
    feature: str,
) -> str:

    return READABLE_FEATURES.get(
        feature,
        feature,
    )


# ============================================================
# SAFE CONVERSION
# ============================================================

def safe_float(
    value: Any,
    default: float = 0.0,
) -> float:

    try:

        if value is None:
            return default

        if isinstance(value, str):
            value = value.strip()

        if value == "":
            return default

        result = float(value)

        if not np.isfinite(result):
            return default

        return result

    except Exception:
        return default


def safe_int(
    value: Any,
    default: int = 0,
) -> int:

    try:

        if value is None:
            return default

        if isinstance(value, str):
            value = value.strip()

        if value == "":
            return default

        return int(float(value))

    except Exception:
        return default


# ============================================================
# FEATURE CREATION
# ============================================================

def create_cleveland_features(
    form: Any,
) -> pd.DataFrame:
    """
    Convert HTML form data into the exact 23-feature
    Cleveland representation used by the application.
    """

    row = {
        feature: 0.0
        for feature in CLEVELAND_FEATURES
    }

    row["age"] = safe_float(
        form.get("age")
    )

    row["sex"] = safe_float(
        form.get("sex")
    )

    row["trestbps"] = safe_float(
        form.get("trestbps")
    )

    row["chol"] = safe_float(
        form.get("chol")
    )

    row["fbs"] = safe_float(
        form.get("fbs")
    )

    row["thalach"] = safe_float(
        form.get("thalach")
    )

    row["exang"] = safe_float(
        form.get("exang")
    )

    row["oldpeak"] = safe_float(
        form.get("oldpeak")
    )

    row["ca"] = safe_float(
        form.get("ca")
    )

    cp = safe_int(
        form.get("cp"),
        0,
    )

    if cp in range(4):
        row[f"cp_{cp}"] = 1.0

    restecg = safe_int(
        form.get("restecg"),
        0,
    )

    if restecg in range(3):
        row[f"restecg_{restecg}"] = 1.0

    slope = safe_int(
        form.get("slope"),
        0,
    )

    if slope in range(3):
        row[f"slope_{slope}"] = 1.0

    thal = safe_int(
        form.get("thal"),
        3,
    )

    if thal == 3:
        row["thal_3.0"] = 1.0

    elif thal == 6:
        row["thal_6.0"] = 1.0

    elif thal == 7:
        row["thal_7.0"] = 1.0

    return pd.DataFrame(
        [row],
        columns=CLEVELAND_FEATURES,
    )


# ============================================================
# MODEL SEARCH
# ============================================================

def model_directories() -> List[str]:

    directories = [
        os.path.join(
            OUTPUT_MODEL_DIR,
            "cleveland",
        ),
        os.path.join(
            MODEL_DIR,
            "cleveland",
        ),
        OUTPUT_MODEL_DIR,
        MODEL_DIR,
        OUTPUT_DIR,
        BASE_DIR,
    ]

    result = []

    for directory in directories:

        if (
            directory
            and os.path.isdir(directory)
            and directory not in result
        ):
            result.append(directory)

    return result


def load_model_file(
    filenames: List[str],
) -> Any:
    """
    Search all common model directories.
    """

    for directory in model_directories():

        for filename in filenames:

            path = os.path.join(
                directory,
                filename,
            )

            if not os.path.isfile(path):
                continue

            try:

                print(
                    "Loading model:",
                    path,
                )

                return joblib.load(
                    path
                )

            except Exception as exc:

                print(
                    "Could not load:",
                    path,
                    exc,
                )

    return None


# ============================================================
# MODEL ATTRIBUTE EXTRACTION
# ============================================================

def get_object_attribute(
    obj: Any,
    names: List[str],
) -> Any:

    if obj is None:
        return None

    for name in names:

        try:

            value = getattr(
                obj,
                name,
                None,
            )

            if value is not None:
                return value

        except Exception:
            pass

    return None


def extract_from_bundle(
    bundle: Any,
) -> Dict[str, Any]:

    result = {}

    if bundle is None:
        return result

    if isinstance(bundle, dict):

        for key, value in bundle.items():

            key_lower = str(
                key
            ).lower()

            if (
                key_lower
                in {
                    "ielf",
                    "model",
                    "ensemble",
                    "classifier",
                }
            ):
                result["ielf"] = value

            elif (
                key_lower
                in {
                    "ebm",
                    "ebm_model",
                }
            ):
                result["ebm"] = value

            elif (
                key_lower
                in {
                    "xgb",
                    "xgboost",
                    "xgb_model",
                }
            ):
                result["xgb"] = value

            elif (
                key_lower
                in {
                    "scaler",
                    "standardscaler",
                }
            ):
                result["scaler"] = value

            elif key_lower in {
                "alpha",
                "weight",
                "ebm_weight",
            }:

                result["alpha"] = value

            elif key_lower in {
                "threshold",
                "decision_threshold",
            }:

                result["threshold"] = value

            elif key_lower in {
                "features",
                "feature_names",
            }:

                result["features"] = value

    else:

        ielf = get_object_attribute(
            bundle,
            [
                "ielf",
                "model",
                "ensemble",
                "classifier",
            ],
        )

        ebm = get_object_attribute(
            bundle,
            [
                "ebm",
                "ebm_model",
            ],
        )

        xgb = get_object_attribute(
            bundle,
            [
                "xgb",
                "xgb_model",
                "xgboost",
            ],
        )

        scaler = get_object_attribute(
            bundle,
            [
                "scaler",
                "standard_scaler",
            ],
        )

        alpha = get_object_attribute(
            bundle,
            [
                "alpha",
                "weight",
                "ebm_weight",
            ],
        )

        threshold = get_object_attribute(
            bundle,
            [
                "threshold",
                "decision_threshold",
            ],
        )

        features = get_object_attribute(
            bundle,
            [
                "features",
                "feature_names",
                "feature_names_in_",
            ],
        )

        if ielf is not None:
            result["ielf"] = ielf

        if ebm is not None:
            result["ebm"] = ebm

        if xgb is not None:
            result["xgb"] = xgb

        if scaler is not None:
            result["scaler"] = scaler

        if alpha is not None:
            result["alpha"] = alpha

        if threshold is not None:
            result["threshold"] = threshold

        if features is not None:
            result["features"] = features

    return result


# ============================================================
# LOAD CLEVELAND MODELS
# ============================================================

def load_cleveland_models() -> Dict[str, Any]:

    models = {
        "ielf": None,
        "ebm": None,
        "xgb": None,
        "scaler": None,
        "features": None,
        "alpha": 0.5,
        "threshold": 0.5,
    }

    # --------------------------------------------------------
    # Try bundle first
    # --------------------------------------------------------

    bundle = load_model_file(
        [
            "ielf.pkl",
            "IELF.pkl",
            "model.pkl",
            "models.pkl",
            "ielf_model.pkl",
            "ensemble.pkl",
        ]
    )

    if bundle is not None:

        extracted = (
            extract_from_bundle(
                bundle
            )
        )

        for key, value in extracted.items():

            if value is not None:
                models[key] = value

    # --------------------------------------------------------
    # Load individual models if necessary
    # --------------------------------------------------------

    if models["ielf"] is None:

        models["ielf"] = load_model_file(
            [
                "ielf.pkl",
                "ielf_model.pkl",
                "ensemble.pkl",
                "model.pkl",
            ]
        )

    if models["ebm"] is None:

        models["ebm"] = load_model_file(
            [
                "ebm.pkl",
                "ebm_model.pkl",
            ]
        )

    if models["xgb"] is None:

        models["xgb"] = load_model_file(
            [
                "xgb.pkl",
                "xgb_model.pkl",
                "xgboost.pkl",
            ]
        )

    if models["scaler"] is None:

        models["scaler"] = load_model_file(
            [
                "scaler.pkl",
                "standard_scaler.pkl",
            ]
        )

    if models["features"] is None:

        models["features"] = load_model_file(
            [
                "features.pkl",
                "feature_names.pkl",
            ]
        )

    # --------------------------------------------------------
    # Try nested attributes from IELF
    # --------------------------------------------------------

    ielf = models["ielf"]

    if ielf is not None:

        if models["ebm"] is None:

            models["ebm"] = get_object_attribute(
                ielf,
                [
                    "ebm",
                    "ebm_model",
                    "ebm_classifier",
                ],
            )

        if models["xgb"] is None:

            models["xgb"] = get_object_attribute(
                ielf,
                [
                    "xgb",
                    "xgb_model",
                    "xgboost",
                    "xgboost_model",
                ],
            )

        alpha = get_object_attribute(
            ielf,
            [
                "alpha",
                "weight",
                "ebm_weight",
            ],
        )

        if alpha is not None:

            try:
                models["alpha"] = float(
                    alpha
                )
            except Exception:
                pass

        threshold = get_object_attribute(
            ielf,
            [
                "threshold",
                "decision_threshold",
            ],
        )

        if threshold is not None:

            try:
                models["threshold"] = float(
                    threshold
                )
            except Exception:
                pass

    # --------------------------------------------------------
    # Normalize alpha
    # --------------------------------------------------------

    try:

        models["alpha"] = float(
            models.get(
                "alpha",
                0.5,
            )
        )

    except Exception:

        models["alpha"] = 0.5

    models["alpha"] = float(
        np.clip(
            models["alpha"],
            0.0,
            1.0,
        )
    )

    # --------------------------------------------------------
    # Normalize threshold
    # --------------------------------------------------------

    try:

        models["threshold"] = float(
            models.get(
                "threshold",
                0.5,
            )
        )

    except Exception:

        models["threshold"] = 0.5

    models["threshold"] = float(
        np.clip(
            models["threshold"],
            0.01,
            0.99,
        )
    )

    print(
        "\n========== MODEL STATUS =========="
    )

    print(
        "IELF:",
        models["ielf"] is not None,
    )

    print(
        "EBM:",
        models["ebm"] is not None,
    )

    print(
        "XGBoost:",
        models["xgb"] is not None,
    )

    print(
        "Scaler:",
        models["scaler"] is not None,
    )

    print(
        "Alpha:",
        models["alpha"],
    )

    print(
        "Threshold:",
        models["threshold"],
    )

    print(
        "==================================\n"
    )

    return models


# ============================================================
# GLOBAL MODEL CACHE
# ============================================================

MODELS = load_cleveland_models()


def reload_models() -> Dict[str, Any]:

    global MODELS

    MODELS = load_cleveland_models()

    return MODELS


# ============================================================
# FEATURE ADAPTATION
# ============================================================

def feature_names_from_model(
    model: Any,
) -> Optional[List[str]]:

    if model is None:
        return None

    candidates = [
        "feature_names_in_",
        "feature_names",
    ]

    for attribute in candidates:

        try:

            value = getattr(
                model,
                attribute,
                None,
            )

            if value is not None:

                values = list(value)

                if values:
                    return [
                        str(item)
                        for item in values
                    ]

        except Exception:
            pass

    # Pipeline / wrapped model

    try:

        steps = getattr(
            model,
            "named_steps",
            None,
        )

        if steps:

            for step in reversed(
                list(steps.values())
            ):

                names = (
                    feature_names_from_model(
                        step
                    )
                )

                if names:
                    return names

    except Exception:
        pass

    return None


def adapt_features(
    model: Any,
    X: pd.DataFrame,
) -> pd.DataFrame:

    if X is None:
        raise ValueError(
            "Feature data is None."
        )

    result = X.copy()

    model_features = (
        feature_names_from_model(
            model
        )
    )

    if model_features:

        # Exact names available

        result = result.reindex(
            columns=model_features,
            fill_value=0.0,
        )

    else:

        # Default application representation

        result = result.reindex(
            columns=CLEVELAND_FEATURES,
            fill_value=0.0,
        )

    result = (
        result.replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .fillna(0.0)
    )

    return result.astype(float)


# ============================================================
# SCALER
# ============================================================

def model_has_pipeline(
    model: Any,
) -> bool:

    if model is None:
        return False

    try:

        if getattr(
            model,
            "steps",
            None,
        ):
            return True

        if getattr(
            model,
            "named_steps",
            None,
        ):
            return True

    except Exception:
        pass

    return False


def apply_scaler(
    X: pd.DataFrame,
    scaler: Any,
) -> pd.DataFrame:

    """
    Apply the external scaler only when its feature count
    exactly matches the input.

    Pipelines are not externally scaled.
    """

    if scaler is None:
        return X

    try:

        # Never apply external scaler blindly to pipeline data.
        if model_has_pipeline(
            scaler
        ):
            return X

        expected = getattr(
            scaler,
            "n_features_in_",
            None,
        )

        if expected is None:
            return X

        if int(expected) != X.shape[1]:

            return X

        transformed = scaler.transform(
            X
        )

        return pd.DataFrame(
            transformed,
            columns=X.columns,
            index=X.index,
        )

    except Exception as exc:

        print(
            "Scaler skipped:",
            exc,
        )

        return X


# ============================================================
# MODEL PROBABILITY
# ============================================================

def probability_from_model(
    model: Any,
    X: pd.DataFrame,
) -> Optional[float]:

    if model is None:
        return None

    # --------------------------------------------------------
    # predict_proba
    # --------------------------------------------------------

    try:

        if hasattr(
            model,
            "predict_proba",
        ):

            probability = model.predict_proba(
                X
            )

            probability = np.asarray(
                probability
            )

            if probability.ndim == 2:

                if probability.shape[1] >= 2:
                    value = probability[
                        0,
                        1,
                    ]
                else:
                    value = probability[
                        0,
                        0,
                    ]

            else:

                value = probability.flatten()[
                    0
                ]

            return float(
                np.clip(
                    value,
                    0.0,
                    1.0,
                )
            )

    except Exception as exc:

        print(
            "predict_proba failed:",
            type(model).__name__,
            exc,
        )

    # --------------------------------------------------------
    # decision_function
    # --------------------------------------------------------

    try:

        if hasattr(
            model,
            "decision_function",
        ):

            score = model.decision_function(
                X
            )

            score = float(
                np.asarray(
                    score
                ).flatten()[0]
            )

            # Logistic conversion

            probability = (
                1.0
                / (
                    1.0
                    + np.exp(
                        -np.clip(
                            score,
                            -50,
                            50,
                        )
                    )
                )
            )

            return float(
                np.clip(
                    probability,
                    0.0,
                    1.0,
                )
            )

    except Exception as exc:

        print(
            "decision_function failed:",
            exc,
        )

    # --------------------------------------------------------
    # predict fallback
    # --------------------------------------------------------

    try:

        prediction = model.predict(
            X
        )

        value = float(
            np.asarray(
                prediction
            ).flatten()[0]
        )

        return float(
            np.clip(
                value,
                0.0,
                1.0,
            )
        )

    except Exception as exc:

        print(
            "predict failed:",
            exc,
        )

    return None


# ============================================================
# FINAL PREDICTION
# ============================================================

def calculate_prediction(
    models: Dict[str, Any],
    X: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Calculate final prediction.

    Priority:
        1. IELF
        2. EBM + XGBoost soft voting
        3. EBM
        4. XGBoost
    """

    if X is None or X.empty:

        raise ValueError(
            "Input feature data is empty."
        )

    scaler = models.get(
        "scaler"
    )

    ebm = models.get(
        "ebm"
    )

    xgb = models.get(
        "xgb"
    )

    ielf = models.get(
        "ielf"
    )

    try:

        alpha = float(
            models.get(
                "alpha",
                0.5,
            )
        )

    except Exception:

        alpha = 0.5

    alpha = float(
        np.clip(
            alpha,
            0.0,
            1.0,
        )
    )

    # --------------------------------------------------------
    # EBM
    # --------------------------------------------------------

    ebm_probability = None

    if ebm is not None:

        try:

            X_ebm = adapt_features(
                ebm,
                X,
            )

            X_ebm = apply_scaler(
                X_ebm,
                scaler,
            )

            ebm_probability = (
                probability_from_model(
                    ebm,
                    X_ebm,
                )
            )

        except Exception as exc:

            print(
                "EBM prediction failed:",
                exc,
            )

    # --------------------------------------------------------
    # XGBoost
    # --------------------------------------------------------

    xgb_probability = None

    if xgb is not None:

        try:

            X_xgb = adapt_features(
                xgb,
                X,
            )

            X_xgb = apply_scaler(
                X_xgb,
                scaler,
            )

            xgb_probability = (
                probability_from_model(
                    xgb,
                    X_xgb,
                )
            )

        except Exception as exc:

            print(
                "XGBoost prediction failed:",
                exc,
            )

    # --------------------------------------------------------
    # IELF
    # --------------------------------------------------------

    ielf_probability = None

    if ielf is not None:

        try:

            X_ielf = adapt_features(
                ielf,
                X,
            )

            X_ielf = apply_scaler(
                X_ielf,
                scaler,
            )

            ielf_probability = (
                probability_from_model(
                    ielf,
                    X_ielf,
                )
            )

        except Exception as exc:

            print(
                "IELF prediction failed:",
                exc,
            )

    # --------------------------------------------------------
    # Final probability
    # --------------------------------------------------------

    prediction_source = ""

    if ielf_probability is not None:

        final_probability = (
            ielf_probability
        )

        prediction_source = (
            "IELF Ensemble"
        )

    elif (
        ebm_probability is not None
        and xgb_probability is not None
    ):

        final_probability = (
            alpha * ebm_probability
            + (1.0 - alpha)
            * xgb_probability
        )

        prediction_source = (
            "EBM + XGBoost Soft Voting"
        )

    elif ebm_probability is not None:

        final_probability = (
            ebm_probability
        )

        prediction_source = "EBM"

    elif xgb_probability is not None:

        final_probability = (
            xgb_probability
        )

        prediction_source = "XGBoost"

    else:

        raise RuntimeError(
            "No trained prediction model is available."
        )

    final_probability = float(
        np.clip(
            final_probability,
            0.0,
            1.0,
        )
    )

    # --------------------------------------------------------
    # Threshold
    # --------------------------------------------------------

    try:

        threshold = float(
            models.get(
                "threshold",
                0.5,
            )
        )

    except Exception:

        threshold = 0.5

    threshold = float(
        np.clip(
            threshold,
            0.01,
            0.99,
        )
    )

    if final_probability >= threshold:

        risk_level = "HIGH RISK"

        prediction_class = 1

    else:

        risk_level = "LOW RISK"

        prediction_class = 0

    return {
        "probability": final_probability,
        "risk_level": risk_level,
        "prediction_class": prediction_class,
        "threshold": threshold,
        "prediction_source": prediction_source,
        "ebm_probability": ebm_probability,
        "xgb_probability": xgb_probability,
        "ielf_probability": ielf_probability,
        "X_model": X.copy(),
    }


# ============================================================
# SHAP
# ============================================================

def generate_shap(
    models: Dict[str, Any],
    X: pd.DataFrame,
) -> Tuple[Optional[str], List[Dict[str, Any]]]:
    """
    Generate SHAP explanation.

    SHAP explains the XGBoost component.

    The application clearly labels this as an XGBoost
    component explanation rather than incorrectly claiming
    that TreeSHAP automatically explains the complete
    custom IELF ensemble.
    """

    if shap is None:

        print(
            "SHAP is not installed."
        )

        return None, []

    if X is None or X.empty:

        return None, []

    xgb_model = models.get(
        "xgb"
    )

    if xgb_model is None:

        print(
            "XGBoost model unavailable for SHAP."
        )

        return None, []

    try:

        # ----------------------------------------------------
        # Prepare XGBoost input
        # ----------------------------------------------------

        X_model = adapt_features(
            xgb_model,
            X,
        )

        X_model = apply_scaler(
            X_model,
            models.get("scaler"),
        )

        X_model = X_model.astype(
            float
        )

        # ----------------------------------------------------
        # TreeSHAP
        # ----------------------------------------------------

        explainer = shap.TreeExplainer(
            xgb_model
        )

        shap_result = (
            explainer.shap_values(
                X_model
            )
        )

        # ----------------------------------------------------
        # Normalize SHAP versions
        # ----------------------------------------------------

        if isinstance(
            shap_result,
            list,
        ):

            if len(shap_result) == 1:

                values = np.asarray(
                    shap_result[0]
                )

            else:

                values = np.asarray(
                    shap_result[-1]
                )

        elif hasattr(
            shap_result,
            "values",
        ):

            values = np.asarray(
                shap_result.values
            )

        else:

            values = np.asarray(
                shap_result
            )

        # ----------------------------------------------------
        # Dimensions
        # ----------------------------------------------------

        if values.ndim == 3:

            values = values[
                0,
                :,
                -1,
            ]

        elif values.ndim == 2:

            values = values[0]

        elif values.ndim == 1:

            values = values

        else:

            print(
                "Unsupported SHAP shape:",
                values.shape,
            )

            return None, []

        values = np.asarray(
            values,
            dtype=float,
        ).flatten()

        feature_names = list(
            X_model.columns
        )

        if len(values) != len(
            feature_names
        ):

            print(
                "SHAP feature mismatch:",
                len(values),
                len(feature_names),
            )

            return None, []

        feature_values = (
            X_model.iloc[0]
            .astype(float)
            .values
        )

        rows = []

        for name, value, contribution in zip(
            feature_names,
            feature_values,
            values,
        ):

            contribution = float(
                contribution
            )

            value = float(
                value
            )

            if contribution > 0:

                direction = (
                    "Increases predicted risk"
                )

            elif contribution < 0:

                direction = (
                    "Decreases predicted risk"
                )

            else:

                direction = (
                    "Little or no contribution"
                )

            rows.append(
                {
                    "feature": readable_feature_name(
                        name
                    ),
                    "raw_feature": name,
                    "value": value,
                    "impact": contribution,
                    "abs_impact": abs(
                        contribution
                    ),
                    "direction": direction,
                }
            )

        rows.sort(
            key=lambda row: row[
                "abs_impact"
            ],
            reverse=True,
        )

        rows = rows[:10]

        if not rows:

            return None, []

        # ----------------------------------------------------
        # Chart
        # ----------------------------------------------------

        labels = [
            row["feature"]
            for row in rows
        ]

        impacts = [
            row["impact"]
            for row in rows
        ]

        plt.figure(
            figsize=(10, 6)
        )

        plt.barh(
            labels[::-1],
            impacts[::-1],
        )

        plt.axvline(
            0,
            linewidth=1,
        )

        plt.xlabel(
            "SHAP contribution"
        )

        plt.ylabel(
            "Feature"
        )

        plt.title(
            "SHAP Local Explanation - XGBoost Component"
        )

        plt.tight_layout()

        buffer = BytesIO()

        plt.savefig(
            buffer,
            format="png",
            dpi=160,
            bbox_inches="tight",
        )

        plt.close()

        buffer.seek(0)

        encoded = base64.b64encode(
            buffer.read()
        ).decode(
            "utf-8"
        )

        image = (
            "data:image/png;base64,"
            + encoded
        )

        return image, rows

    except Exception as exc:

        print(
            "SHAP generation failed:",
            repr(exc),
        )

        return None, []


# ============================================================
# CLEVELAND DATASET FOR LIME
# ============================================================

def get_lime_training_data() -> Optional[pd.DataFrame]:
    """
    Load real Cleveland data.

    Supported:
        1. Already one-hot encoded CSV
        2. Raw Cleveland CSV with:
           age, sex, cp, trestbps, chol, fbs,
           restecg, thalach, exang, oldpeak,
           slope, ca, thal
    """

    candidates = [
        os.path.join(
            DATA_DIR,
            "cleveland.csv",
        ),
        os.path.join(
            DATA_DIR,
            "heart.csv",
        ),
        os.path.join(
            BASE_DIR,
            "cleveland.csv",
        ),
        os.path.join(
            BASE_DIR,
            "heart.csv",
        ),
        os.path.join(
            DATA_DIR,
            "processed.cleveland.data",
        ),
        os.path.join(
            BASE_DIR,
            "processed.cleveland.data",
        ),
    ]

    for path in candidates:

        if not os.path.isfile(path):
            continue

        try:

            print(
                "LIME dataset:",
                path,
            )

            # ------------------------------------------------
            # Try standard CSV
            # ------------------------------------------------

            try:

                df = pd.read_csv(
                    path
                )

            except Exception:

                df = pd.read_csv(
                    path,
                    header=None,
                )

            # ------------------------------------------------
            # Headerless UCI Cleveland dataset
            # ------------------------------------------------

            if df.shape[1] >= 14:

                raw_names = [
                    "age",
                    "sex",
                    "cp",
                    "trestbps",
                    "chol",
                    "fbs",
                    "restecg",
                    "thalach",
                    "exang",
                    "oldpeak",
                    "slope",
                    "ca",
                    "thal",
                    "target",
                ]

                if df.shape[1] == 14:

                    # Determine if columns look numeric.
                    numeric_test = pd.to_numeric(
                        df.iloc[:, 0],
                        errors="coerce",
                    )

                    if numeric_test.notna().sum() > (
                        len(df) * 0.5
                    ):

                        df.columns = raw_names

            # ------------------------------------------------
            # Already 23 features
            # ------------------------------------------------

            if all(
                feature in df.columns
                for feature in CLEVELAND_FEATURES
            ):

                X = df[
                    CLEVELAND_FEATURES
                ].copy()

                X = (
                    X.replace(
                        ["?", "NA", "N/A"],
                        np.nan,
                    )
                    .replace(
                        [np.inf, -np.inf],
                        np.nan,
                    )
                    .apply(
                        pd.to_numeric,
                        errors="coerce",
                    )
                    .fillna(0.0)
                )

                return X.astype(
                    float
                )

            # ------------------------------------------------
            # Raw Cleveland columns
            # ------------------------------------------------

            raw_columns = [
                "age",
                "sex",
                "cp",
                "trestbps",
                "chol",
                "fbs",
                "restecg",
                "thalach",
                "exang",
                "oldpeak",
                "slope",
                "ca",
                "thal",
            ]

            lower_columns = {
                str(column).lower():
                    column
                for column in df.columns
            }

            resolved = {}

            for name in raw_columns:

                if name in df.columns:

                    resolved[name] = name

                elif (
                    name.lower()
                    in lower_columns
                ):

                    resolved[name] = (
                        lower_columns[
                            name.lower()
                        ]
                    )

            if all(
                name in resolved
                for name in raw_columns
            ):

                X = pd.DataFrame(
                    0.0,
                    index=df.index,
                    columns=CLEVELAND_FEATURES,
                )

                numeric_columns = [
                    "age",
                    "sex",
                    "trestbps",
                    "chol",
                    "fbs",
                    "thalach",
                    "exang",
                    "oldpeak",
                    "ca",
                ]

                for column in numeric_columns:

                    X[column] = pd.to_numeric(
                        df[
                            resolved[column]
                        ],
                        errors="coerce",
                    ).fillna(0.0)

                cp_values = pd.to_numeric(
                    df[
                        resolved["cp"]
                    ],
                    errors="coerce",
                ).fillna(0).astype(int)

                for value in range(4):

                    X.loc[
                        cp_values == value,
                        f"cp_{value}",
                    ] = 1.0

                restecg_values = pd.to_numeric(
                    df[
                        resolved["restecg"]
                    ],
                    errors="coerce",
                ).fillna(0).astype(int)

                for value in range(3):

                    X.loc[
                        restecg_values == value,
                        f"restecg_{value}",
                    ] = 1.0

                slope_values = pd.to_numeric(
                    df[
                        resolved["slope"]
                    ],
                    errors="coerce",
                ).fillna(0).astype(int)

                for value in range(3):

                    X.loc[
                        slope_values == value,
                        f"slope_{value}",
                    ] = 1.0

                thal_values = pd.to_numeric(
                    df[
                        resolved["thal"]
                    ],
                    errors="coerce",
                ).fillna(3)

                X.loc[
                    thal_values == 3,
                    "thal_3.0",
                ] = 1.0

                X.loc[
                    thal_values == 6,
                    "thal_6.0",
                ] = 1.0

                X.loc[
                    thal_values == 7,
                    "thal_7.0",
                ] = 1.0

                X = (
                    X.replace(
                        [np.inf, -np.inf],
                        np.nan,
                    )
                    .fillna(0.0)
                )

                return X.astype(
                    float
                )

        except Exception as exc:

            print(
                "LIME data error:",
                path,
                exc,
            )

    print(
        "No compatible Cleveland dataset found for LIME."
    )

    return None


# ============================================================
# LIME
# ============================================================

def generate_lime(
    models: Dict[str, Any],
    X: pd.DataFrame,
) -> List[Dict[str, Any]]:
    """
    Generate LIME explanation.

    IMPORTANT:
    LIME uses the same calculate_prediction() function
    as the application, so the explanation is tied to
    the application's final prediction logic.
    """

    if LimeTabularExplainer is None:

        print(
            "LIME is not installed."
        )

        return []

    if X is None or X.empty:

        return []

    try:

        training_data = (
            get_lime_training_data()
        )

        if training_data is None:

            return []

        training_data = training_data.reindex(
            columns=CLEVELAND_FEATURES,
            fill_value=0.0,
        )

        training_data = (
            training_data
            .replace(
                [np.inf, -np.inf],
                np.nan,
            )
            .fillna(0.0)
            .astype(float)
        )

        # ----------------------------------------------------
        # LIME prediction wrapper
        # ----------------------------------------------------

        def lime_predict(
            samples: np.ndarray,
        ) -> np.ndarray:

            samples = np.asarray(
                samples,
                dtype=float,
            )

            results = []

            for sample in samples:

                row = pd.DataFrame(
                    [sample],
                    columns=CLEVELAND_FEATURES,
                )

                try:

                    prediction = (
                        calculate_prediction(
                            models,
                            row,
                        )
                    )

                    probability = float(
                        prediction[
                            "probability"
                        ]
                    )

                except Exception as exc:

                    print(
                        "LIME prediction error:",
                        exc,
                    )

                    probability = 0.5

                probability = float(
                    np.clip(
                        probability,
                        0.001,
                        0.999,
                    )
                )

                results.append(
                    [
                        1.0 - probability,
                        probability,
                    ]
                )

            return np.asarray(
                results,
                dtype=float,
            )

        # ----------------------------------------------------
        # Explainer
        # ----------------------------------------------------

        explainer = LimeTabularExplainer(
            training_data.values,
            feature_names=CLEVELAND_FEATURES,
            class_names=[
                "No Heart Disease",
                "Heart Disease",
            ],
            mode="classification",
            discretize_continuous=True,
            random_state=42,
        )

        # ----------------------------------------------------
        # Instance
        # ----------------------------------------------------

        instance = (
            X.iloc[0]
            .reindex(
                CLEVELAND_FEATURES,
                fill_value=0.0,
            )
            .astype(float)
            .values
        )

        # ----------------------------------------------------
        # Explanation
        # ----------------------------------------------------

        explanation = (
            explainer.explain_instance(
                instance,
                lime_predict,
                labels=(1,),
                num_features=10,
                num_samples=1000,
            )
        )

        explanation_items = (
            explanation.as_list(
                label=1
            )
        )

        rows = []

        for feature, weight in (
            explanation_items
        ):

            weight = float(
                weight
            )

            if weight > 0:

                direction = (
                    "Supports heart-disease prediction"
                )

            elif weight < 0:

                direction = (
                    "Supports no-disease prediction"
                )

            else:

                direction = (
                    "Little or no contribution"
                )

            rows.append(
                {
                    "feature": feature,
                    "impact": weight,
                    "direction": direction,
                }
            )

        return rows

    except Exception as exc:

        print(
            "LIME generation failed:",
            repr(exc),
        )

        return []


# ============================================================
# SERIALIZATION
# ============================================================

def clean_for_json(
    value: Any,
) -> Any:

    if isinstance(
        value,
        np.generic,
    ):
        return value.item()

    if isinstance(
        value,
        np.ndarray,
    ):
        return value.tolist()

    if isinstance(
        value,
        pd.Series,
    ):
        return value.tolist()

    if isinstance(
        value,
        pd.DataFrame,
    ):
        return value.to_dict(
            orient="records"
        )

    if isinstance(
        value,
        dict,
    ):
        return {
            str(key): clean_for_json(
                item
            )
            for key, item in value.items()
        }

    if isinstance(
        value,
        list,
    ):
        return [
            clean_for_json(item)
            for item in value
        ]

    return value


# ============================================================
# HTML BASE
# ============================================================

BASE_HTML = """
<!DOCTYPE html>
<html lang="en">

<head>

<meta charset="UTF-8">

<meta name="viewport"
      content="width=device-width, initial-scale=1.0">

<title>{{ title or "IELF Heart Disease Prediction" }}</title>

<style>

* {
    box-sizing: border-box;
}

body {
    margin: 0;
    font-family:
        Arial,
        Helvetica,
        sans-serif;
    background: #f5f7fb;
    color: #1f2937;
}

.navbar {
    background: #111827;
    color: white;
    padding: 16px 30px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 15px;
}

.navbar h2 {
    margin: 0;
    font-size: 21px;
}

.nav-links {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
}

.nav-links a {
    color: white;
    text-decoration: none;
    padding: 8px 13px;
    border-radius: 7px;
}

.nav-links a:hover {
    background: #374151;
}

.container {
    max-width: 1200px;
    margin: 30px auto;
    padding: 0 18px;
}

.card {
    background: white;
    border-radius: 14px;
    padding: 24px;
    margin-bottom: 20px;
    box-shadow:
        0 4px 18px
        rgba(0,0,0,0.07);
}

.grid {
    display: grid;
    grid-template-columns:
        repeat(auto-fit, minmax(230px, 1fr));
    gap: 18px;
}

.stat {
    background: white;
    border-radius: 14px;
    padding: 22px;
    box-shadow:
        0 4px 18px
        rgba(0,0,0,0.06);
}

.stat h3 {
    margin: 0 0 8px;
    color: #6b7280;
    font-size: 14px;
}

.stat strong {
    font-size: 28px;
}

input,
select {
    width: 100%;
    padding: 11px;
    border: 1px solid #d1d5db;
    border-radius: 8px;
    margin-top: 6px;
    margin-bottom: 15px;
    background: white;
}

label {
    font-weight: 600;
    font-size: 14px;
}

button,
.btn {
    border: none;
    background: #2563eb;
    color: white;
    padding: 11px 17px;
    border-radius: 8px;
    cursor: pointer;
    text-decoration: none;
    display: inline-block;
}

button:hover,
.btn:hover {
    opacity: 0.9;
}

.btn-secondary {
    background: #4b5563;
}

.btn-danger {
    background: #dc2626;
}

.btn-success {
    background: #059669;
}

table {
    width: 100%;
    border-collapse: collapse;
    margin-top: 12px;
}

th,
td {
    padding: 11px;
    border-bottom:
        1px solid #e5e7eb;
    text-align: left;
}

th {
    background: #f9fafb;
}

.badge {
    display: inline-block;
    padding: 7px 12px;
    border-radius: 20px;
    background: #e5e7eb;
    font-weight: bold;
}

.high {
    background: #fee2e2;
    color: #991b1b;
}

.low {
    background: #dcfce7;
    color: #166534;
}

.warning {
    background: #fef3c7;
    color: #92400e;
}

.info {
    background: #dbeafe;
    color: #1e40af;
}

.alert {
    padding: 14px;
    border-radius: 9px;
    background: #fef3c7;
    margin: 12px 0;
}

.error {
    background: #fee2e2;
    color: #991b1b;
}

.success {
    background: #dcfce7;
    color: #166534;
}

.progress {
    width: 100%;
    height: 18px;
    background: #e5e7eb;
    border-radius: 20px;
    overflow: hidden;
    margin-top: 12px;
}

.progress-bar {
    height: 100%;
    background: #2563eb;
}

.small {
    color: #6b7280;
    font-size: 13px;
}

.explanation {
    max-width: 100%;
    width: 100%;
    height: auto;
    margin-top: 15px;
}

pre {
    background: #111827;
    color: #e5e7eb;
    padding: 16px;
    border-radius: 10px;
    overflow-x: auto;
}

.footer {
    text-align: center;
    color: #6b7280;
    padding: 30px;
    font-size: 13px;
}

.actions {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
}

h1,
h2,
h3 {
    color: #111827;
}

@media(max-width: 700px) {

    .navbar {
        padding: 14px;
    }

    .container {
        margin-top: 20px;
    }

    table {
        display: block;
        overflow-x: auto;
    }

}

</style>

</head>

<body>

<nav class="navbar">

    <h2>
        IELF Heart Disease Prediction
    </h2>

    <div class="nav-links">

        <a href="{{ url_for('home') }}">
            Dashboard
        </a>

        <a href="{{ url_for('patients') }}">
            Patients
        </a>

        <a href="{{ url_for('new_patient') }}">
            New Patient
        </a>

        <a href="{{ url_for('direct_predict') }}">
            Prediction
        </a>

        <a href="{{ url_for('experiments') }}">
            Experiments
        </a>

    </div>

</nav>

<div class="container">

{% block content %}{% endblock %}

</div>

<div class="footer">

    IELF - Interpretable Ensemble Learning Framework
    <br>
    Academic / Research Demonstration Only

</div>

</body>

</html>
"""


# ============================================================
# HOME
# ============================================================

HOME_HTML = """
{% extends_base %}

<div class="card">

    <h1>
        Heart Disease Prediction Using IELF
    </h1>

    <p>
        Interpretable Ensemble Learning Framework
        using EBM, XGBoost, SHAP and LIME.
    </p>

    <div class="actions">

        <a class="btn"
           href="{{ url_for('new_patient') }}">
            Add Patient
        </a>

        <a class="btn btn-secondary"
           href="{{ url_for('direct_predict') }}">
            New Prediction
        </a>

    </div>

</div>

<div class="grid">

    <div class="stat">
        <h3>Total Patients</h3>
        <strong>{{ patient_count }}</strong>
    </div>

    <div class="stat">
        <h3>IELF Model</h3>

        <strong>
            {% if model_status.ielf %}
                Ready
            {% else %}
                Not Found
            {% endif %}
        </strong>
    </div>

    <div class="stat">
        <h3>XGBoost</h3>

        <strong>
            {% if model_status.xgb %}
                Ready
            {% else %}
                Not Found
            {% endif %}
        </strong>
    </div>

    <div class="stat">
        <h3>SHAP</h3>

        <strong>
            {% if shap_available %}
                Ready
            {% else %}
                Not Installed
            {% endif %}
        </strong>
    </div>

    <div class="stat">
        <h3>LIME</h3>

        <strong>
            {% if lime_available %}
                Ready
            {% else %}
                Not Installed
            {% endif %}
        </strong>
    </div>

</div>

<div class="card">

    <h2>
        Model Architecture
    </h2>

    <p>
        Patient Data
        →
        Preprocessing
        →
        EBM + XGBoost
        →
        IELF Prediction
        →
        SHAP + LIME
        →
        Explanation
    </p>

</div>
"""


# ============================================================
# PATIENT FORM
# ============================================================

PATIENT_FORM_HTML = """
{% extends_base %}

<div class="card">

    <h1>
        {{ heading }}
    </h1>

    <form method="POST">

        <div class="grid">

            <div>
                <label>Patient Name</label>
                <input
                    type="text"
                    name="name"
                    value="{{ patient.get('name','') }}"
                    required>
            </div>

            <div>
                <label>Age</label>
                <input
                    type="number"
                    name="age"
                    value="{{ patient.get('age','') }}"
                    min="1"
                    max="120"
                    required>
            </div>

            <div>
                <label>Gender</label>

                <select name="sex">

                    <option value="1"
                    {% if patient.get('sex',1)|string == '1' %}
                    selected
                    {% endif %}>
                        Male
                    </option>

                    <option value="0"
                    {% if patient.get('sex',1)|string == '0' %}
                    selected
                    {% endif %}>
                        Female
                    </option>

                </select>
            </div>

            <div>
                <label>Resting Blood Pressure</label>

                <input
                    type="number"
                    name="trestbps"
                    value="{{ patient.get('trestbps','') }}"
                    required>
            </div>

            <div>
                <label>Cholesterol</label>

                <input
                    type="number"
                    name="chol"
                    value="{{ patient.get('chol','') }}"
                    required>
            </div>

            <div>
                <label>Fasting Blood Sugar</label>

                <select name="fbs">

                    <option value="0"
                    {% if patient.get('fbs',0)|string == '0' %}
                    selected
                    {% endif %}>
                        Normal
                    </option>

                    <option value="1"
                    {% if patient.get('fbs',0)|string == '1' %}
                    selected
                    {% endif %}>
                        High
                    </option>

                </select>
            </div>

            <div>
                <label>Maximum Heart Rate</label>

                <input
                    type="number"
                    name="thalach"
                    value="{{ patient.get('thalach','') }}"
                    required>
            </div>

            <div>
                <label>Exercise Induced Angina</label>

                <select name="exang">

                    <option value="0"
                    {% if patient.get('exang',0)|string == '0' %}
                    selected
                    {% endif %}>
                        No
                    </option>

                    <option value="1"
                    {% if patient.get('exang',0)|string == '1' %}
                    selected
                    {% endif %}>
                        Yes
                    </option>

                </select>
            </div>

            <div>
                <label>Oldpeak</label>

                <input
                    type="number"
                    step="0.1"
                    name="oldpeak"
                    value="{{ patient.get('oldpeak','') }}"
                    required>
            </div>

            <div>
                <label>Major Vessels (CA)</label>

                <select name="ca">

                    {% for value in range(0,4) %}

                    <option
                        value="{{ value }}"
                        {% if patient.get('ca',0)|string == value|string %}
                        selected
                        {% endif %}>
                        {{ value }}
                    </option>

                    {% endfor %}

                </select>
            </div>

            <div>
                <label>Chest Pain Type</label>

                <select name="cp">

                    {% for value in range(0,4) %}

                    <option
                        value="{{ value }}"
                        {% if patient.get('cp',0)|string == value|string %}
                        selected
                        {% endif %}>
                        Type {{ value }}
                    </option>

                    {% endfor %}

                </select>
            </div>

            <div>
                <label>Resting ECG</label>

                <select name="restecg">

                    {% for value in range(0,3) %}

                    <option
                        value="{{ value }}"
                        {% if patient.get('restecg',0)|string == value|string %}
                        selected
                        {% endif %}>
                        Type {{ value }}
                    </option>

                    {% endfor %}

                </select>
            </div>

            <div>
                <label>Slope</label>

                <select name="slope">

                    {% for value in range(0,3) %}

                    <option
                        value="{{ value }}"
                        {% if patient.get('slope',0)|string == value|string %}
                        selected
                        {% endif %}>
                        Type {{ value }}
                    </option>

                    {% endfor %}

                </select>
            </div>

            <div>
                <label>Thalassemia</label>

                <select name="thal">

                    <option value="3"
                    {% if patient.get('thal',3)|string == '3' %}
                    selected
                    {% endif %}>
                        3 - Normal
                    </option>

                    <option value="6"
                    {% if patient.get('thal',3)|string == '6' %}
                    selected
                    {% endif %}>
                        6
                    </option>

                    <option value="7"
                    {% if patient.get('thal',3)|string == '7' %}
                    selected
                    {% endif %}>
                        7
                    </option>

                </select>
            </div>

        </div>

        <button type="submit">
            {{ button_text }}
        </button>

        <a
            class="btn btn-secondary"
            href="{{ url_for('patients') }}">
            Cancel
        </a>

    </form>

</div>
"""


# ============================================================
# PATIENTS PAGE
# ============================================================

PATIENTS_HTML = """
{% extends_base %}

<div class="card">

    <div class="actions">

        <h1 style="margin-right:auto;">
            Patients
        </h1>

        <a class="btn"
           href="{{ url_for('new_patient') }}">
            + Add Patient
        </a>

    </div>

    {% if patients %}

    <table>

        <thead>

            <tr>
                <th>ID</th>
                <th>Name</th>
                <th>Age</th>
                <th>Gender</th>
                <th>Prediction</th>
                <th>Probability</th>
                <th>Actions</th>
            </tr>

        </thead>

        <tbody>

        {% for patient in patients %}

            <tr>

                <td>
                    {{ patient.patient_id }}
                </td>

                <td>
                    {{ patient.name }}
                </td>

                <td>
                    {{ patient.age }}
                </td>

                <td>
                    {% if patient.sex|string == '1' %}
                        Male
                    {% else %}
                        Female
                    {% endif %}
                </td>

                <td>

                    {% if patient.risk_level %}

                        <span class="badge
                        {% if patient.risk_level == 'HIGH RISK' %}
                            high
                        {% else %}
                            low
                        {% endif %}">

                            {{ patient.risk_level }}

                        </span>

                    {% else %}

                        <span class="badge">
                            Not Predicted
                        </span>

                    {% endif %}

                </td>

                <td>

                    {% if patient.probability is not none %}

                        {{ "%.2f"|format(
                            patient.probability * 100
                        ) }}%

                    {% else %}

                        -

                    {% endif %}

                </td>

                <td>

                    <div class="actions">

                        <a class="btn"
                           href="{{ url_for(
                               'patient_detail',
                               patient_id=patient.patient_id
                           ) }}">
                            View
                        </a>

                        <a class="btn btn-success"
                           href="{{ url_for(
                               'patient_predict',
                               patient_id=patient.patient_id
                           ) }}">
                            Predict
                        </a>

                    </div>

                </td>

            </tr>

        {% endfor %}

        </tbody>

    </table>

    {% else %}

        <div class="alert">
            No patients found.
        </div>

    {% endif %}

</div>
"""


# ============================================================
# PATIENT DETAIL
# ============================================================

PATIENT_DETAIL_HTML = """
{% extends_base %}

<div class="card">

    <h1>
        Patient Details
    </h1>

    <div class="grid">

        <div>
            <strong>Patient ID</strong>
            <p>{{ patient.patient_id }}</p>
        </div>

        <div>
            <strong>Name</strong>
            <p>{{ patient.name }}</p>
        </div>

        <div>
            <strong>Age</strong>
            <p>{{ patient.age }}</p>
        </div>

        <div>
            <strong>Gender</strong>
            <p>
                {% if patient.sex|string == '1' %}
                    Male
                {% else %}
                    Female
                {% endif %}
            </p>
        </div>

        <div>
            <strong>Blood Pressure</strong>
            <p>{{ patient.trestbps }}</p>
        </div>

        <div>
            <strong>Cholesterol</strong>
            <p>{{ patient.chol }}</p>
        </div>

        <div>
            <strong>Maximum Heart Rate</strong>
            <p>{{ patient.thalach }}</p>
        </div>

        <div>
            <strong>Oldpeak</strong>
            <p>{{ patient.oldpeak }}</p>
        </div>

    </div>

    <div class="actions">

        <a class="btn btn-success"
           href="{{ url_for(
               'patient_predict',
               patient_id=patient.patient_id
           ) }}">
            Run Prediction
        </a>

        <a class="btn"
           href="{{ url_for(
               'edit_patient',
               patient_id=patient.patient_id
           ) }}">
            Edit
        </a>

        <a class="btn btn-secondary"
           href="{{ url_for('patients') }}">
            Back
        </a>

        {% if patient.risk_level %}

        <a class="btn"
           href="{{ url_for(
               'patient_report',
               patient_id=patient.patient_id
           ) }}">
            PDF Report
        </a>

        {% endif %}

    </div>

</div>

{% if patient.risk_level %}

<div class="card">

    <h2>Latest Prediction</h2>

    <p>

        <span class="badge
        {% if patient.risk_level == 'HIGH RISK' %}
            high
        {% else %}
            low
        {% endif %}">

            {{ patient.risk_level }}

        </span>

    </p>

    <h3>
        Probability:
        {{ "%.2f"|format(
            patient.probability * 100
        ) }}%
    </h3>

    <p class="small">
        Model:
        {{ patient.prediction_source }}
    </p>

</div>

{% endif %}
"""


# ============================================================
# PREDICTION PAGE
# ============================================================

PREDICTION_HTML = """
{% extends_base %}

<div class="card">

    <h1>
        Heart Disease Prediction Result
    </h1>

    <p class="small">
        Patient:
        <strong>{{ patient.name }}</strong>
        {% if patient.patient_id %}
        ({{ patient.patient_id }})
        {% endif %}
    </p>

    <div class="card
    {% if prediction.risk_level == 'HIGH RISK' %}
        high
    {% else %}
        low
    {% endif %}">

        <h2>
            {{ prediction.risk_level }}
        </h2>

        <h3>
            Probability:
            {{ "%.2f"|format(
                prediction.probability * 100
            ) }}%
        </h3>

        <div class="progress">

            <div
                class="progress-bar"
                style="width:
                {{ prediction.probability * 100 }}%;">
            </div>

        </div>

        <p>
            Classification threshold:
            {{ "%.2f"|format(
                prediction.threshold
            ) }}
        </p>

        <p>
            Final prediction generated by:
            <strong>
                {{ prediction.prediction_source }}
            </strong>
        </p>

    </div>

    <div class="grid">

        <div class="stat">

            <h3>IELF Probability</h3>

            <strong>

                {% if prediction.ielf_probability is not none %}

                    {{ "%.2f"|format(
                        prediction.ielf_probability * 100
                    ) }}%

                {% else %}
                    N/A
                {% endif %}

            </strong>

        </div>

        <div class="stat">

            <h3>EBM Probability</h3>

            <strong>

                {% if prediction.ebm_probability is not none %}

                    {{ "%.2f"|format(
                        prediction.ebm_probability * 100
                    ) }}%

                {% else %}
                    N/A
                {% endif %}

            </strong>

        </div>

        <div class="stat">

            <h3>XGBoost Probability</h3>

            <strong>

                {% if prediction.xgb_probability is not none %}

                    {{ "%.2f"|format(
                        prediction.xgb_probability * 100
                    ) }}%

                {% else %}
                    N/A
                {% endif %}

            </strong>

        </div>

    </div>

</div>


<div class="card">

    <h2>
        SHAP Explanation
    </h2>

    <p class="small">
        SHAP explains the contribution of the
        XGBoost component used in the IELF framework.
        Positive values increase the predicted risk;
        negative values decrease it.
    </p>

    {% if shap_image %}

        <img
            class="explanation"
            src="{{ shap_image }}"
            alt="SHAP explanation">

    {% else %}

        <div class="alert error">
            SHAP explanation could not be generated.
            Check the XGBoost model and SHAP installation.
        </div>

    {% endif %}

    {% if shap_rows %}

    <table>

        <thead>

            <tr>
                <th>Feature</th>
                <th>Value</th>
                <th>Contribution</th>
                <th>Interpretation</th>
            </tr>

        </thead>

        <tbody>

        {% for row in shap_rows %}

            <tr>

                <td>
                    {{ row.feature }}
                </td>

                <td>
                    {{ "%.3f"|format(row.value) }}
                </td>

                <td>
                    {{ "%.4f"|format(row.impact) }}
                </td>

                <td>
                    {{ row.direction }}
                </td>

            </tr>

        {% endfor %}

        </tbody>

    </table>

    {% endif %}

</div>


<div class="card">

    <h2>
        LIME Explanation
    </h2>

    <p class="small">
        LIME provides a local explanation based on the
        same prediction function used by this application.
    </p>

    {% if lime_rows %}

    <table>

        <thead>

            <tr>
                <th>Feature Condition</th>
                <th>Impact</th>
                <th>Interpretation</th>
            </tr>

        </thead>

        <tbody>

        {% for row in lime_rows %}

            <tr>

                <td>
                    {{ row.feature }}
                </td>

                <td>
                    {{ "%.4f"|format(row.impact) }}
                </td>

                <td>
                    {{ row.direction }}
                </td>

            </tr>

        {% endfor %}

        </tbody>

    </table>

    {% else %}

        <div class="alert">
            LIME explanation could not be generated.
            Make sure LIME is installed and a compatible
            Cleveland dataset is available.
        </div>

    {% endif %}

</div>


<div class="card">

    <h2>
        Clinical Interpretation Notice
    </h2>

    <p>
        This prediction is generated by a machine-learning
        model for academic/research purposes. It should not
        be treated as a medical diagnosis.
    </p>

    <div class="actions">

        {% if patient.patient_id %}

        <a class="btn"
           href="{{ url_for(
               'patient_report',
               patient_id=patient.patient_id
           ) }}">
            Generate PDF Report
        </a>

        <a class="btn btn-secondary"
           href="{{ url_for(
               'patient_detail',
               patient_id=patient.patient_id
           ) }}">
            Patient Details
        </a>

        {% endif %}

        <a class="btn btn-secondary"
           href="{{ url_for('patients') }}">
            Patients
        </a>

    </div>

</div>
"""


# ============================================================
# EXPERIMENT PAGE
# ============================================================

EXPERIMENT_HTML = """
{% extends_base %}

<div class="card">

    <h1>
        Model Experiments
    </h1>

    <p>
        Run the project's training/experiment script.
    </p>

    <div class="actions">

        <a class="btn"
           href="{{ url_for('run_experiment',
                            dataset='cleveland') }}">
            Run Cleveland
        </a>

        <a class="btn btn-secondary"
           href="{{ url_for('run_experiment',
                            dataset='framingham') }}">
            Run Framingham
        </a>

    </div>

</div>

<div class="card">

    <h2>
        Experiment Status
    </h2>

    <pre>{{ experiments | tojson(indent=2) }}</pre>

</div>
"""


# ============================================================
# TEMPLATE RENDER HELPER
# ============================================================

def render_page(
    template: str,
    **context: Any,
):

    template = template.replace(
        "{% extends_base %}",
        BASE_HTML.replace(
            "{% block content %}",
            "",
        ).replace(
            "{% endblock %}",
            "",
        ),
    )

    return render_template_string(
        template,
        **context,
    )


# ============================================================
# EXPERIMENT STATE
# ============================================================

EXPERIMENTS = {
    "cleveland": {
        "status": "idle",
        "message": "",
        "started_at": None,
        "finished_at": None,
        "output": "",
    },
    "framingham": {
        "status": "idle",
        "message": "",
        "started_at": None,
        "finished_at": None,
        "output": "",
    },
}


def run_experiment_job(
    dataset: str,
) -> None:

    state = EXPERIMENTS[dataset]

    state["status"] = "running"

    state["started_at"] = (
        datetime.now().isoformat()
    )

    state["finished_at"] = None

    state["output"] = ""

    try:

        main_file = os.path.join(
            BASE_DIR,
            "main.py",
        )

        if not os.path.isfile(
            main_file
        ):

            state["status"] = "error"

            state["message"] = (
                "main.py not found."
            )

            return

        process = subprocess.run(
            [
                sys.executable,
                main_file,
                "--only",
                dataset,
            ],
            cwd=BASE_DIR,
            capture_output=True,
            text=True,
            timeout=3600,
        )

        output = (
            process.stdout
            + "\n"
            + process.stderr
        )

        state["output"] = output[
            -10000:
        ]

        if process.returncode == 0:

            state["status"] = "completed"

            state["message"] = (
                "Experiment completed."
            )

        else:

            state["status"] = "error"

            state["message"] = (
                f"Experiment failed "
                f"with exit code "
                f"{process.returncode}."
            )

    except subprocess.TimeoutExpired:

        state["status"] = "error"

        state["message"] = (
            "Experiment timed out."
        )

    except Exception as exc:

        state["status"] = "error"

        state["message"] = str(
            exc
        )

    finally:

        state["finished_at"] = (
            datetime.now().isoformat()
        )


# ============================================================
# ROUTES
# ============================================================

@app.route("/")
def home():

    patients = load_patients()

    model_status = {
        "ielf": MODELS.get("ielf")
        is not None,
        "ebm": MODELS.get("ebm")
        is not None,
        "xgb": MODELS.get("xgb")
        is not None,
        "scaler": MODELS.get("scaler")
        is not None,
    }

    return render_page(
        HOME_HTML,
        title="IELF Dashboard",
        patient_count=len(
            patients
        ),
        model_status=model_status,
        shap_available=(
            shap is not None
        ),
        lime_available=(
            LimeTabularExplainer
            is not None
        ),
    )


# ============================================================
# PATIENTS
# ============================================================

@app.route("/patients")
def patients():

    return render_page(
        PATIENTS_HTML,
        title="Patients",
        patients=load_patients(),
    )


# ============================================================
# NEW PATIENT
# ============================================================

@app.route(
    "/patient/new",
    methods=["GET", "POST"],
)
def new_patient():

    if request.method == "POST":

        patients = load_patients()

        patient_id = (
            generate_patient_id(
                patients
            )
        )

        patient = {
            "patient_id": patient_id,
            "name": request.form.get(
                "name",
                "Unknown",
            ),
            "age": safe_int(
                request.form.get(
                    "age"
                )
            ),
            "sex": safe_int(
                request.form.get(
                    "sex"
                )
            ),
            "trestbps": safe_int(
                request.form.get(
                    "trestbps"
                )
            ),
            "chol": safe_int(
                request.form.get(
                    "chol"
                )
            ),
            "fbs": safe_int(
                request.form.get(
                    "fbs"
                )
            ),
            "thalach": safe_int(
                request.form.get(
                    "thalach"
                )
            ),
            "exang": safe_int(
                request.form.get(
                    "exang"
                )
            ),
            "oldpeak": safe_float(
                request.form.get(
                    "oldpeak"
                )
            ),
            "ca": safe_int(
                request.form.get(
                    "ca"
                )
            ),
            "cp": safe_int(
                request.form.get(
                    "cp"
                )
            ),
            "restecg": safe_int(
                request.form.get(
                    "restecg"
                )
            ),
            "slope": safe_int(
                request.form.get(
                    "slope"
                )
            ),
            "thal": safe_int(
                request.form.get(
                    "thal"
                ),
                3,
            ),
            "created_at": datetime.now().isoformat(),
            "risk_level": None,
            "probability": None,
            "prediction_source": None,
            "shap_explanation": [],
            "lime_explanation": [],
        }

        patients.append(
            patient
        )

        save_patients(
            patients
        )

        return redirect(
            url_for(
                "patient_detail",
                patient_id=patient_id,
            )
        )

    return render_page(
        PATIENT_FORM_HTML,
        title="New Patient",
        heading="Add New Patient",
        button_text="Save Patient",
        patient={},
    )


# ============================================================
# PATIENT DETAIL
# ============================================================

@app.route(
    "/patient/<patient_id>"
)
def patient_detail(
    patient_id: str,
):

    patient = find_patient(
        patient_id
    )

    if patient is None:

        return (
            "Patient not found",
            404,
        )

    return render_page(
        PATIENT_DETAIL_HTML,
        title="Patient Details",
        patient=patient,
    )


# ============================================================
# EDIT PATIENT
# ============================================================

@app.route(
    "/patient/<patient_id>/edit",
    methods=["GET"],
)
def edit_patient(
    patient_id: str,
):

    patient = find_patient(
        patient_id
    )

    if patient is None:

        return (
            "Patient not found",
            404,
        )

    return render_page(
        PATIENT_FORM_HTML,
        title="Edit Patient",
        heading="Edit Patient",
        button_text="Update Patient",
        patient=patient,
    )


# ============================================================
# UPDATE PATIENT
# ============================================================

@app.route(
    "/patient/<patient_id>/update",
    methods=["POST"],
)
def update_patient(
    patient_id: str,
):

    patients = load_patients()

    for patient in patients:

        if str(
            patient.get("patient_id")
        ) != str(patient_id):

            continue

        patient.update(
            {
                "name": request.form.get(
                    "name",
                    patient.get(
                        "name",
                        "",
                    ),
                ),
                "age": safe_int(
                    request.form.get(
                        "age"
                    )
                ),
                "sex": safe_int(
                    request.form.get(
                        "sex"
                    )
                ),
                "trestbps": safe_int(
                    request.form.get(
                        "trestbps"
                    )
                ),
                "chol": safe_int(
                    request.form.get(
                        "chol"
                    )
                ),
                "fbs": safe_int(
                    request.form.get(
                        "fbs"
                    )
                ),
                "thalach": safe_int(
                    request.form.get(
                        "thalach"
                    )
                ),
                "exang": safe_int(
                    request.form.get(
                        "exang"
                    )
                ),
                "oldpeak": safe_float(
                    request.form.get(
                        "oldpeak"
                    )
                ),
                "ca": safe_int(
                    request.form.get(
                        "ca"
                    )
                ),
                "cp": safe_int(
                    request.form.get(
                        "cp"
                    )
                ),
                "restecg": safe_int(
                    request.form.get(
                        "restecg"
                    )
                ),
                "slope": safe_int(
                    request.form.get(
                        "slope"
                    )
                ),
                "thal": safe_int(
                    request.form.get(
                        "thal"
                    ),
                    3,
                ),
                "updated_at": datetime.now().isoformat(),
            }
        )

        # Previous prediction is invalid after
        # changing patient data.
        patient[
            "risk_level"
        ] = None

        patient[
            "probability"
        ] = None

        patient[
            "prediction_source"
        ] = None

        patient[
            "shap_explanation"
        ] = []

        patient[
            "lime_explanation"
        ] = []

        break

    save_patients(
        patients
    )

    return redirect(
        url_for(
            "patient_detail",
            patient_id=patient_id,
        )
    )


# ============================================================
# DELETE PATIENT
# ============================================================

@app.route(
    "/patient/<patient_id>/delete",
    methods=["GET", "POST"],
)
def delete_patient(
    patient_id: str,
):

    patients = load_patients()

    patients = [
        patient
        for patient in patients
        if str(
            patient.get("patient_id")
        ) != str(patient_id)
    ]

    save_patients(
        patients
    )

    return redirect(
        url_for("patients")
    )


# ============================================================
# BUILD PATIENT DATAFRAME
# ============================================================

def patient_to_dataframe(
    patient: Dict[str, Any],
) -> pd.DataFrame:

    return create_cleveland_features(
        patient
    )


# ============================================================
# RUN COMPLETE EXPLANATION
# ============================================================

def run_prediction_with_explanations(
    X: pd.DataFrame,
) -> Tuple[
    Dict[str, Any],
    Optional[str],
    List[Dict[str, Any]],
    List[Dict[str, Any]],
]:

    # Final prediction

    prediction = calculate_prediction(
        MODELS,
        X,
    )

    # SHAP

    shap_image, shap_rows = (
        generate_shap(
            MODELS,
            X,
        )
    )

    # LIME

    lime_rows = generate_lime(
        MODELS,
        X,
    )

    return (
        prediction,
        shap_image,
        shap_rows,
        lime_rows,
    )


# ============================================================
# PATIENT PREDICTION
# ============================================================

@app.route(
    "/patient/<patient_id>/predict",
    methods=["GET", "POST"],
)
def patient_predict(
    patient_id: str,
):

    patient = find_patient(
        patient_id
    )

    if patient is None:

        return (
            "Patient not found",
            404,
        )

    try:

        X = patient_to_dataframe(
            patient
        )

        (
            prediction,
            shap_image,
            shap_rows,
            lime_rows,
        ) = run_prediction_with_explanations(
            X
        )

        # ----------------------------------------------------
        # Save result
        # ----------------------------------------------------

        patients = load_patients()

        for item in patients:

            if str(
                item.get("patient_id")
            ) == str(patient_id):

                item[
                    "risk_level"
                ] = prediction[
                    "risk_level"
                ]

                item[
                    "probability"
                ] = prediction[
                    "probability"
                ]

                item[
                    "threshold"
                ] = prediction[
                    "threshold"
                ]

                item[
                    "prediction_source"
                ] = prediction[
                    "prediction_source"
                ]

                item[
                    "ebm_probability"
                ] = prediction[
                    "ebm_probability"
                ]

                item[
                    "xgb_probability"
                ] = prediction[
                    "xgb_probability"
                ]

                item[
                    "ielf_probability"
                ] = prediction[
                    "ielf_probability"
                ]

                item[
                    "shap_explanation"
                ] = clean_for_json(
                    shap_rows
                )

                item[
                    "lime_explanation"
                ] = clean_for_json(
                    lime_rows
                )

                item[
                    "predicted_at"
                ] = datetime.now().isoformat()

                break

        save_patients(
            patients
        )

        # Refresh patient

        patient = find_patient(
            patient_id
        ) or patient

        return render_page(
            PREDICTION_HTML,
            title="Prediction Result",
            patient=patient,
            prediction=prediction,
            shap_image=shap_image,
            shap_rows=shap_rows,
            lime_rows=lime_rows,
        )

    except Exception as exc:

        print(
            "Prediction error:",
            repr(exc),
        )

        return render_page(
            PREDICTION_HTML,
            title="Prediction Error",
            patient=patient,
            prediction={
                "probability": 0.0,
                "risk_level": "Prediction Error",
                "threshold": 0.5,
                "prediction_source": "Unavailable",
                "ebm_probability": None,
                "xgb_probability": None,
                "ielf_probability": None,
            },
            shap_image=None,
            shap_rows=[],
            lime_rows=[],
        )


# ============================================================
# DIRECT PREDICTION
# ============================================================

@app.route(
    "/predict",
    methods=["GET", "POST"],
)
def direct_predict():

    if request.method == "GET":

        return render_page(
            PATIENT_FORM_HTML,
            title="Prediction",
            heading="Heart Disease Prediction",
            button_text="Predict",
            patient={},
        )

    try:

        patient = {
            "patient_id": None,
            "name": request.form.get(
                "name",
                "Patient",
            ),
        }

        for key in [
            "age",
            "sex",
            "trestbps",
            "chol",
            "fbs",
            "thalach",
            "exang",
            "ca",
            "cp",
            "restecg",
            "slope",
            "thal",
        ]:

            patient[key] = safe_int(
                request.form.get(
                    key
                )
            )

        patient[
            "oldpeak"
        ] = safe_float(
            request.form.get(
                "oldpeak"
            )
        )

        X = patient_to_dataframe(
            patient
        )

        (
            prediction,
            shap_image,
            shap_rows,
            lime_rows,
        ) = run_prediction_with_explanations(
            X
        )

        return render_page(
            PREDICTION_HTML,
            title="Prediction Result",
            patient=patient,
            prediction=prediction,
            shap_image=shap_image,
            shap_rows=shap_rows,
            lime_rows=lime_rows,
        )

    except Exception as exc:

        print(
            "Direct prediction error:",
            repr(exc),
        )

        return render_page(
            PREDICTION_HTML,
            title="Prediction Error",
            patient={
                "name": request.form.get(
                    "name",
                    "Patient",
                )
            },
            prediction={
                "probability": 0.0,
                "risk_level": "Prediction Error",
                "threshold": 0.5,
                "prediction_source": "Unavailable",
                "ebm_probability": None,
                "xgb_probability": None,
                "ielf_probability": None,
            },
            shap_image=None,
            shap_rows=[],
            lime_rows=[],
        )


# ============================================================
# PDF REPORT
# ============================================================

def create_pdf_report(
    patient: Dict[str, Any],
) -> BytesIO:

    if not REPORTLAB_AVAILABLE:

        raise RuntimeError(
            "ReportLab is not installed."
        )

    buffer = BytesIO()

    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=35,
        leftMargin=35,
        topMargin=35,
        bottomMargin=35,
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "TitleCustom",
        parent=styles["Title"],
        alignment=TA_CENTER,
        fontSize=20,
        leading=24,
        spaceAfter=20,
    )

    heading_style = ParagraphStyle(
        "HeadingCustom",
        parent=styles["Heading2"],
        fontSize=14,
        leading=18,
        spaceBefore=12,
        spaceAfter=8,
    )

    normal_style = ParagraphStyle(
        "NormalCustom",
        parent=styles["BodyText"],
        fontSize=9,
        leading=13,
    )

    story = []

    story.append(
        Paragraph(
            "IELF Heart Disease Prediction Report",
            title_style,
        )
    )

    story.append(
        Paragraph(
            "Interpretable Ensemble Learning Framework",
            normal_style,
        )
    )

    story.append(
        Spacer(
            1,
            15,
        )
    )

    # --------------------------------------------------------
    # Patient information
    # --------------------------------------------------------

    story.append(
        Paragraph(
            "Patient Information",
            heading_style,
        )
    )

    patient_data = [
        [
            "Patient ID",
            str(
                patient.get(
                    "patient_id",
                    "-",
                )
            ),
        ],
        [
            "Patient Name",
            str(
                patient.get(
                    "name",
                    "-",
                )
            ),
        ],
        [
            "Age",
            str(
                patient.get(
                    "age",
                    "-",
                )
            ),
        ],
        [
            "Gender",
            (
                "Male"
                if str(
                    patient.get(
                        "sex",
                        "0",
                    )
                )
                == "1"
                else "Female"
            ),
        ],
        [
            "Resting Blood Pressure",
            str(
                patient.get(
                    "trestbps",
                    "-",
                )
            ),
        ],
        [
            "Cholesterol",
            str(
                patient.get(
                    "chol",
                    "-",
                )
            ),
        ],
        [
            "Maximum Heart Rate",
            str(
                patient.get(
                    "thalach",
                    "-",
                )
            ),
        ],
        [
            "Oldpeak",
            str(
                patient.get(
                    "oldpeak",
                    "-",
                )
            ),
        ],
    ]

    table = Table(
        patient_data,
        colWidths=[
            180,
            300,
        ],
    )

    table.setStyle(
        TableStyle(
            [
                (
                    "BACKGROUND",
                    (0, 0),
                    (0, -1),
                    colors.lightgrey,
                ),
                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.5,
                    colors.grey,
                ),
                (
                    "FONTNAME",
                    (0, 0),
                    (-1, -1),
                    "Helvetica",
                ),
                (
                    "FONTSIZE",
                    (0, 0),
                    (-1, -1),
                    9,
                ),
                (
                    "VALIGN",
                    (0, 0),
                    (-1, -1),
                    "TOP",
                ),
            ]
        )
    )

    story.append(table)

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    story.append(
        Paragraph(
            "Prediction Result",
            heading_style,
        )
    )

    probability = safe_float(
        patient.get(
            "probability"
        ),
        0.0,
    )

    result_data = [
        [
            "Risk Classification",
            str(
                patient.get(
                    "risk_level",
                    "Not available",
                )
            ),
        ],
        [
            "Probability",
            f"{probability * 100:.2f}%",
        ],
        [
            "Threshold",
            f"{safe_float(
                patient.get(
                    'threshold',
                    0.5
                ),
                0.5
            ):.2f}",
        ],
        [
            "Prediction Source",
            str(
                patient.get(
                    "prediction_source",
                    "Not available",
                )
            ),
        ],
    ]

    result_table = Table(
        result_data,
        colWidths=[
            180,
            300,
        ],
    )

    result_table.setStyle(
        TableStyle(
            [
                (
                    "BACKGROUND",
                    (0, 0),
                    (0, -1),
                    colors.lightgrey,
                ),
                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.5,
                    colors.grey,
                ),
                (
                    "FONTSIZE",
                    (0, 0),
                    (-1, -1),
                    9,
                ),
            ]
        )
    )

    story.append(
        result_table
    )

    # --------------------------------------------------------
    # Component probabilities
    # --------------------------------------------------------

    story.append(
        Paragraph(
            "Model Component Probabilities",
            heading_style,
        )
    )

    component_data = [
        [
            "Component",
            "Probability",
        ],
        [
            "IELF",
            (
                "N/A"
                if patient.get(
                    "ielf_probability"
                )
                is None
                else f"{safe_float(
                    patient.get(
                        'ielf_probability'
                    )
                ) * 100:.2f}%"
            ),
        ],
        [
            "EBM",
            (
                "N/A"
                if patient.get(
                    "ebm_probability"
                )
                is None
                else f"{safe_float(
                    patient.get(
                        'ebm_probability'
                    )
                ) * 100:.2f}%"
            ),
        ],
        [
            "XGBoost",
            (
                "N/A"
                if patient.get(
                    "xgb_probability"
                )
                is None
                else f"{safe_float(
                    patient.get(
                        'xgb_probability'
                    )
                ) * 100:.2f}%"
            ),
        ],
    ]

    component_table = Table(
        component_data,
        colWidths=[
            180,
            300,
        ],
    )

    component_table.setStyle(
        TableStyle(
            [
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, 0),
                    colors.lightgrey,
                ),
                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.5,
                    colors.grey,
                ),
            ]
        )
    )

    story.append(
        component_table
    )

    # --------------------------------------------------------
    # SHAP
    # --------------------------------------------------------

    shap_rows = patient.get(
        "shap_explanation",
        [],
    )

    if isinstance(
        shap_rows,
        list,
    ) and shap_rows:

        story.append(
            Paragraph(
                "SHAP Explanation",
                heading_style,
            )
        )

        story.append(
            Paragraph(
                "SHAP explains the contribution of "
                "the XGBoost component. Positive values "
                "increase predicted risk while negative "
                "values decrease predicted risk.",
                normal_style,
            )
        )

        shap_data = [
            [
                "Feature",
                "Value",
                "Impact",
                "Interpretation",
            ]
        ]

        for row in shap_rows[:10]:

            shap_data.append(
                [
                    str(
                        row.get(
                            "feature",
                            "",
                        )
                    ),
                    f"{safe_float(
                        row.get(
                            'value',
                            0
                        )
                    ): .3f}",
                    f"{safe_float(
                        row.get(
                            'impact',
                            0
                        )
                    ): .4f}",
                    str(
                        row.get(
                            "direction",
                            "",
                        )
                    ),
                ]
            )

        shap_table = Table(
            shap_data,
            colWidths=[
                125,
                55,
                65,
                225,
            ],
            repeatRows=1,
        )

        shap_table.setStyle(
            TableStyle(
                [
                    (
                        "BACKGROUND",
                        (0, 0),
                        (-1, 0),
                        colors.lightgrey,
                    ),
                    (
                        "GRID",
                        (0, 0),
                        (-1, -1),
                        0.4,
                        colors.grey,
                    ),
                    (
                        "FONTSIZE",
                        (0, 0),
                        (-1, -1),
                        7,
                    ),
                    (
                        "VALIGN",
                        (0, 0),
                        (-1, -1),
                        "TOP",
                    ),
                ]
            )
        )

        story.append(
            shap_table
        )

    # --------------------------------------------------------
    # LIME
    # --------------------------------------------------------

    lime_rows = patient.get(
        "lime_explanation",
        [],
    )

    if isinstance(
        lime_rows,
        list,
    ) and lime_rows:

        story.append(
            Paragraph(
                "LIME Explanation",
                heading_style,
            )
        )

        lime_data = [
            [
                "Feature Condition",
                "Impact",
                "Interpretation",
            ]
        ]

        for row in lime_rows[:10]:

            lime_data.append(
                [
                    str(
                        row.get(
                            "feature",
                            "",
                        )
                    ),
                    f"{safe_float(
                        row.get(
                            'impact',
                            0
                        )
                    ): .4f}",
                    str(
                        row.get(
                            "direction",
                            "",
                        )
                    ),
                ]
            )

        lime_table = Table(
            lime_data,
            colWidths=[
                220,
                70,
                180,
            ],
            repeatRows=1,
        )

        lime_table.setStyle(
            TableStyle(
                [
                    (
                        "BACKGROUND",
                        (0, 0),
                        (-1, 0),
                        colors.lightgrey,
                    ),
                    (
                        "GRID",
                        (0, 0),
                        (-1, -1),
                        0.4,
                        colors.grey,
                    ),
                    (
                        "FONTSIZE",
                        (0, 0),
                        (-1, -1),
                        7,
                    ),
                    (
                        "VALIGN",
                        (0, 0),
                        (-1, -1),
                        "TOP",
                    ),
                ]
            )
        )

        story.append(
            lime_table
        )

    # --------------------------------------------------------
    # Notice
    # --------------------------------------------------------

    story.append(
        Spacer(
            1,
            20,
        )
    )

    story.append(
        Paragraph(
            "Notice: This report is generated for "
            "academic/research purposes. The machine-learning "
            "prediction should not be considered a medical "
            "diagnosis.",
            normal_style,
        )
    )

    story.append(
        Spacer(
            1,
            10,
        )
    )

    story.append(
        Paragraph(
            "Generated: "
            + datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
            normal_style,
        )
    )

    document.build(
        story
    )

    buffer.seek(0)

    return buffer


# ============================================================
# PDF ROUTE
# ============================================================

@app.route(
    "/patient/<patient_id>/report"
)
def patient_report(
    patient_id: str,
):

    patient = find_patient(
        patient_id
    )

    if patient is None:

        return (
            "Patient not found",
            404,
        )

    try:

        pdf_buffer = (
            create_pdf_report(
                patient
            )
        )

        filename = (
            f"IELF_Report_"
            f"{patient_id}.pdf"
        )

        return send_file(
            pdf_buffer,
            mimetype="application/pdf",
            as_attachment=True,
            download_name=filename,
        )

    except Exception as exc:

        return (
            f"PDF generation failed: {exc}",
            500,
        )


# ============================================================
# BACKWARD COMPATIBILITY REPORT ROUTE
# ============================================================

@app.route(
    "/report"
)
def report():

    patient_id = request.args.get(
        "patient_id"
    )

    if patient_id:

        return redirect(
            url_for(
                "patient_report",
                patient_id=patient_id,
            )
        )

    patients = load_patients()

    predicted = [
        patient
        for patient in patients
        if patient.get(
            "risk_level"
        )
    ]

    if not predicted:

        return (
            "No predicted patient is available.",
            404,
        )

    latest = predicted[-1]

    return redirect(
        url_for(
            "patient_report",
            patient_id=latest[
                "patient_id"
            ],
        )
    )


# ============================================================
# EXPERIMENTS
# ============================================================

@app.route(
    "/experiments"
)
def experiments():

    return render_page(
        EXPERIMENT_HTML,
        title="Experiments",
        experiments=EXPERIMENTS,
    )


@app.route(
    "/run/<dataset>"
)
def run_experiment(
    dataset: str,
):

    if dataset not in EXPERIMENTS:

        return (
            "Invalid dataset.",
            400,
        )

    state = EXPERIMENTS[
        dataset
    ]

    if state["status"] == "running":

        return render_page(
            EXPERIMENT_HTML,
            title="Experiment Running",
            experiments=EXPERIMENTS,
        )

    thread = threading.Thread(
        target=run_experiment_job,
        args=(dataset,),
        daemon=True,
    )

    thread.start()

    return redirect(
        url_for(
            "experiments"
        )
    )


# ============================================================
# STATUS
# ============================================================

@app.route(
    "/status"
)
def status():

    return jsonify(
        {
            "application":
                "IELF Heart Disease Prediction",
            "status": "running",
            "models": {
                "ielf":
                    MODELS.get("ielf")
                    is not None,
                "ebm":
                    MODELS.get("ebm")
                    is not None,
                "xgb":
                    MODELS.get("xgb")
                    is not None,
                "scaler":
                    MODELS.get("scaler")
                    is not None,
            },
            "libraries": {
                "shap":
                    shap is not None,
                "lime":
                    LimeTabularExplainer
                    is not None,
                "reportlab":
                    REPORTLAB_AVAILABLE,
            },
            "feature_count":
                len(CLEVELAND_FEATURES),
            "threshold":
                MODELS.get(
                    "threshold",
                    0.5,
                ),
            "alpha":
                MODELS.get(
                    "alpha",
                    0.5,
                ),
        }
    )


# ============================================================
# HEALTH
# ============================================================

@app.route(
    "/health"
)
def health():

    models_ready = any(
        MODELS.get(name)
        is not None
        for name in [
            "ielf",
            "ebm",
            "xgb",
        ]
    )

    return jsonify(
        {
            "status":
                "ok"
                if models_ready
                else "warning",
            "application":
                "IELF Heart Disease Prediction",
            "models": {
                "ielf":
                    MODELS.get("ielf")
                    is not None,
                "ebm":
                    MODELS.get("ebm")
                    is not None,
                "xgb":
                    MODELS.get("xgb")
                    is not None,
                "scaler":
                    MODELS.get("scaler")
                    is not None,
            },
            "libraries": {
                "shap":
                    shap is not None,
                "lime":
                    LimeTabularExplainer
                    is not None,
                "reportlab":
                    REPORTLAB_AVAILABLE,
            },
            "feature_count":
                len(CLEVELAND_FEATURES),
        }
    )


# ============================================================
# API - PATIENTS
# ============================================================

@app.route(
    "/api/patients"
)
def api_patients():

    return jsonify(
        clean_for_json(
            load_patients()
        )
    )


# ============================================================
# API - SINGLE PATIENT
# ============================================================

@app.route(
    "/api/patient/<patient_id>"
)
def api_patient(
    patient_id: str,
):

    patient = find_patient(
        patient_id
    )

    if patient is None:

        return jsonify(
            {
                "error":
                    "Patient not found"
            }
        ), 404

    return jsonify(
        clean_for_json(
            patient
        )
    )


# ============================================================
# API - PREDICT
# ============================================================

@app.route(
    "/api/predict",
    methods=["POST"],
)
def api_predict():

    try:

        data = (
            request.get_json(
                silent=True
            )
            or {}
        )

        X = create_cleveland_features(
            data
        )

        (
            prediction,
            shap_image,
            shap_rows,
            lime_rows,
        ) = run_prediction_with_explanations(
            X
        )

        response = {
            "success": True,
            "prediction": {
                "risk_level":
                    prediction[
                        "risk_level"
                    ],
                "probability":
                    prediction[
                        "probability"
                    ],
                "threshold":
                    prediction[
                        "threshold"
                    ],
                "prediction_source":
                    prediction[
                        "prediction_source"
                    ],
                "ebm_probability":
                    prediction[
                        "ebm_probability"
                    ],
                "xgb_probability":
                    prediction[
                        "xgb_probability"
                    ],
                "ielf_probability":
                    prediction[
                        "ielf_probability"
                    ],
            },
            "shap":
                shap_rows,
            "lime":
                lime_rows,
        }

        return jsonify(
            clean_for_json(
                response
            )
        )

    except Exception as exc:

        return jsonify(
            {
                "success": False,
                "error": str(exc),
            }
        ), 500


# ============================================================
# RELOAD MODELS API
# ============================================================

@app.route(
    "/api/reload-models",
    methods=["POST", "GET"],
)
def api_reload_models():

    reload_models()

    return jsonify(
        {
            "success": True,
            "models": {
                "ielf":
                    MODELS.get("ielf")
                    is not None,
                "ebm":
                    MODELS.get("ebm")
                    is not None,
                "xgb":
                    MODELS.get("xgb")
                    is not None,
                "scaler":
                    MODELS.get("scaler")
                    is not None,
            },
        }
    )


# ============================================================
# ERROR HANDLERS
# ============================================================

@app.errorhandler(404)
def page_not_found(error):

    return (
        """
        <div style="
            font-family:Arial;
            max-width:700px;
            margin:60px auto;
            text-align:center;
        ">
            <h1>404</h1>
            <p>Page not found.</p>
            <a href="/">
                Go to Dashboard
            </a>
        </div>
        """,
        404,
    )


@app.errorhandler(500)
def internal_error(error):

    return (
        """
        <div style="
            font-family:Arial;
            max-width:700px;
            margin:60px auto;
            text-align:center;
        ">
            <h1>500</h1>
            <p>
                An internal application error occurred.
            </p>
            <a href="/">
                Go to Dashboard
            </a>
        </div>
        """,
        500,
    )


# ============================================================
# START APPLICATION
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 65)
    print(
        "IELF Heart Disease Prediction Web Application"
    )
    print("=" * 65)

    print(
        "Base directory:",
        BASE_DIR,
    )

    print(
        "Model directory:",
        MODEL_DIR,
    )

    print(
        "Output model directory:",
        OUTPUT_MODEL_DIR,
    )

    print()
    print(
        "IELF:",
        MODELS.get("ielf")
        is not None,
    )

    print(
        "EBM:",
        MODELS.get("ebm")
        is not None,
    )

    print(
        "XGBoost:",
        MODELS.get("xgb")
        is not None,
    )

    print(
        "Scaler:",
        MODELS.get("scaler")
        is not None,
    )

    print(
        "SHAP:",
        shap is not None,
    )

    print(
        "LIME:",
        LimeTabularExplainer
        is not None,
    )

    print(
        "ReportLab:",
        REPORTLAB_AVAILABLE,
    )

    print()
    print(
        "Dashboard:"
    )

    print(
        "http://127.0.0.1:8080/"
    )

    print(
        "Health:"
    )

    print(
        "http://127.0.0.1:8080/health"
    )

    print(
        "Status:"
    )

    print(
        "http://127.0.0.1:8080/status"
    )

    print("=" * 65)
    print()

    app.run(
        host="127.0.0.1",
        port=8080,
        debug=True,
        use_reloader=False,
    )