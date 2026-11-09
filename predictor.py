"""
predict_cleveland.py
====================

Cleveland heart-disease prediction using the trained IELF model.

Expected trained files:

    outputs/
        models/
            cleveland/
                ielf.pkl
                ebm.pkl
                xgb.pkl
                scaler.pkl
                features.pkl
                alpha.txt
"""

from __future__ import annotations

import os
import joblib
import numpy as np
import pandas as pd


# ============================================================
# DIRECTORIES
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "outputs",
    "models",
    "cleveland"
)


# ============================================================
# LOAD IELF MODEL
# ============================================================

def load_cleveland_model():
    """
    Load the trained Cleveland IELF ensemble.
    """

    model_path = os.path.join(
        MODEL_DIR,
        "ielf.pkl"
    )

    if not os.path.exists(model_path):

        raise FileNotFoundError(
            "\nCleveland IELF model not found.\n\n"
            f"Expected file:\n"
            f"{model_path}\n\n"
            "Run main.py first to train and save the model."
        )

    print(
        "Loading IELF model:",
        model_path
    )

    return joblib.load(
        model_path
    )


# ============================================================
# LOAD SCALER
# ============================================================

def load_cleveland_scaler():
    """
    Load the StandardScaler used during training.
    """

    scaler_path = os.path.join(
        MODEL_DIR,
        "scaler.pkl"
    )

    if not os.path.exists(scaler_path):

        raise FileNotFoundError(
            "\nCleveland scaler not found.\n\n"
            f"Expected file:\n"
            f"{scaler_path}\n\n"
            "The model must be trained using main.py first."
        )

    return joblib.load(
        scaler_path
    )


# ============================================================
# LOAD FEATURE NAMES
# ============================================================

def load_cleveland_features():
    """
    Load the exact feature order used during training.
    """

    features_path = os.path.join(
        MODEL_DIR,
        "features.pkl"
    )

    if not os.path.exists(features_path):

        raise FileNotFoundError(
            "\nCleveland feature file not found.\n\n"
            f"Expected file:\n"
            f"{features_path}"
        )

    features = joblib.load(
        features_path
    )

    if not isinstance(
        features,
        (list, tuple)
    ):

        raise ValueError(
            "features.pkl does not contain "
            "a list of feature names."
        )

    return list(features)


# ============================================================
# CREATE CLEVELAND INPUT
# ============================================================

def create_cleveland_input(
    data,
    expected_features
):
    """
    Convert raw Cleveland patient data into
    exactly the same feature representation
    used during training.
    """

    # --------------------------------------------------------
    # Required raw numerical features
    # --------------------------------------------------------

    required_fields = [
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

    missing = [
        field
        for field in required_fields
        if field not in data
    ]

    if missing:

        raise ValueError(
            "Missing Cleveland input fields: "
            + ", ".join(missing)
        )

    # --------------------------------------------------------
    # Basic numerical features
    # --------------------------------------------------------

    row = {
        "age": float(data["age"]),
        "sex": float(data["sex"]),
        "trestbps": float(data["trestbps"]),
        "chol": float(data["chol"]),
        "fbs": float(data["fbs"]),
        "thalach": float(data["thalach"]),
        "exang": float(data["exang"]),
        "oldpeak": float(data["oldpeak"]),
        "ca": float(data["ca"]),
    }

    # ========================================================
    # CHEST PAIN
    # ========================================================

    cp = int(
        float(data["cp"])
    )

    if cp not in [0, 1, 2, 3]:

        raise ValueError(
            "cp must be one of: 0, 1, 2, 3."
        )

    for value in [0, 1, 2, 3]:

        column = f"cp_{value}"

        if column in expected_features:

            row[column] = int(
                cp == value
            )

    # ========================================================
    # RESTING ECG
    # ========================================================

    restecg = int(
        float(data["restecg"])
    )

    if restecg not in [0, 1, 2]:

        raise ValueError(
            "restecg must be one of: 0, 1, 2."
        )

    for value in [0, 1, 2]:

        column = f"restecg_{value}"

        if column in expected_features:

            row[column] = int(
                restecg == value
            )

    # ========================================================
    # ST SLOPE
    # ========================================================

    slope = int(
        float(data["slope"])
    )

    if slope not in [0, 1, 2]:

        raise ValueError(
            "slope must be one of: 0, 1, 2."
        )

    for value in [0, 1, 2]:

        column = f"slope_{value}"

        if column in expected_features:

            row[column] = int(
                slope == value
            )

    # ========================================================
    # THAL
    # ========================================================

    thal = float(
        data["thal"]
    )

    # The training preprocessing apparently
    # produces thal_3.0, thal_6.0 and thal_7.0.

    allowed_thal = [
        3.0,
        6.0,
        7.0,
    ]

    if thal not in allowed_thal:

        raise ValueError(
            "thal must be one of: 3, 6, 7."
        )

    for value in allowed_thal:

        column = f"thal_{value}"

        if column in expected_features:

            row[column] = int(
                thal == value
            )

    # ========================================================
    # CREATE DATAFRAME
    # ========================================================

    df = pd.DataFrame(
        [row]
    )

    # ========================================================
    # EXACT TRAINING FEATURE ORDER
    # ========================================================

    df = df.reindex(
        columns=expected_features,
        fill_value=0
    )

    # ========================================================
    # NUMERIC CONVERSION
    # ========================================================

    df = df.apply(
        pd.to_numeric,
        errors="coerce"
    )

    if df.isnull().any().any():

        raise ValueError(
            "Invalid numerical value found "
            "in Cleveland input."
        )

    return df


# ============================================================
# PREDICT CLEVELAND
# ============================================================

def predict_cleveland(data):
    """
    Predict heart-disease risk for one Cleveland patient.

    Returns
    -------
    dict

        prediction
        result
        probability
    """

    # --------------------------------------------------------
    # Load trained objects
    # --------------------------------------------------------

    model = load_cleveland_model()

    scaler = load_cleveland_scaler()

    expected_features = (
        load_cleveland_features()
    )

    # --------------------------------------------------------
    # Create raw feature vector
    # --------------------------------------------------------

    X = create_cleveland_input(
        data,
        expected_features
    )

    print()
    print(
        "Raw model input:"
    )

    print(X)

    # ========================================================
    # SCALE INPUT
    # ========================================================

    X_scaled = pd.DataFrame(
        scaler.transform(X),
        columns=expected_features,
        index=X.index
    )

    print()
    print(
        "Scaled model input:"
    )

    print(X_scaled)

    # ========================================================
    # PREDICT
    # ========================================================

    if not hasattr(
        model,
        "predict_proba"
    ):

        raise AttributeError(
            "Loaded IELF model does not provide "
            "predict_proba()."
        )

    probabilities = model.predict_proba(
        X_scaled
    )

    if probabilities.ndim != 2:

        raise ValueError(
            "Unexpected predict_proba output shape: "
            f"{probabilities.shape}"
        )

    if probabilities.shape[1] < 2:

        raise ValueError(
            "Model must return probabilities "
            "for both classes."
        )

    probability = float(
        probabilities[0, 1]
    )

    # --------------------------------------------------------
    # Classification threshold
    # --------------------------------------------------------

    prediction = int(
        probability >= 0.5
    )

    # --------------------------------------------------------
    # Result label
    # --------------------------------------------------------

    if prediction == 1:

        result = "HIGH RISK"

    else:

        result = "LOW RISK"

    # ========================================================
    # RETURN
    # ========================================================

    return {
        "prediction": prediction,
        "result": result,
        "probability": probability,
    }


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    sample_data = {

        "age": 55,

        "sex": 1,

        "cp": 1,

        "trestbps": 140,

        "chol": 250,

        "fbs": 0,

        "restecg": 0,

        "thalach": 150,

        "exang": 0,

        "oldpeak": 0,

        "slope": 0,

        "ca": 0,

        "thal": 3,
    }

    try:

        result = predict_cleveland(
            sample_data
        )

        print()
        print("=" * 50)
        print(
            "CLEVELAND IELF PREDICTION"
        )
        print("=" * 50)

        print(
            "Prediction :",
            result["prediction"]
        )

        print(
            "Result     :",
            result["result"]
        )

        if result["probability"] is not None:

            print(
                "Probability:",
                f"{result['probability'] * 100:.2f}%"
            )

        print("=" * 50)

    except Exception as exc:

        print()
        print(
            "Prediction failed:"
        )

        print(
            str(exc)
        )