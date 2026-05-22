from pathlib import Path
import sys

import joblib
import pandas as pd
from pydantic import ValidationError

from app import MODEL, CATEGORY_ENCODINGS, MODEL_FEATURE_ORDER, PredictRequest, build_feature_vector

FILE_PATH = Path(__file__).parent / "adult.csv"
MODEL_PATH = Path(__file__).parent / "model.pkl"


def check_imports():
    missing = []
    for package in ["fastapi", "uvicorn", "streamlit", "pandas", "sklearn", "joblib"]:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    return missing


def main():
    print("Doctor report for Adult Income project")
    print("======================================")

    missing = check_imports()
    if missing:
        print("[WARN] Missing packages:", ", ".join(missing))
    else:
        print("[OK] All required packages are installed.")

    if not FILE_PATH.exists():
        print(f"[ERROR] Dataset not found: {FILE_PATH}")
        sys.exit(1)
    print(f"[OK] Dataset found: {FILE_PATH}")

    try:
        df = pd.read_csv(FILE_PATH)
        print(f"[OK] Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    except Exception as exc:
        print(f"[ERROR] Failed to read dataset: {exc}")
        sys.exit(1)

    if not MODEL_PATH.exists():
        print(f"[ERROR] Model not found: {MODEL_PATH}")
        sys.exit(1)
    print(f"[OK] Model file found: {MODEL_PATH}")

    try:
        model = joblib.load(MODEL_PATH)
        print(f"[OK] Model loaded: {type(model).__name__}")
    except Exception as exc:
        print(f"[ERROR] Failed to load model: {exc}")
        sys.exit(1)

    print("[OK] Stored categorical mappings:")
    for column, values in CATEGORY_ENCODINGS.items():
        print(f" - {column}: {len(values)} values")

    sample_payload = {
        "age": 39,
        "workclass": "Private",
        "fnlwgt": 77516,
        "education": "Bachelors",
        "educational-num": 13,
        "marital-status": "Never-married",
        "occupation": "Adm-clerical",
        "relationship": "Not-in-family",
        "race": "White",
        "gender": "Female",
        "capital-gain": 2174,
        "capital-loss": 0,
        "hours-per-week": 40,
        "native-country": "United-States",
    }

    try:
        request = PredictRequest(**sample_payload)
        features = build_feature_vector(request)
        input_df = pd.DataFrame([features], columns=MODEL_FEATURE_ORDER)
        prediction = MODEL.predict(input_df)[0]
        print(f"[OK] Sample prediction succeeded: {prediction}")
    except ValidationError as exc:
        print(f"[ERROR] Input validation failed: {exc}")
        sys.exit(1)
    except Exception as exc:
        print(f"[ERROR] Prediction failed: {exc}")
        sys.exit(1)

    print("[OK] Doctor checks passed.")


if __name__ == "__main__":
    main()
