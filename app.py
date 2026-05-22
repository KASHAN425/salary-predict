from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

DATA_PATH = Path(__file__).parent / "adult.csv"
MODEL_PATH = Path(__file__).parent / "model.pkl"

app = FastAPI(
    title="Adult Income API",
    description="API for predicting Adult income bracket using a saved model.",
    version="0.1.0",
)

NUMERIC_FEATURES = [
    "age",
    "fnlwgt",
    "educational-num",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
]
CATEGORICAL_FEATURES = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "gender",
    "native-country",
]
MODEL_FEATURE_ORDER = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "educational-num",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "gender",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
    "native-country",
]


class PredictRequest(BaseModel):
    age: int = Field(..., ge=0, le=120)
    fnlwgt: int = Field(..., ge=0)
    workclass: str
    education: str
    educational_num: int = Field(..., alias="educational-num", ge=0)
    marital_status: str = Field(..., alias="marital-status")
    occupation: str
    relationship: str
    race: str
    gender: str
    capital_gain: int = Field(..., alias="capital-gain", ge=0)
    capital_loss: int = Field(..., alias="capital-loss", ge=0)
    hours_per_week: int = Field(..., alias="hours-per-week", ge=0, le=168)
    native_country: str = Field(..., alias="native-country")
    model_config = {"populate_by_name": True}


class PredictResponse(BaseModel):
    prediction: str
    confidence: Optional[float] = None


def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")

    data = pd.read_csv(DATA_PATH)
    data.replace(["?", " ?"], pd.NA, inplace=True)
    for column in data.select_dtypes(include="object").columns:
        data[column] = data[column].fillna(data[column].mode()[0]).astype(str).str.strip()
    return data


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
    return joblib.load(MODEL_PATH)


def build_category_encodings(data: pd.DataFrame) -> Dict[str, Dict[str, int]]:
    encodings: Dict[str, Dict[str, int]] = {}
    for column in CATEGORICAL_FEATURES:
        values = sorted(data[column].astype(str).str.strip().unique().tolist())
        encodings[column] = {value: index for index, value in enumerate(values)}
    return encodings


def encode_categorical_value(column: str, value: str) -> int:
    cleaned = str(value).strip()
    if cleaned not in CATEGORY_ENCODINGS[column]:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported value for '{column}': '{cleaned}'. "
                f"Supported values: {sorted(CATEGORY_ENCODINGS[column].keys())}"
            ),
        )
    return CATEGORY_ENCODINGS[column][cleaned]


def build_feature_vector(payload: PredictRequest) -> List[Any]:
    return [
        payload.age,
        encode_categorical_value("workclass", payload.workclass),
        payload.fnlwgt,
        encode_categorical_value("education", payload.education),
        payload.educational_num,
        encode_categorical_value("marital-status", payload.marital_status),
        encode_categorical_value("occupation", payload.occupation),
        encode_categorical_value("relationship", payload.relationship),
        encode_categorical_value("race", payload.race),
        encode_categorical_value("gender", payload.gender),
        payload.capital_gain,
        payload.capital_loss,
        payload.hours_per_week,
        encode_categorical_value("native-country", payload.native_country),
    ]


DATA = load_data()
MODEL = load_model()
CATEGORY_ENCODINGS = build_category_encodings(DATA)


@app.get("/", summary="Service overview")
def root() -> Dict[str, Any]:
    return {
        "service": "Adult Income API",
        "description": "Predict whether an adult earns >50K using a saved model.",
        "endpoints": [
            "/health",
            "/metadata",
            "/supported-values",
            "/sample",
            "/statistics",
            "/predict",
        ],
    }


@app.get("/health", summary="Health check")
def health() -> Dict[str, Any]:
    return {"status": "ok", "rows": len(DATA), "model_loaded": True}


@app.get("/metadata", summary="Dataset metadata")
def metadata() -> Dict[str, Any]:
    return {
        "shape": list(DATA.shape),
        "columns": list(DATA.columns),
        "income_values": DATA["income"].unique().tolist(),
    }


@app.get("/supported-values", summary="Supported categorical values")
def supported_values() -> Dict[str, List[str]]:
    return {column: sorted(list(CATEGORY_ENCODINGS[column].keys())) for column in CATEGORICAL_FEATURES}


@app.get("/sample", summary="Sample dataset rows")
def sample(limit: int = 10) -> List[Dict[str, Any]]:
    if limit < 1 or limit > 100:
        raise HTTPException(status_code=400, detail="limit must be between 1 and 100")
    return DATA.head(limit).to_dict(orient="records")


@app.get("/statistics", summary="Dataset statistics")
def statistics() -> Dict[str, Any]:
    return {
        "counts_by_income": DATA["income"].value_counts().to_dict(),
        "average_hours_per_week": float(DATA["hours-per-week"].mean()),
        "average_age": float(DATA["age"].mean()),
    }


@app.post("/predict", response_model=PredictResponse, summary="Predict income bracket")
def predict(payload: PredictRequest) -> PredictResponse:
    try:
        features = build_feature_vector(payload)
        input_df = pd.DataFrame([features], columns=MODEL_FEATURE_ORDER)
        proba = None
        if hasattr(MODEL, "predict_proba"):
            proba = MODEL.predict_proba(input_df)[0].max().item()
        pred = MODEL.predict(input_df)[0]
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction error: {exc}")

    return PredictResponse(prediction=str(pred), confidence=float(proba) if proba is not None else None)
