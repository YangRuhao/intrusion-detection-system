from __future__ import annotations

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from ids.config import ARTIFACTS_DIR


app = FastAPI(title="IDS CICIDS2017 ML API", version="1.0.0")


def load_artifacts():
    model_path = ARTIFACTS_DIR / "ids_model.joblib"
    meta_path = ARTIFACTS_DIR / "metadata.joblib"
    if not model_path.exists() or not meta_path.exists():
        raise FileNotFoundError("Model artifacts not found. Train first.")
    pipe = joblib.load(model_path)
    meta = joblib.load(meta_path)
    return pipe, meta


# flexible schema: CIC feature set is large; accept arbitrary key-values
class PredictRequest(BaseModel):
    features: dict


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(req: PredictRequest):
    try:
        pipe, _ = load_artifacts()
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e))

    X = pd.DataFrame([req.features])
    if hasattr(pipe, "predict_proba"):
        score = float(pipe.predict_proba(X)[:, 1][0])
    else:
        score = float(pipe.decision_function(X)[0])

    pred = 1 if score >= 0.5 else 0
    return {
        "prediction": "attack" if pred == 1 else "benign",
        "probability_attack": score,
    }
