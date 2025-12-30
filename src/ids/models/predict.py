from __future__ import annotations

import json
from pathlib import Path
import joblib
import pandas as pd

from ids.config import ARTIFACTS_DIR


def load_model():
    model_path = ARTIFACTS_DIR / "ids_model.joblib"
    meta_path = ARTIFACTS_DIR / "metadata.joblib"
    pipe = joblib.load(model_path)
    meta = joblib.load(meta_path)
    return pipe, meta


def predict_one(json_path: Path) -> None:
    pipe, meta = load_model()

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    X = pd.DataFrame([payload])  # single row

    score = pipe.predict_proba(X)[:, 1][0] if hasattr(pipe, "predict_proba") else float(pipe.decision_function(X)[0])
    pred = int(score >= 0.5)

    print({"prediction": "attack" if pred == 1 else "benign", "probability_attack": float(score)})


if __name__ == "__main__":
    # Example:
    # python -m ids.models.predict .\example_request.json
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("json_path", type=str)
    a = p.parse_args()
    predict_one(Path(a.json_path))
