from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.utils.class_weight import compute_class_weight

from ids.config import ARTIFACTS_DIR, FIGURES_DIR
from ids.data.load import load_raw_csv
from ids.data.split import SplitConfig, split_train_val_test, make_binary_label
from ids.features.preprocess import build_preprocessor, make_xy


@dataclass(frozen=True)
class TrainConfig:
    model_name: str = "random_forest"  # logistic | random_forest
    use_smote: bool = False
    random_state: int = 42
    n_estimators: int = 300
    sample_frac: float | None = None


def build_model(model_name: str, y_bin_train: np.ndarray, cfg: TrainConfig):
    if model_name == "logistic":
        return LogisticRegression(
            max_iter=5000,
            class_weight="balanced",
        )

    if model_name == "random_forest":
        classes = np.unique(y_bin_train)
        weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_bin_train)
        class_weight = {int(c): float(w) for c, w in zip(classes, weights)}
        return RandomForestClassifier(
            n_estimators=cfg.n_estimators,
            random_state=cfg.random_state,
            n_jobs=-1,
            class_weight=class_weight,
        )

    raise ValueError("model_name must be one of: logistic, random_forest")


def train_and_validate(cfg: TrainConfig) -> Path:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    df = load_raw_csv(sample_frac=cfg.sample_frac)
    train_df, val_df, test_df, label_col = split_train_val_test(df, SplitConfig(random_state=cfg.random_state))

    X_train, y_train = make_xy(train_df, label_col)
    X_val, y_val = make_xy(val_df, label_col)

    y_train_bin = make_binary_label(y_train)
    y_val_bin = make_binary_label(y_val)

    preprocessor = build_preprocessor(X_train)
    model = build_model(cfg.model_name, y_train_bin, cfg)

    if cfg.use_smote:
        pipe = ImbPipeline(steps=[
            ("preprocess", preprocessor),
            ("smote", SMOTE(random_state=cfg.random_state)),
            ("model", model),
        ])
    else:
        pipe = ImbPipeline(steps=[
            ("preprocess", preprocessor),
            ("model", model),
        ])

    pipe.fit(X_train, y_train_bin)

    val_scores = pipe.predict_proba(X_val)[:, 1] if hasattr(pipe, "predict_proba") else pipe.decision_function(X_val)
    ap = average_precision_score(y_val_bin, val_scores)
    print(f"Validation Average Precision (PR-AUC): {ap:.4f}")

    artifact_path = ARTIFACTS_DIR / "ids_model.joblib"
    metadata_path = ARTIFACTS_DIR / "metadata.joblib"

    joblib.dump(pipe, artifact_path)
    joblib.dump(
        {
            "label_col": label_col,
            "model_name": cfg.model_name,
            "use_smote": cfg.use_smote,
            "random_state": cfg.random_state,
            "n_estimators": cfg.n_estimators,
            "validation_average_precision": float(ap),
            "sample_frac": cfg.sample_frac,
        },
        metadata_path,
    )

    test_path = ARTIFACTS_DIR / "test_split.csv"
    test_df.to_csv(test_path, index=False)

    print(f"Saved model: {artifact_path}")
    print(f"Saved metadata: {metadata_path}")
    print(f"Saved test split: {test_path}")
    return artifact_path


def parse_args() -> TrainConfig:
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["logistic", "random_forest"], default="random_forest")
    p.add_argument("--use-smote", action="store_true")
    p.add_argument("--n-estimators", type=int, default=300)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--sample-frac", type=float, default=None, help="Train on a fraction of the dataset for faster iteration (e.g., 0.2).")
    a = p.parse_args()
    return TrainConfig(
        model_name=a.model,
        use_smote=a.use_smote,
        n_estimators=a.n_estimators,
        random_state=a.random_state,
        sample_frac=a.sample_frac,
    )


if __name__ == "__main__":
    cfg = parse_args()
    train_and_validate(cfg)
