from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    average_precision_score,
    precision_recall_curve,
)

from ids.config import ARTIFACTS_DIR, FIGURES_DIR
from ids.data.load import load_raw_csv
from ids.data.split import split_train_val_test, SplitConfig, make_binary_label
from ids.features.preprocess import make_xy


def load_artifacts():
    model_path = ARTIFACTS_DIR / "ids_model.joblib"
    meta_path = ARTIFACTS_DIR / "metadata.joblib"
    if not model_path.exists() or not meta_path.exists():
        raise FileNotFoundError("Model artifacts not found. Run: python -m ids.models.train")

    pipe = joblib.load(model_path)
    meta = joblib.load(meta_path)
    return pipe, meta


def evaluate(threshold: float = 0.5, sample_frac: float | None = None) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    pipe, meta = load_artifacts()
    label_col = meta["label_col"]

    # Prefer saved test split
    test_path = ARTIFACTS_DIR / "test_split.csv"
    if test_path.exists():
        test_df = pd.read_csv(test_path)
        # normalize columns
        test_df.columns = test_df.columns.astype(str).str.strip()
    else:
        df = load_raw_csv(sample_frac=sample_frac)
        _, _, test_df, label_col = split_train_val_test(df, SplitConfig(random_state=int(meta.get("random_state", 42))))

    X_test, y_test = make_xy(test_df, label_col)
    y_test_bin = make_binary_label(y_test)

    if hasattr(pipe, "predict_proba"):
        y_score = pipe.predict_proba(X_test)[:, 1]
    else:
        y_score = pipe.decision_function(X_test)

    y_pred = (y_score >= threshold).astype(int)

    ap = average_precision_score(y_test_bin, y_score)
    print(f"\nTest Average Precision (PR-AUC): {ap:.4f}")
    print(f"Threshold: {threshold:.2f}\n")
    print(classification_report(y_test_bin, y_pred, digits=4, target_names=["benign", "attack"]))

    # Confusion matrix
    disp = ConfusionMatrixDisplay.from_predictions(y_test_bin, y_pred, values_format="d")
    plt.title("Confusion Matrix (Attack vs Benign)")
    cm_path = FIGURES_DIR / "confusion_matrix.png"
    plt.savefig(cm_path, bbox_inches="tight", dpi=200)
    plt.close()
    print(f"Saved: {cm_path}")

    # Precision-Recall curve
    precision, recall, thresh = precision_recall_curve(y_test_bin, y_score)
    plt.figure()
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    pr_path = FIGURES_DIR / "precision_recall_curve.png"
    plt.savefig(pr_path, bbox_inches="tight", dpi=200)
    plt.close()
    print(f"Saved: {pr_path}")

    # Optional: threshold tuning plot (precision/recall vs threshold)
    # thresh has length = len(precision)-1
    if len(thresh) > 0:
        plt.figure()
        plt.plot(thresh, precision[:-1], label="precision")
        plt.plot(thresh, recall[:-1], label="recall")
        plt.xlabel("Threshold")
        plt.ylabel("Score")
        plt.title("Precision/Recall vs Threshold")
        plt.legend()
        tr_path = FIGURES_DIR / "threshold_curve.png"
        plt.savefig(tr_path, bbox_inches="tight", dpi=200)
        plt.close()
        print(f"Saved: {tr_path}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--sample-frac", type=float, default=None)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(threshold=args.threshold, sample_frac=args.sample_frac)
