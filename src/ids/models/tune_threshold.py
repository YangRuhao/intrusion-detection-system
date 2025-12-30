from __future__ import annotations

import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import precision_recall_curve

from ids.config import ARTIFACTS_DIR
from ids.data.split import make_binary_label
from ids.features.preprocess import make_xy

def main(target_recall: float = 0.99):
    pipe = joblib.load(ARTIFACTS_DIR / "ids_model.joblib")
    meta = joblib.load(ARTIFACTS_DIR / "metadata.joblib")
    label_col = str(meta["label_col"]).strip()

    test_df = pd.read_csv(ARTIFACTS_DIR / "test_split.csv", low_memory=False)
    test_df.columns = test_df.columns.str.strip()

    X_test, y_test = make_xy(test_df, label_col)
    y_true = make_binary_label(y_test)

    y_score = pipe.predict_proba(X_test)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)

    # thresholds has len = len(precision)-1
    precision = precision[:-1]
    recall = recall[:-1]

    # Find smallest threshold that achieves target recall
    mask = recall >= target_recall
    if not mask.any():
        print(f"Could not achieve recall >= {target_recall}")
        return

    best_idx = np.argmax(mask)  # earliest True
    best_threshold = thresholds[best_idx]
    print(f"Threshold for recall >= {target_recall}: {best_threshold:.6f}")
    print(f"Precision: {precision[best_idx]:.4f}, Recall: {recall[best_idx]:.4f}")

if __name__ == "__main__":
    main()
