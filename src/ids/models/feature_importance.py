from __future__ import annotations

import joblib
import numpy as np
import pandas as pd

from ids.config import ARTIFACTS_DIR, REPORTS_DIR

def main(top_k: int = 20):
    model_path = ARTIFACTS_DIR / "ids_model.joblib"
    if not model_path.exists():
        raise FileNotFoundError("Train first: python -m ids.models.train")

    pipe = joblib.load(model_path)

    # imblearn Pipeline -> named_steps
    preprocess = pipe.named_steps["preprocess"]
    model = pipe.named_steps["model"]

    # Get feature names after preprocessing
    feature_names = preprocess.get_feature_names_out()

    if not hasattr(model, "feature_importances_"):
        raise TypeError("This model does not expose feature_importances_. Use random_forest.")

    importances = model.feature_importances_
    idx = np.argsort(importances)[::-1][:top_k]

    out = pd.DataFrame({
        "feature": feature_names[idx],
        "importance": importances[idx],
    })

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REPORTS_DIR / "feature_importance_top20.csv"
    out.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")
    print(out)

if __name__ == "__main__":
    main()
