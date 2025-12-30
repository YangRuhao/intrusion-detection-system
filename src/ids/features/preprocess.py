from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass(frozen=True)
class PreprocessConfig:
    scale_numeric: bool = True


def build_preprocessor(
    X: pd.DataFrame, cfg: PreprocessConfig = PreprocessConfig()
) -> ColumnTransformer:
    X = X.copy()
    X.columns = X.columns.astype(str).str.strip()

    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    numeric_transformer = Pipeline(
        steps=[("scaler", StandardScaler())] if cfg.scale_numeric else []
    )

    if len(numeric_transformer.steps) == 0:
        numeric_transformer = "passthrough"

    categorical_transformer = (
        OneHotEncoder(handle_unknown="ignore") if categorical_cols else "drop"
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def make_xy(df: pd.DataFrame, label_col: str) -> Tuple[pd.DataFrame, pd.Series]:
    df = df.copy()
    df.columns = df.columns.astype(str).str.strip()
    label_col = str(label_col).strip()

    X = df.drop(columns=[label_col])
    y = df[label_col]
    return X, y
