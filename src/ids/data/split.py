from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


# Add dataset-specific label option
LABEL_CANDIDATES = [
    "Attack Type",  # <-- your dataset
    "Label",
    "label",
    "Class",
    "class",
    "target",
    "Target",
    "Attack",
    "attack",
    "Category",
    "category",
]

# In your dataset benign appears as "Normal Traffic"
BENIGN_VALUES = {"normal traffic", "benign", "normal"}


@dataclass(frozen=True)
class SplitConfig:
    test_size: float = 0.20
    val_size: float = 0.20  # fraction of remaining train split
    random_state: int = 42


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.astype(str).str.strip()
    return df


def detect_label_column(df: pd.DataFrame) -> str:
    cols = list(df.columns)

    for c in LABEL_CANDIDATES:
        if c in cols:
            return c

    # last resort: any col containing "attack" or "label" or "class"
    for c in cols:
        lc = c.lower()
        if "attack" in lc or "label" in lc or "class" in lc or "target" in lc:
            return c

    raise ValueError(
        "Could not detect label column.\n"
        f"First 30 columns: {cols[:30]}\n"
        f"Last 30 columns: {cols[-30:]}\n"
        "Hint: Your label seems to be at the end."
    )


def make_binary_label(y: pd.Series) -> np.ndarray:
    """
    Binary IDS label: 0=benign, 1=attack
    """
    y_str = y.astype(str).str.strip().str.lower()
    return (~y_str.isin(BENIGN_VALUES)).astype(int).to_numpy()


def split_train_val_test(
    df: pd.DataFrame, cfg: SplitConfig = SplitConfig()
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    df = normalize_columns(df)

    label_col = detect_label_column(df)

    # Clean
    df = df.replace([np.inf, -np.inf], np.nan).dropna(axis=0)

    # Stratify on binary label
    y_bin = make_binary_label(df[label_col])

    train_df, test_df = train_test_split(
        df,
        test_size=cfg.test_size,
        random_state=cfg.random_state,
        stratify=y_bin,
    )

    y_bin_train = make_binary_label(train_df[label_col])
    train_df, val_df = train_test_split(
        train_df,
        test_size=cfg.val_size,
        random_state=cfg.random_state,
        stratify=y_bin_train,
    )

    return train_df, val_df, test_df, label_col
