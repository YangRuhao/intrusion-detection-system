from __future__ import annotations

from pathlib import Path
import pandas as pd

from ids.config import RAW_DIR, DatasetConfig


def load_raw_csv(path: Path | None = None, sample_frac: float | None = None) -> pd.DataFrame:
    cfg = DatasetConfig()
    csv_path = path if path is not None else (RAW_DIR / cfg.raw_filename)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Raw dataset not found at {csv_path}. Put the CSV in data/raw/."
        )

    df = pd.read_csv(csv_path, low_memory=False)
    # normalize column names early
    df.columns = df.columns.astype(str).str.strip()

    if sample_frac is not None:
        df = df.sample(frac=sample_frac, random_state=42)

    return df
