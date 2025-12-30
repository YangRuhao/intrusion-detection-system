from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]  # repo root
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

MODELS_DIR = PROJECT_ROOT / "models"
ARTIFACTS_DIR = MODELS_DIR / "artifacts"

REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"


@dataclass(frozen=True)
class DatasetConfig:
    kaggle_dataset: str = "ericanacletoribeiro/cicids2017-cleaned-and-preprocessed"
    raw_filename: str = "cicids2017_cleaned.csv"  # per Kaggle dataset description
