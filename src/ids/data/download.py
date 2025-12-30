from __future__ import annotations

import zipfile
from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi

from ids.config import RAW_DIR, DatasetConfig


def download_from_kaggle(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = DatasetConfig()
    api = KaggleApi()
    api.authenticate()

    # Downloads a zip of the dataset files
    api.dataset_download_files(cfg.kaggle_dataset, path=str(out_dir), quiet=False)

    # Kaggle usually saves as <dataset>.zip in the target folder
    # Find the most recent zip in out_dir
    zips = sorted(out_dir.glob("*.zip"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not zips:
        raise FileNotFoundError("No zip file was downloaded from Kaggle.")
    zip_path = zips[0]

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(out_dir)

    # Optional: remove zip to save space
    zip_path.unlink(missing_ok=True)

    csv_path = out_dir / cfg.raw_filename
    if not csv_path.exists():
        # Some Kaggle downloads may extract into subfolders; search for it
        matches = list(out_dir.rglob(cfg.raw_filename))
        if matches:
            matches[0].replace(csv_path)
        else:
            raise FileNotFoundError(
                f"Expected {cfg.raw_filename} not found after extraction."
            )

    print(f"Downloaded and extracted to: {csv_path}")


if __name__ == "__main__":
    download_from_kaggle(RAW_DIR)
