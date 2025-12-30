from __future__ import annotations
import pandas as pd
from ids.config import ARTIFACTS_DIR


def main():
    test_path = ARTIFACTS_DIR / "test_split.csv"
    if not test_path.exists():
        raise FileNotFoundError("Run training first to generate test_split.csv")

    test_df = pd.read_csv(test_path, low_memory=False)
    test_df.columns = test_df.columns.str.strip()

    # quick duplicate check inside test
    dup_count = test_df.duplicated().sum()
    print(f"Duplicates within test split: {dup_count}")

    # Can extend this to compare with a saved train split if you save it too
    print("Done")


if __name__ == "__main__":
    main()
