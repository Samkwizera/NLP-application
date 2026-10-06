from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw" / "KINNEWS" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"


def load_split(name: str) -> pd.DataFrame:
    # name is one of train, val, test, train_original
    path = PROCESSED_DIR / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - run `python scripts/prepare_data.py` first")
    return pd.read_csv(path, keep_default_na=False)
