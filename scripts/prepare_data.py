# Downloads KINNEWS and builds deduplicated, leak-free train/val/test splits.
# The official train set has ~55% duplicate rows, some with conflicting labels,
# and a few articles that also appear in test.
import json
import sys
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from kinnews import LABELS  # noqa: E402
from kinnews.data import PROCESSED_DIR, RAW_DIR, ROOT  # noqa: E402
from kinnews.text import normalize  # noqa: E402

URL = "https://github.com/saradhix/kinnews_kirnews/raw/master/KINNEWS.zip"
SEED = 42
VAL_SIZE = 0.15


def download():
    zip_path = ROOT / "data" / "raw" / "KINNEWS.zip"
    if not RAW_DIR.exists():
        zip_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {URL}")
        urllib.request.urlretrieve(URL, zip_path)
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(zip_path.parent)


def load_official(split: str) -> pd.DataFrame:
    df = pd.read_csv(RAW_DIR / f"{split}.csv")
    df["label_id"] = df["label"] - 1  # official ids are 1..14
    df["label"] = df["label_id"].map(lambda i: LABELS[i])
    df["title"] = df["title"].map(normalize)
    df["content"] = df["content"].map(normalize)
    df["text"] = df["title"] + ". " + df["content"]
    return df[["url", "label", "label_id", "title", "content", "text"]]


def main():
    download()
    train_raw, test_raw = load_official("train"), load_official("test")
    stats = {"official_train_rows": len(train_raw), "official_test_rows": len(test_raw)}

    labels_per_url = train_raw.groupby("url")["label_id"].nunique()
    conflicting = set(labels_per_url[labels_per_url > 1].index)
    train = train_raw[~train_raw["url"].isin(conflicting)].drop_duplicates("url")
    stats["train_conflicting_label_articles_dropped"] = len(conflicting)

    test = test_raw.drop_duplicates("url").drop_duplicates("text")
    leaked = train["url"].isin(test["url"]) | train["text"].isin(test["text"])
    stats["train_test_overlap_dropped"] = int(leaked.sum())
    train = train[~leaked]

    train, val = train_test_split(train, test_size=VAL_SIZE, stratify=train["label_id"], random_state=SEED)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    # train_original keeps the duplicates so we can measure their effect (E6)
    splits = {"train": train, "val": val, "test": test, "train_original": train_raw}
    for name, df in splits.items():
        df.to_csv(PROCESSED_DIR / f"{name}.csv", index=False)
        stats[f"{name}_rows"] = len(df)
        stats[f"{name}_label_counts"] = {l: int(c) for l, c in df["label"].value_counts().reindex(LABELS, fill_value=0).items()}

    n_words = train["text"].str.split().str.len()
    stats["train_words_per_article"] = {k: round(float(v), 1) for k, v in n_words.describe().items()}

    (PROCESSED_DIR / "stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps({k: v for k, v in stats.items() if not isinstance(v, dict)}, indent=2))


if __name__ == "__main__":
    main()
