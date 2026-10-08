# Checks how much a random split of the official (duplicated) train set inflates accuracy.
# The KINNEWS paper validated on a random 9:1 split like this one.
import sys
from pathlib import Path

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from kinnews.data import load_split  # noqa: E402
from train_baseline import word_tfidf  # noqa: E402


def main():
    orig = load_split("train_original")
    train, val = train_test_split(orig, test_size=0.1, random_state=0)
    model = Pipeline([("tfidf", word_tfidf()), ("clf", LinearSVC(C=1))])
    model.fit(train["text"], train["label_id"])
    acc = accuracy_score(val["label_id"], model.predict(val["text"]))
    leaked = val["url"].isin(train["url"]).mean()
    print(f"random 9:1 split with duplicates: accuracy {acc:.4f}")
    print(f"share of val articles that also appear in train: {leaked:.3f}")


if __name__ == "__main__":
    main()
