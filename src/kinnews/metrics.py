import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score  # noqa: E402

from kinnews import LABELS  # noqa: E402
from kinnews.data import ROOT  # noqa: E402

RESULTS_DIR = ROOT / "results"


def compute(y_true, y_pred) -> dict:
    # macro-F1 is the main metric: rare topics like fashion (~30 articles) count as much as politics
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "macro_f1": round(f1_score(y_true, y_pred, average="macro"), 4),
        "weighted_f1": round(f1_score(y_true, y_pred, average="weighted"), 4),
        "per_class": classification_report(
            y_true, y_pred, labels=range(len(LABELS)), target_names=LABELS, output_dict=True, zero_division=0
        ),
    }


def save_run(name: str, config: dict, val: dict, test: dict, y_true=None, y_pred=None):
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / f"{name}.json").write_text(json.dumps({"name": name, "config": config, "val": val, "test": test}, indent=2))
    if y_true is not None:
        plot_confusion(y_true, y_pred, RESULTS_DIR / "figures" / f"cm_{name}.png", title=name)
    print(f"{name:40s} val macro-F1 {val['macro_f1']:.4f} | test acc {test['accuracy']:.4f} macro-F1 {test['macro_f1']:.4f}")


def plot_confusion(y_true, y_pred, path: Path, title: str = ""):
    cm = confusion_matrix(y_true, y_pred, labels=range(len(LABELS)), normalize="true")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 8))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", xticklabels=LABELS, yticklabels=LABELS, cbar=False, ax=ax,
                annot_kws={"size": 7})
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Row-normalised confusion matrix (test) - {title}")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
