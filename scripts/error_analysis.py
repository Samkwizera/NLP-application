# Error analysis of the final model (T5) on the test set.
# Writes results/error_analysis.md and a few figures to results/figures/.
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from kinnews import LABELS  # noqa: E402
from kinnews.data import ROOT, load_split  # noqa: E402

RUN = "t5_best_maxlen512_weighted"
BASELINE = "e1_tfidf_word_logreg"
RESULTS = ROOT / "results"
FIGS = RESULTS / "figures"


def source(urls):
    return urls.str.extract(r"https?://(?:www\.)?([^/]+)")[0]


def main():
    test, train = load_split("test"), load_split("train")
    p = pd.read_csv(RESULTS / f"preds_{RUN}.csv", keep_default_na=False)
    assert (p["url"].values == test["url"].values).all()
    p["text"] = test["text"].values
    p["source"] = source(p["url"])
    p["seen_source"] = p["source"].isin(set(source(train["url"])))
    p["correct"] = p["label"] == p["pred"]

    final = json.load(open(RESULTS / f"{RUN}.json"))["test"]["per_class"]
    base = json.load(open(RESULTS / f"{BASELINE}.json"))["test"]["per_class"]
    per_class = pd.DataFrame({
        "n_train": train["label"].value_counts().reindex(LABELS),
        "n_test": [int(final[c]["support"]) for c in LABELS],
        "baseline_f1": [round(base[c]["f1-score"], 2) for c in LABELS],
        "precision": [round(final[c]["precision"], 2) for c in LABELS],
        "recall": [round(final[c]["recall"], 2) for c in LABELS],
        "f1": [round(final[c]["f1-score"], 2) for c in LABELS],
    }, index=LABELS)

    by_source = p.groupby(["source", "seen_source"])["correct"].agg(accuracy="mean", n="size").round(3)
    by_trunc = p.groupby("truncated")["correct"].agg(accuracy="mean", n="size").round(3)
    p["confidence_bin"] = pd.cut(p["confidence"], [0, 0.5, 0.7, 0.9, 1.0])
    by_conf = p.groupby("confidence_bin", observed=True)["correct"].agg(accuracy="mean", n="size").round(3)
    confusions = p[~p["correct"]].groupby(["label", "pred"]).size().sort_values(ascending=False).head(10)

    # confident mistakes are the most informative: the model "knows" something the label disagrees with
    examples = []
    for true, pred in confusions.index[:6]:
        rows = p[(p["label"] == true) & (p["pred"] == pred)].sort_values("confidence", ascending=False).head(2)
        for _, r in rows.iterrows():
            examples.append(f"- true **{true}** -> predicted **{pred}** ({r['confidence']:.2f}, {r['source']}): {r['title']}")
    good = []
    for c in ["tourism", "environment", "fashion", "culture"]:
        r = p[(p["label"] == c) & p["correct"]].sort_values("confidence", ascending=False).iloc[0]
        good.append(f"- **{c}** ({r['confidence']:.2f}): {r['title']}")

    FIGS.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    per_class[["baseline_f1", "f1"]].rename(columns={"baseline_f1": "E1 TF-IDF baseline", "f1": "T5 AfriBERTa"}).plot.bar(ax=ax)
    ax.set_ylabel("Test F1")
    ax.set_title("Per-class test F1: baseline vs final model")
    fig.tight_layout()
    fig.savefig(FIGS / "per_class_f1.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.bar([str(b) for b in by_conf.index], by_conf["accuracy"])
    ax.set_xlabel("Model confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy by confidence (test)")
    fig.tight_layout()
    fig.savefig(FIGS / "accuracy_by_confidence.png", dpi=150)
    plt.close(fig)

    report = f"""# Error analysis - {RUN}

## Per-class results (test)
{per_class.to_markdown()}

## Accuracy by news source
Sources marked seen_source=True also appear in the training data.
{by_source.to_markdown()}

## Accuracy by truncation (article longer than 512 tokens)
{by_trunc.to_markdown()}

## Accuracy by confidence
{by_conf.to_markdown()}

## Most common confusions
{confusions.rename('count').to_frame().to_markdown()}

## Confident mistakes
{chr(10).join(examples)}

## Confident correct predictions on rare topics
{chr(10).join(good)}
"""
    (RESULTS / "error_analysis.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
