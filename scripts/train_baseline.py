# TF-IDF baselines and experiments E1-E6. Hyperparameters are picked on val only.
import pickle
import sys
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from kinnews.data import ROOT, load_split  # noqa: E402
from kinnews.metrics import compute, save_run  # noqa: E402
from kinnews.text import tokenize  # noqa: E402


def tok_split(text):
    return tokenize(text, split_elision=True)


def tok_nosplit(text):
    return tokenize(text, split_elision=False)


def word_tfidf(tokenizer=tok_split):
    return TfidfVectorizer(tokenizer=tokenizer, lowercase=False, token_pattern=None,
                           ngram_range=(1, 2), min_df=2, max_features=200_000, sublinear_tf=True)


def char_tfidf():
    return TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3, max_features=300_000,
                           sublinear_tf=True, lowercase=True)


def tune_and_eval(name, make_pipeline, grid, train, val, test, config):
    best = None
    for value in grid:
        pipe = make_pipeline(value)
        pipe.fit(train["text"], train["label_id"])
        score = compute(val["label_id"], pipe.predict(val["text"]))
        if best is None or score["macro_f1"] > best[1]["macro_f1"]:
            best = (value, score, pipe)
    value, val_scores, pipe = best
    test_pred = pipe.predict(test["text"])
    save_run(name, {**config, "selected_hparam": value}, val_scores, compute(test["label_id"], test_pred),
             test["label_id"], test_pred)
    return pipe


def main():
    train, val, test = load_split("train"), load_split("val"), load_split("test")
    C_GRID = [0.5, 1, 4, 16]

    def logreg(features, class_weight=None):
        return lambda C: Pipeline([("tfidf", features()),
                                   ("clf", LogisticRegression(C=C, max_iter=2000, class_weight=class_weight))])

    best = tune_and_eval("e1_tfidf_word_logreg", logreg(word_tfidf), C_GRID, train, val, test,
                         {"features": "tfidf word 1-2gram", "clf": "logreg", "grid_C": C_GRID})

    tune_and_eval("e2_tfidf_word_nb",
                  lambda a: Pipeline([("tfidf", word_tfidf()), ("clf", MultinomialNB(alpha=a))]),
                  [0.01, 0.05, 0.1, 0.5], train, val, test, {"features": "tfidf word 1-2gram", "clf": "multinomial NB"})
    tune_and_eval("e2_tfidf_word_svm",
                  lambda C: Pipeline([("tfidf", word_tfidf()), ("clf", LinearSVC(C=C))]),
                  [0.1, 0.5, 1, 4], train, val, test, {"features": "tfidf word 1-2gram", "clf": "linear SVM"})

    tune_and_eval("e3_tfidf_word_logreg_no_elision_split", logreg(lambda: word_tfidf(tok_nosplit)), C_GRID,
                  train, val, test, {"features": "tfidf word 1-2gram, elision NOT split", "clf": "logreg"})

    tune_and_eval("e4_tfidf_char_logreg", logreg(char_tfidf), C_GRID, train, val, test,
                  {"features": "tfidf char_wb 2-5gram", "clf": "logreg"})

    tune_and_eval("e5_tfidf_word_logreg_balanced", logreg(word_tfidf, "balanced"), C_GRID, train, val, test,
                  {"features": "tfidf word 1-2gram", "clf": "logreg class_weight=balanced"})

    # same model as E1 but trained on the official train set with its duplicates
    orig = load_split("train_original")
    orig = orig[~orig["url"].isin(val["url"])]  # keep val unseen so model selection is still fair
    tune_and_eval("e6_tfidf_word_logreg_with_duplicates", logreg(word_tfidf), C_GRID, orig, val, test,
                  {"features": "tfidf word 1-2gram", "clf": "logreg", "train": "official train incl. duplicates"})

    models_dir = ROOT / "models"
    models_dir.mkdir(exist_ok=True)
    with open(models_dir / "baseline_tfidf_logreg.pkl", "wb") as f:
        pickle.dump(best, f)


if __name__ == "__main__":
    main()
