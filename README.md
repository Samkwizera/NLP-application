# Kinyarwanda News Topic Classification (KINNEWS)

Classifies Kinyarwanda news articles into 14 topics (politics, sport, economy, health, …) using the
KINNEWS corpus. Compares classical TF-IDF models, BiLSTMs with Word2Vec and attention, and fine-tuned
multilingual / African-centric Transformers. Pays particular attention to **generalisation across news sources**.

| | |
|---|---|
| **Live demo** | https://kinnews-classifier.streamlit.app/ (free tier: if it has been asleep, click "wake up" and wait ~1 min) |
| **Demo video** | _TODO_ |
| **Report (PDF)** | [report/report.pdf](report/report.pdf) |
| **Fine-tuned model** | https://huggingface.co/Samkwizera/kinnews-topic-classifier |

## Problem
Most of the news written in Kinyarwanda (about 12M speakers) has no automatic topic tagging. Topic classification
helps with news aggregation, search and media monitoring. It is also a standard benchmark for low-resource NLP.

## Dataset
KINNEWS (Niyongabo et al., 2020) contains 21,268 articles from 20 Rwandan news sources, labelled with 14 topics.
Data audit (`scripts/prepare_data.py`):

| Issue in the official release | Count | Our handling |
|---|---|---|
| Duplicate training rows (same URL) | 9,387 of 17,014 | deduplicated |
| Training articles whose copies carry different labels | 160 | removed |
| Articles in both train and test | 3 | removed from train |
| Test articles from news sources never seen in training | 54% | kept: the test set measures **cross-source** generalisation |

Final splits: **train 6,344 / val 1,120 (stratified, same sources as train) / test 3,464 (official test, deduplicated)**.

## Repository layout
```
src/kinnews/        shared code: text normalisation, data loading, metrics
scripts/
  prepare_data.py       download + audit + build splits
  train_baseline.py     TF-IDF experiments E1-E6 (CPU, ~50 min)
  train_bilstm.py       BiLSTM experiments E7-E9
  train_transformer.py  Transformer fine-tuning T1-T5
notebooks/gpu_experiments.ipynb   runs E7-E9 and T1-T5 on a free GPU (Kaggle/Colab)
notebooks/kaggle_run1.ipynb       first Kaggle run (E7-E9, T1-T3 outputs)
notebooks/kaggle_run2.ipynb       full Kaggle run (E7-E9, T1-T5), the reported results
app/app.py          Streamlit web app (app/requirements.txt is what Streamlit Cloud installs)
results/            metrics JSON per experiment, confusion matrices, predictions
```

## Reproduce
```bash
python -m venv .venv && .venv/Scripts/activate      # Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt
python scripts/prepare_data.py
python scripts/train_baseline.py
# GPU experiments: import notebooks/gpu_experiments.ipynb into Kaggle, Save & Run All
streamlit run app/app.py
```

## Results
Macro-F1 is the main metric because the topics are very imbalanced (31 vs 1,503 training articles).
Every choice (hyperparameters, best model) was made on **val**. Test was only used for reporting.
Val shares news sources with train, while most of test comes from unseen sources, so the test scores are lower.

| Exp | Model | Val macro-F1 | Test macro-F1 | Test acc |
|---|---|---|---|---|
| E1 | TF-IDF word 1-2grams + LogReg (**baseline**) | 0.583 | 0.426 | 0.697 |
| E2 | TF-IDF + Naive Bayes | 0.563 | 0.426 | 0.701 |
| E2 | TF-IDF + Linear SVM | 0.665 | 0.478 | 0.718 |
| E3 | E1 without splitting apostrophe elision | 0.557 | 0.414 | 0.684 |
| E4 | TF-IDF char 2-5grams + LogReg | 0.630 | 0.451 | 0.705 |
| E5 | E1 + balanced class weights | 0.682 | 0.480 | 0.716 |
| E6 | E1 trained with the official duplicates | 0.586 | 0.425 | 0.690 |
| E7 | BiLSTM, random embeddings, max pooling | 0.509 | 0.383 | 0.609 |
| E8 | BiLSTM, Word2Vec embeddings, max pooling | 0.617 | 0.461 | 0.622 |
| E9 | BiLSTM, Word2Vec embeddings, attention pooling | 0.633 | 0.535 | 0.717 |
| T1 | XLM-R base | 0.440 | 0.409 | 0.699 |
| T2 | AfroXLMR base | 0.525 | 0.453 | 0.737 |
| T3 | AfriBERTa base | 0.605 | 0.467 | 0.726 |
| T4 | AfriBERTa base, 512 tokens | 0.606 | 0.524 | 0.733 |
| **T5** | **AfriBERTa base, 512 tokens, class-weighted loss (deployed)** | **0.703** | **0.584** | **0.751** |

Main findings:
- Pretraining that includes Kinyarwanda helps a lot: XLM-R < AfroXLMR < AfriBERTa.
- Class weighting gave the largest single gain for both TF-IDF (E1 -> E5) and AfriBERTa (T4 -> T5).
- Word2Vec and attention both helped the BiLSTM (E7 -> E8 -> E9).
- The deduplicated data loses nothing compared with the duplicated official set (E1 vs E6).
- Repeated runs moved by up to ~2 points, so smaller differences are treated as noise.

Error analysis ([results/error_analysis.md](results/error_analysis.md)): accuracy is 88% on the test source seen in
training and 61-66% on unseen sources. Many errors come from source-specific labels (e.g. advice columns and
horoscopes labelled *education*) and articles that cover two topics (fashion/entertainment, environment/economy).

## Acknowledgements
KINNEWS dataset (Niyongabo et al., 2020). Pretrained models: XLM-R (Conneau et al., 2020),
AfroXLMR (Alabi et al., 2022), AfriBERTa (Ogueji et al., 2021). Libraries: Hugging Face Transformers, scikit-learn,
PyTorch, gensim, Streamlit.
