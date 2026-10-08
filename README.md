# Kinyarwanda News Topic Classification (KINNEWS)

Classifies Kinyarwanda news articles into 14 topics (politics, sport, economy, health, …) using the
KINNEWS corpus. Compares classical TF-IDF models, BiLSTMs with Word2Vec and attention, and fine-tuned
multilingual / African-centric Transformers. Pays particular attention to **generalisation across news sources**.

| | |
|---|---|
| **Live demo** | _TODO: Hugging Face Space link_ |
| **Demo video** | _TODO_ |
| **Report (PDF)** | [report/report.pdf](report/report.pdf) |
| **Fine-tuned model** | _TODO: Hugging Face Hub link_ |

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
_TODO: experiment table_

## Acknowledgements
KINNEWS dataset (Niyongabo et al., 2020). Pretrained models: XLM-R (Conneau et al., 2020),
AfroXLMR (Alabi et al., 2022), AfriBERTa (Ogueji et al., 2021). Libraries: Hugging Face Transformers, scikit-learn,
PyTorch, gensim, Streamlit.
