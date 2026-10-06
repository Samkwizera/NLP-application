# BiLSTM classifier (E7-E9): embeddings -> BiLSTM -> max or attention pooling -> linear.
# e.g. python scripts/train_bilstm.py --run-name e8_bilstm_w2v_max --pooling max --w2v
import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from kinnews import LABELS  # noqa: E402
from kinnews.data import ROOT, load_split  # noqa: E402
from kinnews.metrics import compute, save_run  # noqa: E402
from kinnews.text import tokenize  # noqa: E402

PAD, UNK = 0, 1


def build_vocab(token_lists, min_freq=2, max_size=50_000):
    counts = Counter(t for toks in token_lists for t in toks)
    words = [w for w, c in counts.most_common(max_size) if c >= min_freq]
    return {w: i + 2 for i, w in enumerate(words)}  # 0 = <pad>, 1 = <unk>


def encode(token_lists, vocab, max_len):
    ids = np.zeros((len(token_lists), max_len), dtype=np.int64)
    for i, toks in enumerate(token_lists):
        seq = [vocab.get(t, UNK) for t in toks[:max_len]]
        ids[i, :len(seq)] = seq
    return torch.from_numpy(ids)


def word2vec_matrix(token_lists, vocab, dim, seed):
    # skip-gram trained on train articles only so val/test text is never seen
    from gensim.models import Word2Vec
    w2v = Word2Vec(token_lists, vector_size=dim, window=5, min_count=2, sg=1, epochs=10, workers=4, seed=seed)
    rng = np.random.default_rng(seed)
    matrix = rng.normal(0, 0.1, (len(vocab) + 2, dim)).astype(np.float32)
    matrix[PAD] = 0
    found = 0
    for word, idx in vocab.items():
        if word in w2v.wv:
            matrix[idx] = w2v.wv[word]
            found += 1
    print(f"Word2Vec coverage: {found}/{len(vocab)} vocabulary words")
    return torch.from_numpy(matrix)


class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden, n_classes, pooling, dropout, pretrained=None):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD)
        if pretrained is not None:
            self.embedding.weight.data.copy_(pretrained)
        self.lstm = nn.LSTM(emb_dim, hidden, batch_first=True, bidirectional=True)
        self.pooling = pooling
        if pooling == "attention":
            self.att_proj = nn.Linear(2 * hidden, 2 * hidden)
            self.att_vec = nn.Linear(2 * hidden, 1, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(2 * hidden, n_classes)

    def forward(self, ids, return_attention=False):
        mask = ids != PAD                                       # (B, T)
        h, _ = self.lstm(self.dropout(self.embedding(ids)))     # (B, T, 2H)
        if self.pooling == "max":
            pooled = h.masked_fill(~mask.unsqueeze(-1), -1e4).max(dim=1).values
            weights = None
        else:
            # additive attention: a_t = softmax(w . tanh(W h_t))
            scores = self.att_vec(torch.tanh(self.att_proj(h))).squeeze(-1)   # (B, T)
            weights = torch.softmax(scores.masked_fill(~mask, -1e4), dim=1)
            pooled = (weights.unsqueeze(-1) * h).sum(dim=1)
        logits = self.out(self.dropout(pooled))
        return (logits, weights) if return_attention else logits


def predict(model, loader, device):
    model.eval()
    preds = []
    with torch.no_grad():
        for ids, _ in loader:
            preds.append(model(ids.to(device)).argmax(-1).cpu())
    return torch.cat(preds).numpy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-name", required=True)
    p.add_argument("--pooling", choices=["max", "attention"], default="max")
    p.add_argument("--w2v", action="store_true", help="initialise embeddings with Word2Vec")
    p.add_argument("--max-len", type=int, default=400)
    p.add_argument("--emb-dim", type=int, default=300)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train, val, test = load_split("train"), load_split("val"), load_split("test")
    toks = {name: [tokenize(t) for t in df["text"]] for name, df in [("train", train), ("val", val), ("test", test)]}

    vocab = build_vocab(toks["train"])
    pretrained = word2vec_matrix(toks["train"], vocab, args.emb_dim, args.seed) if args.w2v else None

    def loader(name, df, shuffle):
        ds = TensorDataset(encode(toks[name], vocab, args.max_len), torch.tensor(df["label_id"].values))
        return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle)

    train_dl, val_dl, test_dl = loader("train", train, True), loader("val", val, False), loader("test", test, False)

    model = BiLSTMClassifier(len(vocab) + 2, args.emb_dim, args.hidden, len(LABELS), args.pooling,
                             args.dropout, pretrained).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    best_f1, best_state, bad_epochs = -1, None, 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for ids, y in train_dl:
            ids, y = ids.to(device), y.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(ids), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # LSTMs can have exploding gradients
            optimizer.step()
            total += loss.item() * len(y)
        val_f1 = compute(val["label_id"], predict(model, val_dl, device))["macro_f1"]
        print(f"epoch {epoch:2d}  train loss {total / len(train):.4f}  val macro-F1 {val_f1:.4f}")
        if val_f1 > best_f1:
            best_f1, best_state, bad_epochs = val_f1, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print("early stopping")
                break

    model.load_state_dict(best_state)
    test_pred = predict(model, test_dl, device)
    config = {**vars(args), "vocab_size": len(vocab), "params": sum(p.numel() for p in model.parameters())}
    save_run(args.run_name, config, compute(val["label_id"], predict(model, val_dl, device)),
             compute(test["label_id"], test_pred), test["label_id"], test_pred)


if __name__ == "__main__":
    main()
