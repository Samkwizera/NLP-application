# Fine-tunes a pretrained encoder + classification head (T1-T5). Needs a GPU, see notebooks/finetune_colab.ipynb
# e.g. python scripts/train_transformer.py --model Davlan/afro-xlmr-base --run-name t2_afroxlmr_base
import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from transformers import (AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding,
                          EarlyStoppingCallback, Trainer, TrainingArguments, set_seed)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from kinnews import LABELS  # noqa: E402
from kinnews.data import ROOT, load_split  # noqa: E402
from kinnews.metrics import compute, save_run  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="Hugging Face model id")
    p.add_argument("--run-name", required=True)
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save-model", action="store_true", help="keep the final model in models/<run-name>")
    p.add_argument("--class-weights", action="store_true", help="balanced class weights in the loss")
    return p.parse_args()


class WeightedTrainer(Trainer):
    # same as Trainer but the cross-entropy loss is weighted per class
    def __init__(self, class_weights, **kwargs):
        super().__init__(**kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss_fn = torch.nn.CrossEntropyLoss(weight=self.class_weights.to(outputs.logits.device))
        loss = loss_fn(outputs.logits.float(), labels)
        return (loss, outputs) if return_outputs else loss


def main():
    args = parse_args()
    set_seed(args.seed)
    train, val, test = load_split("train"), load_split("val"), load_split("test")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=len(LABELS),
        id2label=dict(enumerate(LABELS)), label2id={l: i for i, l in enumerate(LABELS)},
    )

    def to_dataset(df):
        ds = Dataset.from_dict({"text": df["text"].tolist(), "label": df["label_id"].tolist()})
        return ds.map(lambda b: tokenizer(b["text"], truncation=True, max_length=args.max_length),
                      batched=True, remove_columns=["text"])

    train_ds, val_ds, test_ds = to_dataset(train), to_dataset(val), to_dataset(test)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        m = compute(labels, logits.argmax(-1))
        return {"accuracy": m["accuracy"], "macro_f1": m["macro_f1"]}

    out_dir = ROOT / "models" / args.run_name
    training_args = TrainingArguments(
        output_dir=str(out_dir),
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        weight_decay=args.weight_decay,
        # newer transformers dropped warmup_ratio, so turn the ratio into a step count
        warmup_steps=int(math.ceil(len(train_ds) / args.batch_size) * args.epochs * args.warmup_ratio),
        lr_scheduler_type="linear",
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,        # keep the epoch with the best val macro-F1
        metric_for_best_model="macro_f1",
        fp16=torch.cuda.is_available(),
        logging_steps=50,
        report_to="none",
        seed=args.seed,
    )
    trainer_kwargs = dict(
        model=model, args=training_args, train_dataset=train_ds, eval_dataset=val_ds,
        processing_class=tokenizer, data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics, callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )
    if args.class_weights:
        # same formula as sklearn's "balanced": n_samples / (n_classes * count), so rare topics weigh more
        counts = np.bincount(train["label_id"], minlength=len(LABELS))
        weights = torch.tensor(len(train) / (len(LABELS) * counts), dtype=torch.float32)
        trainer = WeightedTrainer(weights, **trainer_kwargs)
    else:
        trainer = Trainer(**trainer_kwargs)
    trainer.train()

    val_pred = trainer.predict(val_ds).predictions.argmax(-1)
    test_out = trainer.predict(test_ds)
    test_pred = test_out.predictions.argmax(-1)
    config = {k: v for k, v in vars(args).items() if k != "save_model"}
    config["best_checkpoint"] = trainer.state.best_model_checkpoint
    save_run(args.run_name, config, compute(val["label_id"], val_pred), compute(test["label_id"], test_pred),
             test["label_id"], test_pred)

    # per-article predictions for the error analysis
    probs = torch.softmax(torch.tensor(test_out.predictions, dtype=torch.float32), -1).numpy()
    preds = test[["url", "label", "title"]].copy()
    preds["pred"] = [LABELS[i] for i in test_pred]
    preds["confidence"] = probs.max(-1).round(4)
    preds["n_tokens"] = [min(len(x), args.max_length) for x in test_ds["input_ids"]]
    preds["truncated"] = [len(tokenizer(t)["input_ids"]) > args.max_length for t in test["text"]]
    preds.to_csv(ROOT / "results" / f"preds_{args.run_name}.csv", index=False)

    if args.save_model:
        trainer.save_model(str(out_dir / "final"))
        tokenizer.save_pretrained(str(out_dir / "final"))
    np.save(ROOT / "results" / f"probs_{args.run_name}.npy", probs)


if __name__ == "__main__":
    main()
