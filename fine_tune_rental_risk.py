#!/usr/bin/env python3
"""
Fine-tune a small legal risk classifier for rental agreements.

Input dataset: CSV at data/rental_risk_dataset.csv with columns:
    text,label
where label is one of: SAFE, RISKY

This script:
- Loads and splits the dataset (train/val/test).
- Fine-tunes DistilBERT for 2-class classification.
- Computes accuracy/precision/recall/F1 without extra dependencies.
- Saves the model (with label mappings) to ./models/rental-risk-bert
- Writes metrics to ./models/rental-risk-bert/train_metrics.json

Run:
    python fine_tune_rental_risk.py
Optional args:
    --csv data/rental_risk_dataset.csv
    --outdir models/rental-risk-bert
    --epochs 3
    --batch 16
    --lr 5e-5
    --maxlen 256
"""

import os
import json
import argparse
import random
from dataclasses import dataclass
from typing import Dict, List

import torch
import pandas as pd
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)

# ----------------------------
# Utilities (no sklearn needed)
# ----------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def compute_metrics_binary(preds: List[int], labels: List[int]) -> Dict[str, float]:
    assert len(preds) == len(labels)
    tp = sum((p == 1 and y == 1) for p, y in zip(preds, labels))
    tn = sum((p == 0 and y == 0) for p, y in zip(preds, labels))
    fp = sum((p == 1 and y == 0) for p, y in zip(preds, labels))
    fn = sum((p == 0 and y == 1) for p, y in zip(preds, labels))
    total = len(labels)
    acc = (tp + tn) / max(total, 1)
    prec = tp / max((tp + fp), 1)
    rec = tp / max((tp + fn), 1)
    f1 = 2 * prec * rec / max((prec + rec), 1e-9)
    return {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn, "total": total
    }

# ----------------------------
# Torch datasets
# ----------------------------
@dataclass
class TextClsDataset(torch.utils.data.Dataset):
    encodings: Dict[str, torch.Tensor]
    labels: List[int]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

# ----------------------------
# Main
# ----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="data/rental_risk_dataset.csv")
    parser.add_argument("--outdir", default="models/rental-risk-bert")
    parser.add_argument("--base", default="distilbert-base-uncased")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--maxlen", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.outdir, exist_ok=True)

    # 1) Load dataset
    df = pd.read_csv(args.csv).dropna(subset=["text", "label"])
    df["label"] = df["label"].str.strip().str.upper()

    # Supported labels
    label2id = {"SAFE": 0, "RISKY": 1}
    id2label = {v: k for k, v in label2id.items()}

    # Filter unexpected labels
    df = df[df["label"].isin(label2id.keys())].reset_index(drop=True)

    # 2) Deterministic split: 80/10/10
    rng = random.Random(args.seed)
    idxs = list(range(len(df)))
    rng.shuffle(idxs)
    n = len(idxs)
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)
    train_idx = idxs[:n_train]
    val_idx = idxs[n_train:n_train + n_val]
    test_idx = idxs[n_train + n_val:]

    def subset(id_list):
        sub = df.iloc[id_list]
        texts = sub["text"].tolist()
        labels = [label2id[l] for l in sub["label"].tolist()]
        return texts, labels

    train_texts, train_labels = subset(train_idx)
    val_texts, val_labels = subset(val_idx)
    test_texts, test_labels = subset(test_idx)

    # 3) Tokeniser / model
    tokenizer = AutoTokenizer.from_pretrained(args.base)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.base,
        num_labels=2,
        id2label=id2label,
        label2id=label2id,
    )

    # 4) Tokenise
    def tok(texts):
        return tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=args.maxlen,
            return_tensors="pt"
        )

    train_enc = tok(train_texts)
    val_enc = tok(val_texts)
    test_enc = tok(test_texts)

    train_ds = TextClsDataset(train_enc, train_labels)
    val_ds = TextClsDataset(val_enc, val_labels)
    test_ds = TextClsDataset(test_enc, test_labels)

    # 5) Training arguments
    training_args = TrainingArguments(
        output_dir=os.path.join(args.outdir, "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        learning_rate=args.lr,
        weight_decay=0.1,
        logging_steps=50,
        seed=args.seed,
        disable_tqdm=False,
        eval_strategy="epoch",  # <--- Add this
        save_strategy="epoch",        # <--- Add this
        load_best_model_at_end=True,  # <--- Add this
        metric_for_best_model="f1",
    )

    # 6) Define compute_metrics callback using our manual function
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = logits.argmax(axis=-1)
        return compute_metrics_binary(preds.tolist(), labels.tolist())

    # 7) Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.evaluate()  # runs validation once after training

    # 8) Evaluate on test set
    test_out = trainer.predict(test_ds)
    metrics = compute_metrics_binary(
        preds=test_out.predictions.argmax(axis=-1).tolist(),
        labels=test_out.label_ids.tolist()
    )

    # 9) Save model + tokenizer + metrics
    model.save_pretrained(args.outdir)
    tokenizer.save_pretrained(args.outdir)
    with open(os.path.join(args.outdir, "train_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # 10) Short console summary
    print("Saved fine-tuned model to:", args.outdir)
    print("Test metrics:", json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

