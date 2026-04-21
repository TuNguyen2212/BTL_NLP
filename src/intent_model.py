"""Huấn luyện và inference PhoBERT cho Intent Classification."""

import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    MODEL_DIR,
    PHOBERT_MODEL_NAME,
    ANNOTATED_INTENT_PATH,
    INTENT_LABELS,
)

INTENT_PHOBERT_PATH = os.path.join(MODEL_DIR, "intent_phobert")

LABEL2ID = {l: i for i, l in enumerate(INTENT_LABELS)}
ID2LABEL = {i: l for l, i in LABEL2ID.items()}


def load_data(path: str = ANNOTATED_INTENT_PATH):
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)

    texts = [d["clause"] for d in raw]
    labels = [d["intent"] for d in raw]

    from collections import Counter

    counts = Counter(labels)
    print(f"[Intent Model] Loaded {len(texts)} samples")
    for lbl in INTENT_LABELS:
        print(f"  {lbl:25}: {counts.get(lbl, 0)}")

    return texts, labels


def make_compute_metrics():
    from sklearn.metrics import f1_score, accuracy_score

    def compute_metrics(eval_pred):
        logits, label_ids = eval_pred
        preds = np.argmax(logits, axis=-1)

        acc = accuracy_score(label_ids, preds)
        f1_mac = f1_score(label_ids, preds, average="macro", zero_division=0)
        f1_wgt = f1_score(label_ids, preds, average="weighted", zero_division=0)

        return {
            "accuracy": round(acc, 3),
            "f1_macro": round(f1_mac, 3),
            "f1_weighted": round(f1_wgt, 3),
        }

    return compute_metrics


def compute_class_weights(labels: list[str]) -> list[float]:
    from collections import Counter

    counts = Counter(labels)
    total = len(labels)
    n_cls = len(INTENT_LABELS)
    weights = []
    for lbl in INTENT_LABELS:
        cnt = counts.get(lbl, 1)
        weights.append(total / (n_cls * cnt))
    print(
        f"[Intent Model] Class weights: "
        + "  ".join(f"{l}={w:.2f}" for l, w in zip(INTENT_LABELS, weights))
    )
    return weights


def make_weighted_trainer(class_weights: list[float]):
    import torch
    from transformers import Trainer

    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.get("labels")
            outputs = model(**inputs)
            logits = outputs.get("logits")

            weights = torch.tensor(
                class_weights, dtype=torch.float, device=logits.device
            )
            loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
            loss = loss_fn(logits, labels)

            return (loss, outputs) if return_outputs else loss

    return WeightedTrainer


def load_label_config(model_path: str = INTENT_PHOBERT_PATH) -> tuple[dict, dict]:
    """Load label mappings from the exported model if available."""
    label_path = os.path.join(model_path, "label_config.json")
    if not os.path.exists(label_path):
        return LABEL2ID, ID2LABEL

    with open(label_path, encoding="utf-8") as f:
        raw = json.load(f)

    label2id = raw.get("label2id", LABEL2ID)
    id2label_raw = raw.get("id2label", ID2LABEL)
    id2label = {int(k): v for k, v in id2label_raw.items()}
    return label2id, id2label


def unwrap_pipeline_result(result):
    """Flatten nested pipeline outputs such as [[{...}]] -> {...}."""
    while isinstance(result, list):
        result = result[0]
    return result


def train(
    model_path: str = INTENT_PHOBERT_PATH,
    num_epochs: int = 15,
    batch_size: int = 8,
    learning_rate: float = 2e-5,
):

    from transformers import (
        AutoTokenizer,
        AutoModelForSequenceClassification,
        TrainingArguments,
        DataCollatorWithPadding,
    )
    from datasets import Dataset
    from sklearn.model_selection import StratifiedShuffleSplit

    texts, labels = load_data()

    label_ids = [LABEL2ID[l] for l in labels]
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(sss.split(texts, label_ids))

    train_texts = [texts[i] for i in train_idx]
    train_labels = [label_ids[i] for i in train_idx]
    val_texts = [texts[i] for i in val_idx]
    val_labels = [label_ids[i] for i in val_idx]

    print(f"\n  Train: {len(train_texts)} | Val: {len(val_texts)}")

    print(f"\n[Intent Model] Loading {PHOBERT_MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(PHOBERT_MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        PHOBERT_MODEL_NAME,
        num_labels=len(INTENT_LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
    )

    def tokenize_fn(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=128,
            padding=False,
        )

    train_ds = Dataset.from_dict({"text": train_texts, "labels": train_labels})
    val_ds = Dataset.from_dict({"text": val_texts, "labels": val_labels})
    train_ds = train_ds.map(tokenize_fn, batched=True)
    val_ds = val_ds.map(tokenize_fn, batched=True)
    train_ds = train_ds.remove_columns(["text"])
    val_ds = val_ds.remove_columns(["text"])

    class_weights = compute_class_weights(labels)
    warmup_steps = int((len(train_ds) // batch_size) * num_epochs * 0.1)
    os.makedirs(model_path, exist_ok=True)
    args = TrainingArguments(
        output_dir=model_path,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=0.01,
        warmup_steps=warmup_steps,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        logging_steps=5,
        report_to="none",
        seed=42,
    )

    data_collator = DataCollatorWithPadding(tokenizer)
    compute_metrics = make_compute_metrics()
    WeightedTrainer = make_weighted_trainer(class_weights)

    trainer = WeightedTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print(f"\n[Intent Model] Training {num_epochs} epochs...")
    trainer.train()

    trainer.save_model(model_path)
    tokenizer.save_pretrained(model_path)
    with open(os.path.join(model_path, "label_config.json"), "w", encoding="utf-8") as f:
        json.dump({"label2id": LABEL2ID, "id2label": ID2LABEL}, f, ensure_ascii=False)
    print(f"[Intent Model] Saved -> {model_path}")

    return trainer


def evaluate_and_compare(model_path: str = INTENT_PHOBERT_PATH):
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
    from sklearn.metrics import classification_report
    import joblib
    from config import INTENT_MODEL_PATH

    texts, labels = load_data()
    _, id2label = load_label_config(model_path)

    print(f"\n[Intent Model] Loading PhoBERT from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    cls_pipe = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        device=-1,
        top_k=1,
        truncation=True,
        max_length=128,
    )

    phobert_preds = []
    for text in texts:
        result = unwrap_pipeline_result(cls_pipe(text))
        label = result["label"]
        if label.startswith("LABEL_"):
            idx = int(label.split("_")[1])
            label = id2label.get(idx, label)
        phobert_preds.append(label)

    tfidf_preds = []
    if os.path.exists(INTENT_MODEL_PATH):
        tfidf_model = joblib.load(INTENT_MODEL_PATH)
        tfidf_preds = tfidf_model.predict(texts).tolist()
    else:
        print("  [WARN] TF-IDF model không tìm thấy, bỏ qua so sánh")

    print("\n" + "=" * 60)
    print("  INTENT CLASSIFICATION - So sánh model")
    print("=" * 60)

    print("\n  [PhoBERT]")
    print(
        classification_report(
            labels, phobert_preds, target_names=INTENT_LABELS, zero_division=0
        )
    )

    if tfidf_preds:
        print("  [TF-IDF + Logistic Regression]")
        print(
            classification_report(
                labels, tfidf_preds, target_names=INTENT_LABELS, zero_division=0
            )
        )

    from sklearn.metrics import f1_score, accuracy_score

    print("  Bảng tóm tắt (macro F1):")
    print(f"  {'Model':25} {'Accuracy':>10} {'F1 macro':>10} {'F1 weighted':>12}")
    print(f"  {'-'*60}")

    pb_acc = accuracy_score(labels, phobert_preds)
    pb_f1m = f1_score(labels, phobert_preds, average="macro", zero_division=0)
    pb_f1w = f1_score(labels, phobert_preds, average="weighted", zero_division=0)
    print(f"  {'PhoBERT':25} {pb_acc:>10.3f} {pb_f1m:>10.3f} {pb_f1w:>12.3f}")

    if tfidf_preds:
        tf_acc = accuracy_score(labels, tfidf_preds)
        tf_f1m = f1_score(labels, tfidf_preds, average="macro", zero_division=0)
        tf_f1w = f1_score(labels, tfidf_preds, average="weighted", zero_division=0)
        print(f"  {'TF-IDF + LR':25} {tf_acc:>10.3f} {tf_f1m:>10.3f} {tf_f1w:>12.3f}")

    print("=" * 60)

    return {
        "phobert": {"accuracy": pb_acc, "f1_macro": pb_f1m},
        "tfidf": {
            "accuracy": tf_acc if tfidf_preds else None,
            "f1_macro": tf_f1m if tfidf_preds else None,
        },
    }


_cls_pipeline = None


def predict_intent_phobert(clause: str, model_path: str = INTENT_PHOBERT_PATH) -> dict:
    """Dự đoán intent bằng PhoBERT."""
    global _cls_pipeline
    label2id, id2label = load_label_config(model_path)
    if _cls_pipeline is None:
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model không tìm thấy tại {model_path}.\n"
                f"Hãy chạy: python src/intent_model.py --train"
            )
        from transformers import (
            AutoTokenizer,
            AutoModelForSequenceClassification,
            pipeline,
        )

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        model.config.label2id = label2id
        model.config.id2label = id2label
        _cls_pipeline = pipeline(
            "text-classification",
            model=model,
            tokenizer=tokenizer,
            device=-1,
            top_k=1,
            truncation=True,
            max_length=128,
        )
        print(f"[Intent Model] Loaded <- {model_path}")

    if not clause.strip():
        return {"intent": "Obligation", "confidence": 0.0, "source": "default"}

    result = unwrap_pipeline_result(_cls_pipeline(clause))
    label = result["label"]
    if label.startswith("LABEL_"):
        idx = int(label.split("_")[1])
        label = id2label.get(idx, label)

    return {
        "intent": label,
        "confidence": round(float(result["score"]), 4),
        "source": "phobert",
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="Fine-tune PhoBERT")
    parser.add_argument("--eval", action="store_true", help="So sánh TF-IDF vs PhoBERT")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=2e-5)
    args = parser.parse_args()

    if not args.train and not args.eval:
        parser.print_help()
        sys.exit(0)

    if args.train:
        train(num_epochs=args.epochs, learning_rate=args.lr)

    if args.eval:
        evaluate_and_compare()
