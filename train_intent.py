import os
import sys
import io
import json
import joblib
import argparse

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score, accuracy_score
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    ANNOTATED_INTENT_PATH,
    INTENT_MODEL_PATH,
    INTENT_LABELS,
    INTENT_TFIDF_PARAMS,
    INTENT_LR_PARAMS,
    INTENT_KEYWORDS,
)


def load_data(path: str = ANNOTATED_INTENT_PATH):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    texts = [d["clause"] for d in data]
    labels = [d["intent"] for d in data]

    print(f"[Train] Loaded {len(texts)} samples từ {path}")
    counts = Counter(labels)
    for label in INTENT_LABELS:
        print(f"  {label:25}: {counts.get(label, 0)}")

    return texts, labels


def build_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("tfidf", TfidfVectorizer(**INTENT_TFIDF_PARAMS)),
            ("clf", LogisticRegression(**INTENT_LR_PARAMS)),
        ]
    )


def train(texts, labels, model_path: str = INTENT_MODEL_PATH):
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    pipeline = build_pipeline()
    pipeline.fit(texts, labels)

    joblib.dump(pipeline, model_path)
    print(f"\n[Train] Model saved -> {model_path}")
    return pipeline


def evaluate(pipeline, texts, labels):
    preds = pipeline.predict(texts)

    acc = accuracy_score(labels, preds)
    f1_mac = f1_score(labels, preds, average="macro", zero_division=0)
    f1_w = f1_score(labels, preds, average="weighted", zero_division=0)

    print(f"\n[Eval] TF-IDF+LR trên training data (tham khảo, train=test):")
    print(f"  Accuracy    : {acc:.3f}")
    print(f"  F1 macro    : {f1_mac:.3f}")
    print(f"  F1 weighted : {f1_w:.3f}")

    print(f"\n[Eval] Classification report:")
    print(classification_report(labels, preds, target_names=INTENT_LABELS, zero_division=0))

    return preds


def rule_based_predict(clause: str) -> str:
    for intent in INTENT_LABELS:
        for kw in INTENT_KEYWORDS.get(intent, []):
            if kw in clause:
                return intent
    return "Obligation"


def compare_with_rules(texts, labels, tfidf_preds):
    rule_preds = [rule_based_predict(t) for t in texts]

    rule_acc = accuracy_score(labels, rule_preds)
    model_acc = accuracy_score(labels, tfidf_preds)
    rule_f1 = f1_score(labels, rule_preds, average="macro", zero_division=0)
    model_f1 = f1_score(labels, tfidf_preds, average="macro", zero_division=0)

    print("\n[Compare] Rule-based vs TF-IDF+LR (tham khảo, train=test):")
    print(f"  {'Metric':<15} {'Rule-based':<12} {'TF-IDF+LR':<12}")
    print(f"  {'-'*15} {'-'*12} {'-'*12}")
    print(f"  {'Accuracy':<15} {rule_acc:<12.3f} {model_acc:<12.3f}")
    print(f"  {'F1 macro':<15} {rule_f1:<12.3f} {model_f1:<12.3f}")

    diffs = [
        (t, r, m, g)
        for t, r, m, g in zip(texts, rule_preds, tfidf_preds, labels)
        if r != m
    ]
    if diffs:
        print(f"\n  Clauses có kết quả khác nhau ({len(diffs)}):")
        for t, r, m, g in diffs[:5]:
            print(f"    Rule={r:25} | Model={m:25} | Gold={g}")
            print(f"    '{t[:70]}'")


def compare_with_phobert(texts, labels, tfidf_preds):
    from src.phobert_intent import PhoBERTIntentClassifier

    print("\n" + "=" * 80)
    print("[Compare] PhoBERT vs TF-IDF+LR (tham khảo, train=test)")
    print("=" * 80)

    phobert = PhoBERTIntentClassifier()

    if not phobert.is_available():
        print(f"\nPhoBERT model không tìm thấy tại {phobert.model_path}")
        print("Train model trên Colab và tải model về để so sánh.")
        return

    tfidf_acc = accuracy_score(labels, tfidf_preds)
    tfidf_f1_macro = f1_score(labels, tfidf_preds, average="macro", zero_division=0)
    tfidf_f1_weighted = f1_score(labels, tfidf_preds, average="weighted", zero_division=0)

    print("\n[1] TF-IDF+LR:")
    print(f"  Accuracy    : {tfidf_acc:.4f}")
    print(f"  F1 macro    : {tfidf_f1_macro:.4f}")
    print(f"  F1 weighted : {tfidf_f1_weighted:.4f}")

    print("\n[2] PhoBERT:")
    phobert_results = phobert.evaluate(texts, labels)

    print(f"  Accuracy    : {phobert_results['accuracy']:.4f}")
    print(f"  F1 macro    : {phobert_results['f1_macro']:.4f}")
    print(f"  F1 weighted : {phobert_results['f1_weighted']:.4f}")

    print("\n[3] So sánh (tham khảo, cả 2 predict trên training data):")
    print(f"  {'Metric':<15} {'TF-IDF+LR':<12} {'PhoBERT':<12} {'Diff':<12}")
    print(f"  {'-'*15} {'-'*12} {'-'*12} {'-'*12}")

    acc_diff = phobert_results["accuracy"] - tfidf_acc
    f1m_diff = phobert_results["f1_macro"] - tfidf_f1_macro
    f1w_diff = phobert_results["f1_weighted"] - tfidf_f1_weighted

    print(
        f"  {'Accuracy':<15} {tfidf_acc:<12.4f} {phobert_results['accuracy']:<12.4f} {acc_diff:+.4f}"
    )
    print(
        f"  {'F1 macro':<15} {tfidf_f1_macro:<12.4f} {phobert_results['f1_macro']:<12.4f} {f1m_diff:+.4f}"
    )
    print(
        f"  {'F1 weighted':<15} {tfidf_f1_weighted:<12.4f} {phobert_results['f1_weighted']:<12.4f} {f1w_diff:+.4f}"
    )

    phobert_preds = phobert_results["predictions"]
    disagreements = [
        (t, tf, pb, g)
        for t, tf, pb, g in zip(texts, tfidf_preds, phobert_preds, labels)
        if tf != pb
    ]

    if disagreements:
        print(f"\n[4] Disagreements giữa 2 models ({len(disagreements)} clauses):")
        for i, (text, tf_pred, pb_pred, gold) in enumerate(disagreements[:10]):
            correct_marker_tf = "OK" if tf_pred == gold else "X"
            correct_marker_pb = "OK" if pb_pred == gold else "X"
            print(f"\n  [{i+1}] TF-IDF={tf_pred:<25} {correct_marker_tf}")
            print(f"      PhoBERT={pb_pred:<25} {correct_marker_pb}")
            print(f"      Gold={gold}")
            print(f"      Text: '{text[:100]}...'")

    print("\n[5] Classification Report - PhoBERT:")
    print(phobert_results["report"])

    print("\n" + "=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--eval",
        action="store_true",
        help="Evaluate trên training data (tham khảo)",
    )
    parser.add_argument(
        "--compare", action="store_true", help="So sánh Rule-based vs TF-IDF+LR"
    )
    parser.add_argument(
        "--phobert",
        action="store_true",
        help="So sánh PhoBERT vs TF-IDF+LR (cần có model PhoBERT đã train)",
    )
    args = parser.parse_args()

    texts, labels = load_data()
    pipeline = train(texts, labels)

    tfidf_preds = None
    if args.eval or args.compare or args.phobert:
        tfidf_preds = evaluate(pipeline, texts, labels)

    if args.compare:
        compare_with_rules(texts, labels, tfidf_preds)

    if args.phobert:
        compare_with_phobert(texts, labels, tfidf_preds)
