"""Huấn luyện mô hình Intent bằng TF-IDF và Logistic Regression."""

import os
import sys
import json
import joblib
import argparse
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import classification_report
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


def evaluate_cv(texts, labels, n_splits: int = 5):
    """Đánh giá bằng Stratified K-Fold."""
    n_splits = min(n_splits, min(Counter(labels).values()))
    pipeline = build_pipeline()

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = cross_validate(
        pipeline,
        texts,
        labels,
        cv=cv,
        scoring=["accuracy", "f1_macro", "f1_weighted"],
        return_train_score=False,
    )

    print(f"\n[Eval] Stratified {n_splits}-Fold Cross-Validation:")
    print(
        f"  Accuracy    : {np.mean(scores['test_accuracy']):.3f} +/- {np.std(scores['test_accuracy']):.3f}"
    )
    print(
        f"  F1 macro    : {np.mean(scores['test_f1_macro']):.3f} +/- {np.std(scores['test_f1_macro']):.3f}"
    )
    print(
        f"  F1 weighted : {np.mean(scores['test_f1_weighted']):.3f} +/- {np.std(scores['test_f1_weighted']):.3f}"
    )

    return scores


def full_report(pipeline, texts, labels):
    preds = pipeline.predict(texts)
    print("\n[Eval] Classification report (train set - chỉ để tham khảo):")
    print(
        classification_report(
            labels, preds, target_names=INTENT_LABELS, zero_division=0
        )
    )


def rule_based_predict(clause: str) -> str:
    for intent in INTENT_LABELS:
        for kw in INTENT_KEYWORDS.get(intent, []):
            if kw in clause:
                return intent
    return "Obligation"  # default


def compare_with_rules(texts, labels, pipeline):
    rule_preds = [rule_based_predict(t) for t in texts]
    model_preds = pipeline.predict(texts)

    rule_correct = sum(r == g for r, g in zip(rule_preds, labels))
    model_correct = sum(m == g for m, g in zip(model_preds, labels))

    print("\n[Compare] Rule-based vs TF-IDF+LR (trên toàn bộ training set):")
    print(
        f"  Rule-based accuracy : {rule_correct}/{len(labels)} = {rule_correct/len(labels):.3f}"
    )
    print(
        f"  TF-IDF+LR accuracy  : {model_correct}/{len(labels)} = {model_correct/len(labels):.3f}"
    )

    diffs = [
        (t, r, m, g)
        for t, r, m, g in zip(texts, rule_preds, model_preds, labels)
        if r != m
    ]
    if diffs:
        print(f"\n  Clauses có kết quả khác nhau ({len(diffs)}):")
        for t, r, m, g in diffs[:5]:
            print(f"    Rule={r:25} | Model={m:25} | Gold={g}")
            print(f"    '{t[:70]}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--eval",
        action="store_true",
        help="Chạy cross-validation và in report sau khi train",
    )
    parser.add_argument(
        "--compare", action="store_true", help="So sánh Rule-based vs TF-IDF+LR"
    )
    args = parser.parse_args()

    texts, labels = load_data()
    pipeline = train(texts, labels)

    if args.eval:
        evaluate_cv(texts, labels)
        full_report(pipeline, texts, labels)

    if args.compare:
        compare_with_rules(texts, labels, pipeline)
