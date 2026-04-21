"""Inference cho Intent Classification bằng TF-IDF + LR."""

import os
import sys
import json
import joblib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    CLAUSES_PATH,
    INTENT_OUTPUT_PATH,
    INTENT_MODEL_PATH,
    INTENT_LABELS,
    INTENT_KEYWORDS,
)


_model = None


def _get_model():
    global _model
    if _model is None:
        if not os.path.exists(INTENT_MODEL_PATH):
            raise FileNotFoundError(
                f"Model không tìm thấy tại {INTENT_MODEL_PATH}.\n"
                f"Hãy chạy: python train_intent.py"
            )
        _model = joblib.load(INTENT_MODEL_PATH)
        print(f"[Intent] Model loaded <- {INTENT_MODEL_PATH}")
    return _model


def _rule_based_predict(clause: str) -> str:
    for intent in INTENT_LABELS:
        for kw in INTENT_KEYWORDS.get(intent, []):
            if kw in clause:
                return intent
    return "Obligation"


def predict_intent(clause: str, confidence_threshold: float = 0.5) -> dict:
    """Dự đoán intent cho một clause."""
    if not clause or not clause.strip():
        return {"intent": "Obligation", "confidence": 0.0, "source": "default"}

    model = _get_model()

    proba = model.predict_proba([clause])[0]
    classes = model.classes_
    max_idx = proba.argmax()
    confidence = float(proba[max_idx])
    predicted = classes[max_idx]

    if confidence < confidence_threshold:
        rule_pred = _rule_based_predict(clause)
        return {
            "intent": rule_pred,
            "confidence": confidence,
            "source": "rule",
        }

    return {
        "intent": predicted,
        "confidence": round(confidence, 4),
        "source": "model",
    }


def run_intent(
    clauses_path: str = CLAUSES_PATH, output_path: str = INTENT_OUTPUT_PATH
) -> list[dict]:
    """Chạy dự đoán intent cho toàn bộ clause."""
    with open(clauses_path, encoding="utf-8") as f:
        clauses = [line.strip() for line in f if line.strip()]

    results = []
    txt_lines = []

    for i, clause in enumerate(clauses):
        pred = predict_intent(clause)
        results.append(
            {
                "id": i + 1,
                "clause": clause,
                "intent": pred["intent"],
                "confidence": pred["confidence"],
                "source": pred["source"],
            }
        )
        txt_lines.append(f"{clause}\t{pred['intent']}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(txt_lines))

    json_path = output_path.replace(".txt", "_detail.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    from collections import Counter

    counts = Counter(r["intent"] for r in results)
    rule_count = sum(1 for r in results if r["source"] == "rule")
    avg_conf = sum(r["confidence"] for r in results) / len(results)

    print(f"[Intent] Processed {len(results)} clauses -> {output_path}")
    print(f"  Avg confidence : {avg_conf:.3f}")
    print(f"  Rule fallback  : {rule_count}/{len(results)} clauses")
    print(f"  Distribution   :")
    for label in INTENT_LABELS:
        print(f"    {label:25}: {counts.get(label, 0)}")

    return results


def evaluate(annotated_path: str) -> dict:
    from sklearn.metrics import classification_report

    with open(annotated_path, encoding="utf-8") as f:
        gold_data = json.load(f)

    gold_labels, pred_labels = [], []
    for item in gold_data:
        pred = predict_intent(item["clause"])
        gold_labels.append(item["intent"])
        pred_labels.append(pred["intent"])

    report_str = classification_report(
        gold_labels,
        pred_labels,
        target_names=INTENT_LABELS,
        zero_division=0,
    )
    print("\n[Intent] Evaluation report:")
    print(report_str)

    return {"gold": gold_labels, "pred": pred_labels}


if __name__ == "__main__":
    import argparse
    from config import ANNOTATED_INTENT_PATH

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--eval", action="store_true", help="Evaluate against annotated_intent.json"
    )
    args = parser.parse_args()

    run_intent()

    if args.eval:
        evaluate(ANNOTATED_INTENT_PATH)
