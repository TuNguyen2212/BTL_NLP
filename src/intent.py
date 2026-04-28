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


_tfidf_model = None
_phobert_cls = None


def _get_tfidf_model():
    global _tfidf_model
    if _tfidf_model is None:
        if not os.path.exists(INTENT_MODEL_PATH):
            raise FileNotFoundError(
                f"TF-IDF model không tìm thấy tại {INTENT_MODEL_PATH}.\n"
                f"Hãy chạy: python train_intent.py"
            )
        _tfidf_model = joblib.load(INTENT_MODEL_PATH)
        print(f"[Intent] TF-IDF model loaded <- {INTENT_MODEL_PATH}")
    return _tfidf_model


def _get_phobert():
    global _phobert_cls
    if _phobert_cls is None:
        try:
            from src.phobert_intent import PhoBERTIntentClassifier

            _phobert_cls = PhoBERTIntentClassifier()
            if _phobert_cls.is_available():
                _phobert_cls.load()
            else:
                _phobert_cls = False
        except ImportError:
            _phobert_cls = False
    return _phobert_cls if _phobert_cls else None


def _rule_based_predict(clause: str) -> str:
    for intent in INTENT_LABELS:
        for kw in INTENT_KEYWORDS.get(intent, []):
            if kw in clause:
                return intent
    return "Obligation"


def predict_intent(clause: str, confidence_threshold: float = 0.5) -> dict:
    if not clause or not clause.strip():
        return {"intent": "Obligation", "confidence": 0.0, "source": "default"}

    candidates = []

    # TF-IDF prediction
    if os.path.exists(INTENT_MODEL_PATH):
        model = _get_tfidf_model()
        proba = model.predict_proba([clause])[0]
        classes = model.classes_
        max_idx = proba.argmax()
        candidates.append({
            "intent": classes[max_idx],
            "confidence": round(float(proba[max_idx]), 4),
            "source": "tfidf",
        })

    # PhoBERT prediction
    phobert = _get_phobert()
    if phobert:
        candidates.append(phobert.predict_single(clause))

    # Rule-based fallback
    candidates.append({
        "intent": _rule_based_predict(clause),
        "confidence": 0.0,
        "source": "rule",
    })

    if len(candidates) >= 2 and candidates[0]["source"] == "tfidf" and candidates[1]["source"] == "phobert":
        tfidf_pred = candidates[0]
        phobert_pred = candidates[1]
        if tfidf_pred["intent"] == phobert_pred["intent"]:
            # Both agree → ensemble
            return {
                "intent": tfidf_pred["intent"],
                "confidence": round(max(tfidf_pred["confidence"], phobert_pred["confidence"]), 4),
                "source": "ensemble",
            }
        else:
            # Disagree → pick higher conf
            best = max([tfidf_pred, phobert_pred], key=lambda x: x["confidence"])
            return best

    # Single model available
    for c in candidates:
        if c["confidence"] >= confidence_threshold:
            return c

    return candidates[-1]  # rule-based fallback


def run_intent(
    clauses_path: str = CLAUSES_PATH, output_path: str = INTENT_OUTPUT_PATH
) -> list[dict]:
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
    sources = Counter(r["source"] for r in results)
    avg_conf = sum(r["confidence"] for r in results) / len(results)

    print(f"[Intent] Processed {len(results)} clauses -> {output_path}")
    print(f"  Avg confidence : {avg_conf:.3f}")
    print(f"  Sources        : {dict(sources)}")
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
