import re
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import NER_PATTERNS, NER_LABELS, CLAUSES_PATH, NER_OUTPUT_PATH


def _find_all_matches(clause: str) -> list[dict]:
    candidates = []
    for label in NER_LABELS:
        patterns = NER_PATTERNS.get(label, [])
        for pat in patterns:
            for m in re.finditer(pat, clause, flags=re.IGNORECASE):
                candidates.append(
                    {
                        "text": m.group(),
                        "label": label,
                        "start": m.start(),
                        "end": m.end(),
                    }
                )
    return candidates


def _resolve_overlaps(candidates: list[dict]) -> list[dict]:
    label_priority = {
        "PARTY": 5,
        "MONEY": 4,
        "DATE": 3,
        "LAW": 3,
        "RATE": 2,
        "PENALTY": 1,
    }

    sorted_cands = sorted(
        candidates,
        key=lambda x: (
            -label_priority.get(x["label"], 0),
            -(x["end"] - x["start"]),
            x["start"],
        ),
    )

    selected = []
    for cand in sorted_cands:
        overlap = False
        for sel in selected:
            if cand["start"] < sel["end"] and sel["start"] < cand["end"]:
                overlap = True
                break
        if not overlap:
            selected.append(cand)

    selected.sort(key=lambda x: x["start"])
    return selected


def _is_valid_entity(entity: dict, clause: str) -> bool:
    text = entity["text"]
    label = entity["label"]

    if label == "MONEY":
        if (
            text in ["phụ cấp"]
            and ":" not in clause[entity["start"] : entity["end"] + 20]
        ):
            return False

    return True


def extract_entities(clause: str) -> list[dict]:
    if not clause or not clause.strip():
        return []
    candidates = _find_all_matches(clause)
    resolved = _resolve_overlaps(candidates)

    valid_entities = [e for e in resolved if _is_valid_entity(e, clause)]
    return valid_entities


def run_ner(
    clauses_path: str = CLAUSES_PATH, output_path: str = NER_OUTPUT_PATH
) -> list[dict]:
    with open(clauses_path, encoding="utf-8") as f:
        clauses = [line.strip() for line in f if line.strip()]

    results = []
    for i, clause in enumerate(clauses):
        entities = extract_entities(clause)
        results.append(
            {
                "id": i + 1,
                "clause": clause,
                "entities": entities,
            }
        )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"[NER] Processed {len(results)} clauses -> {output_path}")
    return results


def evaluate(annotated_path: str) -> dict:
    with open(annotated_path, encoding="utf-8") as f:
        gold_data = json.load(f)

    label_stats = {label: {"tp": 0, "fp": 0, "fn": 0} for label in NER_LABELS}

    for item in gold_data:
        clause = item["clause"]
        gold_entities = item.get("entities", [])
        pred_entities = extract_entities(clause)

        gold_set = {(e["start"], e["end"], e["label"]) for e in gold_entities}
        pred_set = {(e["start"], e["end"], e["label"]) for e in pred_entities}

        for label in NER_LABELS:
            gold_l = {k for k in gold_set if k[2] == label}
            pred_l = {k for k in pred_set if k[2] == label}
            label_stats[label]["tp"] += len(gold_l & pred_l)
            label_stats[label]["fp"] += len(pred_l - gold_l)
            label_stats[label]["fn"] += len(gold_l - pred_l)

    report = {}
    macro_p = macro_r = macro_f1 = 0
    active_labels = 0

    for label, s in label_stats.items():
        tp, fp, fn = s["tp"], s["fp"], s["fn"]
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        report[label] = {
            "precision": round(p, 3),
            "recall": round(r, 3),
            "f1": round(f1, 3),
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }
        if tp + fp + fn > 0:
            macro_p += p
            macro_r += r
            macro_f1 += f1
            active_labels += 1

    if active_labels > 0:
        report["macro"] = {
            "precision": round(macro_p / active_labels, 3),
            "recall": round(macro_r / active_labels, 3),
            "f1": round(macro_f1 / active_labels, 3),
        }

    return report


if __name__ == "__main__":
    import argparse
    from config import ANNOTATED_NER_PATH

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--eval",
        action="store_true",
        help="Evaluate against annotated_ner.json after running",
    )
    args = parser.parse_args()

    run_ner()

    if args.eval:
        print("\n[NER] Evaluation:")
        report = evaluate(ANNOTATED_NER_PATH)
        for label, scores in report.items():
            if label == "macro":
                print(
                    f"\n  {'MACRO':10} | P={scores['precision']:.3f}  R={scores['recall']:.3f}  F1={scores['f1']:.3f}"
                )
            else:
                print(
                    f"  {label:10} | P={scores['precision']:.3f}  R={scores['recall']:.3f}  F1={scores['f1']:.3f}"
                    f"  (tp={scores['tp']} fp={scores['fp']} fn={scores['fn']})"
                )
