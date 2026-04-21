import os
import sys
import json
import argparse
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    CLAUSES_PATH,
    DEPENDENCY_PATH,
    NER_OUTPUT_PATH,
    SRL_OUTPUT_PATH,
    INTENT_OUTPUT_PATH,
    ANNOTATED_NER_PATH,
    ANNOTATED_INTENT_PATH,
    INTENT_MODEL_PATH,
)

def preflight_check():
    print("=" * 60)
    print(" INPUT CHECK")
    print("=" * 60)

    required = {
        "clauses.txt (A1 output)": CLAUSES_PATH,
        "dependency.json (A1 output)": DEPENDENCY_PATH,
    }

    all_ok = True
    for name, path in required.items():
        ok = os.path.exists(path)
        print(f"  {'[OK]' if ok else '[!!]'}  {name}")
        if not ok:
            all_ok = False

    model_ok = os.path.exists(INTENT_MODEL_PATH)
    hint = "" if model_ok else " <- will be trained automatically before intent inference"
    print(f"  {'[OK]' if model_ok else '[--]'}  Intent TF-IDF model{hint}")

    print()
    if not all_ok:
        print("[ERROR] Missing required inputs. Please run Assignment 1 first.")
        sys.exit(1)


def run_task_ner(eval_mode: bool = False):
    print("-" * 60)
    print(" TASK 2.1 - NER")
    print("-" * 60)

    t0 = time.time()
    from src.ner import run_ner

    results = run_ner()

    elapsed = time.time() - t0
    print(f"  Completed in {elapsed:.2f}s")

    if eval_mode and os.path.exists(ANNOTATED_NER_PATH):
        print()
        from src.ner import evaluate

        report = evaluate(ANNOTATED_NER_PATH)
        print("  NER evaluation:")
        for label, scores in report.items():
            if label == "macro":
                print(
                    f"    {'MACRO':10} | P={scores['precision']:.3f}  "
                    f"R={scores['recall']:.3f}  F1={scores['f1']:.3f}"
                )
            elif scores["tp"] + scores["fp"] + scores["fn"] > 0:
                print(
                    f"    {label:10} | P={scores['precision']:.3f}  "
                    f"R={scores['recall']:.3f}  F1={scores['f1']:.3f}"
                )
    print()
    return results


def run_task_intent(eval_mode: bool = False):
    print("-" * 60)
    print(" TASK 2.3 - INTENT CLASSIFICATION")
    print("-" * 60)

    t0 = time.time()

    if not os.path.exists(INTENT_MODEL_PATH):
        print("  TF-IDF model not found, training one before inference...")
        from train_intent import load_data, train

        texts, labels = load_data()
        train(texts, labels)
        print()

    from src.intent import run_intent

    results = run_intent()

    elapsed = time.time() - t0
    print(f"  Completed in {elapsed:.2f}s")

    if eval_mode and os.path.exists(ANNOTATED_INTENT_PATH):
        print()
        from src.intent import evaluate

        evaluate(ANNOTATED_INTENT_PATH)

    print()
    return results


def run_task_srl(eval_mode: bool = False):
    print("-" * 60)
    print(" TASK 2.2 - SEMANTIC ROLE LABELING")
    print("-" * 60)

    if not os.path.exists(NER_OUTPUT_PATH):
        print("  NER output not found, running NER first...")
        run_task_ner()

    from src.srl import run_srl

    t0 = time.time()
    results = run_srl()
    elapsed = time.time() - t0

    print(f"  Completed in {elapsed:.2f}s")
    print()
    return results


def print_summary():
    print("=" * 60)
    print(" PIPELINE SUMMARY")
    print("=" * 60)

    outputs = {
        "ner_results.json": NER_OUTPUT_PATH,
        "intent_classification.txt": INTENT_OUTPUT_PATH,
        "srl_results.json": SRL_OUTPUT_PATH,
    }
    for name, path in outputs.items():
        if os.path.exists(path):
            size = os.path.getsize(path)
            print(f"  [OK]  {name:35} ({size/1024:.1f} KB)")
        else:
            print(f"  [--]  {name:35} (not generated)")

    if os.path.exists(NER_OUTPUT_PATH):
        with open(NER_OUTPUT_PATH, encoding="utf-8") as f:
            ner_data = json.load(f)
        total_ents = sum(len(r["entities"]) for r in ner_data)
        print(f"\n  NER   : {len(ner_data)} clauses, {total_ents} entities")

    if os.path.exists(SRL_OUTPUT_PATH):
        with open(SRL_OUTPUT_PATH, encoding="utf-8") as f:
            srl_data = json.load(f)
        has_agent = sum(1 for r in srl_data if "Agent" in r["roles"])
        negated = sum(1 for r in srl_data if r["negated"])
        print(f"  SRL   : {has_agent} with Agent, {negated} negated")

    if os.path.exists(INTENT_OUTPUT_PATH):
        from collections import Counter

        with open(INTENT_OUTPUT_PATH, encoding="utf-8") as f:
            lines = f.read().splitlines()
        counts = Counter(l.split("\t")[1] for l in lines if "\t" in l)
        print(f"  Intent: " + "  ".join(f"{k}={v}" for k, v in counts.most_common()))

    print()


def main():
    parser = argparse.ArgumentParser(
        description="Assignment 2 - NLP pipeline for legal contracts"
    )
    parser.add_argument(
        "--task",
        choices=["ner", "intent", "srl", "all"],
        default="all",
        help="Task to run (default: all)",
    )
    parser.add_argument(
        "--eval", action="store_true", help="Run evaluation after each task"
    )
    args = parser.parse_args()

    preflight_check()

    t_total = time.time()

    if args.task in ("ner", "all"):
        run_task_ner(eval_mode=args.eval)

    if args.task in ("srl", "all"):
        run_task_srl(eval_mode=args.eval)

    if args.task in ("intent", "all"):
        run_task_intent(eval_mode=args.eval)

    total_elapsed = time.time() - t_total
    print_summary()
    print(f"  Total time: {total_elapsed:.2f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
