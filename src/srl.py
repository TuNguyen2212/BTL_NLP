"""Semantic Role Labeling theo hướng hybrid."""

import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    CLAUSES_PATH,
    DEPENDENCY_PATH,
    NER_OUTPUT_PATH,
    SRL_OUTPUT_PATH,
    CONDITION_MARKERS,
    CONDITION_END_MARKERS,
)


def _resolve_to_entity(token: str, clause: str, entities: list[dict]) -> str:
    """Mở rộng token dependency thành span entity gần nhất nếu có."""
    token_pos = clause.find(token)
    if token_pos == -1:
        return token

    best = None
    best_dist = float("inf")
    for ent in entities:
        dist = abs(ent["start"] - token_pos)
        if dist < best_dist and ent["text"].startswith(token.strip()):
            best_dist = dist
            best = ent["text"]

    return best if best else token


def _find_party_entities(entities: list[dict]) -> list[str]:
    return [e["text"] for e in entities if e["label"] == "PARTY"]


def _find_date_entities(entities: list[dict]) -> list[str]:
    return [e["text"] for e in entities if e["label"] == "DATE"]


def _find_money_entities(entities: list[dict]) -> list[str]:
    return [e["text"] for e in entities if e["label"] == "MONEY"]


def _extract_condition(clause: str) -> str | None:
    """Lấy phần điều kiện đứng trước từ khóa `thì` nếu có."""
    for marker in CONDITION_MARKERS:
        if clause.startswith(marker) or f" {marker} " in clause:
            for end_marker in CONDITION_END_MARKERS:
                idx = clause.find(end_marker)
                if idx != -1:
                    condition = clause[:idx].strip()
                    for m in CONDITION_MARKERS:
                        if condition.startswith(m):
                            condition = condition[len(m) :].strip()
                    return condition if condition else None
    return None


def _is_negated(deps: list[dict]) -> bool:
    return any(d["dep"] in ("advmod:neg", "neg") for d in deps)


def extract_roles(clause: str, deps: list[dict], entities: list[dict]) -> dict:
    """Suy ra predicate và các vai nghĩa chính cho một clause."""
    roles = {
        role: None for role in ["Agent", "Theme", "Recipient", "Time", "Condition"]
    }

    root_token = next((d["token"] for d in deps if d["dep"] == "root"), None)
    negated = _is_negated(deps)
    nsubj_tokens = [d["token"] for d in deps if d["dep"] == "nsubj"]
    party_entities = _find_party_entities(entities)

    if nsubj_tokens and party_entities:
        resolved = _resolve_to_entity(nsubj_tokens[0], clause, entities)
        roles["Agent"] = resolved if resolved in party_entities else party_entities[0]
    elif party_entities:
        roles["Agent"] = party_entities[0]

    obj_tokens = [d["token"] for d in deps if d["dep"] == "obj"]
    money_entities = _find_money_entities(entities)

    if money_entities:
        roles["Theme"] = money_entities[0]
    elif obj_tokens:
        roles["Theme"] = _resolve_to_entity(obj_tokens[0], clause, entities)

    obl_iobj = [d["token"] for d in deps if d["dep"] == "obl:iobj"]
    if obl_iobj and len(party_entities) >= 2:
        agent = roles["Agent"]
        for p in party_entities:
            if p != agent:
                roles["Recipient"] = p
                break
    elif obl_iobj:
        roles["Recipient"] = _resolve_to_entity(obl_iobj[0], clause, entities)

    date_entities = _find_date_entities(entities)
    if date_entities:
        roles["Time"] = date_entities[0]
    else:
        tmod = next((d["token"] for d in deps if d["dep"] == "obl:tmod"), None)
        if tmod:
            roles["Time"] = tmod

    roles["Condition"] = _extract_condition(clause)
    roles = {k: v for k, v in roles.items() if v is not None}

    return {
        "predicate": root_token,
        "negated": negated,
        "roles": roles,
    }


def run_srl(
    clauses_path: str = CLAUSES_PATH,
    dependency_path: str = DEPENDENCY_PATH,
    ner_path: str = NER_OUTPUT_PATH,
    output_path: str = SRL_OUTPUT_PATH,
) -> list[dict]:
    """Chạy SRL cho toàn bộ clause và ghi ra file JSON."""
    with open(clauses_path, encoding="utf-8") as f:
        clauses = [line.strip() for line in f if line.strip()]

    with open(dependency_path, encoding="utf-8") as f:
        dep_data = json.load(f)

    with open(ner_path, encoding="utf-8") as f:
        ner_data = json.load(f)

    dep_map = {d["clause"]: d["dependencies"] for d in dep_data}
    ner_map = {n["clause"]: n["entities"] for n in ner_data}

    results = []
    for i, clause in enumerate(clauses):
        deps = dep_map.get(clause, [])
        entities = ner_map.get(clause, [])

        srl = extract_roles(clause, deps, entities)
        results.append(
            {
                "id": i + 1,
                "clause": clause,
                "predicate": srl["predicate"],
                "negated": srl["negated"],
                "roles": srl["roles"],
            }
        )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    role_counts = {}
    for r in results:
        for role in r["roles"]:
            role_counts[role] = role_counts.get(role, 0) + 1

    negated_count = sum(1 for r in results if r["negated"])

    print(f"[SRL] Processed {len(results)} clauses -> {output_path}")
    print(f"  Negated clauses : {negated_count}")
    print(f"  Role coverage   :")
    for role, cnt in sorted(role_counts.items(), key=lambda x: -x[1]):
        pct = cnt / len(results) * 100
        print(f"    {role:12}: {cnt:3} clauses ({pct:.0f}%)")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--show", type=int, default=5, help="In N kết quả mẫu sau khi chạy"
    )
    args = parser.parse_args()

    results = run_srl()

    print(f"\n[SRL] {args.show} kết quả mẫu:")
    samples = [r for r in results if len(r["roles"]) >= 2][: args.show]
    for r in samples:
        neg_str = " [NEGATED]" if r["negated"] else ""
        print(f"\n  [{r['id']:02d}] {r['clause'][:70]}")
        print(f"       Predicate : {r['predicate']}{neg_str}")
        for role, val in r["roles"].items():
            print(f"       {role:12}: {val}")
