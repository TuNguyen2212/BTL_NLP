import json


def load_clauses(path):
    clauses = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if line:
                clauses.append({"id": i, "text": line})
    return clauses


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def merge_all(clauses_path, ner_path, srl_path, intent_path):
    clauses = load_clauses(clauses_path)
    ner = load_json(ner_path)
    srl = load_json(srl_path)
    intent = load_json(intent_path)

    ner_map = {x["id"]: x for x in ner}
    srl_map = {x["id"]: x for x in srl}
    intent_map = {x["id"]: x for x in intent}

    enriched = []

    contract_id = "lease"
    contract_name = "HĐ Thuê mặt bằng"

    for c in clauses:
        cid = c["id"]
        text_lower = c["text"].lower()

        if "giao kết hợp đồng lao động" in text_lower:
            contract_id = "labor"
            contract_name = "HĐ Lao động"

        enriched.append(
            {
                "clause_id": f"C{cid:03}",
                "text": c["text"],
                "contract_id": contract_id,
                "contract_name": contract_name,
                "entities": ner_map.get(cid, {}).get("entities", []),
                "srl": {
                    "predicate": srl_map.get(cid, {}).get("predicate"),
                    "roles": srl_map.get(cid, {}).get("roles", {}),
                    "negated": srl_map.get(cid, {}).get("negated", False),
                },
                "intent": intent_map.get(cid, {}).get("intent"),
                "intent_confidence": intent_map.get(cid, {}).get("confidence"),
            }
        )

    return enriched


def save_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
