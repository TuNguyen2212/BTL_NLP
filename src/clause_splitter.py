import re

CONNECTORS = ["và", "hoặc", "nhưng", "nếu", "khi", "trong trường hợp"]


def split_sentences(text):
    text = re.sub(r'\s+', ' ', text.strip())
    return re.split(r'[.!?]\s*', text)


def split_clauses(sentence):
    pattern = r',\s*|(?=\b(?:' + '|'.join(CONNECTORS) + r')\b)'
    clauses = re.split(pattern, sentence)

    results = []
    for c in clauses:
        c = c.strip()
        if not c:
            continue
        if not c.endswith("."):
            c += "."
        c = c[0].upper() + c[1:]
        results.append(c)

    return results