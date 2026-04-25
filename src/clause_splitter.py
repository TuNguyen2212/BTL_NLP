import re

CONNECTORS = ["và", "hoặc", "nhưng", "nếu", "khi", "trong trường hợp"]
_SPLIT_CONNECTORS = ["trong trường hợp", "không", "và", "hoặc", "nhưng", "nếu", "khi"]
_LEADING_COORD = re.compile(r"^(?:và|hoặc|nhưng)\s+", re.IGNORECASE)


def protect_numbers(text):
    return re.sub(r"(\d)\.(\d)", r"\1<dot>\2", text)


def restore_numbers(text):
    return text.replace("<dot>", ".")


def protect_decimal_comma(text):
    return re.sub(r"(\d),(\d)", r"\1<comma>\2", text)


def restore_decimal_comma(text):
    return text.replace("<comma>", ",")


def normalize_text(text):
    text = re.sub(r"\s+", " ", text.strip())
    text = re.sub(r"([.!?])(?!\s)", r"\1 ", text)
    return text


def split_sentences(text):
    text = protect_numbers(text)
    text = normalize_text(text)
    sentences = re.split(r"(?<=[.!?])\s+", text)
    sentences = [restore_numbers(s) for s in sentences]
    return [s.strip() for s in sentences if s.strip()]


def split_clauses(sentence):
    if not sentence or not sentence.strip():
        return []

    sentence = protect_numbers(sentence)
    sentence = protect_decimal_comma(sentence)

    pattern = r",\s*(?=\b(?:" + "|".join(_SPLIT_CONNECTORS) + r")\b)"
    clauses = re.split(pattern, sentence)

    results = []
    for c in clauses:
        c = c.strip()
        if not c:
            continue

        c = _LEADING_COORD.sub("", c).strip()
        if not c:
            continue

        if len(c.split()) < 3:
            continue

        if not re.search(r"[.!?]$", c):
            c += "."

        c = c[0].upper() + c[1:]
        c = restore_numbers(c)
        c = restore_decimal_comma(c)
        results.append(c)

    if not results:
        c = restore_numbers(sentence)
        c = restore_decimal_comma(c)
        if not re.search(r"[.!?]$", c):
            c += "."
        return [c[0].upper() + c[1:]]

    return results