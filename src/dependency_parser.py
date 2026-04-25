import stanza
import unicodedata

_nlp = None
_SUP2_PLACEHOLDER = "SSUP2SS"


def _get_nlp():
    global _nlp
    if _nlp is None:
        _nlp = stanza.Pipeline(
            lang="vi",
            processors="tokenize,pos,lemma,depparse",
            use_gpu=False,
        )
    return _nlp


def _preprocess(text):
    text = unicodedata.normalize("NFC", text)
    text = text.replace("²", _SUP2_PLACEHOLDER)
    return text


def _postprocess_token(tok):
    return tok.replace(_SUP2_PLACEHOLDER, "²")


def _fix_n_dot(deps):
    result = [dict(d) for d in deps]
    for i, tok in enumerate(result):
        if tok["token"] == "n." and i > 0:
            prev = result[i - 1]
            if not prev["token"].endswith("."):
                prev["token"] += "n"
                tok["token"] = "."
    return result


def _merge_ben_tokens(deps):
    merge_positions = [
        i
        for i in range(len(deps) - 1)
        if deps[i]["token"] == "Bên" and deps[i + 1]["token"] in ("A", "B")
    ]
    if not merge_positions:
        return deps

    result = [dict(d) for d in deps]

    for i in reversed(merge_positions):
        removed_1idx = i + 2
        result[i]["token"] = f"Bên {result[i + 1]['token']}"
        del result[i + 1]
        for tok in result:
            h = tok["head"]
            if h == 0:
                continue
            if h == removed_1idx:
                tok["head"] = removed_1idx - 1
            elif h > removed_1idx:
                tok["head"] -= 1

    return result


def parse_dependency(sentence):
    sentence = _preprocess(sentence)
    doc = _get_nlp()(sentence)

    results = []
    for sent in doc.sentences:
        for word in sent.words:
            results.append(
                {
                    "token": _postprocess_token(word.text),
                    "head": word.head,
                    "dep": word.deprel,
                }
            )

    results = _fix_n_dot(results)
    return _merge_ben_tokens(results)
