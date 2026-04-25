from underthesea import pos_tag

_FORCED_O = {
    "khi",
    "trước khi",
    "sau khi",
    "nếu",
    "nếu như",
    "mà",
    "thì",
    "rằng",
}


def np_chunk(sentence):
    if isinstance(sentence, list):
        sentence = " ".join(sentence)

    pos_tags = pos_tag(sentence)

    result = []
    inside_np = False

    for word, pos in pos_tags:
        if pos.startswith("N") and word.lower() not in _FORCED_O:
            if not inside_np:
                result.append((word, "B-NP"))
                inside_np = True
            else:
                result.append((word, "I-NP"))
        else:
            result.append((word, "O"))
            inside_np = False

    return result
