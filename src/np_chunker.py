from underthesea import word_tokenize, pos_tag


def np_chunk(sentence):
    if isinstance(sentence, list):
        sentence = " ".join(sentence)

    tokens = word_tokenize(sentence, format="text")

    pos_tags = pos_tag(tokens)

    result = []
    inside_np = False

    for word, pos in pos_tags:
        if pos.startswith("N"):
            if not inside_np:
                result.append((word, "B-NP"))
                inside_np = True
            else:
                result.append((word, "I-NP"))
        else:
            result.append((word, "O"))
            inside_np = False

    return result