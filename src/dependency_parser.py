import stanza

nlp = stanza.Pipeline(
    lang="vi",
    processors="tokenize,pos,lemma,depparse",
    use_gpu=False
)


def parse_dependency(sentence):
    doc = nlp(sentence)

    results = []

    for sent in doc.sentences:
        for word in sent.words:
            results.append({
                "token": word.text,
                "head": word.head,        # index của head
                "dep": word.deprel        # quan hệ (root, nsubj, obj,...)
            })

    return results