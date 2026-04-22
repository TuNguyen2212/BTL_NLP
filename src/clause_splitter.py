import re

CONNECTORS = ["và", "hoặc", "nhưng", "nếu", "khi", "trong trường hợp"]


def normalize_text(text):
    # Xoá khoảng trắng thừa
    text = re.sub(r'\s+', ' ', text.strip())

    # Đảm bảo sau dấu chấm, ?, ! luôn có 1 dấu cách
    text = re.sub(r'([.!?])(?!\s)', r'\1 ', text)

    return text


def split_sentences(text):
    text = normalize_text(text)

    # Chỉ tách khi:
    # - Có dấu .!? 
    # - Sau đó là dấu cách
    sentences = re.split(r'(?<=[.!?])\s+', text)

    # Loại bỏ câu rỗng
    return [s.strip() for s in sentences if s.strip()]


def split_clauses(sentence):
    pattern = r',\s*|(?=\b(?:' + '|'.join(CONNECTORS) + r')\b)'
    clauses = re.split(pattern, sentence)

    results = []
    for c in clauses:
        c = c.strip()
        if not c:
            continue

        # Đảm bảo kết thúc bằng dấu chấm
        if not re.search(r'[.!?]$', c):
            c += '.'

        # Viết hoa chữ cái đầu
        c = c[0].upper() + c[1:]

        results.append(c)

    return results