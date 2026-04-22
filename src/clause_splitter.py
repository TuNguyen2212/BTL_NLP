import re

CONNECTORS = ["và", "hoặc", "nhưng", "nếu", "khi", "trong trường hợp"]


# Protect numeric dots (e.g., 20.000.000)
def protect_numbers(text):
    return re.sub(r'(\d)\.(\d)', r'\1<dot>\2', text)


# Restore numeric dots
def restore_numbers(text):
    return text.replace('<dot>', '.')


def normalize_text(text):
    # Xoá khoảng trắng thừa
    text = re.sub(r'\s+', ' ', text.strip())

    # Đảm bảo sau dấu chấm, ?, ! luôn có 1 dấu cách
    text = re.sub(r'([.!?])(?!\s)', r'\1 ', text)

    return text


def split_sentences(text):
    # Protect numbers BEFORE normalize & split
    text = protect_numbers(text)

    text = normalize_text(text)

    # Chỉ tách khi:
    # - Có dấu .!? 
    # - Sau đó là dấu cách
    sentences = re.split(r'(?<=[.!?])\s+', text)

    # Restore numbers AFTER split
    sentences = [restore_numbers(s) for s in sentences]

    # Loại bỏ câu rỗng
    return [s.strip() for s in sentences if s.strip()]


def split_clauses(sentence):
    # Protect numbers before clause split
    sentence = protect_numbers(sentence)

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

        # Restore numbers before append
        c = restore_numbers(c)

        results.append(c)

    return results