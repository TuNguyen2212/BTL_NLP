import re

_CONTRACT_SEP = re.compile(r"\n*={5,}\n*")

_ARTICLE_HEADER = re.compile(r"^Điều\s+\d+\.\s+[^\d\n]")

_SUBCLAUSE_PREFIX = re.compile(r"^\d+(?:\.\d+)+\.?\s+")

_LETTER_PREFIX = re.compile(r"^\([a-z]\)\s+")

_PARTY_DETAIL_PATTERNS = [
    re.compile(r"^(Ông|Bà|Họ và tên)\s*:", re.IGNORECASE),
    re.compile(r"^Ngày sinh\s*:", re.IGNORECASE),
    re.compile(r"^Số (CCCD|CMT|CMND)\s*:", re.IGNORECASE),
    re.compile(r"^Địa chỉ (thường trú|trụ sở|:)", re.IGNORECASE),
    re.compile(r"^Số điện thoại\s*:", re.IGNORECASE),
    re.compile(r"^Mã số doanh nghiệp\s*:", re.IGNORECASE),
    re.compile(r"^Người đại diện (theo pháp luật)?\s*:", re.IGNORECASE),
    re.compile(r"^Trình độ chuyên môn\s*:", re.IGNORECASE),
    re.compile(r"^Chức vụ\s*:", re.IGNORECASE),
]

_ADMIN_HEADER_PATTERNS = [
    re.compile(r"^CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM", re.IGNORECASE),
    re.compile("^Độc lập\\s*[\u2013-]", re.IGNORECASE),
    re.compile("^[\u2014\u2013-]{2,}$"),
    re.compile(r"^={3,}$"),
    re.compile(r"^HỢP ĐỒNG\s+", re.IGNORECASE),
    re.compile(r"^Số\s*:\s*\d+", re.IGNORECASE),
    re.compile(r"^Hôm nay,?\s*ngày", re.IGNORECASE),
]

_PARTY_BLOCK_HEADER = re.compile(
    r"^(?:BÊN\s+(?:CHO THUÊ|THUÊ|A|B)|NGƯỜI\s+(?:SỞ DỤNG LAO ĐỘNG|LAO ĐỘNG))\s*[\(:]",
    re.IGNORECASE,
)

_SIGNATURE_MARKER = re.compile(r"^ĐẠI DIỆN BÊN", re.IGNORECASE)


def _strip_prefix(line: str) -> str:
    line = _SUBCLAUSE_PREFIX.sub("", line)
    line = _LETTER_PREFIX.sub("", line)
    return line.strip()


def _normalize_end_punct(text: str) -> str:
    text = text.strip()
    if text and text[-1] in (";", ":"):
        text = text[:-1] + "."
    return text


def _is_admin_header(line: str) -> bool:
    return any(p.match(line) for p in _ADMIN_HEADER_PATTERNS)


def _is_party_detail(line: str) -> bool:
    return any(p.match(line) for p in _PARTY_DETAIL_PATTERNS)


def _find_content_start(lines: list) -> int:
    for i, line in enumerate(lines):
        s = line.strip()
        if re.match(r"^Hai bên\s+thống nhất", s, re.IGNORECASE):
            return i
        if re.match(r"^Điều\s+1\.", s):
            return i
    return 0


def _find_content_end(lines: list) -> int:
    for i, line in enumerate(lines):
        if _SIGNATURE_MARKER.match(line.strip()):
            return i
    return len(lines)


def clean_contract_block(block: str) -> list:
    lines = block.splitlines()
    start = _find_content_start(lines)
    end = _find_content_end(lines)

    result = []
    for line in lines[start:end]:
        s = line.strip()

        if not s:
            continue

        if _ARTICLE_HEADER.match(s):
            continue
        if _PARTY_BLOCK_HEADER.match(s):
            continue
        if _is_party_detail(s):
            continue
        if _is_admin_header(s):
            continue

        cleaned = _strip_prefix(s)
        cleaned = _normalize_end_punct(cleaned)

        if cleaned:
            result.append(cleaned)

    return result


def clean_contracts(raw_text: str) -> str:
    raw_text = raw_text.replace("\r\n", "\n").replace("\r", "\n")
    blocks = _CONTRACT_SEP.split(raw_text)

    all_paragraphs = []
    for block in blocks:
        if block.strip():
            paragraphs = clean_contract_block(block)
            all_paragraphs.extend(paragraphs)

    return "\n".join(all_paragraphs)
