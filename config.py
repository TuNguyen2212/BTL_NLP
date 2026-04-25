import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(BASE_DIR, "input")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

INPUT_PATH = os.path.join(INPUT_DIR, "raw_contracts.txt")
CHUNKS_PATH = os.path.join(OUTPUT_DIR, "chunks.txt")
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")

CLAUSES_PATH = os.path.join(OUTPUT_DIR, "clauses.txt")
DEPENDENCY_PATH = os.path.join(OUTPUT_DIR, "dependency.json")
NER_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "ner_results.json")
SRL_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "srl_results.json")
INTENT_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "intent_classification.txt")

ANNOTATED_NER_PATH = os.path.join(DATA_DIR, "annotated_ner.json")
ANNOTATED_INTENT_PATH = os.path.join(DATA_DIR, "annotated_intent.json")
INTENT_MODEL_PATH = os.path.join(MODEL_DIR, "intent_tfidf.pkl")
PHOBERT_MODEL_PATH = os.path.join(MODEL_DIR, "intent_phobert")

NER_LABELS = ["PARTY", "MONEY", "DATE", "RATE", "PENALTY", "LAW"]

NER_PATTERNS = {
    "PARTY": [
        r"một trong hai bên",
        r"cả hai bên",
        r"hai bên",
        r"mỗi bên",
        r"bên vi phạm",
        r"bên còn lại",
        r"bên kia",
        r"bên chấm dứt",
        r"Bên [AB]",
        r"Người lao động",
        r"Người sử dụng lao động",
        r"Công ty",
        r"Nhân viên",
    ],
    "MONEY": [
        r"\d[\d.,]*\s*(?:VNĐ|đồng|triệu|tỷ)(?:\s*/\s*tháng)?",
        r"tiền đặt cọc",
        r"tiền thuê nhà",
        r"tiền thuê",
        r"tiền lương",
        r"tiền phạt",
        r"tiền mặt",
        r"tiền điện",
        r"tiền nước",
        r"lương cơ bản",
        r"mức lương",
        r"khoản tiền",
        r"số tiền",
        r"phụ cấp[\w\s]{0,20}",
        r"thu nhập",
    ],
    "DATE": [
        r"trước ít nhất \d+[\s\(][\w\s]*\)?\s*(?:ngày|tháng|năm)",
        r"kể từ ngày[\w\s]+ký kết",
        r"ngày \d+\s+hàng tháng",
        r"ngày \d{1,2}/\d{1,2}/\d{4}",
        r"ngày \d+",
        r"hàng năm",
        r"\d+[\s\(][\w\s]*\)?\s*(?:ngày|tháng|năm)",
        r"hết thời hạn thuê",
        r"thời hạn thuê",
        r"trước thời hạn",
        r"khi ký hợp đồng",
        r"đúng thời gian đã thỏa thuận",
    ],
    "RATE": [
        r"\d+(?:[.,]\d+)?\s*%\s*/\s*(?:ngày|tháng|năm)",
        r"\d+(?:[.,]\d+)?\s*%",
    ],
    "PENALTY": [
        r"[Bb]ồi thường",
        r"khấu trừ",
        r"khoản phạt",
        r"tiền phạt",
        r"mức phạt",
        r"phạt chậm",
    ],
    "LAW": [
        r"Điều \d+(?:\.\d+)?",
        r"khoản \d+(?:\.\d+)?",
        r"Bộ luật\s+\w+(?:\s+(?!và\b|hoặc\b|của\b|theo\b|các\b|hiện\b|này\b)\w+){0,5}",
        r"(?<!ộ )(?<!pháp )Luật\s+\w+(?:\s+(?!và\b|hoặc\b|của\b|theo\b|các\b|hiện\b|này\b)\w+){0,5}",
        r"Nghị định(?:\s+số\s+[\d/\w-]+)?(?:\s+ngày\s+\d{1,2}/\d{1,2}/\d{4})?",
        r"Quyết định(?:\s+số\s+[\d/\w-]+)?",
        r"Thông tư(?:\s+số\s+[\d/\w-]+)?",
    ],
}

SRL_ROLES = ["Agent", "Predicate", "Theme", "Recipient", "Time", "Condition"]

SRL_DEP_TO_ROLE = {
    "nsubj": "Agent",
    "obj": "Theme",
    "obl:iobj": "Recipient",
    "obl": "Time",
    "advcl": "Condition",
}

CONDITION_MARKERS = ["Nếu", "nếu", "Khi", "khi", "Trừ", "trừ"]
CONDITION_END_MARKERS = ["thì", "Thì"]

INTENT_LABELS = ["Obligation", "Prohibition", "Right", "Termination Condition"]

INTENT_KEYWORDS = {
    "Obligation": ["phải", "có nghĩa vụ", "bắt buộc", "cần phải", "có trách nhiệm"],
    "Prohibition": [
        "không được",
        "cấm",
        "không có quyền",
        "không thể",
        "không được phép",
    ],
    "Right": ["có quyền", "được phép", "được quyền", "có thể"],
    "Termination Condition": [
        "chấm dứt",
        "hết thời hạn",
        "hết hiệu lực",
        "hủy hợp đồng",
        "kết thúc hợp đồng",
        "thanh lý hợp đồng",
    ],
}

PHOBERT_MODEL_NAME = "vinai/phobert-base"

INTENT_TFIDF_PARAMS = {
    "ngram_range": (1, 2),
    "max_features": 5000,
    "sublinear_tf": True,
}

INTENT_LR_PARAMS = {
    "max_iter": 1000,
    "C": 1.0,
    "class_weight": "balanced",
}
