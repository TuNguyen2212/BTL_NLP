# 📄 Bài 1 - Pipeline NLP cho Hợp Đồng Pháp Lý

## 📌 Tổng quan

Đây là một phần trong bài tập lớn gồm 3 module NLP cho xử lý hợp đồng pháp lý tiếng Việt.

Module này tập trung vào việc tiền xử lý văn bản, gồm 3 bước:

1. Tách mệnh đề (Clause Splitting)
2. Chunk danh từ (IOB tagging)
3. Phân tích phụ thuộc (Dependency Parsing bằng Stanza)

---

## 📁 Cấu trúc thư mục

```text
project/
├── input/
│   └── raw_contracts.txt
├── output/
│   ├── clauses.txt
│   ├── chunks.txt
│   └── dependency.json
├── src/
│   ├── clause_splitter.py
│   ├── np_chunker.py
│   ├── dependency_parser.py
│   └── utils.py
├── preprocess.py
└── requirements.txt
```
---

## ⚙️ Cài đặt

pip install -r requirements.txt
python setup_stanza.py

---

## ▶️ Cách chạy

python preprocess.py

---

## 📥 Input

File: input/raw_contracts.txt

Ví dụ:
Bên B sẽ thanh toán toàn bộ tiền thuê trước ngày 5 hàng tháng, và nếu thanh toán trễ hạn, mức phạt 1% mỗi ngày sẽ được áp dụng.

---

## 📤 Output

### 1. clauses.txt

Bên B sẽ thanh toán toàn bộ tiền thuê trước ngày 5 hàng tháng.
Nếu thanh toán trễ hạn, mức phạt 1% mỗi ngày sẽ được áp dụng.

---

### 2. chunks.txt (IOB format)

Bên    B-NP
B      I-NP
sẽ     O
thanh  O
toán   O
toàn   B-NP
bộ     I-NP
tiền   I-NP
thuê   I-NP
.      O

---

### 3. dependency.json

[
  {
    "clause": "Bên B sẽ thanh toán toàn bộ tiền thuê.",
    "dependencies": [
      {
        "token": "Bên",
        "head": 3,
        "dep": "nsubj"
      },
      {
        "token": "thanh",
        "head": 0,
        "dep": "root"
      },
      {
        "token": "tiền thuê",
        "head": 4,
        "dep": "obj"
      }
    ]
  }
]

---

## 🧠 Methodology

### 1. Clause Splitting
- Rule-based tách câu theo dấu câu và liên từ:
  - "và", "nếu", "khi", ...

### 2. Chunking
- POS tagging bằng Underthesea
- Gán nhãn IOB:
  - B-NP: bắt đầu cụm danh từ
  - I-NP: bên trong cụm danh từ
  - O: ngoài cụm danh từ

### 3. Dependency Parsing
- Sử dụng thư viện Stanza (Stanford NLP)
- Trích xuất:
  - token
  - head
  - dep (root, nsubj, obj, advcl,...)

---

## ⚠️ Lưu ý

- Phải chạy `setup_stanza.py` trước khi chạy `preprocess.py`
- Model tải lần đầu khá nặng (~500MB)