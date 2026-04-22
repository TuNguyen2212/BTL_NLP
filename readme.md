# 📄 NLP Pipeline cho Hợp Đồng Pháp Lý Tiếng Việt

## 📌 Tổng quan

Project này xây dựng một pipeline NLP hoàn chỉnh cho xử lý hợp đồng pháp lý tiếng Việt, gồm 2 giai đoạn chính:

Giai đoạn 1: Tiền xử lý (Preprocessing)
- Tách câu và mệnh đề (Clause Splitting)
- Chunk danh từ (IOB tagging)
- Phân tích phụ thuộc (Dependency Parsing - Stanza)

Giai đoạn 2: Trích xuất & hiểu ngữ nghĩa
- Nhận diện thực thể (NER)
- Gán vai nghĩa (SRL)
- Phân loại ý định (Intent Classification)

---

## 🧱 Pipeline tổng thể

Raw Contracts
      ↓
[Preprocess - Bài 1]
  - Sentence Split
  - Clause Split
  - NP Chunking
  - Dependency Parsing
      ↓
Clauses + Dependency
      ↓
[Extraction - Bài 2]
  - NER
  - SRL
  - Intent Classification
      ↓
Structured Legal Information

---

## 📁 Cấu trúc thư mục

project/
├── input/
│   └── raw_contracts.txt
├── data/
│   ├── annotated_ner.json
│   └── annotated_intent.json
├── models/
│   ├── intent_tfidf.pkl
│   └── intent_phobert/
├── output/
│   ├── clauses.txt
│   ├── chunks.txt
│   ├── dependency.json
│   ├── ner_results.json
│   ├── srl_results.json
│   ├── intent_classification.txt
│   └── intent_classification_detail.json
├── src/
│   ├── clause_splitter.py
│   ├── np_chunker.py
│   ├── dependency_parser.py
│   ├── ner.py
│   ├── srl.py
│   ├── intent.py
│   ├── intent_model.py
│   └── utils.py
├── preprocess.py
├── extract.py
├── train_intent.py
├── setup_stanza.py
├── config.py
└── requirements.txt

---

## ⚙️ Cài đặt

pip install -r requirements.txt
python setup_stanza.py

---

## ▶️ Cách chạy pipeline

1. Chạy tiền xử lý (Bài 1)
python preprocess.py

Output:
- clauses.txt
- chunks.txt
- dependency.json

2. Train model Intent (nếu chưa có)
python train_intent.py

3. Chạy trích xuất thông tin (Bài 2)
python extract.py

Hoặc:
python extract.py --eval

4. Chạy từng module
python extract.py --task ner
python extract.py --task srl
python extract.py --task intent

5. So sánh TF-IDF vs PhoBERT
python src/intent_model.py --eval

---

## 📥 Input

File:
input/raw_contracts.txt

Ví dụ:
Bên B sẽ thanh toán toàn bộ tiền thuê trước ngày 5 hàng tháng, và nếu thanh toán trễ hạn, mức phạt 1% mỗi ngày sẽ được áp dụng.

---

## 📤 Output

Giai đoạn 1:

clauses.txt
Bên B sẽ thanh toán toàn bộ tiền thuê trước ngày 5 hàng tháng.
Nếu thanh toán trễ hạn, mức phạt 1% mỗi ngày sẽ được áp dụng.

chunks.txt (IOB)
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

dependency.json
[
  {
    "clause": "Bên B sẽ thanh toán toàn bộ tiền thuê.",
    "dependencies": [
      {"token": "Bên", "head": 3, "dep": "nsubj"},
      {"token": "thanh", "head": 0, "dep": "root"},
      {"token": "tiền thuê", "head": 4, "dep": "obj"}
    ]
  }
]

---

Giai đoạn 2:

ner_results.json
- PARTY, MONEY, DATE, RATE...

srl_results.json
- Agent, Theme, Time, Condition...

intent_classification.txt
clause<TAB>intent

intent_classification_detail.json
- confidence + model source

---

## 🧠 Methodology

Clause Splitting
- Rule-based (regex + liên từ)
- Có xử lý số tiền (20.000.000) để tránh tách sai

NP Chunking
- POS tagging bằng Underthesea
- IOB: B-NP, I-NP, O

Dependency Parsing
- Stanza
- token, head, dep

NER
- Rule-based + regex pháp lý
- PARTY, MONEY, DATE, RATE...

SRL
- Heuristic từ dependency + NER

Intent Classification
- TF-IDF + Logistic Regression
- Labels:
  - Obligation
  - Prohibition
  - Right
  - Termination Condition
- Có PhoBERT để so sánh

---

## ⚠️ Lưu ý

- Phải chạy setup_stanza.py trước
- Model Stanza ~500MB
- Bài 2 phụ thuộc output Bài 1
- Đã fix lỗi số tiền:
  - 20.000.000
  - 1.000.000.000

---

## 🎯 Hướng phát triển

- PhoBERT cho NER
- SRL deep learning
- Legal-BERT tiếng Việt
- Knowledge graph hợp đồng