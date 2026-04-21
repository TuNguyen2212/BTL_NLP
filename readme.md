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

---

# Bài 2 - Trích xuất thông tin và Phân tích ngữ nghĩa

## Tổng quan

Bài 2 sử dụng đầu ra của Bài 1 để thực hiện 3 tác vụ:

1. Nhận diện thực thể (Named Entity Recognition - NER)
2. Gán vai nghĩa ngữ nghĩa (Semantic Role Labeling - SRL)
3. Phân loại ý định (Intent Classification)

Pipeline hiện tại gồm:

1. NER rule-based
2. SRL hybrid dựa trên dependency parse và NER
3. Intent baseline bằng TF-IDF kết hợp Logistic Regression

Ngoài ra, nhóm có thêm nhánh PhoBERT cho Intent Classification để phục vụ so sánh với baseline.

---

## Cấu trúc thư mục

```text
project/
├── data/
│   ├── annotated_ner.json
│   └── annotated_intent.json
├── models/
│   ├── intent_tfidf.pkl
│   └── intent_phobert/
├── output/
│   ├── clauses.txt
│   ├── dependency.json
│   ├── ner_results.json
│   ├── srl_results.json
│   ├── intent_classification.txt
│   └── intent_classification_detail.json
├── src/
│   ├── ner.py
│   ├── srl.py
│   ├── intent.py
│   └── intent_model.py
├── config.py
├── train_intent.py
└── extract.py
```

---

## Cài đặt

pip install -r requirements.txt

---

## Cách chạy

### 1. Train baseline cho Intent

python train_intent.py

---

### 2. Chạy toàn bộ pipeline Bài 2

python extract.py

---

### 3. Chạy kèm evaluation

python extract.py --eval

---

### 4. Chạy từng tác vụ riêng

python extract.py --task ner
python extract.py --task srl
python extract.py --task intent

---

### 5. So sánh baseline với PhoBERT cho Intent

python src/intent_model.py --eval

---

## Input

Bài 2 sử dụng đầu ra từ Bài 1:

- output/clauses.txt
- output/dependency.json

Ngoài ra còn dùng dữ liệu gán nhãn để đánh giá:

- data/annotated_ner.json
- data/annotated_intent.json

---

## Output

### 1. ner_results.json

Lưu danh sách thực thể được trích xuất từ từng clause.

---

### 2. srl_results.json

Lưu predicate, negation và các vai nghĩa chính như:

- Agent
- Theme
- Recipient
- Time
- Condition

---

### 3. intent_classification.txt

Lưu nhãn intent cho từng clause theo định dạng:

clause<TAB>intent

---

### 4. intent_classification_detail.json

Lưu thêm confidence và nguồn dự đoán của baseline Intent.

---

## Methodology

### 1. NER

- Rule-based matching bằng regex và pattern từ miền hợp đồng pháp lý
- Các nhãn chính:
  - PARTY
  - MONEY
  - DATE
  - RATE
  - PENALTY
  - LAW

### 2. SRL

- Dựa trên dependency parse từ Bài 1
- Kết hợp với NER để mở rộng span thực thể
- Suy ra các vai nghĩa chính bằng heuristic

### 3. Intent Classification

- Baseline: TF-IDF + Logistic Regression
- Các nhãn intent:
  - Obligation
  - Prohibition
  - Right
  - Termination Condition
- Có thêm nhánh PhoBERT để so sánh với baseline

---

## Lưu ý

- Bài 2 chỉ chạy đúng khi Bài 1 đã sinh ra clauses.txt và dependency.json
- Nếu chưa có models/intent_tfidf.pkl, pipeline sẽ tự train baseline khi chạy task intent