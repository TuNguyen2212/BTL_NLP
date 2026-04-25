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
  - Contract Cleaner
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
```text
project/
├── input/                                    # Văn bản hợp đồng thô
│   └── raw_contracts.txt
├── data/                                     # Dữ liệu annotated
│   ├── annotated_ner.json
│   └── annotated_intent.json
├── models/                                   # Models đã train
│   ├── intent_tfidf.pkl
│   └── intent_phobert/                       # PhoBERT (train trên Colab)
├── output/                                   # Kết quả pipeline
│   ├── clauses.txt
│   ├── chunks.txt
│   ├── dependency.json
│   ├── ner_results.json
│   ├── srl_results.json
│   ├── intent_classification.txt
│   └── intent_classification_detail.json
├── src/
│   ├── clause_splitter.py                    # Bài 1.1
│   ├── np_chunker.py                         # Bài 1.2
│   ├── dependency_parser.py                  # Bài 1.3
│   ├── contract_cleaner.py                   # Tiền xử lý văn bản
│   ├── ner.py                                # Bài 2.1
│   ├── srl.py                                # Bài 2.2
│   ├── intent.py                             # Bài 2.3 — inference
│   ├── phobert_intent.py                     # PhoBERT classifier (inference)
│   └── utils.py
├── preprocess.py                   # Chạy Bài 1
├── extract.py                      # Chạy Bài 2
├── train_intent.py                 # Train TF-IDF + so sánh PhoBERT
├── train_phobert_intent_colab.ipynb # Train PhoBERT trên Colab
├── setup_stanza.py                 # Download Stanza model
├── config.py                       # Cấu hình chung
└── requirements.txt
```
---

## ⚙️ Cài đặt

pip install -r requirements.txt
python setup_stanza.py

---

## ▶️ Cách chạy pipeline

1. Chạy tiền xử lý (Bài 1)

```bash
python preprocess.py
```

Output: `clauses.txt`, `chunks.txt`, `dependency.json`

2. Train model Intent (nếu chưa có)

```bash
# Train TF-IDF + Logistic Regression
python train_intent.py

# Train + cross-validation + so sánh rule-based
python train_intent.py --eval --compare

# So sánh TF-IDF vs PhoBERT (cần có model PhoBERT)
python train_intent.py --phobert
```

**PhoBERT:** Train trên Google Colab bằng `train_phobert_intent_colab.ipynb`, sau đó copy folder model về `models/intent_phobert/`.

3. Chạy trích xuất thông tin (Bài 2)
```bash
# Chạy toàn bộ pipeline
python extract.py

# Chạy + evaluation
python extract.py --eval

# Chạy từng task
python extract.py --task ner
python extract.py --task srl
python extract.py --task intent
```

## 🧠 Methodology

| Task                  | Phương pháp                                                         |
| --------------------- | ------------------------------------------------------------------- |
| Clause Splitting      | Rule-based (regex + liên từ), xử lý số tiền                         |
| NP Chunking           | POS tagging (Underthesea) + IOB labeling                            |
| Dependency Parsing    | Stanza (pre-trained Vietnamese)                                     |
| NER                   | Rule-based + regex pháp lý (PARTY, MONEY, DATE, RATE, PENALTY, LAW) |
| SRL                   | Heuristic từ dependency tree + NER entities                         |
| Intent Classification | TF-IDF+LR baseline + PhoBERT fine-tuned                             |

**Intent Labels:** Obligation, Prohibition, Right, Termination Condition

---

## ⚠️ Lưu ý

- Chạy `setup_stanza.py` trước lần đầu (download ~500MB model)
- Bài 2 phụ thuộc output Bài 1 — chạy `preprocess.py` trước
- PhoBERT model (~500MB) cần train trên Colab rồi copy về local
- Inference PhoBERT hỗ trợ CPU only (`low_cpu_mem_usage=True`)

---

## 🎯 Hướng phát triển

- PhoBERT cho NER
- SRL deep learning
- Legal-BERT tiếng Việt
- Knowledge graph hợp đồng