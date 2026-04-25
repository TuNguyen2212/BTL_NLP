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

````text
project/
├── input/                          # Thư mục chứa văn bản hợp đồng thô đầu vào
│   └── raw_contracts.txt
├── data/                           # Dữ liệu đã được gán nhãn (annotated)
│   ├── annotated_ner.json
│   └── annotated_intent.json
├── models/                         # Các mô hình đã được huấn luyện (pre-trained/trained)
│   ├── intent_tfidf.pkl
│   └── intent_phobert/             # Thư mục chứa trọng số mô hình PhoBERT
├── output/                         # Thư mục lưu kết quả sau khi chạy pipeline
│   ├── clauses.txt                 # Kết quả tách mệnh đề/câu
│   ├── chunks.txt                  # Kết quả trích xuất Noun Phrase (NP Chunker)
│   ├── dependency.json             # Cây phụ thuộc cú pháp
│   ├── ner_results.json            # Kết quả nhận dạng thực thể có tên (NER)
│   ├── srl_results.json            # Kết quả gán nhãn vai nghĩa (SRL)
│   ├── intent_classification.txt   # Kết quả phân loại ý định (dạng text)
│   └── intent_classification_detail.json # Chi tiết phân loại ý định
├── src/                            # Source code chứa các module chính
│   ├── clause_splitter.py          # Bài 1.1: Tách câu/mệnh đề
│   ├── np_chunker.py               # Bài 1.2: Trích xuất cụm danh từ
│   ├── dependency_parser.py        # Bài 1.3: Phân tích cú pháp phụ thuộc
│   ├── contract_cleaner.py         # Tiền xử lý, làm sạch văn bản hợp đồng
│   ├── ner.py                      # Bài 2.1: Nhận dạng thực thể (NER)
│   ├── srl.py                      # Bài 2.2: Gán nhãn vai nghĩa (SRL)
│   ├── intent.py                   # Bài 2.3: Phân loại ý định (Inference dùng TF-IDF)
│   ├── phobert_intent.py           # Phân loại ý định (Inference dùng PhoBERT)
│   └── utils.py                    # Các hàm tiện ích hỗ trợ
├── preprocess.py                   # Script thực thi Pipeline Bài 1 (Tiền xử lý)
├── extract.py                      # Script thực thi Pipeline Bài 2 (Trích xuất thông tin)
├── train_intent.py                 # Script huấn luyện mô hình TF-IDF & so sánh với PhoBERT
├── train_phobert_intent_colab.ipynb# Notebook huấn luyện mô hình PhoBERT trên Google Colab
├── setup_stanza.py                 # Script tải các resource/model cần thiết cho Stanza
├── config.py                       # File cấu hình chung cho toàn bộ project (đường dẫn, tham số)
└── requirements.txt                # Danh sách các thư viện Python cần thiết

---

## ⚙️ Cài đặt

pip install -r requirements.txt
python setup_stanza.py

---

## ▶️ Cách chạy pipeline

1. Chạy tiền xử lý (Bài 1)

```bash
python preprocess.py
````

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
