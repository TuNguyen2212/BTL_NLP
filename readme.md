# 📄 NLP Pipeline cho Hợp Đồng Pháp Lý Tiếng Việt

## 📌 Tổng quan

Project này xây dựng một pipeline NLP hoàn chỉnh cho xử lý hợp đồng pháp lý tiếng Việt, gồm 3 giai đoạn chính:

Giai đoạn 1: Tiền xử lý (Preprocessing)

- Tách câu và mệnh đề (Clause Splitting)
- Chunk danh từ (IOB tagging)
- Phân tích phụ thuộc (Dependency Parsing - Stanza)

Giai đoạn 2: Trích xuất & hiểu ngữ nghĩa

- Nhận diện thực thể (NER)
- Gán vai nghĩa (SRL)
- Phân loại ý định (Intent Classification)

Giai đoạn 3A: RAG Chatbot – Retrieval (Assignment 3A)

- Gộp dữ liệu từ pipeline thành enriched_clauses.json
- Embedding clauses bằng model tiếng Việt / đa ngôn ngữ
- Lưu vector vào FAISS
- Truy vấn top-k clauses theo query

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
↓
[RAG - Bài 3A]

- Merge dữ liệu
- Embedding
- Vector DB (FAISS)
- Retriever
↓
Top-k Clauses (input cho LLM)
---

## 📁 Cấu trúc thư mục

```text
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
├── vector_db/
│   ├── faiss.index
│   ├── metadata.json
│   └── enriched_clauses.json
├── src/
│   ├── clause_splitter.py
│   ├── np_chunker.py
│   ├── dependency_parser.py
│   ├── contract_cleaner.py
│   ├── ner.py
│   ├── srl.py
│   ├── intent.py
│   ├── phobert_intent.py
│   ├── utils.py
│   ├── merger.py
│   ├── embedder.py
│   ├── vector_store.py
│   └── evaluator.py
├── preprocess.py
├── extract.py
├── train_intent.py
├── train_phobert_intent_colab.ipynb
├── build_vector_db.py
├── retriever.py
├── evaluate_retrieval.py
├── setup_stanza.py
├── config.py
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
```

```bash
# Train + cross-validation + so sánh rule-based
python train_intent.py --eval --compare
```

```bash
# So sánh TF-IDF vs PhoBERT (cần có model PhoBERT)
python train_intent.py --phobert
```

**PhoBERT:** Train trên Google Colab bằng `train_phobert_intent_colab.ipynb`, sau đó copy folder model về `models/intent_phobert/`.

3. Chạy trích xuất thông tin (Bài 2)

```bash
# Chạy toàn bộ pipeline
python extract.py
```

```bash
# Chạy + evaluation
python extract.py --eval
```

# Chạy từng task

```bash
python extract.py --task ner
```

```bash
python extract.py --task srl
```

```bash
python extract.py --task intent
```

4. RAG Retrieval (Bài 3A)
```bash
# Xây dựng vector database (merge + embedding + FAISS)
python build_vector_db.py
```

```bash
# Chạy retriever với query mẫu
python retriever.py
```

```bash
# Chạy evaluation cho retriever
python evaluate_retrieval.py
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
| Retrieval             | Embedding (E5) + FAISS (cosine similarity)                |
**Intent Labels:** Obligation, Prohibition, Right, Termination Condition

---

## 🧾 Enriched Clause Schema (Bài 3A)

{
  "clause_id": "C001",
  "text": "...",
  "entities": [...],
  "srl": {
    "predicate": "...",
    "roles": {...},
    "negated": false
  },
  "intent": "...",
  "intent_confidence": 0.91
}

---

## ⚠️ Lưu ý

- Chạy `setup_stanza.py` trước lần đầu (download ~500MB model)
- Bài 2 phụ thuộc output Bài 1 — chạy `preprocess.py` trước
- PhoBERT model (~500MB) cần train trên Colab rồi copy về local
- Inference PhoBERT hỗ trợ CPU only (`low_cpu_mem_usage=True`)
- Bài 3 phụ thuộc output của Bài 1 & 2
- Phải chạy preprocess.py và extract.py trước
- Vector DB được lưu trong vector_db/
- Sử dụng FAISS cho similarity search
- Embedding model: intfloat/multilingual-e5-base

---

## 🎯 Hướng phát triển

- PhoBERT cho NER
- SRL deep learning
- Legal-BERT tiếng Việt
- Knowledge graph hợp đồng
- Hybrid retrieval (BM25 + embedding)
- Reranking (cross-encoder)
- RAG chatbot (LLM + Streamlit)
