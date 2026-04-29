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

Giai đoạn 3B: RAG Chatbot – Generation (Assignment 3B)

- LLM trả lời có trích dẫn mệnh đề [Cxxx]
- Kiểm tra ảo giác (hallucination check) tự động
- Giao diện Streamlit hỗ trợ chọn hợp đồng, xem hợp đồng gốc

---

## 🚀 Chạy nhanh RAG Chatbot

### 1. Cài đặt

```bash
pip install -r requirements.txt
```

### 2. Cấu hình API Key (chọn 1 trong 2 cách)

**Cách A — File `.env`:**

```bash
# Tạo file .env ở thư mục gốc project
OPENROUTER_API_KEY=sk-or-v1-xxxxxxxx
```

**Cách B — Nhập trên giao diện:**
Khởi chạy app, nhập API key vào ô "API Key" trên sidebar.

> Tạo key miễn phí tại: https://openrouter.ai/keys
> (Đăng ký bằng Google, không cần thẻ tín dụng hoặc lấy key free trong report)

### 3. Khởi chạy

```bash
streamlit run app.py
```

Mở trình duyệt tại `http://localhost:8501`.

**Lưu ý:** Lần đầu chạy sẽ tải model embedding (~1.2GB), cần đợi 1-2 phút.

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
[RAG Retrieval - Bài 3A]

- Merge dữ liệu
- Embedding
- Vector DB (FAISS)
- Retriever (BM25 + Dense + RRF + Reranker)
  ↓
  Top-k Clauses
  ↓
[RAG Generation - Bài 3B]

- LLM trả lời có trích dẫn
- Kiểm tra ảo giác (hallucination check)
- Giao diện Streamlit
  ↓
  Câu trả lời có trích dẫn + cảnh báo ảo giác

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
├── rag_pipeline.py
├── generator.py
├── app.py
├── setup_stanza.py
├── config.py
├── .env
└── requirements.txt
```

---

## ⚙️ Cài đặt

```bash
pip install -r requirements.txt
python setup_stanza.py
```

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

```bash
# Chạy từng task
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

5. RAG Generation (Bài 3B)

```bash
# Chạy chatbot Streamlit
streamlit run app.py
```

Mở trình duyệt tại `http://localhost:8501`. Cần cấu hình API key OpenRouter (xem mục "Chạy nhanh RAG Chatbot" phía trên).

```bash
# Chạy chatbot CLI (không cần Streamlit)
python rag_pipeline.py
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
| Retrieval             | Hybrid (BM25 + E5 embedding) + RRF + Cross-encoder reranking        |
| Generation            | LLM (OpenRouter) + prompt engineering + hallucination check         |

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
- Bài 3A phụ thuộc output của Bài 1 & 2 — chạy `preprocess.py` và `extract.py` trước
- Bài 3B cần API key OpenRouter (xem mục "Chạy nhanh RAG Chatbot")
- Vector DB được lưu trong vector_db/
- Sử dụng FAISS cho similarity search
- Embedding model: intfloat/multilingual-e5-large
- **RAM:** Cần tối thiểu ~4GB RAM trống để chạy Streamlit app (embedding model ~1.2GB + reranker ~470MB + overhead). Trên máy 8GB RAM, nên đóng bớt ứng dụng khác trước khi chạy.
---

## 🎯 Hướng phát triển

- PhoBERT / Transformer cho NER (thay thế rule-based regex)
- SRL dựa trên deep learning (thay thế heuristic từ dependency tree)
- Fine-tune Legal-BERT tiếng Việt cho domain pháp lý
- Knowledge graph liên kết các điều khoản hợp đồng
- Hỗ trợ nhiều định dạng hợp đồng đầu vào (PDF, DOCX)
- Lưu lịch sử hội thoại và feedback người dùng để cải thiện retrieval
