import re
import numpy as np
from collections import defaultdict

from rank_bm25 import BM25Okapi

from src.embedder import Embedder
from src.vector_store import VectorStore, load_metadata

def _tokenize(text: str) -> list[str]:
    return re.findall(r'\w+', text.lower())

class ClauseRetriever:
    def __init__(self):
        print("[Retriever] Initializing...")

        self.index = VectorStore.load("vector_db/faiss.index")
        self.metadata = load_metadata("vector_db/metadata.json")

        corpus = [_tokenize(item["text"]) for item in self.metadata]
        self.bm25 = BM25Okapi(corpus)

        from sentence_transformers import CrossEncoder
        self.reranker = CrossEncoder("cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")

        self.embedder = Embedder()

        print(f"[Retriever] Loaded {len(self.metadata)} clauses (FAISS + BM25 + Reranker)")

    def encode_query(self, query: str):
        return np.array([self.embedder.embed_query(query)])

    def get_contract_names(self):
        names = {}
        for item in self.metadata:
            cid = item.get("contract_id")
            cname = item.get("contract_name")
            if cid and cname and cid not in names:
                names[cid] = cname
        return names

    def _bm25_search(self, query: str, top_k: int, contract_id: str = None):
        tokens = _tokenize(query)
        scores = self.bm25.get_scores(tokens)
        ranked = np.argsort(scores)[::-1]

        results = {}
        for rank, idx in enumerate(ranked):
            idx = int(idx)
            if idx >= len(self.metadata):
                continue
            item = self.metadata[idx]
            if contract_id and item.get("contract_id") != contract_id:
                continue
            results[idx] = rank + 1
            if len(results) >= top_k:
                break
        return results

    def _dense_search(self, query: str, top_k: int, contract_id: str = None):
        query_vector = self.encode_query(query)
        fetch_k = top_k * 3 if contract_id else top_k
        scores, indices = self.index.search(query_vector, fetch_k)

        results = {}
        for rank, idx in enumerate(indices[0]):
            idx = int(idx)
            if idx < 0 or idx >= len(self.metadata):
                continue
            item = self.metadata[idx]
            if contract_id and item.get("contract_id") != contract_id:
                continue
            results[idx] = rank + 1
            if len(results) >= top_k:
                break
        return results

    def _reciprocal_rank_fusion(self, *rank_dicts, k: int = 60):
        fused = defaultdict(float)
        for rd in rank_dicts:
            for idx, rank in rd.items():
                fused[idx] += 1.0 / (k + rank)
        return dict(sorted(fused.items(), key=lambda x: x[1], reverse=True))

    def search(self, query: str, top_k: int = 5, contract_id: str = None):
        candidate_k = top_k * 3
        dense_ranks = self._dense_search(query, candidate_k, contract_id)
        bm25_ranks = self._bm25_search(query, candidate_k, contract_id)

        fused = self._reciprocal_rank_fusion(dense_ranks, bm25_ranks)

        candidate_indices = list(fused.keys())[:top_k * 2]
        if candidate_indices:
            pairs = [(query, self.metadata[idx]["text"]) for idx in candidate_indices]
            rerank_scores = self.reranker.predict(pairs)
            scored = sorted(zip(candidate_indices, rerank_scores), key=lambda x: x[1], reverse=True)
        else:
            scored = []

        results = []
        for idx, score in scored[:top_k]:
            item = self.metadata[idx]
            results.append({
                "rank": len(results) + 1,
                "clause_id": item["clause_id"],
                "text": item["text"],
                "score": float(score),
                "intent": item.get("intent"),
                "entities": item.get("entities", []),
                "contract_id": item.get("contract_id"),
                "contract_name": item.get("contract_name"),
            })
        return results

    def format_results(self, query: str, results: list):
        output = []

        output.append("=" * 90)
        output.append(f"QUERY: {query}")
        output.append("=" * 90)

        for r in results:
            output.append("")
            output.append("-" * 90)
            output.append(f"Rank: {r['rank']}")
            output.append(f"Clause ID: {r['clause_id']}")
            output.append(f"Score: {r['score']:.4f}")
            output.append(f"Intent: {r['intent']}")
            output.append("")
            output.append("Text:")
            output.append(r["text"])

            if r["entities"]:
                output.append("")
                output.append("Entities:")
                for e in r["entities"]:
                    output.append(f"   - {e.get('label', 'UNKNOWN')}: {e.get('text', '')}")

        output.append("=" * 90)

        return "\n".join(output)


def run_demo_mode(retriever):
    print("\n[Demo] Running demo mode\n")

    test_queries = [
        "khi nào phải thanh toán tiền thuê",
        "thời hạn thuê bao lâu",
        "phạt chậm thanh toán như thế nào",
        "điều kiện chấm dứt hợp đồng"
    ]

    for query in test_queries:
        results = retriever.search(query, top_k=5)
        print(retriever.format_results(query, results))
        print("\n\n")


def run_chat_mode(retriever):
    print("\n[Chat] Legal RAG Chatbot")
    print("Type 'exit' to quit\n")

    while True:
        query = input("You: ")

        if query.lower() == "exit":
            print("Bye!")
            break

        results = retriever.search(query, top_k=5)
        print(retriever.format_results(query, results))


def main():
    retriever = ClauseRetriever()

    print("\nSelect mode:")
    print("1 - Demo mode (fixed queries)")
    print("2 - Chat mode (interactive)")

    choice = input("\nEnter choice (1/2): ")

    if choice == "1":
        run_demo_mode(retriever)
    else:
        run_chat_mode(retriever)


if __name__ == "__main__":
    main()