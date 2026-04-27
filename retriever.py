import numpy as np
from collections import defaultdict

from src.embedder import Embedder
from src.vector_store import VectorStore, load_metadata


class ClauseRetriever:
    def __init__(self):
        print("📦 Initializing ClauseRetriever...")

        self.embedder = Embedder()
        self.index = VectorStore.load("vector_db/faiss.index")
        self.metadata = load_metadata("vector_db/metadata.json")

        print(f"✅ Loaded {len(self.metadata)} clauses")

    def encode_query(self, query: str):
        return np.array([self.embedder.embed_query(query)])

    def search(self, query: str, top_k: int = 5):
        query_vector = self.encode_query(query)

        scores, indices = self.index.search(query_vector, top_k)

        results = []

        for rank, idx in enumerate(indices[0]):
            item = self.metadata[idx]

            results.append({
                "rank": rank + 1,
                "clause_id": item["clause_id"],
                "text": item["text"],
                "score": float(scores[0][rank]),
                "intent": item.get("intent"),
                "entities": item.get("entities", [])
            })

        return results

    def format_results(self, query: str, results: list):
        output = []

        output.append("=" * 90)
        output.append(f"🔎 QUERY: {query}")
        output.append("=" * 90)

        for r in results:
            output.append("")
            output.append("-" * 90)
            output.append(f"📌 Rank: {r['rank']}")
            output.append(f"🆔 Clause ID: {r['clause_id']}")
            output.append(f"🎯 Score: {r['score']:.4f}")
            output.append(f"🧠 Intent: {r['intent']}")
            output.append("")
            output.append("📄 Text:")
            output.append(r["text"])

            if r["entities"]:
                output.append("")
                output.append("🏷 Entities:")
                for e in r["entities"]:
                    output.append(f"   - {e.get('label', 'UNKNOWN')}: {e.get('text', '')}")

        output.append("=" * 90)

        return "\n".join(output)


# =========================
# MODE 1: DEMO MODE
# =========================
def run_demo_mode(retriever):
    print("\n🚀 RUNNING DEMO MODE\n")

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


# =========================
# MODE 2: CHAT MODE
# =========================
def run_chat_mode(retriever):
    print("\n💬 LEGAL RAG CHATBOT")
    print("Type 'exit' to quit\n")

    while True:
        query = input("You: ")

        if query.lower() == "exit":
            print("👋 Bye!")
            break

        results = retriever.search(query, top_k=5)
        print(retriever.format_results(query, results))


# =========================
# MAIN
# =========================
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