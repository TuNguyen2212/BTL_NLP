from src.merger import merge_all, save_json
from src.embedder import Embedder
from src.vector_store import VectorStore, save_metadata
import os


def main():
    print("[VectorDB] Merging data...")

    enriched = merge_all(
        "output/clauses.txt",
        "output/ner_results.json",
        "output/srl_results.json",
        "output/intent_classification_detail.json"
    )

    os.makedirs("vector_db", exist_ok=True)

    save_json(enriched, "vector_db/enriched_clauses.json")

    print("[VectorDB] Embedding...")

    embedder = Embedder()
    texts = [c["text"] for c in enriched]

    vectors = embedder.embed_passages(texts)

    print("[VectorDB] Building FAISS index...")

    store = VectorStore(dim=len(vectors[0]))
    store.add(vectors)

    store.save("vector_db/faiss.index")
    save_metadata(enriched, "vector_db/metadata.json")

    print("[VectorDB] Done.")


if __name__ == "__main__":
    main()