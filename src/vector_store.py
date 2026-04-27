import faiss
import json
import numpy as np


class VectorStore:
    def __init__(self, dim):
        self.index = faiss.IndexFlatIP(dim)

    def add(self, vectors):
        self.index.add(np.array(vectors))

    def search(self, query_vec, top_k):
        scores, indices = self.index.search(query_vec, top_k)
        return scores, indices

    def save(self, index_path):
        faiss.write_index(self.index, index_path)

    @staticmethod
    def load(index_path):
        return faiss.read_index(index_path)


def save_metadata(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_metadata(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)