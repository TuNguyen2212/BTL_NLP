from fastembed import TextEmbedding

class Embedder:
    def __init__(self):
        self.model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")

    def embed_passages(self, texts):
        return list(self.model.embed(texts))

    def embed_query(self, query):
        return list(self.model.embed([query]))[0]