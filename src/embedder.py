class Embedder:
    def __init__(self):
        from fastembed import TextEmbedding
        self.model = TextEmbedding(model_name="intfloat/multilingual-e5-large")

    def embed_passages(self, texts):
        prefixed = [f"passage: {t}" for t in texts]
        return list(self.model.embed(prefixed))

    def embed_query(self, query):
        prefixed = f"query: {query}"
        return list(self.model.embed([prefixed]))[0]