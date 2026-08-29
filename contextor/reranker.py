class Reranker:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self.model_name, device="cpu")
        return self._model

    def rerank(self, query: str, hits: list[dict], top_k: int) -> list[dict]:
        model = self._load()
        pairs = [(query, hit["text"]) for hit in hits]
        scores = model.predict(pairs)

        for hit, score in zip(hits, scores):
            hit["rerank_score"] = float(score)

        return sorted(hits, key=lambda h: h["rerank_score"], reverse=True)[:top_k]