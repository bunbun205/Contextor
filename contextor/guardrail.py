import re
from .embedder import Embedder

def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot / (norm_a * norm_b)

def _split_sentences(text: str) -> list[str]:
    sentences = re.split(r"(?<=[.?!])\s+", text.strip())
    return [s for s in sentences if s]

class Guardrail:
    def __init__(self, enabled: bool, strict_mode: bool, embedder: Embedder, threshold: float = 0.5):
        self.enabled = enabled
        self.strict_mode = strict_mode
        self.embedder = embedder
        self.threshold = threshold

    def check(self, answer: str, context_chunks: list[dict]) -> dict:
        if not self.enabled:
            return {"answer": answer, "flagged_claims": [], "grounded": True}

        answer_sentences = _split_sentences(answer)
        if not answer_sentences or not context_chunks:
            return {"answer": answer, "flagged_claims": [], "grounded": True}

        context_sentences = []
        for chunk in context_chunks:
            context_sentences.extend(_split_sentences(chunk["text"]))

        if not context_sentences:
            return {"answer": answer, "flagged_claims": [], "grounded": True}

        answer_embeddings = self.embedder.embed_texts(answer_sentences)
        context_embeddings = self.embedder.embed_texts(context_sentences)

        flagged = []
        for sentence, sent_emb in zip(answer_sentences, answer_embeddings):
            best_similarity = max(
                _cosine_similarity(sent_emb, ctx_emb) for ctx_emb in context_embeddings
            )
            if best_similarity < self.threshold:
                flagged.append(sentence)

        grounded = len(flagged) == 0

        if self.strict_mode and not grounded:
            return{
                "answer": "I can't confidently answer this from the provided context alone.",
                "flagged_claims": flagged,
                "grounded": False,
            }

        return {"answer": answer, "flagged_claims": flagged, "grounded": grounded}