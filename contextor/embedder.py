from .chunker import Chunk

def _resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested

    import torch

    if not torch.cuda.is_available():
        return "cpu"

    major, minor = torch.cuda.get_device_capability(0)
    arch = f"sm_{major}.{minor}"
    supported = torch.cuda.get_arch_list()

    if arch in supported:
        return "cuda"

    print(
        f"Note: GPU compute capability  {arch} is not supported in this "
        f"PyTorch build (supports: {supported}). Falling back to CPU."
    )

    return "cpu"

class Embedder:
    def __init__(self, model_name: str, device: str = "auto"):
        self.model_name = model_name
        self.device = _resolve_device(device)
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        model = self._load()
        return model.encode(texts, show_progress_bar=False).tolist()

    def embed_chunks(self, chunks: list[Chunk]) -> list[list[float]]:
        return self.embed_texts([c.text for c in chunks])

    def embed_query(self, query: str) -> list[float]:
        return self.embed_texts([query])[0]
