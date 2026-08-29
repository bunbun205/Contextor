from .config import Config
from.loader import load_corpus
from .chunker import chunk_documents
from .embedder import Embedder
from .vectorstore import VectorStore
from .generator import create_generator
from .guardrail import Guardrail

class ContextorPipeline:
    def __init__(self, config: Config):
        self.config = config
        from .reranker import Reranker
        self.reranker = Reranker(config["retrieval"]["reranker_model"])

        self.embedder = Embedder(
            model_name=config["embedding"]["model"],
            device=config["embedding"].get("device", "auto"),
        )
        self.vectorstore = VectorStore(
            persist_dir=config["vectorstore"]["persist_dir"],
            collection_name=config.collection_name,
            embedder=self.embedder,
        )
        self.generator = create_generator(config)
        self.guardrail = Guardrail(
            enabled=config["guardrail"]["enabled"],
            strict_mode=config["guardrail"]["strict_mode"],
            embedder=self.embedder,
        )

    def ingest(self) -> int:
        docs = load_corpus(self.config.corpus_path)
        chunks = chunk_documents(
            docs,
            chunk_size=self.config["chunking"]["chunk_size"],
            chunk_overlap=self.config["chunking"]["chunk_overlap"],
        )
        self.vectorstore.index(chunks)
        return len(chunks)

    def query(self, question: str) -> dict:
        top_k = self.config["retrieval"]["top_k"]
        hits = self.vectorstore.query(question, top_k=top_k)

        if self.config["retrieval"]["rerank"]:
            hits = self.reranker.rerank(question, hits, top_k=self.config["retrieval"]["rerank_top_k"])
        else:
            hits = hits[:self.config["retrieval"]["rerank_top_k"]]

        answer = self.generator.generate(question, hits)
        result = self.guardrail.check(answer, hits)
        result["retrieved_chunks"] = hits
        return result