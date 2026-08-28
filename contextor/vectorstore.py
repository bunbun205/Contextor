from pathlib import Path
from .chunker import Chunk
from .embedder import Embedder

class VectorStore:
    def __init__(self, persist_dir: Path, collection_name: str, embedder: Embedder):
        import chromadb

        self.embedder = embedder
        self.client = chromadb.PersistentClient(path=str(persist_dir))
        self.collection = self.client.get_or_create_collection(name=collection_name)

    def index(self, chunks: list[Chunk]) -> None:
        embeddings = self.embedder.embed_chunks(chunks)
        self.collection.upsert(
            ids = [c.id for c in chunks],
            embeddings = embeddings,
            documents = [c.text for c in chunks],
            metadatas = [{"source": c.source} for c in chunks],
        )

    def query(self, query_text: str, top_k: int) -> list[dict]:
        query_embedding = self.embedder.embed_query(query_text)
        results = self.collection.query(query_embeddings=[query_embedding], n_results=top_k)

        hits = []
        for i in range(len(results["ids"][0])):
            hits.append({
                "id": results["ids"][0][i],
                "text": results["documents"][0][i],
                "source": results["metadatas"][0][i]["source"],
                "distance": results["distances"][0][i],
            })

        return hits
