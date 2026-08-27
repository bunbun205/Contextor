from dataclasses import dataclass
from .loader import Document

SEPERATORS = [ "\n\n", "\n", ". ", " " ]

@dataclass
class Chunk:
    id: str
    source: str
    text: str

def _split_text(text:str, chunk_size: int, seperators: list[str]) -> list[str]:
    if(len(text) <= chunk_size):
        return [text]

    if not seperators:
        return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

    sep, rest = seperators[0], seperators[1:]
    parts = text.split(sep)
    chunks, current = [], ""

    for part in parts:
        candidate = current + sep + part if current else part
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            if current:
                chunks.append(current)
            if len(part) > chunk_size:
                chunks.extend(_split_text(part, chunk_size, rest))
                current = ""
            else:
                current = part

    if current:
        chunks.append(current)

    return chunks

def chunk_documents(docs: list[Document], chunk_size: int, chunk_overlap: int) -> list[Chunk]:
    chunks: list[Chunk] = []
    for doc in docs:
        pieces = _split_text(doc.text, chunk_size, SEPERATORS)

        overlapped = []
        for i, piece in enumerate(pieces):
            if i > 0 and chunk_overlap > 0:
                tail = pieces[i - 1][-chunk_overlap:]
                piece = tail + " " + piece
            overlapped.append(piece)

        for i, piece in enumerate(overlapped):
            chunks.append(Chunk(id=f"{doc.source}::{i}", source=doc.source, text=piece))

    return chunks