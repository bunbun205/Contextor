from contextor.config import Config
from contextor.loader import load_corpus
from contextor.chunker import chunk_documents

c = Config.load('config.yaml')
docs = load_corpus(c.corpus_path)

# use a small chunk_size on purpose so our short test file actually splits
chunks = chunk_documents(docs, chunk_size=200, chunk_overlap=30)

print(f'Produced {len(chunks)} chunks from {len(docs)} doc(s)\n')
for c in chunks:
    print(f'--- {c.id} ({len(c.text)} chars) ---')
    print(c.text)
    print()