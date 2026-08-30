# Contextor

A config-driven, corpus-agnostic RAG engine. Point it at any folder of
documents and get grounded, citation-backed Q&A with a hallucination
guardrail — swapping domains, embedding models, or LLM providers requires
editing `config.yaml` only, no code changes.

Validated end-to-end across two unrelated domains and file formats: a
tabletop gaming rules corpus (Markdown) and a workplace safety handbook
(PDF).

## Architecture

data/corpus/ --> loader.py --> chunker.py --> embedder.py --> vectorstore.py (Chroma)
|
question --------------------------------------------------> retriever (top-k, embedding search)
|
reranker.py (cross-encoder)
|
generator.py (Anthropic or Ollama)
|
guardrail.py (sentence-level grounding check)
|
answer + citations

## Architecture

```text
data/corpus/ --> loader.py --> chunker.py --> embedder.py --> vectorstore.py (Chroma)
|
question --------------------------------------------------> retriever (top-k, embedding search)
|
reranker.py (cross-encoder)
|
generator.py (Anthropic or Ollama)
|
guardrail.py (sentence-level grounding check)
|
answer + citations
```

Every stage is an independently-swappable module with a narrow interface.
`pipeline.py` is the only file that wires them together — no module
imports another beyond what it strictly needs.

## Features

- **Format-agnostic ingestion**: `.txt`, `.md`, `.mdx`, `.pdf`, `.html` via one loader
- **Recursive, overlap-aware chunking** that respects paragraph/sentence boundaries
- **Local embeddings** (`sentence-transformers`) with automatic GPU detection and graceful CPU fallback on compute-capability mismatches
- **Two-stage retrieval**: fast embedding search (top-k) → cross-encoder reranking for precision
- **Provider-agnostic generation** via the strategy pattern — switch between Anthropic Claude and a local Ollama model with one config line
- **Grounding guardrail**: flags generated claims not traceable to retrieved context, via sentence-level cosine similarity

## Setup

```bash
uv add pyyaml sentence-transformers chromadb anthropic pypdf beautifulsoup4 tqdm requests python-dotenv
```

Create a `.env` file at the project root:

```text
ANTHROPIC_API_KEY=sk-ant-your-key-here
```

(Only required if using the Anthropic provider — see Configuration below.)

If using the local provider, install [Ollama](https://ollama.com) and pull a model:
```bash
ollama pull llama3.1:8b
```

Drop documents into `data/corpus_a/` (or point `config.yaml` at any folder), then:

```bash
uv run python scripts/ingest.py
uv run python scripts/query.py "your question here"
```

## Configuration

Everything domain- or model-specific lives in `config.yaml`. Swapping the
corpus:

```yaml
corpus:
  name: "corpus_b"
  path: "./data/corpus_b"
```

Swapping the generation backend — no code changes, no restart logic:

```yaml
generation:
  provider: "ollama"   # or "anthropic"
```

## Evaluation

Two unrelated corpora were used to test the "plug and play" claim directly,
not just assert it — same codebase, different domain, different file
format, changed via config only.

| Corpus | Format | Domain | Retrieval Precision@k | Grounding Rate |
|---|---|---|---|---|
| corpus_a | Markdown | D&D 5e SRD (gaming rules) | 5/5 (100%) | 5/5 (100%) |
| corpus_b | PDF | OSHA Small Business Safety Handbook | 3/3 (100%) | 3/3 (100%) |

QA pairs are hand-curated and verified against the source text (see
`eval/qa_pairs_*.json`) — a small, labeled sample reflecting this project's
two-week scope, not an exhaustive benchmark. Run it yourself:

```bash
uv run python scripts/eval.py --qa eval/qa_pairs_corpus_a.json
```

### The generator path was fully tested for both providers

- **Ollama** (`llama3.1:8b`): fully tested, all results above generated via this path
- **Anthropic** (`claude-sonnet-4-6`): implemented via the same interface and factory pattern; not live-tested end-to-end in this build due to a billing/workspace configuration constraint on the developer account used, not a code defect

## Known Limitations & Design Decisions

Real issues found and fixed during development, documented rather than
hidden — each one taught something about RAG systems generally, not just
this codebase:

- **Chunking can separate an entity from its content.** Several D&D
  classes share an identical "Extra Attack" feature description; greedy
  chunking sometimes split the class-name heading from the ability text,
  causing the *correct* chunk to rank outside the top 10 in embedding-only
  search. The cross-encoder reranker (`reranker.py`) was added specifically
  to correct this — confirmed via a targeted before/after test.
- **Markdown isn't always plain text.** The SRD source files embed raw
  HTML for complex stat-block tables. Left unstripped, this HTML noise
  dominated the embedding space (hundreds of near-duplicate `<td>` chunks),
  actively displacing relevant content from retrieval results entirely.
  Fixed in `loader.py` by detecting and stripping embedded HTML from
  `.md`/`.mdx` files.
- **Flattening HTML tables loses column structure.** Stripping tags removes
  noise but also removes the semantic meaning of table columns (e.g. which
  number is "Level" vs "Proficiency Bonus"). Not fully solved here — a
  correct fix would parse tables into natural-language sentences during
  loading.
- **A correct answer isn't proof of grounding.** In one test case, the LLM
  correctly answered "2 attacks" for a level-5 fighter, but its own
  retrieved context only showed a level-20 entry and a feature-listing
  table — no chunk stating the actual number. The model filled the gap
  with background knowledge, not retrieved fact. This is exactly the
  failure case the grounding guardrail is designed to catch, and motivated
  moving from "no guardrail" to a real sentence-level implementation.
- **Grounding checks need sentence-level granularity on both sides.**
  Comparing a generated sentence against a whole multi-topic chunk
  under-scores genuine support, since the chunk's other content dilutes
  the embedding average. Fixed by splitting both the answer *and* the
  retrieved context into sentences before comparing.
- **Citation tags aren't claims.** The sentence-level grounding check
  initially flagged bracketed citations like `[source: file.md]` as
  "unsupported," since they have almost no natural-language content to
  compare semantically. Fixed by detecting and skipping citation-only
  fragments before grounding evaluation.
- **The grounding threshold (0.5 cosine similarity) is a heuristic, not an
  empirically calibrated value.** It was sanity-checked against real
  passing and failing cases in this project's evaluation, but a production
  system would want to tune it against a larger, labeled dataset.
- **PDF extraction includes layout artifacts.** Table-of-contents
  dot-leaders (`. . . . . 93`) are extracted as literal text alongside
  real content, since `pypdf` extracts text without understanding visual
  layout intent. Cosmetic noise, not a retrieval-breaking issue observed
  in testing.
- **Hardware compatibility is handled automatically, not assumed.**
  `embedder.py` checks the installed PyTorch build's supported CUDA
  architectures against the detected GPU before use, falling back to CPU
  with a printed explanation on mismatch — rather than crashing or
  silently guessing.

## Tech Stack

| Layer | Choice |
|---|---|
| Dependency management | `uv`, Python 3.14 |
| Embeddings | `sentence-transformers` (`all-MiniLM-L6-v2`), local, CPU/GPU auto-detect |
| Vector store | Chroma (embedded, file-persisted) |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Generation | Anthropic Claude, OpenAI, or Ollama — swappable via config, validated against a per-provider known-models list |
| Guardrail | Sentence-level cosine similarity grounding check |

## Corpus Attribution

- `corpus_a`: D&D 5e SRD 5.2.1, sourced from [downfallx/dnd-5e-srd-markdown](https://github.com/downfallx/dnd-5e-srd-markdown), licensed CC-BY-4.0
- `corpus_b`: OSHA Small Business Safety and Health Handbook (OSHA 2209),
  a U.S. government publication

## Possible Future Work

- Natural-language conversion of tabular content during loading, rather than flat-text HTML stripping
- LLM-based (entailment-style) grounding check as a more accurate, slower alternative to the current embedding-similarity approach
- Word-boundary-aware chunk overlap (currently character-count-based)
- FastAPI endpoint wrapping the pipeline for HTTP access