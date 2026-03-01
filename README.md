# AIRA: Anesthesia Assistant (RAG + Diagram-Aware Retrieval)

AIRA is a retrieval-augmented assistant for anesthesia and critical-care study workflows.
It ingests textbook PDFs, indexes chunked content in a local Chroma vector store, retrieves relevant sections for user queries, and returns grounded answers with citations.

This project also supports **diagram-aware ingestion**:
- Figures/diagrams are extracted during ingestion.
- Diagram paths are stored in retrieval metadata.
- Relevant diagrams are rendered in the Gradio chat output when available.

## Features

- PDF ingestion with bookmark-aware section structuring
- Page-wise fallback ingestion when bookmarks are absent
- Token-based chunking with metadata preservation
- Local vector search using ChromaDB + sentence-transformer embeddings
- Query rewriting before retrieval for better recall
- Citation-rich responses (book, hierarchy, page range)
- Diagram extraction and preview links in chat
- Conversation persistence for chat sessions
- Optional voice input via Whisper

## Tech Stack

- Python 3.11+
- [Gradio](https://www.gradio.app/) for UI
- [PyMuPDF (fitz)](https://pymupdf.readthedocs.io/) for PDF parsing
- [ChromaDB](https://www.trychroma.com/) for vector storage
- [SentenceTransformers](https://www.sbert.net/) (`BAAI/bge-base-en-v1.5`) for embeddings
- Ollama-backed generation (configured in `llm/`)

## Project Structure

```text
.
├── gradio_app.py                   # Main chat UI + orchestration
├── ingestion/
│   ├── ingestion_pipeline.py       # End-to-end ingestion script
│   ├── bookmark_parser.py          # TOC/bookmark tree parsing
│   ├── document_builder.py         # Bookmark-based text+metadata docs
│   ├── page_builder.py             # Page-wise fallback docs
│   ├── chunker.py                  # Token chunking
│   └── diagram_extractor.py        # Diagram extraction/render fallback
├── retrieval/
│   └── retriever.py                # Embedding query + similarity filtering
├── vectorstore/
│   └── build_vectorstore.py        # Chroma write path
├── llm/
│   ├── llm_service.py
│   ├── prompt_builder.py
│   └── query_rewriter.py
├── storage/
│   └── conversation_store.py       # Chat history persistence
├── data_bank/                      # Source PDFs + extracted diagrams
└── vectorstore/                    # Persistent Chroma data
```

## Setup

1. Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install dependencies (based on your existing environment/package files):

```bash
pip install -U pip
# install your project dependencies here
```

3. Ensure required runtime tools/models are available:

- Ollama running with your configured chat model (default in code: `mistral`)
- Embedding model `BAAI/bge-base-en-v1.5` available in local cache

### Runtime Configuration (Environment Variables)

The app now supports environment-driven runtime config:

- `OLLAMA_MODEL` (default: `mistral`)
- `RETRIEVER_TOP_K` (default: `8`)
- `SIMILARITY_THRESHOLD` (default: `0.35`)
- `CHROMA_PERSIST_DIRECTORY` (default: `vectorstore`)
- `CHROMA_COLLECTION_NAME` (default: `medical_knowledge`)
- `GRADIO_SHARE` (default: `true`)
- `WHISPER_MODEL` (default: `base`)
- `STRUCTURED_LOGS` (default: `true`)
- `ALLOWED_BOOK_IDS` (optional CSV; if omitted, books are discovered from Chroma metadata)

Example:

```bash
export OLLAMA_MODEL=mistral
export RETRIEVER_TOP_K=8
export SIMILARITY_THRESHOLD=0.35
export ALLOWED_BOOK_IDS="miller,barash-clinical-anaesthesiology-231220_124533,the-icu-book-5e-2025-paul-l-marino-algrawany"
python gradio_app.py
```

## Run the App

```bash
python gradio_app.py
```

Open the local URL shown in terminal (typically `http://127.0.0.1:7860`).

## Ingestion Workflow

Run ingestion as a module from repo root:

```bash
python -m ingestion.ingestion_pipeline
```

### What ingestion does

1. Generates `book_id` from PDF filename
2. Parses PDF bookmarks (or falls back to page-wise docs)
3. Extracts text + metadata + diagram paths
4. Chunks documents while preserving metadata
5. Deletes prior vectors for that `book_id`
6. Re-embeds and stores fresh chunks in Chroma

### Diagram handling

- Extracts raster images from pages where present.
- If no raster image exists but figure-like text is detected (`figure`, `fig.`, `diagram`, `algorithm`), it saves a rendered page fallback image.
- Diagrams are saved under:

```text
data_bank/diagrams/<book_id>/
```

## Retrieval and Chat Behavior

- Retriever returns top-k chunks with metadata.
- Metadata includes `diagram_paths` (serialized for Chroma compatibility).
- Chat output includes:
  - grounded answer text
  - sources block (book + hierarchy + pages)
  - diagrams section with links and inline previews if available

For Gradio 6.x compatibility, file rendering uses:

```text
/gradio_api/file=<absolute_path>
```

and `allowed_paths` is configured in `gradio_app.py`.

Structured JSON logs are emitted for request lifecycle and model upstream events
(`query_received`, `retrieval_done`, `generation_done`, `llm_request_retryable_error`, etc.).

## Common Commands

### Check if diagram metadata exists for a book

```bash
python - <<'PY'
import json, chromadb
from chromadb.config import Settings

book_id = 'miller'
client = chromadb.Client(Settings(persist_directory='vectorstore', is_persistent=True))
col = client.get_collection('medical_knowledge')
metas = col.get(where={'book_id': book_id}, include=['metadatas'])['metadatas'] or []

with_key = sum(1 for m in metas if 'diagram_paths' in m)
with_non_empty = 0
for m in metas:
    v = m.get('diagram_paths')
    if isinstance(v, str):
        try:
            arr = json.loads(v)
            if isinstance(arr, list) and arr:
                with_non_empty += 1
        except Exception:
            if v:
                with_non_empty += 1

print('total:', len(metas))
print('with diagram_paths key:', with_key)
print('with non-empty diagram_paths:', with_non_empty)
PY
```

### Stop a running ingestion process

```bash
pkill -f "python -m ingestion.ingestion_pipeline"
```

## Eval Workflow

This repo includes a lightweight eval runner in `evals/run_evals.py` that validates
retrieval behavior (and optionally answer generation) against a case file.
It also computes retrieval ranking quality metrics such as `MRR`, `DCG`, `NDCG`,
`Hit Rate`, and `Precision@k` for evaluable cases.

### 1) Edit eval cases

Update:

```text
evals/cases.json
```

Each case supports:
- `query`: user input
- `book_id`: optional book filter (`null` for all books)
- `expect_refused`: expected retrieval refusal behavior
- `expected_terms`: terms that should appear in retrieved context
- `expected_source_any`: at least one source hint that should appear in metadata/text
- `expected_answer_terms`: terms expected in generated answer (when LLM eval is enabled)
- `forbidden_answer_terms`: terms that must not appear in generated answer
- `require_diagram`: when `true`, retrieval must return at least one valid diagram path

### 2) Run retrieval evals (recommended default)

```bash
python -m evals.run_evals
```

Run only the first N cases (useful for very large suites):

```bash
python -m evals.run_evals --cases evals/cases_bulk.json --max-cases 240
```

### 3) Run end-to-end evals (rewrite + retrieval + answer)

Requires Ollama running with your configured model.

```bash
python -m evals.run_evals --include-llm --model mistral
```

### 4) Generate a large multi-book eval set

Auto-generate hundreds of questions from indexed chunks in all books:

```bash
python -m evals.generate_bulk_cases --per-book 180 --diagram-cases-miller 100
```

This writes:

```text
evals/cases_bulk.json
```

### 5) Read reports

Reports are written to:

```text
evals/reports/eval_report_<timestamp>.json
```

Exit code behavior:
- `0`: all cases passed
- `2`: at least one case failed
- `1`: setup/runtime failure (missing collection/model/dependency, etc.)

Ranking metrics are included in:
- report summary: `summary.ranking`
- each case: `results[].ranking_metrics`

## Troubleshooting

### 1) `ModuleNotFoundError: No module named 'ingestion'`
Run as module from root:

```bash
python -m ingestion.ingestion_pipeline
```

### 2) Chroma metadata errors for list values
`diagram_paths` and similar list metadata are JSON-serialized before insert.

### 3) Diagrams not rendering in UI
- Confirm ingestion completed for target book.
- Confirm retrieval results include non-empty `diagram_paths`.
- Confirm URL pattern is `/gradio_api/file=...`.
- Confirm `allowed_paths` includes `data_bank/diagrams`.

### 4) Embedding model load/network issues
Current code is configured to load embedding model from **local cache** (`local_files_only=True`).
Ensure the model exists in your local transformers cache.

## Notes

- Re-ingesting a book overwrites only that book’s vectors in Chroma (same `book_id`).
- Other books remain intact.
- Runtime artifacts (vectorstore binaries, cache, diagrams, chat history) are intentionally excluded from source control via `.gitignore`.
