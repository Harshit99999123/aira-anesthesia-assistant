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
