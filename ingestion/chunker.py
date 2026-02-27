import tiktoken
from typing import List, Dict


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 150) -> List[str]:
    """
    Split text into token-based chunks.
    """

    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(text)

    chunks = []
    start = 0

    while start < len(tokens):
        end = start + chunk_size
        chunk_tokens = tokens[start:end]
        chunk_text = encoding.decode(chunk_tokens)
        chunks.append(chunk_text)

        start += chunk_size - overlap

    return chunks


def chunk_documents(
    documents: List[Dict],
    chunk_size: int = 800,
    overlap: int = 150
) -> List[Dict]:
    """
    Apply chunking while preserving ALL metadata dynamically.
    Ensures citation metadata remains intact.
    """

    chunked_documents = []

    for doc in documents:
        text_chunks = chunk_text(doc["text"], chunk_size, overlap)

        # Copy all metadata except text
        metadata = {k: v for k, v in doc.items() if k != "text"}

        for idx, chunk in enumerate(text_chunks):
            chunked_documents.append({
                **metadata,
                "chunk_index": idx,
                "text": chunk
            })

    return chunked_documents