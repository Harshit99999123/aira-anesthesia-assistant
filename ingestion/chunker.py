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


def chunk_documents(documents: List[Dict], chunk_size: int = 800, overlap: int = 150) -> List[Dict]:
    """
    Apply chunking to enriched documents.
    Each chunk keeps full metadata.
    """

    chunked_documents = []

    for doc in documents:
        chunks = chunk_text(doc["text"], chunk_size, overlap)

        for idx, chunk in enumerate(chunks):
            chunked_documents.append({
                "book": doc["book"],
                "volume": doc["volume"],
                "section": doc["section"],
                "chapter": doc["chapter"],
                "heading": doc["heading"],
                "start_page": doc["start_page"],
                "end_page": doc["end_page"],
                "chunk_index": idx,
                "text": chunk
            })

    return chunked_documents