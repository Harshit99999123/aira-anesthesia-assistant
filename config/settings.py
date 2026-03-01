import os
from dataclasses import dataclass
from typing import List, Optional


def _get_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    value = raw.strip().lower()
    return value in {"1", "true", "yes", "on"}


def _get_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw.strip())
    except ValueError:
        return default


def _get_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw.strip())
    except ValueError:
        return default


def _get_csv(name: str) -> Optional[List[str]]:
    raw = os.getenv(name)
    if raw is None:
        return None
    values = [item.strip() for item in raw.split(",")]
    values = [item for item in values if item]
    return values or None


@dataclass(frozen=True)
class AppSettings:
    ollama_model: str
    retriever_top_k: int
    similarity_threshold: float
    chroma_persist_directory: str
    chroma_collection_name: str
    gradio_share: bool
    whisper_model: str
    structured_logs: bool
    allowed_book_ids: Optional[List[str]]


def load_settings() -> AppSettings:
    return AppSettings(
        ollama_model=os.getenv("OLLAMA_MODEL", "mistral"),
        retriever_top_k=_get_int("RETRIEVER_TOP_K", 8),
        similarity_threshold=_get_float("SIMILARITY_THRESHOLD", 0.35),
        chroma_persist_directory=os.getenv("CHROMA_PERSIST_DIRECTORY", "vectorstore"),
        chroma_collection_name=os.getenv("CHROMA_COLLECTION_NAME", "medical_knowledge"),
        gradio_share=_get_bool("GRADIO_SHARE", True),
        whisper_model=os.getenv("WHISPER_MODEL", "base"),
        structured_logs=_get_bool("STRUCTURED_LOGS", True),
        allowed_book_ids=_get_csv("ALLOWED_BOOK_IDS"),
    )


settings = load_settings()
