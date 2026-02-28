import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from typing import List, Dict
import uuid
import json


class VectorStoreBuilder:

    def __init__(
        self,
        persist_directory: str = "vectorstore",
        collection_name: str = "medical_knowledge",
    ):

        self.client = chromadb.Client(
            Settings(
                persist_directory=persist_directory,
                is_persistent=True
            )
        )

        self.collection = self.client.get_or_create_collection(
            name=collection_name
        )

        print("Loading embedding model (local cache)...")
        self.model = SentenceTransformer(
            "BAAI/bge-base-en-v1.5",
            local_files_only=True
        )

    # --------------------------------------------------
    # Delete existing book before re-ingesting
    # --------------------------------------------------

    def delete_book(self, book_id: str):
        print(f"Deleting existing chunks for book_id={book_id} (if any)...")
        self.collection.delete(
            where={"book_id": book_id}
        )

    # --------------------------------------------------
    # Embed and store
    # --------------------------------------------------

    def embed_and_store(
        self,
        chunked_docs: List[Dict],
        book_id: str,
        book_name: str,
        batch_size: int = 64
    ):

        # Step 1: Remove old version of this book
        self.delete_book(book_id)

        total = len(chunked_docs)
        print(f"Embedding {total} chunks for book_id={book_id}...")

        for i in range(0, total, batch_size):
            batch = chunked_docs[i:i + batch_size]

            texts = [doc["text"] for doc in batch]

            # Preserve all metadata dynamically
            metadatas = []
            for doc in batch:
                metadata = {}
                for key, value in doc.items():
                    if key == "text":
                        continue
                    # Chroma rejects empty list metadata values.
                    # Serialize all lists to JSON strings for stable storage.
                    if isinstance(value, list):
                        metadata[key] = json.dumps(value)
                    else:
                        metadata[key] = value
                metadata["book_id"] = book_id
                metadata["book_name"] = book_name
                metadatas.append(metadata)

            embeddings = self.model.encode(texts, show_progress_bar=False)

            ids = [str(uuid.uuid4()) for _ in batch]

            self.collection.add(
                documents=texts,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids
            )

            print(f"Processed {min(i + batch_size, total)} / {total}")

        print("Embedding complete.")
