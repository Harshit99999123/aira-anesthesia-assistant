import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from typing import List, Dict
import uuid
import shutil
import os


class VectorStoreBuilder:

    def __init__(self,
                 persist_directory: str = "vectorstore",
                 collection_name: str = "miller_anesthesia",
                 rebuild: bool = False):

        self.persist_directory = persist_directory
        self.collection_name = collection_name

        if rebuild and os.path.exists(persist_directory):
            print("Rebuild flag enabled. Deleting existing vectorstore...")
            shutil.rmtree(persist_directory)

        self.client = chromadb.Client(
            Settings(
                persist_directory=persist_directory,
                is_persistent=True
            )
        )

        self.collection = self.client.get_or_create_collection(
            name=self.collection_name
        )

        if self.collection.count() > 0 and not rebuild:
            print("Collection already contains data.")
            print("If you want to rebuild, set rebuild=True.")

        print("Loading embedding model...")
        self.model = SentenceTransformer("BAAI/bge-base-en-v1.5")

    def embed_and_store(self, chunked_docs: List[Dict], batch_size: int = 64):

        total = len(chunked_docs)
        print(f"Embedding {total} chunks...")

        for i in range(0, total, batch_size):
            batch = chunked_docs[i:i + batch_size]

            texts = [doc["text"] for doc in batch]
            metadatas = [
                {
                    "book": doc["book"],
                    "volume": doc["volume"],
                    "section": doc["section"],
                    "chapter": doc["chapter"],
                    "heading": doc["heading"],
                    "start_page": doc["start_page"],
                    "end_page": doc["end_page"],
                    "chunk_index": doc["chunk_index"]
                }
                for doc in batch
            ]

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