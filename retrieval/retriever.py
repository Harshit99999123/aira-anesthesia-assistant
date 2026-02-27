import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from typing import Dict, Optional


class Retriever:

    def __init__(
        self,
        persist_directory: str = "vectorstore",
        collection_name: str = "medical_knowledge",
        similarity_threshold: float = 0.38,
        top_k: int = 5
    ):

        self.similarity_threshold = similarity_threshold
        self.top_k = top_k

        self.client = chromadb.Client(
            Settings(
                persist_directory=persist_directory,
                is_persistent=True
            )
        )

        self.collection = self.client.get_collection(name=collection_name)

        print("Loading embedding model for retrieval...")
        self.model = SentenceTransformer("BAAI/bge-base-en-v1.5")

    # --------------------------------------------------
    # Query Expansion
    # --------------------------------------------------

    def _expand_query(self, query: str) -> str:
        if len(query.split()) <= 6:
            return f"Provide detailed explanation including dosage, mechanism, and clinical use: {query}"
        return query

    # --------------------------------------------------
    # Retrieval
    # --------------------------------------------------

    def retrieve(self, query: str, book_id: Optional[str] = None) -> Dict:

        query = self._expand_query(query)
        query_embedding = self.model.encode([query])[0]

        query_kwargs = {
            "query_embeddings": [query_embedding],
            "n_results": 12
        }

        # If specific book requested → filter
        if book_id:
            query_kwargs["where"] = {"book_id": book_id}

        results = self.collection.query(**query_kwargs)

        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        similarities = [1 - d for d in distances]

        if not similarities or max(similarities) < self.similarity_threshold:
            return {
                "status": "refused",
                "message": "I cannot find sufficiently relevant information in the indexed medical sources.",
                "similarity": max(similarities) if similarities else 0.0
            }

        sorted_results = sorted(
            zip(documents, metadatas, similarities),
            key=lambda x: x[2],
            reverse=True
        )

        filtered_results = []

        for doc, metadata, sim in sorted_results:

            if sim < self.similarity_threshold:
                continue

            if self._is_reference_chunk(doc):
                continue

            filtered_results.append({
                "text": doc,
                "metadata": metadata,
                "similarity": sim
            })

            if len(filtered_results) >= self.top_k:
                break

        if not filtered_results:
            return {
                "status": "refused",
                "message": "I cannot find sufficiently relevant information in the indexed medical sources.",
                "similarity": max(similarities)
            }

        return {
            "status": "success",
            "max_similarity": filtered_results[0]["similarity"],
            "results": filtered_results
        }

    # --------------------------------------------------
    # Reference Filter
    # --------------------------------------------------

    def _is_reference_chunk(self, text: str) -> bool:

        year_count = sum(text.count(str(y)) for y in range(1950, 2025))
        if year_count > 5:
            return True

        if text.count(";") > 8:
            return True

        sentences = text.split(".")
        if len(sentences) > 10:
            avg_len = sum(len(s.split()) for s in sentences) / len(sentences)
            if avg_len < 6:
                return True

        return False