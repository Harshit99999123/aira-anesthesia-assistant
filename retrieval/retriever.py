import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from typing import Dict, Optional, List


class Retriever:

    def __init__(
        self,
        persist_directory: str = "vectorstore",
        collection_name: str = "medical_knowledge",
        similarity_threshold: float = 0.38,
        top_k: int = 5,
        min_support_chunks: int = 1,
        min_avg_similarity: float = 0.0,
    ):

        self.similarity_threshold = similarity_threshold
        self.top_k = top_k
        self.min_support_chunks = max(1, min_support_chunks)
        self.min_avg_similarity = min_avg_similarity

        self.client = chromadb.Client(
            Settings(
                persist_directory=persist_directory,
                is_persistent=True
            )
        )

        self.collection = self.client.get_collection(name=collection_name)

        print("Loading embedding model for retrieval (local cache)...")
        self.model = SentenceTransformer(
            "BAAI/bge-base-en-v1.5",
            local_files_only=True
        )

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

        support_check = self._evaluate_context_support(filtered_results)
        if support_check["status"] == "refused":
            return support_check

        return {
            "status": "success",
            "max_similarity": filtered_results[0]["similarity"],
            "avg_similarity": sum(item["similarity"] for item in filtered_results) / len(filtered_results),
            "supporting_chunks": len(filtered_results),
            "results": filtered_results
        }

    def list_book_ids(self) -> List[str]:
        rows = self.collection.get(include=["metadatas"])
        metadatas = rows.get("metadatas") or []
        seen = set()
        ordered = []
        for metadata in metadatas:
            book_id = metadata.get("book_id")
            if isinstance(book_id, str):
                value = book_id.strip()
                if value and value not in seen:
                    seen.add(value)
                    ordered.append(value)
        return ordered

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

    # --------------------------------------------------
    # Context Sufficiency Gate
    # --------------------------------------------------

    def _evaluate_context_support(self, results: List[Dict]) -> Dict:
        chunk_count = len(results)
        avg_similarity = sum(item["similarity"] for item in results) / chunk_count

        if chunk_count < self.min_support_chunks:
            return {
                "status": "refused",
                "message": (
                    "The retrieved context is too limited to support a reliable answer "
                    "from the indexed medical sources."
                ),
                "reason": "insufficient_supporting_chunks",
                "supporting_chunks": chunk_count,
                "avg_similarity": avg_similarity,
            }

        if avg_similarity < self.min_avg_similarity:
            return {
                "status": "refused",
                "message": (
                    "The retrieved context is not strong enough to support a reliable answer "
                    "from the indexed medical sources."
                ),
                "reason": "insufficient_average_similarity",
                "supporting_chunks": chunk_count,
                "avg_similarity": avg_similarity,
            }

        return {"status": "success"}
