import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from typing import List, Dict


class Retriever:

    def __init__(self,
                 persist_directory: str = "vectorstore",
                 collection_name: str = "miller_anesthesia",
                 similarity_threshold: float = 0.4,
                 top_k: int = 5):

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

    def _expand_query(self, query: str) -> str:
        """
        Expand short medical queries to improve embedding stability.
        """
        if len(query.split()) <= 6:
            return f"Provide detailed explanation including dosage, mechanism, and clinical use: {query}"
        return query

    def retrieve(self, query: str) -> Dict:

        query = self._expand_query(query)

        query_embedding = self.model.encode([query])[0]

        # Retrieve more candidates for stability
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=12
        )

        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        similarities = [1 - d for d in distances]

        # If ALL candidates below threshold → refuse
        if max(similarities) < 0.38:
            return {
                "status": "refused",
                "message": "I cannot find relevant information in Miller Anesthesia.",
                "similarity": max(similarities)
            }

        # Sort by similarity descending
        sorted_results = sorted(
            zip(documents, metadatas, similarities),
            key=lambda x: x[2],
            reverse=True
        )

        filtered_results = []

        for doc, metadata, sim in sorted_results:

            if sim < 0.38:
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
                "message": "I cannot find relevant information in Miller Anesthesia.",
                "similarity": max(similarities)
            }

        return {
            "status": "success",
            "max_similarity": filtered_results[0]["similarity"],
            "results": filtered_results
        }

    def _is_reference_chunk(self, text: str) -> bool:
        """
        Detect likely bibliography/reference chunk.
        """

        # Many years
        year_count = sum(text.count(str(y)) for y in range(1950, 2025))
        if year_count > 5:
            return True

        # Too many semicolons (common in citations)
        if text.count(";") > 8:
            return True

        # Very short average sentence length
        sentences = text.split(".")
        if len(sentences) > 10:
            avg_len = sum(len(s.split()) for s in sentences) / len(sentences)
            if avg_len < 6:
                return True

        return False