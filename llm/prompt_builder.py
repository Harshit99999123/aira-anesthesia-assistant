from typing import List, Dict


class PromptBuilder:

    SYSTEM_INSTRUCTIONS = """
You are AIRA, an anesthesia resident assistant.

STRICT RULES:
- Answer ONLY using the provided context from the indexed medical sources.
- If the answer is not present in the context, say:
  "The answer is not available in the indexed medical sources."
- Do NOT use outside knowledge.
- Do NOT speculate.
- Use ALL relevant details from the provided context.
- Provide a comprehensive, structured explanation.
- Write in clear academic medical language.
- Include mechanisms, physiological explanations, and clinical implications if present in the context.
- Aim for a detailed answer of approximately one page if the context supports it.
"""

    @staticmethod
    def _format_citation(metadata: Dict) -> str:
        """
        Dynamically render citation from hierarchy.
        """

        book_name = metadata.get("book_name", "Unknown Source")
        hierarchy = metadata.get("hierarchy", [])

        hierarchy_str = " → ".join(hierarchy) if hierarchy else "Untitled Section"

        start_page = metadata.get("start_page", "?")
        end_page = metadata.get("end_page", "?")

        return (
            f"{book_name}\n"
            f"{hierarchy_str}\n"
            f"(Pages {start_page}-{end_page})"
        )

    @classmethod
    def build_context(cls, chunks: List[Dict]) -> str:
        formatted_chunks = []

        for chunk in chunks:
            metadata = chunk["metadata"]

            citation = cls._format_citation(metadata)

            formatted_chunks.append(
                f"[SOURCE]\n{citation}\n\n{chunk['text']}\n"
            )

        return "\n\n".join(formatted_chunks)

    @classmethod
    def build_prompt(cls, query: str, chunks: List[Dict]) -> str:

        context = cls.build_context(chunks)

        return f"""
{cls.SYSTEM_INSTRUCTIONS}

CONTEXT:
{context}

QUESTION:
{query}

ANSWER (Provide a detailed, structured explanation):
"""