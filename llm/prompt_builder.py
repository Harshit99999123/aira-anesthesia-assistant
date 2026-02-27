from typing import List, Dict


class PromptBuilder:

    SYSTEM_INSTRUCTIONS = """
    You are AIRA, an anesthesia resident assistant.

    STRICT RULES:
    - Answer ONLY using the provided context from Miller Anesthesia.
    - If the answer is not present in the context, say:
      "The answer is not available in Miller Anesthesia."
    - Do NOT use outside knowledge.
    - Do NOT speculate.
    - Use ALL relevant details from the provided context.
    - Provide a comprehensive, structured explanation.
    - Write in clear academic medical language.
    - Include mechanisms, physiological explanations, and clinical implications if present in the context.
    - Aim for a detailed answer of approximately one page if the context supports it.
    """

    @staticmethod
    def build_context(chunks: List[Dict]) -> str:
        formatted_chunks = []

        for chunk in chunks:
            metadata = chunk["metadata"]

            citation = (
                f"{metadata['volume']} | "
                f"{metadata['section']} | "
                f"{metadata['chapter']} | "
                f"{metadata['heading']} "
                f"(Pages {metadata['start_page']}-{metadata['end_page']})"
            )

            formatted_chunks.append(
                f"[SOURCE]\n{citation}\n{chunk['text']}\n"
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