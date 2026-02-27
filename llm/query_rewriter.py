from llm.ollama_client import OllamaClient


class QueryRewriter:

    def __init__(self, model: str = "mistral"):
        self.client = OllamaClient(model=model)

    def rewrite(self, message: str, history: list) -> str:
        """
        Rewrite follow-up queries into standalone medical questions.
        """

        # Extract previous user message
        previous_user_msgs = [
            msg["content"]
            for msg in history
            if msg["role"] == "user"
        ]

        if not previous_user_msgs:
            return message

        previous_question = previous_user_msgs[-1]

        rewrite_prompt = f"""
You are a medical query rewriting assistant.

Your task is to rewrite follow-up questions into
standalone medical questions.

Previous user question:
{previous_question}

Follow-up question:
{message}

You must ONLY replace ambiguous references (such as "it", "that", "the above").
Do NOT add new medical information.
Do NOT add assumptions.
Do NOT change the meaning.
If the question is already standalone, return it unchanged.

Rewritten standalone question:
"""

        rewritten = self.client.generate(rewrite_prompt)

        return rewritten.strip()