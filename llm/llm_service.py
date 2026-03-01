from llm.prompt_builder import PromptBuilder
from llm.ollama_client import OllamaClient


class LLMService:

    def __init__(self, model: str = "mistral"):
        self.client = OllamaClient(model=model)

    def generate_answer_stream(self, query: str, retrieval_response):

        if retrieval_response["status"] == "refused":
            yield retrieval_response["message"]
            return

        prompt = PromptBuilder.build_prompt(
            query,
            retrieval_response["results"]
        )

        try:
            emitted = False
            for token in self.client.generate_stream(prompt):
                emitted = True
                yield token

            if not emitted:
                yield (
                    "I’m unable to generate a response right now due to an upstream model issue. "
                    "Please try again."
                )
        except Exception:
            yield (
                "I’m unable to generate a response right now due to an upstream model issue. "
                "Please try again."
            )
