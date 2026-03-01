import json
import random
import time
from typing import Generator

import requests
from observability.logging import log_event


class OllamaClient:

    def __init__(
        self,
        model: str = "mistral",
        base_url: str = "http://localhost:11434/api/generate",
        max_retries: int = 3,
        connect_timeout: float = 3.05,
        read_timeout: float = 45.0,
        circuit_fail_threshold: int = 5,
        circuit_cooldown_seconds: int = 30,
    ):
        self.model = model
        self.base_url = base_url
        self.max_retries = max_retries
        self.connect_timeout = connect_timeout
        self.read_timeout = read_timeout
        self.circuit_fail_threshold = circuit_fail_threshold
        self.circuit_cooldown_seconds = circuit_cooldown_seconds

        self._fail_count = 0
        self._circuit_open_until = 0.0

    def _is_circuit_open(self) -> bool:
        return time.time() < self._circuit_open_until

    def _record_success(self) -> None:
        self._fail_count = 0
        self._circuit_open_until = 0.0

    def _record_failure(self) -> None:
        self._fail_count += 1
        if self._fail_count >= self.circuit_fail_threshold:
            self._circuit_open_until = time.time() + self.circuit_cooldown_seconds

    def _retry_delay(self, attempt: int) -> float:
        # Exponential backoff + jitter to reduce synchronized retry spikes.
        base = 0.5 * (2 ** attempt)
        return base + random.uniform(0.0, 0.25 * base)

    def _post_with_retries(self, payload: dict, stream: bool = False) -> requests.Response:
        if self._is_circuit_open():
            log_event("llm_circuit_open", model=self.model)
            raise RuntimeError("LLM circuit is open due to repeated upstream failures.")

        last_error = None
        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    self.base_url,
                    json=payload,
                    stream=stream,
                    timeout=(self.connect_timeout, self.read_timeout),
                )
                response.raise_for_status()
                self._record_success()
                if attempt > 0:
                    log_event("llm_retry_recovered", model=self.model, attempts=attempt + 1)
                return response
            except (requests.Timeout, requests.ConnectionError, requests.HTTPError) as exc:
                last_error = exc
                self._record_failure()
                log_event(
                    "llm_request_retryable_error",
                    model=self.model,
                    attempt=attempt + 1,
                    error=str(exc),
                )
                if attempt < self.max_retries - 1:
                    time.sleep(self._retry_delay(attempt))
                continue
            except requests.RequestException as exc:
                self._record_failure()
                log_event("llm_request_error", model=self.model, error=str(exc))
                raise RuntimeError(f"Ollama request failed: {exc}") from exc

        log_event("llm_request_failed", model=self.model, error=str(last_error))
        raise RuntimeError(f"Ollama request failed after retries: {last_error}")

    def generate(self, prompt: str) -> str:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,
                "num_predict": 200
            }
        }

        response = self._post_with_retries(payload, stream=False)
        try:
            parsed = response.json()
        except ValueError as exc:
            raise RuntimeError("Invalid JSON response from Ollama.") from exc

        return parsed.get("response", "")

    def generate_stream(self, prompt: str) -> Generator[str, None, None]:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": True,
            "options": {
                "num_predict": 1200,
                "temperature": 0.2,
                "top_p": 0.9
            }
        }

        response = self._post_with_retries(payload, stream=True)

        for line in response.iter_lines():
            if not line:
                continue
            try:
                data = json.loads(line.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
            yield data.get("response", "")
