"""
Ollama LLM provider — fully local, free, offline inference.

Requires a running Ollama server (https://ollama.com).
Default URL: http://localhost:11434

Models to pull:
    ollama pull nomic-embed-text   # embeddings (768-dim)
    ollama pull mistral            # generation
"""

import requests
from typing import List

from stores.llm.base import BaseLLMProvider


class OllamaProvider(BaseLLMProvider):
    """Local LLM inference via Ollama REST API."""

    def __init__(
        self,
        base_url: str,
        embeddings_model: str,
        generation_model: str,
        embedding_dimension: int,
        max_response_tokens: int,
        temperature: float,
    ):
        self.base_url = base_url.rstrip("/")
        self.embeddings_model = embeddings_model
        self._generation_model = generation_model
        self._embedding_dimension = embedding_dimension
        self.max_response_tokens = max_response_tokens
        self.temperature = temperature

    # ── BaseLLMProvider interface ────────────────────────────────────────────

    @property
    def embedding_dimension(self) -> int:
        return self._embedding_dimension

    @property
    def generation_model(self) -> str:
        return self._generation_model

    def embed_text(self, text: str) -> List[float]:
        resp = requests.post(
            f"{self.base_url}/api/embed",
            json={"model": self.embeddings_model, "input": text},
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()["embeddings"][0]

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        resp = requests.post(
            f"{self.base_url}/api/embed",
            json={"model": self.embeddings_model, "input": texts},
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()["embeddings"]

    def generate(self, prompt: str, max_tokens: int = None) -> str:
        resp = requests.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self._generation_model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_predict": max_tokens or self.max_response_tokens,
                    "temperature": self.temperature,
                },
            },
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()["response"].strip()
