"""
OpenAI-compatible LLM provider.

Works with any endpoint that speaks the OpenAI API format:
  - OpenAI proper (GPT-4, etc.)
  - Google Gemini via its OpenAI-compatible gateway
  - LM Studio / vLLM / text-generation-inference
"""

import time
from openai import OpenAI, RateLimitError
from typing import List

from stores.llm.base import BaseLLMProvider


class OpenAIProvider(BaseLLMProvider):
    """LLM Factory: point OPENAI_API_BASE at any OpenAI-compatible endpoint."""

    def __init__(
        self,
        api_key: str,
        api_base: str,
        embeddings_model: str,
        generation_model: str,
        embedding_dimension: int,
        max_response_tokens: int,
        temperature: float,
    ):
        self.client = OpenAI(api_key=api_key, base_url=api_base)
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

    def _embed_with_retry(self, input, retries: int = 5, base_delay: float = 10.0):
        for attempt in range(retries):
            try:
                return self.client.embeddings.create(
                    model=self.embeddings_model,
                    input=input,
                )
            except RateLimitError:
                if attempt == retries - 1:
                    raise
                wait = base_delay * (2 ** attempt)
                print(f"  [rate limit] waiting {wait:.0f}s before retry {attempt + 1}/{retries - 1}...")
                time.sleep(wait)

    def embed_text(self, text: str) -> List[float]:
        response = self._embed_with_retry(text)
        return response.data[0].embedding

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        response = self._embed_with_retry(texts)
        # Sort by index to preserve original order (Gemini returns None for index)
        sorted_data = sorted(response.data, key=lambda x: x.index if x.index is not None else 0)
        return [item.embedding for item in sorted_data]

    def generate(self, prompt: str, max_tokens: int = None) -> str:
        response = self.client.chat.completions.create(
            model=self._generation_model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens or self.max_response_tokens,
            temperature=self.temperature,
        )
        return response.choices[0].message.content.strip()
