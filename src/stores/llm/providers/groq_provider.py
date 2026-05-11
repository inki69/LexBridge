"""
Groq LLM provider — fast cloud inference via Groq Cloud API.

Groq does not offer embedding models, so this provider is split:
  - Generation  → Groq Cloud  (llama-3.3-70b-versatile, mixtral, etc.)
  - Embeddings  → Gemini / any OpenAI-compatible endpoint you already have

This keeps the corpus vectors consistent: if you indexed with Gemini 3072-dim
embeddings, queries must also be embedded by Gemini — not Groq.

Set LLM_BACKEND=groq in .env and add your GROQ_API_KEY to switch.
"""

import time
from openai import OpenAI, RateLimitError
from typing import List

from stores.llm.base import BaseLLMProvider


class GroqProvider(BaseLLMProvider):
    """Groq for generation + OpenAI-compatible endpoint for embeddings."""

    def __init__(
        self,
        groq_api_key: str,
        groq_api_base: str,
        generation_model: str,
        embed_api_key: str,
        embed_api_base: str,
        embeddings_model: str,
        embedding_dimension: int,
        max_response_tokens: int,
        temperature: float,
    ):
        self.gen_client = OpenAI(api_key=groq_api_key, base_url=groq_api_base)
        self.embed_client = OpenAI(api_key=embed_api_key, base_url=embed_api_base)
        self._generation_model = generation_model
        self.embeddings_model = embeddings_model
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

    # ── Embeddings (Gemini / OpenAI-compatible) ──────────────────────────────

    def _embed_with_retry(self, input, retries: int = 5, base_delay: float = 10.0):
        for attempt in range(retries):
            try:
                return self.embed_client.embeddings.create(
                    model=self.embeddings_model,
                    input=input,
                )
            except RateLimitError:
                if attempt == retries - 1:
                    raise
                wait = base_delay * (2 ** attempt)
                print(f"  [embed rate limit] waiting {wait:.0f}s before retry {attempt + 1}/{retries - 1}...")
                time.sleep(wait)

    def embed_text(self, text: str) -> List[float]:
        response = self._embed_with_retry(text)
        return response.data[0].embedding

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        response = self._embed_with_retry(texts)
        sorted_data = sorted(response.data, key=lambda x: x.index if x.index is not None else 0)
        return [item.embedding for item in sorted_data]

    # ── Generation (Groq) ────────────────────────────────────────────────────

    def generate(self, prompt: str, max_tokens: int = None) -> str:
        response = self.gen_client.chat.completions.create(
            model=self._generation_model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens or self.max_response_tokens,
            temperature=self.temperature,
        )
        return response.choices[0].message.content.strip()
