"""
Abstract base class for the LLM Factory Pattern.

The Factory Pattern allows swapping between different LLM backends
(Gemini, OpenAI, Ollama, LM Studio, etc.) by changing only the
configuration—no code changes needed in the rest of the application.

All providers expose the same interface: embed_text, embed_batch, generate.
"""

from abc import ABC, abstractmethod
from typing import List


class BaseLLMProvider(ABC):
    """Unified interface that every LLM provider must implement."""

    @property
    @abstractmethod
    def embedding_dimension(self) -> int:
        """Return the dimensionality of the embedding vectors."""
        ...

    @property
    @abstractmethod
    def generation_model(self) -> str:
        """Return the name of the generation model in use."""
        ...

    @abstractmethod
    def embed_text(self, text: str) -> List[float]:
        """Embed a single piece of text into a vector."""
        ...

    @abstractmethod
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of texts into vectors (preserving order)."""
        ...

    @abstractmethod
    def generate(self, prompt: str, max_tokens: int = None) -> str:
        """Generate a text response for the given prompt."""
        ...
