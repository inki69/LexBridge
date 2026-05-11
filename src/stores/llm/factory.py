"""
LLM Factory — returns the correct provider based on configuration.

Usage:
    provider = LLMFactory.create(settings)

Supported backends (selected via LLM_BACKEND env var):
    "openai"  → OpenAI-compatible endpoint (Gemini, OpenAI, LM Studio, vLLM)
    "groq"    → Groq Cloud generation + Gemini embeddings (Groq has no embed models)
    "ollama"  → Local Ollama server (free, fully offline)
"""

from helpers.config import Settings
from stores.llm.base import BaseLLMProvider


class LLMFactory:
    """
    Factory that instantiates the appropriate LLM provider.

    Implements the GoF Factory Method pattern: the concrete class is resolved
    at runtime based on a configuration string, keeping the rest of the
    codebase decoupled from any specific LLM SDK.
    """

    @staticmethod
    def create(settings: Settings) -> BaseLLMProvider:
        backend = getattr(settings, "LLM_BACKEND", "openai").lower()

        if backend == "groq":
            from stores.llm.providers.groq_provider import GroqProvider
            return GroqProvider(
                groq_api_key=settings.GROQ_API_KEY,
                groq_api_base=settings.GROQ_API_BASE,
                generation_model=settings.GROQ_GENERATE_MODEL,
                embed_api_key=settings.OPENAI_API_KEY,
                embed_api_base=settings.OPENAI_API_BASE,
                embeddings_model=settings.EMBEDDINGS_MODEL,
                embedding_dimension=settings.EMBEDDING_DIMENSION,
                max_response_tokens=settings.MAX_RESPONSE_TOKENS,
                temperature=settings.TEMPERATURE,
            )

        if backend == "ollama":
            from stores.llm.providers.ollama_provider import OllamaProvider
            return OllamaProvider(
                base_url=settings.OLLAMA_BASE_URL,
                embeddings_model=settings.EMBEDDINGS_MODEL,
                generation_model=settings.GENERATE_RESPONSE_MODEL,
                embedding_dimension=settings.EMBEDDING_DIMENSION,
                max_response_tokens=settings.MAX_RESPONSE_TOKENS,
                temperature=settings.TEMPERATURE,
            )

        # Default: OpenAI-compatible endpoint (Gemini, OpenAI, LM Studio, etc.)
        from stores.llm.providers.openai_provider import OpenAIProvider
        return OpenAIProvider(
            api_key=settings.OPENAI_API_KEY,
            api_base=settings.OPENAI_API_BASE,
            embeddings_model=settings.EMBEDDINGS_MODEL,
            generation_model=settings.GENERATE_RESPONSE_MODEL,
            embedding_dimension=settings.EMBEDDING_DIMENSION,
            max_response_tokens=settings.MAX_RESPONSE_TOKENS,
            temperature=settings.TEMPERATURE,
        )
