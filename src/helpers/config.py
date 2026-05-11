from pydantic_settings import BaseSettings
from typing import List


class Settings(BaseSettings):
    APP_NAME: str = "LexBridge"
    APP_VERSION: str = "1.0.0"

    FILE_ALLOWED_EXTENSIONS: List[str] = [
        "text/plain",
        "application/pdf",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",  # .docx
    ]
    FILE_MAX_SIZE_MB: int = 10
    FILE_CHUNK_SIZE: int = 512

    MONGODB_URI: str = "mongodb://localhost:27017"
    MONGODB_DB_NAME: str = "lexbridge"

    # ── LLM Factory Pattern ─────────────────────────────────────────────────
    # Set LLM_BACKEND to switch providers with zero code changes:
    #   "openai"  → OpenAI-compatible endpoint (Gemini, OpenAI, LM Studio)
    #   "ollama"  → Local Ollama server
    LLM_BACKEND: str = "openai"

    # OpenAI-compatible settings (used when LLM_BACKEND="openai")
    OPENAI_API_KEY: str = ""
    OPENAI_API_BASE: str = "https://generativelanguage.googleapis.com/v1beta/openai/"
    GENERATE_RESPONSE_MODEL: str = "models/gemini-2.5-flash"
    EMBEDDINGS_MODEL: str = "models/gemini-embedding-001"
    EMBEDDING_DIMENSION: int = 3072
    MAX_INPUT_TOKENS: int = 4096
    MAX_RESPONSE_TOKENS: int = 2048
    TEMPERATURE: float = 0.1

    # Groq settings (used when LLM_BACKEND="groq")
    # Embeddings still route through OPENAI_API_KEY/OPENAI_API_BASE (Gemini)
    # because Groq has no embedding models.
    # Model options: qwen/qwen3-32b | llama-3.3-70b-versatile | llama-3.1-8b-instant | meta-llama/llama-4-scout-17b-16e-instruct
    GROQ_API_KEY: str = ""
    GROQ_API_BASE: str = "https://api.groq.com/openai/v1"
    GROQ_GENERATE_MODEL: str = "qwen/qwen3-32b"

    # Ollama settings (used when LLM_BACKEND="ollama")
    OLLAMA_BASE_URL: str = "http://localhost:11434"

    # Vector store
    VECTOR_DB_PATH: str = "assets/db/qdrant_data"
    VECTOR_DISTANCE_METRIC: str = "cosine"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


def get_settings() -> Settings:
    return Settings()
