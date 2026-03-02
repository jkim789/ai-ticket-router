"""
Application configuration using Pydantic BaseSettings.

All configuration is loaded from environment variables with sensible defaults.
"""

from typing import Literal

from groq import AsyncGroq
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # LLM Configuration
    GROQ_API_KEY: str  # Groq API key (free tier)
    OPENAI_API_KEY: str  # For embeddings

    # Application
    APP_ENV: Literal["development", "production"] = "development"
    APP_PORT: int = 8000
    APP_HOST: str = "0.0.0.0"
    LOG_LEVEL: str = "INFO"

    # ChromaDB
    # Default to "chromadb" (Docker service name). For local embedded Chroma,
    # override this in your .env with CHROMA_HOST=127.0.0.1 or localhost.
    CHROMA_HOST: str = "chromadb"
    CHROMA_PORT: int = 8000
    CHROMA_COLLECTION_NAME: str = "novatech_knowledge_base"

    # Agent Configuration
    CONFIDENCE_THRESHOLD: float = 0.75
    MAX_KB_RESULTS: int = 5
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    LLM_MODEL: str = "llama-3.3-70b-versatile"

    # Rate Limiting
    RATE_LIMIT_RPM: int = 10  # Max requests per minute per IP

    # Database
    DATABASE_URL: str = "sqlite+aiosqlite:///./data/tickets.db"

    class Config:
        """Pydantic config."""
        env_file = ".env"
        case_sensitive = True


# Global settings instance
settings = Settings()


def get_llm_client() -> AsyncGroq:
    """
    Create a Groq async client for LLM API calls.

    Returns:
        Configured AsyncGroq client
    """
    return AsyncGroq(api_key=settings.GROQ_API_KEY)
