from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """
    Central configuration for the RAG chatbot.

    Values are loaded from environment variables / .env file when available.
    """

    # Base paths
    base_dir: Path = Field(default_factory=lambda: Path(__file__).resolve().parent.parent)

    # OpenAI / LLM
    openai_api_key: str = Field(..., env="OPENAI_API_KEY")
    openai_model: str = Field("gpt-4o", env="OPENAI_MODEL_NAME")
    openai_temperature: float = Field(0.2, env="OPENAI_TEMPERATURE")
    openai_max_tokens: int = Field(768, env="OPENAI_MAX_TOKENS")

    # Pinecone
    pinecone_api_key: str = Field(..., env="PINECONE_API_KEY")
    pinecone_index_name: str = Field("therapy-rag", env="PINECONE_INDEX_NAME")
    pinecone_index_host: str | None = Field(None, env="PINECONE_INDEX_HOST")
    pinecone_namespace: str = Field("default", env="PINECONE_NAMESPACE")

    # Embeddings (runtime retrieval must match indexing)
    embedding_model_name: str = Field("BAAI/bge-m3", env="EMBEDDING_MODEL_NAME")

    # Dense retrieval behavior
    dense_top_k: int = Field(5, env="DENSE_TOP_K")
    dense_score_threshold: float = Field(0.4, env="DENSE_SCORE_THRESHOLD")

    # Domain / safety flags (logic is in prompts + code, these are just toggles)
    domain_name: str = Field("פסיכולוגיה ובריאות נפשית", env="DOMAIN_NAME")
    language: Literal["he"] = "he"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Cached accessor so we don't repeatedly parse environment variables.
    """

    return Settings()

