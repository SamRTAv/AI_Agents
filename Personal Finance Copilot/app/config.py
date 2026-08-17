"""Central configuration.

Every environment-dependent choice lives here so that swapping an LLM provider,
an embedding backend or a storage path never requires touching graph code.
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ---- LLM ----
    llm_provider: Literal["groq", "openai"] = "groq"
    groq_api_key: str | None = None
    groq_model: str = "llama-3.3-70b-versatile"
    openai_api_key: str | None = None
    openai_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.0

    # ---- Embeddings (phase 4) ----
    embedding_provider: Literal["huggingface", "openai"] = "huggingface"
    hf_embedding_model: str = "BAAI/bge-small-en-v1.5"
    openai_embedding_model: str = "text-embedding-3-small"

    # ---- Storage ----
    corpus_dir: Path = BASE_DIR / "data" / "corpus"
    index_dir: Path = BASE_DIR / "data" / "faiss_index"
    checkpoint_db: Path = BASE_DIR / "checkpoints.db"

    # ---- Observability ----
    # Read by the langsmith SDK directly; mirrored here so /health can report it.
    langsmith_tracing: bool = False
    langsmith_api_key: str | None = None
    langsmith_project: str = "personal-finance-copilot"

    @property
    def active_model_name(self) -> str:
        return self.groq_model if self.llm_provider == "groq" else self.openai_model

    def require_llm_key(self) -> None:
        """Fail loudly at startup rather than on the first user message."""
        if self.llm_provider == "groq" and not self.groq_api_key:
            raise RuntimeError(
                "LLM_PROVIDER=groq but GROQ_API_KEY is empty. "
                "Get a free key at https://console.groq.com/keys and put it in .env"
            )
        if self.llm_provider == "openai" and not self.openai_api_key:
            raise RuntimeError(
                "LLM_PROVIDER=openai but OPENAI_API_KEY is empty. Set it in .env"
            )


@lru_cache
def get_settings() -> Settings:
    return Settings()


def get_llm(settings: Settings | None = None):
    """Build the chat model.

    Both providers expose the same tool-calling interface, so everything
    downstream (nodes, graph, tools) is provider-agnostic.
    """
    settings = settings or get_settings()
    settings.require_llm_key()

    if settings.llm_provider == "groq":
        from langchain_groq import ChatGroq

        return ChatGroq(
            model=settings.groq_model,
            temperature=settings.llm_temperature,
            api_key=settings.groq_api_key,
        )

    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=settings.openai_model,
        temperature=settings.llm_temperature,
        api_key=settings.openai_api_key,
    )
