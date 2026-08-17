"""Embedding model factory.

Separate from get_llm() because Groq serves no embeddings — the chat provider
and the embedding provider are always independent choices here.

Critically, ingestion and serving must use the *same* model: an index built
with one embedding space is meaningless to another. Both sides call this.
"""

from __future__ import annotations

from app.config import Settings, get_settings


def get_embeddings(settings: Settings | None = None):
    settings = settings or get_settings()

    if settings.embedding_provider == "huggingface":
        from langchain_huggingface import HuggingFaceEmbeddings

        return HuggingFaceEmbeddings(
            model_name=settings.hf_embedding_model,
            encode_kwargs={"normalize_embeddings": True},
        )

    from langchain_openai import OpenAIEmbeddings

    if not settings.openai_api_key:
        raise RuntimeError("EMBEDDING_PROVIDER=openai but OPENAI_API_KEY is empty.")

    return OpenAIEmbeddings(
        model=settings.openai_embedding_model,
        api_key=settings.openai_api_key,
    )
