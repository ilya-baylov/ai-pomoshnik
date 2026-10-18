"""Выбор провайдера эмбеддингов по настройке."""

from __future__ import annotations

import functools
from urllib.parse import urlparse

from app.core.config import get_settings
from app.embeddings.base import EmbeddingProvider


@functools.lru_cache(maxsize=1)
def get_embedding_provider() -> EmbeddingProvider:
    settings = get_settings()
    profile = settings.profile()

    if settings.embedding_provider == "local":
        from app.embeddings.local import LocalEmbeddings

        return LocalEmbeddings(profile, cache_dir=str(settings.db_path.parent / "models"))

    if settings.embedding_provider == "openai":
        # Любой OpenAI-совместимый сервис: OpenAI, OpenRouter, Ollama…
        from app.embeddings.openai_compat import OpenAIEmbeddings

        base_url = settings.embedding_base_url or "https://api.openai.com/v1"
        key = settings.embedding_api_key
        if not key and urlparse(base_url).hostname == "api.openai.com":
            key = settings.openai_api_key
        if not key and urlparse(base_url).hostname == urlparse(settings.llm_base_url).hostname:
            key = settings.llm_api_key
        return OpenAIEmbeddings(
            profile, base_url=base_url, api_key=key
        )

    if settings.embedding_provider == "gemini":
        from app.embeddings.gemini import GeminiEmbeddings

        if settings.service_account_path:
            # Сервисный аккаунт приоритетнее: своя квота проекта Vertex AI
            return GeminiEmbeddings(
                profile, api_keys=[], service_account_path=str(settings.service_account_path)
            )
        return GeminiEmbeddings(profile, api_keys=settings.embedding_keys)

    raise ValueError(
        f"Неизвестный EMBEDDING_PROVIDER: {settings.embedding_provider}. "
        "Допустимо: gemini, openai, local"
    )
