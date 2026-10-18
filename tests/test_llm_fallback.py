"""Логика резервного провайдера LLM.

Проверяется выбор цепочки и её ключа. Сетевые вызовы не тестируются —
только конфигурация, из которой они собираются.
"""

from __future__ import annotations

import pytest

from app.core.config import Settings
from app.generation.llm import fallback_chain, primary_chain


def make_settings(monkeypatch, **overrides) -> Settings:
    values = {
        "LLM_API_KEY": "primary-key",
        "LLM_BASE_URL": "https://openrouter.ai/api/v1",
        "LLM_MODEL": "z-ai/glm-5.2:free",
        "EMBEDDING_PROVIDER": "gemini",
        "EMBEDDING_MODEL": "gemini-embedding-001",
        "EMBEDDING_DIM": "1536",
        "LLM_FALLBACK_ENABLED": "true",
    }
    values.update(overrides)
    for key, value in values.items():
        monkeypatch.setenv(key, str(value))
    return Settings(_env_file=None)


def test_fallback_is_gemini_on_embeddings_key(monkeypatch):
    """Связка OpenRouter + Gemini: резерв берёт ключ эмбеддингов."""
    settings = make_settings(monkeypatch, EMBEDDING_API_KEY="gemini-key")

    fallback = fallback_chain(settings)
    assert fallback is not None
    assert "generativelanguage.googleapis.com" in fallback.base_url
    assert fallback.api_key == "gemini-key"
    assert fallback.model == "gemini-3.6-flash"


def test_fallback_skipped_when_same_provider(monkeypatch):
    """Основная модель уже Gemini — резерв в ту же квоту бессмыслен."""
    settings = make_settings(
        monkeypatch,
        LLM_BASE_URL="https://generativelanguage.googleapis.com/v1beta/openai/",
        LLM_MODEL="gemini-3.6-flash",
    )
    assert fallback_chain(settings) is None


def test_fallback_disabled_by_flag(monkeypatch):
    settings = make_settings(monkeypatch, EMBEDDING_API_KEY="gemini-key",
                             LLM_FALLBACK_ENABLED="false")
    assert fallback_chain(settings) is None


def test_fallback_explicit_key_wins(monkeypatch):
    settings = make_settings(
        monkeypatch,
        EMBEDDING_API_KEY="gemini-key",
        LLM_FALLBACK_API_KEY="own-key",
        LLM_FALLBACK_MODEL="gemini-3.5-flash",
    )
    fallback = fallback_chain(settings)
    assert fallback is not None
    assert fallback.api_key == "own-key"
    assert fallback.model == "gemini-3.5-flash"


def test_fallback_needs_some_key(monkeypatch):
    """OpenRouter без EMBEDDING_API_KEY: ключа для резерва нет."""
    settings = make_settings(monkeypatch, EMBEDDING_API_KEY="")
    assert fallback_chain(settings) is None


def test_fallback_model_cannot_be_empty(monkeypatch):
    settings = make_settings(
        monkeypatch, EMBEDDING_API_KEY="gemini-key", LLM_FALLBACK_MODEL=""
    )
    assert fallback_chain(settings) is None


def test_primary_chain_reflects_settings(monkeypatch):
    settings = make_settings(monkeypatch)
    primary = primary_chain(settings)
    assert primary.base_url == "https://openrouter.ai/api/v1"
    assert primary.model == "z-ai/glm-5.2:free"


def test_gemini_llm_key_reaches_embeddings(monkeypatch):
    """Дефолтная связка: один ключ Gemini обслуживает и LLM, и эмбеддинги."""
    settings = make_settings(
        monkeypatch,
        LLM_API_KEY="shared-key",
        LLM_BASE_URL="https://generativelanguage.googleapis.com/v1beta/openai/",
    )
    assert settings.embeddings_key == "shared-key"


def test_rate_limit_on_primary_goes_to_fallback(monkeypatch):
    """429 на основной модели: запрос должен уйти на резервную."""
    from app.generation import llm

    settings = make_settings(monkeypatch, EMBEDDING_API_KEY="gemini-key")
    monkeypatch.setattr(llm, "get_settings", lambda: settings)

    async def fake_attempt(chain, prompt, temperature, max_tokens):
        if chain.model == "z-ai/glm-5.2:free":
            raise llm.LLMUnavailable("лимит запросов")
        return f"ответ от {chain.model}"

    monkeypatch.setattr(llm, "_attempt", fake_attempt)

    import asyncio

    text, model = asyncio.run(llm.complete("промпт"))
    assert text == "ответ от gemini-3.6-flash"
    assert model == "gemini-3.6-flash"


def test_both_providers_down_raises_single_error(monkeypatch):
    import asyncio

    import pytest

    from app.generation import llm

    settings = make_settings(monkeypatch, EMBEDDING_API_KEY="gemini-key")
    monkeypatch.setattr(llm, "get_settings", lambda: settings)

    async def fake_attempt(chain, prompt, temperature, max_tokens):
        raise llm.LLMUnavailable(f"недоступна {chain.model}")

    monkeypatch.setattr(llm, "_attempt", fake_attempt)

    with pytest.raises(llm.LLMUnavailable) as exc_info:
        asyncio.run(llm.complete("промпт"))
    message = str(exc_info.value)
    assert "z-ai/glm-5.2:free" in message
    assert "gemini-3.6-flash" in message
