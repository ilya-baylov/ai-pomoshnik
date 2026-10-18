"""Клиент языковой модели с резервным провайдером.

Основной провайдер задаётся LLM_BASE_URL; по умолчанию официальный OpenAI.
Резерв выключен по умолчанию и должен включаться явно.

Начиная с openai 3.x транспорт — HTTPX2; свой HTTP-клиент мы не передаём,
поэтому миграция с 2.x нас не задевает.
"""

from __future__ import annotations

import asyncio
import functools
import logging
from contextvars import ContextVar
from collections.abc import AsyncIterator
from contextlib import aclosing
from dataclasses import dataclass
from urllib.parse import urlparse

from app.core.config import Settings, get_settings
from app.generation.prompts import ModelPrompt

log = logging.getLogger(__name__)
_usage_events: ContextVar[list | None] = ContextVar("llm_usage_events", default=None)


def start_usage() -> None:
    _usage_events.set([])


def usage_totals() -> dict[str, int]:
    events = _usage_events.get() or []
    return {key: sum(e.get(key, 0) for e in events)
            for key in ("requests", "input_tokens", "cached_tokens", "output_tokens")}


def _record_usage(usage, model: str) -> None:
    if usage is None:
        return
    details = getattr(usage, "prompt_tokens_details", None)
    event = {"requests": 1, "input_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
             "cached_tokens": int(getattr(details, "cached_tokens", 0) or 0),
             "output_tokens": int(getattr(usage, "completion_tokens", 0) or 0)}
    events = _usage_events.get()
    if events is not None:
        events.append(event)
    log.info("LLM usage %s input=%s cached=%s output=%s", model,
             event["input_tokens"], event["cached_tokens"], event["output_tokens"])


class LLMUnavailable(RuntimeError):
    """Провайдер недоступен или исчерпан лимит."""


@dataclass(frozen=True)
class LLMChain:
    """Один провайдер: адрес, ключ, модель."""

    base_url: str
    api_key: str
    model: str


def primary_chain(settings: Settings | None = None) -> LLMChain:
    settings = settings or get_settings()
    return LLMChain(
        base_url=settings.llm_base_url,
        api_key=(settings.llm_api_key or settings.openai_api_key)
        if urlparse(settings.llm_base_url).hostname == "api.openai.com"
        else settings.llm_api_key,
        model=settings.llm_model,
    )


def fallback_chain(settings: Settings | None = None) -> LLMChain | None:
    """Резервная цепочка или None, когда резерв не имеет смысла.

    Не настраивается, если: выключена, ведёт в того же провайдера (второй
    раз стучаться в ту же квоту бессмысленно) или для неё нет ключа.
    """
    settings = settings or get_settings()
    if not settings.llm_fallback_enabled:
        return None

    base_url = (settings.llm_fallback_base_url or "").rstrip("/")
    if not base_url or base_url == settings.llm_base_url.rstrip("/"):
        return None

    # По умолчанию — ключ эмбеддингов: при связке OpenRouter + Gemini
    # это ключ Gemini, и резервная квота с ним независима
    api_key = settings.llm_fallback_api_key
    if not api_key and urlparse(base_url).hostname == "generativelanguage.googleapis.com" and settings.embedding_provider == "gemini":
        api_key = settings.embeddings_key
    if not api_key or not settings.llm_fallback_model:
        return None

    return LLMChain(
        base_url=base_url,
        api_key=api_key,
        model=settings.llm_fallback_model,
    )


@functools.lru_cache(maxsize=4)
def _build_client(base_url: str, api_key: str):
    from openai import AsyncOpenAI

    return AsyncOpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=60.0,
        max_retries=0,  # повторами управляем сами, с пониманием причины отказа
    )


def _request_kwargs(chain: LLMChain, prompt: str | ModelPrompt, temperature: float, budget: int) -> dict:
    if budget <= 0:
        raise ValueError("Бюджет токенов должен быть положительным")
    if isinstance(prompt, ModelPrompt):
        role = "developer" if urlparse(chain.base_url).hostname == "api.openai.com" else "system"
        messages = [{"role": role, "content": prompt.instructions},
                    {"role": "user", "content": prompt.user_input}]
    else:
        messages = [{"role": "user", "content": prompt}]
    kwargs = {"model": chain.model, "messages": messages}
    if urlparse(chain.base_url).hostname == "api.openai.com":
        kwargs.update(max_completion_tokens=budget, store=False)
        if chain.model.startswith(("gpt-6-luna", "gpt-5.6-luna")):
            kwargs["reasoning_effort"] = get_settings().llm_reasoning_effort
        # Эти семейства не принимают обычный temperature при стандартных настройках.
        if not chain.model.startswith(("o1", "o3", "o4", "gpt-5", "gpt-6")):
            kwargs["temperature"] = temperature
    else:
        kwargs.update(max_tokens=budget, temperature=temperature)
    return kwargs


async def _attempt(
    chain: LLMChain, prompt: str | ModelPrompt, temperature: float, max_tokens: int
) -> str:
    """Один провайдер. Повторы — только на сетевые сбои и 5xx.

    429 намеренно не ретраится: у бесплатных моделей лимит чаще суточный,
    ждать его сброса бессмысленно — быстрее уйти на резервного провайдера.

    Отдельно обрабатываются думающие модели (Gemini 3.x и подобные): при
    маленьком max_tokens весь бюджет уходит на размышление, контент
    приходит пустым с finish_reason=length. Ловим это и перезапрашиваем
    с увеличенным бюджетом.
    """
    from openai import (
        APIConnectionError,
        APITimeoutError,
        APIStatusError,
        RateLimitError,
    )

    client = _build_client(chain.base_url, chain.api_key)
    last_error: Exception | None = None
    budget = max_tokens

    for attempt in range(4):
        try:
            response = await client.chat.completions.create(**_request_kwargs(chain, prompt, temperature, budget))
            _record_usage(getattr(response, "usage", None), chain.model)
            if not response.choices:
                raise LLMUnavailable(f"{chain.model}: пустой ответ")
            choice = response.choices[0]
            text = (choice.message.content or "").strip()

            if choice.finish_reason == "length":
                new_budget = min(budget * 4 + 256, 8192)
                if attempt == 3 or new_budget <= budget:
                    raise LLMUnavailable(f"{chain.model}: ответ обрезан по токенам")
                # Думающие модели (Gemini 3.x и подобные) тратят токены на
                # размышление: при малом бюджете контент пустой или обрывок
                # черновика. Перезапрашиваем с увеличенным бюджетом.
                budget = new_budget
                log.info(
                    "%s: ответ обрезан по токенам, повторяю с бюджетом %s",
                    chain.model,
                    budget,
                )
                continue
            if choice.finish_reason == "content_filter" or getattr(choice.message, "refusal", None):
                raise LLMUnavailable(f"{chain.model}: ответ отклонён")
            if text:
                return text
            raise LLMUnavailable(f"{chain.model}: пустой ответ")
        except RateLimitError as exc:
            raise LLMUnavailable(f"лимит запросов ({chain.model})") from exc
        except (APIConnectionError, APITimeoutError) as exc:
            last_error = exc
        except APIStatusError as exc:
            if exc.status_code >= 500 and attempt < 3:
                last_error = exc
            else:
                raise LLMUnavailable(f"{chain.model}: HTTP {exc.status_code}") from exc

        if attempt < 3:
            await asyncio.sleep(2 * (attempt + 1))

    raise LLMUnavailable(f"{chain.model}: сетевая ошибка") from last_error


async def _attempt_stream(
    chain: LLMChain, prompt: str | ModelPrompt, temperature: float, max_tokens: int
) -> AsyncIterator[str]:
    from openai import (
        APIConnectionError,
        APIStatusError,
        APITimeoutError,
        RateLimitError,
    )

    client = _build_client(chain.base_url, chain.api_key)
    response = None
    received_text = False
    finished = False
    try:
        response = await client.chat.completions.create(
            **_request_kwargs(chain, prompt, temperature, max_tokens), stream=True,
            **({"stream_options": {"include_usage": True}} if urlparse(chain.base_url).hostname == "api.openai.com" else {}),
        )
        async for chunk in response:
            _record_usage(getattr(chunk, "usage", None), chain.model)
            if not chunk.choices:
                continue
            choice = chunk.choices[0]
            if choice.finish_reason in ("length", "content_filter"):
                raise LLMUnavailable(f"{chain.model}: поток не завершил ответ ({choice.finish_reason})")
            if choice.finish_reason == "stop":
                finished = True
            delta = choice.delta
            if delta and getattr(delta, "refusal", None):
                raise LLMUnavailable(f"{chain.model}: ответ отклонён")
            if delta and delta.content:
                received_text = received_text or bool(delta.content.strip())
                yield delta.content
        if not received_text or not finished:
            raise LLMUnavailable(f"{chain.model}: пустой или незавершённый поток")
    except LLMUnavailable:
        raise
    except RateLimitError as exc:
        raise LLMUnavailable(f"лимит запросов ({chain.model})") from exc
    except (APIConnectionError, APITimeoutError, APIStatusError) as exc:
        raise LLMUnavailable(f"{chain.model}: ошибка связи ({type(exc).__name__})") from exc
    except Exception as exc:
        raise LLMUnavailable(f"{chain.model}: ошибка ответа ({type(exc).__name__})") from exc
    finally:
        if response is not None:
            await response.close()


async def complete(
    prompt: str | ModelPrompt, temperature: float = 0.2, max_tokens: int = 1200
) -> tuple[str, str]:
    """Запрос к модели. Возвращает (текст, какая модель ответила).

    Сначала основной провайдер; при отказе — резервный, если настроен.
    LLMUnavailable, если не ответил никто.
    """
    settings = get_settings()
    primary = primary_chain(settings)
    if not primary.api_key:
        raise LLMUnavailable("Не задан OPENAI_API_KEY (или LLM_API_KEY) в .env")

    try:
        return await _attempt(primary, prompt, temperature, max_tokens), primary.model
    except LLMUnavailable as primary_error:
        fallback = fallback_chain(settings)
        if fallback is None:
            raise

        log.warning(
            "Модель %s недоступна (%s), переключаюсь на резервную %s",
            primary.model,
            str(primary_error)[:120],
            fallback.model,
        )
        try:
            return (
                await _attempt(fallback, prompt, temperature, max_tokens),
                fallback.model,
            )
        except LLMUnavailable as fallback_error:
            raise LLMUnavailable(
                "Основная и резервная модели недоступны: "
                f"{primary.model} ({str(primary_error)[:80]}), "
                f"{fallback.model} ({str(fallback_error)[:80]})"
            ) from fallback_error


async def complete_stream(
    prompt: str | ModelPrompt, temperature: float = 0.2, max_tokens: int = 1200
) -> AsyncIterator[tuple[str, str]]:
    """Потоковый запрос к модели. Отдаёт (кусок текста, какая модель отвечает).

    Если основной провайдер отказывает до отправки первого чанка, переключается
    на резервный.
    """
    settings = get_settings()
    primary = primary_chain(settings)
    if not primary.api_key:
        raise LLMUnavailable("Не задан OPENAI_API_KEY (или LLM_API_KEY) в .env")

    fallback = fallback_chain(settings)
    yielded_any = False
    primary_failure = ""

    try:
        async with aclosing(_attempt_stream(primary, prompt, temperature, max_tokens)) as stream:
            async for delta in stream:
                yielded_any = True
                yield delta, primary.model
        return
    except LLMUnavailable as primary_error:
        primary_failure = str(primary_error)[:80]
        if yielded_any or fallback is None:
            raise
        log.warning(
            "Модель %s недоступна в потоке (%s), переключаюсь на резервную %s",
            primary.model,
            str(primary_error)[:120],
            fallback.model,
        )

    try:
        async with aclosing(_attempt_stream(fallback, prompt, temperature, max_tokens)) as stream:
            async for delta in stream:
                yield delta, fallback.model
    except LLMUnavailable as fallback_error:
        raise LLMUnavailable(
            "Основная и резервная модели недоступны в потоке: "
            f"{primary.model} ({primary_failure}), "
            f"{fallback.model} ({str(fallback_error)[:80]})"
        ) from fallback_error


async def health_check() -> tuple[bool, str]:
    """Проверка для админки и диагностики: отвечает хотя бы одна цепочка."""
    try:
        _, model = await complete(
            "ответь одним словом: работает", temperature=0.0, max_tokens=256
        )
        return True, model
    except Exception as exc:  # noqa: BLE001
        return False, str(exc)[:200]
