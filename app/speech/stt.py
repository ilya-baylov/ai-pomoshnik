"""Распознавание речи через официальный OpenAI или явно выбранный Groq."""

from __future__ import annotations

import logging

import httpx

from app.core.config import get_settings

log = logging.getLogger(__name__)

GROQ_TRANSCRIPTION_URL = "https://api.groq.com/openai/v1/audio/transcriptions"
MAX_AUDIO_BYTES = 25 * 1_000_000


class STTUnavailable(RuntimeError):
    """Сбой сервиса распознавания речи."""


async def transcribe_audio(
    audio_bytes: bytes,
    filename: str = "voice.ogg",
    content_type: str = "audio/ogg",
    language: str = "ru",
) -> str:
    """Отправка аудио в Groq Whisper API и получение текста."""
    settings = get_settings()
    if not audio_bytes or len(audio_bytes) > MAX_AUDIO_BYTES:
        raise STTUnavailable("Аудио пустое или превышает лимит 25 МБ")
    if settings.stt_provider == "openai":
        from openai import AsyncOpenAI, OpenAIError
        from app.generation.llm import primary_chain
        api_key = settings.stt_api_key or settings.openai_api_key
        if not api_key:
            from urllib.parse import urlparse
            if urlparse(settings.llm_base_url).hostname == "api.openai.com":
                api_key = primary_chain(settings).api_key
        if not api_key:
            raise STTUnavailable("Не задан OPENAI_API_KEY в .env")
        try:
            async with AsyncOpenAI(api_key=api_key, base_url="https://api.openai.com/v1",
                                   timeout=60.0, max_retries=1) as client:
                response = await client.audio.transcriptions.create(
                    model=settings.stt_model, file=(filename, audio_bytes, content_type),
                    language=language, response_format="json",
                )
            text = (response.text or "").strip()
            if not text:
                raise STTUnavailable("Голос не распознан или аудио пустое")
            return text
        except OpenAIError as exc:
            raise STTUnavailable(f"OpenAI: ошибка распознавания ({type(exc).__name__})") from exc
    if settings.stt_provider != "groq":
        raise STTUnavailable("STT_PROVIDER должен быть openai или groq")
    api_key = settings.groq_api_key
    if not api_key:
        raise STTUnavailable("Не задан GROQ_API_KEY в .env")

    model = settings.groq_whisper_model or "whisper-large-v3"

    try:
        async with httpx.AsyncClient(timeout=45.0) as client:
            response = await client.post(
                GROQ_TRANSCRIPTION_URL,
                headers={"Authorization": f"Bearer {api_key}"},
                files={"file": (filename, audio_bytes, content_type)},
                data={
                    "model": model,
                    "language": language,
                    "response_format": "json",
                },
            )
            if response.status_code != 200:
                log.error(
                    "Groq Whisper вернул статус %s",
                    response.status_code,
                )
                raise STTUnavailable(f"Groq API ошибка ({response.status_code})")

            data = response.json()
            text = (data.get("text") or "").strip()
            if not text:
                raise STTUnavailable("Голос не распознан или аудио пустое")
            return text
    except STTUnavailable:
        raise
    except httpx.RequestError as exc:
        log.exception("Сетевая ошибка при обращении к Groq Whisper")
        raise STTUnavailable(f"Сетевая ошибка: {exc}") from exc
    except Exception as exc:
        log.exception("Непредвиденная ошибка распознавания речи")
        raise STTUnavailable(f"Ошибка распознавания: {exc}") from exc
