"""Тесты распознавания речи через Groq Whisper."""

from __future__ import annotations

import io
import wave
import pytest

from app.speech import STTUnavailable, transcribe_audio


def _create_dummy_wav() -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\x00\x00" * 8000)
    return buf.getvalue()


@pytest.mark.asyncio
async def test_transcribe_audio_missing_key(monkeypatch):
    from app.core import config

    current = config.get_settings()
    monkeypatch.setattr(current, "stt_provider", "groq")
    monkeypatch.setattr(current, "groq_api_key", "")
    with pytest.raises(STTUnavailable, match="Не задан GROQ_API_KEY"):
        await transcribe_audio(b"dummy", filename="test.wav", content_type="audio/wav")


@pytest.mark.asyncio
async def test_transcribe_audio_mock(monkeypatch):
    import httpx
    from app.core.config import get_settings
    monkeypatch.setattr(get_settings(), "stt_provider", "groq")
    monkeypatch.setattr(get_settings(), "groq_api_key", "test-key")

    class MockResponse:
        status_code = 200

        def json(self):
            return {"text": "какое расписание в ростове"}

    async def mock_post(*args, **kwargs):
        return MockResponse()

    monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)

    text = await transcribe_audio(
        _create_dummy_wav(),
        filename="test.wav",
        content_type="audio/wav",
    )
    assert text == "какое расписание в ростове"
