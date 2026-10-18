"""Проверки границы между найденными материалами и ответом модели."""
from unittest.mock import AsyncMock

import pytest

from app.core.config import get_bot_config
from app.core.contracts import RetrievedChunk
from app.generation import answer


def evidence(chunk_id, text, score, section="Общее"):
    return RetrievedChunk(chunk_id, text, "Инструкция", None, None, score, section)


@pytest.mark.asyncio
@pytest.mark.parametrize("chunks", [[], [evidence(1, "Случайное совпадение", 0.1)]])
async def test_missing_evidence_does_not_call_model(monkeypatch, chunks):
    model = AsyncMock()
    monkeypatch.setattr(answer.llm, "complete", model)
    result = await answer.build_answer("Какой у нас порядок согласования?", chunks, require_company_data=True)
    assert result.is_fallback
    assert result.used_chunk_ids == []
    model.assert_not_awaited()


@pytest.mark.asyncio
async def test_weak_match_is_not_sent_alongside_relevant_evidence(monkeypatch):
    model = AsyncMock(return_value=("Заявку согласует руководитель.", "test-model"))
    monkeypatch.setattr(answer.llm, "complete", model)
    result = await answer.build_answer(
        "Кто согласует заявку?",
        [evidence(1, "Заявку согласует руководитель.", 0.9),
         evidence(2, "Посторонний документ: UNRELATED_INTERNAL_DATA", 0.1)],
        require_company_data=True,
    )
    prompt = model.await_args.args[0]
    assert "Заявку согласует руководитель" in prompt.user_input
    assert "UNRELATED_INTERNAL_DATA" not in prompt.user_input
    assert result.used_chunk_ids == [1]
    assert len(result.sources) == 1


@pytest.mark.asyncio
async def test_sensitive_section_returns_quote_without_model(monkeypatch):
    monkeypatch.setitem(get_bot_config()._data, "sensitive_sections", ["Регламенты"])
    model = AsyncMock()
    monkeypatch.setattr(answer.llm, "complete", model)
    result = await answer.build_answer(
        "Кто согласует заявку?",
        [evidence(1, "Заявку согласует руководитель.", 0.9, "Регламенты")],
        require_company_data=True,
    )
    assert "Заявку согласует руководитель." in result.text
    assert not result.is_fallback
    model.assert_not_awaited()


@pytest.mark.asyncio
async def test_unconfirmed_number_is_rejected_after_repair_attempt(monkeypatch):
    model = AsyncMock(return_value=("Стоимость согласования — 987654 рублей.", "test-model"))
    monkeypatch.setattr(answer.llm, "complete", model)
    result = await answer.build_answer(
        "Сколько стоит согласование?",
        [evidence(1, "Для согласования обратитесь к руководителю. Стоимость не указана.", 0.9)],
        require_company_data=True,
    )
    assert model.await_count == 2
    assert result.is_fallback
    assert "987654" not in result.text
