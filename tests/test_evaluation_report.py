import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.core.contracts import RetrievedChunk
from eval import run


@pytest.mark.asyncio
@pytest.mark.parametrize("wrong_answer", [False, True])
async def test_report_checks_answer_not_just_retrieved_title(tmp_path, monkeypatch, wrong_answer):
    from app.generation import answer
    from app.retrieval import search, vector

    cases = tmp_path / "questions.yaml"
    cases.write_text("- question: Кто согласует заявку?\n  expect_source: Инструкция\n  expected_terms: [руководител]\n", encoding="utf-8")
    monkeypatch.setattr(run, "QUESTIONS_FILE", cases)
    index = SimpleNamespace(load=AsyncMock())
    monkeypatch.setattr(vector, "get_vector_index", lambda: index)
    chunk = RetrievedChunk(1, "Заявку согласует руководитель.", "Инструкция", None, None, 0.9)
    monkeypatch.setattr(search, "search", AsyncMock(return_value=[chunk]))
    model_answer = SimpleNamespace(text="Заявку согласует руководитель." if not wrong_answer else "Неизвестно.", is_fallback=False)
    monkeypatch.setattr(answer, "build_answer", AsyncMock(return_value=model_answer))
    output = tmp_path / "report.json"
    status = await run.run_eval(answers=True, output=str(output))
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["results"][0]["source_match"]
    assert report["results"][0]["answer_match"] == (not wrong_answer)
    assert report["failed"] == int(wrong_answer)
    assert status == int(wrong_answer)
