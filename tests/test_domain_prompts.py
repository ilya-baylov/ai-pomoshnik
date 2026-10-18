from unittest.mock import AsyncMock

import pytest

from app.core.config import get_bot_config
from app.core.contracts import RetrievedChunk
from app.generation import answer


@pytest.mark.asyncio
@pytest.mark.parametrize("template", [None, "domains/service_business.j2"])
async def test_industry_rules_are_explicit_opt_in(monkeypatch, template):
    config = get_bot_config()
    monkeypatch.setitem(config._data["generation"], "domain_rules_template", template)
    model = AsyncMock(return_value=("Заявку согласует руководитель.", "test-model"))
    monkeypatch.setattr(answer.llm, "complete", model)
    chunk = RetrievedChunk(1, "Заявку согласует руководитель.", "Инструкция", None, None, 0.9)
    await answer.build_answer("Кто согласует заявку?", [chunk], require_company_data=True)
    instructions = model.await_args.args[0].instructions
    assert "Демо-компания" in instructions
    assert "сведения из документов" in instructions
    assert ("Дополнительные правила для сервисного бизнеса" in instructions) == bool(template)
    if template is None:
        assert "салон" not in instructions.lower()
        assert "beauty" not in instructions.lower()
