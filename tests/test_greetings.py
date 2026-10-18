from types import SimpleNamespace as NS
from unittest.mock import AsyncMock
import pytest
from app.channels.telegram import handlers


@pytest.mark.parametrize('text',['привет','Привет!','приветик','Приветики 🙂',' здравствуйте ','Добрый день!','hello','Привет, бот'])
def test_simple_greetings(text):
    assert handlers.is_greeting(text)


@pytest.mark.parametrize('text',['привет, какие акции в Казани?','добрый день, как оформить отпуск?','Как работает бот?'])
def test_greeting_with_question_is_not_discarded(text):
    assert not handlers.is_greeting(text)


@pytest.mark.asyncio
async def test_greeting_sends_persistent_message_without_search_or_quota(monkeypatch):
    def unexpected():raise AssertionError('Greeting must not call question service')
    monkeypatch.setattr(handlers,'get_service',unexpected)
    message=NS(answer=AsyncMock(return_value=NS(message_id=123)))
    await handlers._process_question_locked(message,NS(id=1),'Привет!')
    message.answer.assert_awaited_once()
    assert message.answer.call_args.args[0] == 'Привет! 🙂 Чем помочь?'
