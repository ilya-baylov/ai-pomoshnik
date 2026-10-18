from unittest.mock import AsyncMock, MagicMock
import pytest
from aiogram.types import CallbackQuery, Message
from app.channels.telegram.panels import render


@pytest.mark.asyncio
async def test_navigation_edits_current_text_without_new_message():
    event = MagicMock(spec=CallbackQuery)
    event.message = MagicMock(spec=Message)
    event.answer = AsyncMock()
    event.message.photo = []
    event.message.edit_text = AsyncMock(return_value='edited')
    event.message.answer = AsyncMock()
    result = await render(event, 'Статус', None)
    assert result == 'edited'
    event.message.edit_text.assert_awaited_once_with('Статус', reply_markup=None)
    event.message.answer.assert_not_awaited()
    event.answer.assert_awaited_once()


@pytest.mark.asyncio
async def test_legacy_photo_is_replaced_once_and_deleted():
    event = MagicMock(spec=CallbackQuery)
    event.message = MagicMock(spec=Message)
    event.answer = AsyncMock()
    event.message.photo = [object()]
    event.message.answer = AsyncMock(return_value='text panel')
    event.message.delete = AsyncMock()
    assert await render(event, 'Управление', None) == 'text panel'
    event.message.answer.assert_awaited_once_with('Управление', reply_markup=None)
    event.message.delete.assert_awaited_once()


@pytest.mark.asyncio
async def test_admin_command_sends_text_only():
    message = MagicMock(spec=Message)
    message.answer = AsyncMock(return_value='text panel')
    message.answer_photo = AsyncMock()
    assert await render(message, 'Управление', None) == 'text panel'
    message.answer.assert_awaited_once()
    message.answer_photo.assert_not_awaited()
