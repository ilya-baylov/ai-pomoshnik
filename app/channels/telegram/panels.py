"""Текстовая админка: обновлять экран в одном сообщении."""
from __future__ import annotations

import contextlib
from aiogram.exceptions import TelegramBadRequest
from aiogram.types import CallbackQuery, Message


async def render(event, text, markup):
    message = event.message if isinstance(event, CallbackQuery) else event
    if isinstance(event, Message):
        return await message.answer(text, reply_markup=markup)
    try:
        if not message.photo:
            return await message.edit_text(text, reply_markup=markup)
        # Telegram не позволяет превратить фотографию в текстовое сообщение.
        # Сначала показать новый экран; затем убрать предыдущий.
        result = await message.answer(text, reply_markup=markup)
        try:
            await message.delete()
        except TelegramBadRequest:
            with contextlib.suppress(TelegramBadRequest):
                await message.edit_reply_markup(reply_markup=None)
        return result
    except TelegramBadRequest as exc:
        description = str(exc).lower()
        if 'message is not modified' in description or 'canceled by new edit message request' in description:
            # При быстрых нажатиях Telegram оставляет более новое обновление.
            return message
        raise
    finally:
        with contextlib.suppress(TelegramBadRequest): await event.answer()
