"""Доступ по списку сотрудников, приглашению или явному режиму регистрации."""

from __future__ import annotations

from aiogram import BaseMiddleware
from aiogram.types import CallbackQuery, Message, TelegramObject

from app.core.config import get_bot_config
from app.core.contracts import IncomingMessage, Role
from app.service import get_service


class AccessMiddleware(BaseMiddleware):
    """Проверяет доступ до обработки сообщения."""

    async def __call__(self, handler, event: TelegramObject, data: dict):
        from_user = getattr(event, "from_user", None)
        if from_user is None:
            return await handler(event, data)

        service = get_service()
        incoming = IncomingMessage(
            channel="telegram",
            external_user_id=str(from_user.id),
            chat_id=str(from_user.id),
            text="",
        )
        user = await service.resolve_user(
            incoming,
            username=from_user.username,
            full_name=from_user.full_name,
        )

        if not user.is_active and user.role == Role.PENDING.value and isinstance(event, Message):
            import re
            from app.storage import repo
            from app.storage.db import session_scope
            text = (event.text or "").strip()
            if text.startswith("/start") and " " in text:
                text = text.split(maxsplit=1)[1]
            if re.fullmatch(r"(?:[A-Fa-f0-9]{6}|[A-Fa-f0-9]{16})", text):
                async with session_scope() as session:
                    session.add(user)
                    accepted = await repo.redeem_invite(session, text, user)
                if accepted:
                    await event.answer(get_bot_config().message("greeting"))
                    return None
                await event.answer("Код недействителен или уже использован.")
                return None
        if user.is_active:
            data["user"] = user
            data["is_admin"] = user.role == Role.ADMIN.value
            return await handler(event, data)

        # Доступ закрыт админом через карточку пользователя
        message = get_bot_config().message("access_denied")
        if isinstance(event, Message):
            await event.answer(message)
        elif isinstance(event, CallbackQuery):
            await event.answer(message, show_alert=True)
        return None
