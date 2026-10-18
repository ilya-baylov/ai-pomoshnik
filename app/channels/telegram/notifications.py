"""Уведомления админам.

Каждый тип с защитой от спама: вопросы без ответа группируются по времени,
ошибки провайдера не сыплются на каждый запрос.
"""

from __future__ import annotations

import html
import hashlib
import json
import logging
import time
from datetime import datetime, timezone

from aiogram import Bot
from sqlalchemy import select

from app.core.config import get_bot_config, get_settings
from app.storage.db import session_scope
from app.storage.models import Query, User

log = logging.getLogger(__name__)

# Когда последний раз отправляли уведомление данного типа
_last_sent: dict[str, float] = {}


async def _admin_ids(bot: Bot) -> list[int]:
    """Явный список получателей уведомлений имеет приоритет над админами."""
    configured = get_bot_config().get("notifications.recipient_ids", None)
    if configured is not None:
        values = configured if isinstance(configured, list) else []
        ids = set()
        for value in values:
            try:
                chat_id = int(value)
            except (TypeError, ValueError):
                continue
            if chat_id > 0:
                ids.add(chat_id)
        return sorted(ids)
    settings = get_settings()
    ids = set(settings.admin_ids)
    async with session_scope() as session:
        rows = (
            await session.execute(
                select(User.external_id).where(
                    User.role == "admin", User.channel == "telegram"
                )
            )
        ).all()
    for (external_id,) in rows:
        try:
            ids.add(int(external_id))
        except ValueError:
            continue
    return sorted(ids)


async def send_to_admins(bot: Bot, text: str, **kwargs) -> None:
    for admin_id in await _admin_ids(bot):
        try:
            message = await bot.send_message(admin_id, text, **kwargs)
            # Telegram Bot API не выдаёт историю: сохраняем ID для точечного удаления.
            try:
                path = get_settings().db_path.parent / "technical-notifications.jsonl"
                path.parent.mkdir(parents=True, exist_ok=True)
                record = {"chat_id": admin_id, "message_id": message.message_id,
                          "sent_at": datetime.now(timezone.utc).isoformat(),
                          "type": text.split("\n", 1)[0],
                          "text_sha256": hashlib.sha256(text.encode("utf8")).hexdigest()}
                with path.open("a", encoding="utf8") as journal:
                    journal.write(json.dumps(record, ensure_ascii=False) + "\n")
            except Exception:
                log.exception("Не удалось сохранить ID технического уведомления")
        except Exception as exc:  # noqa: BLE001 — один недоступный админ не мешает остальным
            log.warning("Не удалось отправить уведомление %s: %s", admin_id, exc)


def _throttled(key: str, minutes: float) -> bool:
    """True, если ещё рано отправлять повторно."""
    if minutes <= 0:
        return False
    now = time.monotonic()
    last = _last_sent.get(key, 0.0)
    if now - last < minutes * 60:
        return True
    _last_sent[key] = now
    return False


async def notify_unanswered(bot: Bot, question: str) -> None:
    config = get_bot_config()
    if not config.get("notifications.unanswered", True):
        return
    throttle = float(config.get("notifications.unanswered_throttle_minutes", 30))
    if _throttled("unanswered", throttle):
        return

    await send_to_admins(
        bot,
        "<b>Вопрос без ответа</b>\n\n"
        f"{html.escape(question[:300])}\n\n"
        "Список целиком — в /admin → Без ответа.",
    )


async def notify_negative_feedback(bot: Bot) -> None:
    config = get_bot_config()
    if not config.get('notifications.negative_feedback', True) or _throttled('negative_feedback', 30):
        return
    await send_to_admins(bot, '<b>Ответ требует проверки</b>\n\nСотрудник оставил замечание. Список — в /admin → Качество ответов.')


async def notify_ingest_done(bot: Bot, result, *, manual: bool = False) -> None:
    """Отчёт после индексации. При ручном запуске отчёт уже виден в чате."""
    config = get_bot_config()
    if manual or not config.get("notifications.ingest_done", True):
        return
    if not result.changed and not result.errors:
        return

    lines = ["<b>Индексация завершена</b>", ""]
    lines.append(f"изменено файлов: {result.files_changed}")
    lines.append(f"фрагментов создано: {result.chunks_created}")
    if result.files_deleted:
        lines.append(f"удалено документов: {result.files_deleted}")
    if result.warnings:
        lines.append("")
        lines.append(f"⚠ предупреждений: {len(result.warnings)}")
        lines.extend(f"• {html.escape(w[:150])}" for w in result.warnings[:3])
    if result.errors:
        lines.append("")
        lines.append(f"⚠ ошибок: {len(result.errors)}")
        lines.extend(f"• {html.escape(e[:150])}" for e in result.errors[:3])

    await send_to_admins(bot, "\n".join(lines))


async def notify_error(bot: Bot, title: str, detail: str) -> None:
    """Сбой провайдера. С паузой, чтобы не залить чат при массовом отказе."""
    config = get_bot_config()
    if not config.get("notifications.errors", True):
        return
    if _throttled(f"error:{title}", 15):
        return

    await send_to_admins(
        bot, f"<b>⚠ {html.escape(title)}</b>\n\n<code>{html.escape(detail[:400])}</code>"
    )


async def notify_startup(bot: Bot, checks: list[tuple[str, bool, str]]) -> None:
    """Отчёт при запуске: сразу видно, что настроено, а что нет."""
    lines = ["<b>Бот запущен</b>", ""]
    for label, ok, detail in checks:
        lines.append(f"{'●' if ok else '⚠'} {label}: {html.escape(detail)}")

    if any(not ok for _, ok, _ in checks):
        lines += ["", "Подробности — в /admin → Проверка."]
    else:
        lines += ["", "Панель управления — /admin"]

    await send_to_admins(bot, "\n".join(lines))
