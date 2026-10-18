"""Обработчики сотрудника.

Сотрудник просто пишет вопрос. Оценка предлагается только после явного
замечания к предыдущему ответу, обычные ответы приходят без кнопок.
"""

from __future__ import annotations

import asyncio
import contextlib
import html
import io
import logging
import random
import re
import time
import weakref
from collections import OrderedDict

from aiogram import F, Router
from aiogram.filters import Command, CommandStart
from aiogram.methods import SendMessageDraft
from aiogram.types import CallbackQuery, Message, MessageGenerationStopped, Update
from sqlalchemy import func, select

from app.channels.telegram.formatting import render_parts
from app.channels.telegram import keyboards
from app.core.config import get_bot_config
from app.generation.llm import LLMUnavailable
from app.service import RateLimited, get_service
from app.speech import STTUnavailable, transcribe_audio
from app.storage import repo
from app.storage.db import session_scope
from app.storage.models import Document, User

log = logging.getLogger(__name__)
router = Router(name="staff")

MIN_QUESTION_LENGTH = 1


def is_greeting(text: str) -> bool:
    normalized = " ".join(text.casefold().strip().strip(".,!? 👋🙌🙂").split())
    return normalized in {
        "привет", "приветик", "приветики", "привет бот", "привет, бот", "здравствуй", "здравствуйте",
        "добрый день", "добрый вечер", "доброе утро", "hello", "hi", "салам", "салям", "салют",
    }


def is_answer_criticism(text: str) -> bool:
    folded = text.casefold().strip()
    # Narrow patterns: a business question about an error is not answer feedback.
    return bool(re.search(
        r'^(?:это\s+)?(?:неверно|неправильно|неправда|неполный ответ|не хватает информации)(?:[.!?,\s]|$)'
        r'|^(?:это\s+не\s+так|плохой ответ)(?:[.!?,\s]|$)'
        r'|^(?:ты\s+)?(?:ошибся|ошиблась|ошибаешься|неправильно ответил|неправильно ответила|не так ответил|неправильно отвечаешь)(?:[.!?,\s]|$)'
        r'|^(?:(?:твой|этот|предыдущий)\s+)?ответ\s+(?:неверн\w*|неправильн\w*|неполн\w*|ошибочн\w*|бесполезн\w*)'
        r'|^(?:в\s+)?(?:твоём|твоем|этом|предыдущем)\s+ответе\s+(?:ошибк\w*|не хватает)', folded))


_feedback_comments: OrderedDict[tuple[int, int], str] = OrderedDict()


async def _offer_feedback(message: Message, user: User, comment: str) -> None:
    async with session_scope() as session:
        query = await repo.latest_feedback_query(session, user.id)
    if query is None:
        await _send_reply(message, 'Уточни, какой ответ неверный и что нужно исправить.')
        return
    key = (user.id, query.id)
    _feedback_comments[key] = comment[:2000]
    _feedback_comments.move_to_end(key)
    while len(_feedback_comments) > 512:
        _feedback_comments.popitem(last=False)
    question = html.escape(query.question[:140])
    await _send_reply(message, f'Что не так с ответом на «{question}»?',
                      reply_markup=keyboards.answer_feedback(query.id))


async def _send_reply(message: Message, text: str, **kwargs) -> None:
    sent = await message.answer(text, **kwargs)
    log.info("Ответ отправлен в Telegram: message_id=%s", sent.message_id)


@router.message(CommandStart())
async def cmd_start(message: Message, user: User) -> None:
    config = get_bot_config()
    text = config.message('greeting')
    await _send_reply(message, text)


@router.message(Command("help"))
async def cmd_help(message: Message, user: User) -> None:
    """Что бот знает — по разделам базы. Помогает понять, о чём спрашивать."""
    async with session_scope() as session:
        rows = (
            await session.execute(
                select(Document.section, func.count(Document.id))
                .group_by(Document.section)
                .order_by(func.count(Document.id).desc())
            )
        ).all()

    lines = ["<b>Что я знаю</b>"]
    if rows:
        for section, count in rows:
            lines.append(f"• {section or 'без раздела'} — {count} док.")
    else:
        lines.append("База пуста. Обратитесь к администратору.")

    lines += [
        "",
        "Просто напишите вопрос своими словами.",
        "Помогаю по рабочим вопросам компании: услуги, клиенты, идеи и тексты для салонов.",
        "",
        "/new — начать разговор заново, забыв контекст",
        "/profile — сохранить город, филиал и должность",
    ]
    await message.answer("\n".join(lines))


@router.message(Command("new"))
async def cmd_new(message: Message, user: User) -> None:
    await get_service().new_conversation(user)
    await message.answer("Контекст сброшен. Задайте новый вопрос.")


_profile_inputs: dict[tuple[int, int], str] = {}


def _profile_text(profile: dict) -> str:
    return ("<b>Мой рабочий профиль</b>\n\n"
            f"Город: {html.escape(profile.get('city') or 'не выбран')}\n"
            f"Филиал: {html.escape(profile.get('branch') or 'не указан')}\n"
            f"Должность: {html.escape(profile.get('job_title') or 'не указана')}\n\n"
            "Использую эти сведения по умолчанию. Другой город можно указать прямо в вопросе.")


@router.message(Command("profile"))
async def cmd_profile(message: Message, user: User) -> None:
    if message.from_user:
        _profile_inputs.pop((message.chat.id, message.from_user.id), None)
    async with session_scope() as session:
        profile = await repo.staff_profile(session, user.id)
    await message.answer(_profile_text(profile), reply_markup=keyboards.profile_menu())


@router.callback_query(F.data.startswith("profile:"))
async def cb_profile(callback: CallbackQuery, user: User) -> None:
    if not callback.message:
        await callback.answer()
        return
    data = callback.data or ''
    key = (callback.message.chat.id, callback.from_user.id)
    if data == "profile:edit:city":
        _profile_inputs.pop(key, None)
        await callback.message.answer("Выбери свой рабочий город:", reply_markup=keyboards.profile_cities())
    elif data in {"profile:edit:branch", "profile:edit:job_title"}:
        field = data.rsplit(':', 1)[1]
        _profile_inputs[key] = field
        label = "филиал или его адрес" if field == "branch" else "должность"
        await callback.message.answer(f"Напиши {label} одним сообщением. /cancel — отменить.")
    elif data.startswith("profile:city:"):
        from app.core.staff_context import CITIES
        raw = data.rsplit(':', 1)[1]
        if not raw.isdigit() or int(raw) >= len(CITIES):
            await callback.answer("Город не найден", show_alert=True)
            return
        _profile_inputs.pop(key, None)
        async with session_scope() as session:
            profile = await repo.update_staff_profile(session, user.id, "city", CITIES[int(raw)])
        await callback.message.answer(_profile_text(profile), reply_markup=keyboards.profile_menu())
    elif data == "profile:clear":
        _profile_inputs.pop(key, None)
        async with session_scope() as session:
            for field in ("city", "branch", "job_title"):
                await repo.update_staff_profile(session, user.id, field, None)
        await callback.message.answer(_profile_text({}), reply_markup=keyboards.profile_menu())
    await callback.answer()


@router.message(Command("cancel"))
async def cmd_cancel(message: Message, user: User) -> None:
    if message.from_user:
        _profile_inputs.pop((message.chat.id, message.from_user.id), None)
    await message.answer("Ввод отменён.")


@router.callback_query(F.data.startswith("fb:"))
async def cb_answer_feedback(callback: CallbackQuery, user: User) -> None:
    match = re.fullmatch(r'fb:(wrong|incomplete):(\d+)', callback.data or '')
    if not match:
        await callback.answer("Оценка не распознана", show_alert=True)
        return
    async with session_scope() as session:
        accepted = await repo.submit_feedback(session, int(match[2]), user.id, match[1],
                    comment=_feedback_comments.get((user.id, int(match[2]))))
    if not accepted:
        await callback.answer("Можно оценить только свой ответ", show_alert=True)
        return
    _feedback_comments.pop((user.id, int(match[2])), None)
    if callback.message:
        with contextlib.suppress(Exception):
            await callback.message.edit_reply_markup(reply_markup=None)
    await callback.answer("Спасибо! Ответ добавлен в список на проверку.")
    from app.channels.telegram.notifications import notify_negative_feedback
    await notify_negative_feedback(callback.bot)


_active_generations: dict[tuple[int, int], asyncio.Task] = {}


@router.stopped_message_generation()
async def on_generation_stopped(event: MessageGenerationStopped) -> None:
    """Пользователь нажал кнопку «Остановить» в интерфейсе стриминга."""
    key = (event.chat.id, event.draft_id)
    task = _active_generations.pop(key, None)
    if task and not task.done():
        task.cancel()
        log.info(
            "Генерация отменена пользователем: chat_id=%s, draft_id=%s",
            event.chat.id,
            event.draft_id,
        )


_question_locks: weakref.WeakValueDictionary[int, asyncio.Lock] = weakref.WeakValueDictionary()


async def _process_question(message: Message, user: User, question: str) -> None:
    lock = _question_locks.setdefault(user.id, asyncio.Lock())
    async with lock:
        await _process_question_locked(message, user, question)


async def _process_question_locked(message: Message, user: User, question: str) -> None:
    if is_greeting(question):
        await _send_reply(message, 'Привет! 🙂 Чем помочь?')
        return
    if is_answer_criticism(question):
        await _offer_feedback(message, user, question)
        return
    service = get_service()
    try:
        await service.check_limits(user)
    except RateLimited as exc:
        await message.answer(exc.message)
        return

    draft_id = random.randint(1, 2**31 - 1)
    last_draft_time = 0.0
    last_draft_len = 0
    draft_failed = False

    async def _on_draft(full_text: str) -> None:
        nonlocal last_draft_time, last_draft_len, draft_failed
        if draft_failed:
            return
        now = time.monotonic()
        # Дросселирование: не чаще раза в 350 мс и при приросте текста
        if (now - last_draft_time < 0.35) and (len(full_text) - last_draft_len < 30):
            return
        if len(full_text) == last_draft_len:
            return

        text_preview = full_text[:4000]
        try:
            await message.bot(
                SendMessageDraft(
                    chat_id=message.chat.id,
                    draft_id=draft_id,
                    text=text_preview,
                    message_thread_id=message.message_thread_id,
                    can_stop=True,
                    parse_mode=None,
                )
            )
            last_draft_time = now
            last_draft_len = len(full_text)
        except Exception:
            # Если чат не поддерживает draft (или ошибка Bot API),
            # отключаем черновики и полагаемся на typing-индикатор
            log.debug("sendMessageDraft недоступен или вернул ошибку", exc_info=True)
            draft_failed = True

    # Индикатор набора: без него неясно, работает бот или молчит
    typing = asyncio.create_task(_keep_typing(message))
    current_task = asyncio.current_task()
    if current_task:
        _active_generations[(message.chat.id, draft_id)] = current_task

    try:
        answer = await service.ask_stream(user, question, on_draft=_on_draft)
    except asyncio.CancelledError:
        log.info("Генерация отменена для chat_id=%s", message.chat.id)
        return
    except LLMUnavailable:
        await message.answer(get_bot_config().message("service_error"))
        return
    except Exception:  # noqa: BLE001 — сбой не должен выглядеть как молчание
        log.exception("Ошибка обработки вопроса")
        await message.answer(get_bot_config().message("service_error"))
        return
    finally:
        _active_generations.pop((message.chat.id, draft_id), None)
        typing.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await typing

    parts = render_parts(answer.text)
    for part in parts:
        await _send_reply(message, part)
    if answer.is_fallback:
        await _notify_unanswered(message, question)


@router.message(F.text & ~F.text.startswith("/"))
async def on_question(message: Message, user: User) -> None:
    question = (message.text or "").strip()
    key = (message.chat.id, message.from_user.id) if message.from_user else None
    field = _profile_inputs.get(key) if key else None
    if field:
        try:
            async with session_scope() as session:
                profile = await repo.update_staff_profile(session, user.id, field, question)
        except ValueError:
            await message.answer("Напиши короткое значение одним сообщением: филиал до 160 символов, должность до 100. /cancel — отменить.")
            return
        _profile_inputs.pop(key, None)
        await message.answer(_profile_text(profile), reply_markup=keyboards.profile_menu())
        return
    if len(question) < MIN_QUESTION_LENGTH:
        await message.answer("Сформулируйте вопрос чуть подробнее.")
        return

    await _process_question(message, user, question)


@router.message(F.voice | F.audio | F.video_note)
async def on_voice(message: Message, user: User) -> None:
    lock = _question_locks.setdefault(user.id, asyncio.Lock())
    async with lock:
        await _on_voice_locked(message, user)


async def _on_voice_locked(message: Message, user: User) -> None:
    """Голосовые сообщения, аудио и кружочки через выбранный сервис речи."""
    service = get_service()
    try:
        await service.check_limits(user)
    except RateLimited as exc:
        await message.answer(exc.message)
        return

    media = message.voice or message.video_note or message.audio
    if not media:
        return
    from app.speech.stt import MAX_AUDIO_BYTES
    if getattr(media, "file_size", 0) and media.file_size > MAX_AUDIO_BYTES:
        await message.answer("Аудиофайл больше 25 МБ. Отправьте запись покороче.")
        return

    status = await message.answer("🎤 <i>Слушаю…</i>")
    try:
        file = await message.bot.get_file(media.file_id)
        if not file.file_path:
            await status.edit_text("Не удалось загрузить аудиофайл.")
            return

        buffer = io.BytesIO()
        await message.bot.download_file(file.file_path, destination=buffer)
        audio_bytes = buffer.getvalue()
        if len(audio_bytes) > MAX_AUDIO_BYTES:
            await status.edit_text("Аудиофайл больше 25 МБ. Отправьте запись покороче.")
            return
        if not audio_bytes:
            await status.edit_text("Аудиозапись пустая.")
            return

        if message.voice:
            filename, content_type = "voice.ogg", "audio/ogg"
        elif message.video_note:
            filename, content_type = "video_note.mp4", "video/mp4"
        else:
            filename = getattr(media, "file_name", None) or "audio.mp3"
            content_type = getattr(media, "mime_type", None) or "audio/mpeg"

        transcribed = await transcribe_audio(
            audio_bytes,
            filename=filename,
            content_type=content_type,
            language="ru",
        )
    except STTUnavailable as exc:
        log.warning("Ошибка STT: %s", exc)
        await status.edit_text("Не удалось распознать речь. Попробуйте написать вопрос текстом.")
        return
    except Exception:
        log.exception("Не удалось обработать голосовое сообщение")
        await status.edit_text("Произошла ошибка при обработке звука. Попробуйте написать текстом.")
        return

    question = transcribed.strip()
    if len(question) < MIN_QUESTION_LENGTH:
        await status.edit_text(
            f"🎤 <i>«{html.escape(question)}»</i>\n\nСформулируйте вопрос чуть подробнее."
        )
        return

    # Показываем распознанный вопрос сотруднику
    await status.edit_text(f"🎤 <i>«{html.escape(question)}»</i>")
    # Передаём в стандартный RAG-пайплайн
    await _process_question_locked(message, user, question)


async def _keep_typing(message: Message) -> None:
    """Держать индикатор: Telegram гасит его через ~5 секунд."""
    try:
        while True:
            await message.bot.send_chat_action(message.chat.id, "typing")
            await asyncio.sleep(4)
    except asyncio.CancelledError:
        raise
    except Exception:  # noqa: BLE001 — индикатор не критичен
        return


async def _notify_unanswered(message: Message, question: str) -> None:
    from app.channels.telegram.notifications import notify_unanswered

    await notify_unanswered(message.bot, question)
