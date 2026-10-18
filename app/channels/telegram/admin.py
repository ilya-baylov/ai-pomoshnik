"""Админка: дашборд, индексация с живым прогрессом, приём документов.

Дашборд перерисовывает одно сообщение вместо отправки новых — чат остаётся
чистым. Прогресс индексации обновляется не чаще раза в 1.5 секунды: Telegram
ограничивает частоту правок сообщения.
"""

from __future__ import annotations

import asyncio
import contextlib
import functools
import html
import logging
import re
import time
import tempfile
from datetime import date, datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from aiogram import Bot, F, Router
from aiogram.filters import BaseFilter, Command
from aiogram.types import CallbackQuery, Message, BufferedInputFile
from sqlalchemy import func, select

from app.channels.telegram import keyboards, views
from app.channels.telegram.formatting import progress_bar
from app.core.config import get_bot_config, get_settings
from app.core.contracts import Role
from app.ingest.loaders.registry import SUPPORTED_EXTENSIONS
from app.ingest.pipeline import IngestProgress, Ingestor, IngestResult
from app.retrieval.vector import get_vector_index
from app.storage import repo
from app.storage.db import session_scope
from app.storage.models import Chunk, Document, IngestRun, User

log = logging.getLogger(__name__)
router = Router(name="admin")

# Одна индексация за раз: параллельный запуск попортит индекс
_ingest_lock = asyncio.Lock()

# Ожидание своего лимита: id пользователя, чей лимит админ сейчас введёт
_limit_input_targets: dict[tuple[int, int], int] = {}


def _limit_input_key(message: Message) -> tuple[int, int]:
    return message.chat.id, message.from_user.id


class UserLimitInput(BaseFilter):
    """Фильтр: админ в режиме ввода своего лимита."""

    async def __call__(self, message: Message) -> bool:
        return _limit_input_key(message) in _limit_input_targets if message.from_user else False


def admin_only(handler):
    """Пропускать только админов.

    functools.wraps обязателен: aiogram читает сигнатуру обработчика через
    интроспекцию, и без сохранения метаданных он не увидит параметры.
    """

    @functools.wraps(handler)
    async def wrapper(event, *args, **kwargs):
        if not kwargs.get("is_admin"):
            if isinstance(event, CallbackQuery):
                await event.answer("Только для администраторов", show_alert=True)
            return None
        return await handler(event, *args, **kwargs)

    return wrapper


async def _dashboard_data(days: int = 7) -> dict:
    async with session_scope() as session:
        data = await repo.stats(session, days=days)
        last_run = (
            await session.execute(
                select(IngestRun)
                .where(IngestRun.finished_at.is_not(None))
                .order_by(IngestRun.finished_at.desc())
                .limit(1)
            )
        ).scalar_one_or_none()
    data["last_ingest"] = (
        last_run.finished_at.replace(tzinfo=timezone.utc).astimezone(ZoneInfo('Asia/Qyzylorda')).strftime("%d.%m %H:%M") if last_run else None
    )
    async with session_scope() as session:
        titles = (
            await session.execute(select(Document.title))
        ).scalars().all()
    stale = _stale_period_titles(list(titles))
    data["stale_period"] = len(stale)
    from app.core.knowledge_review import load_queue, is_pending
    data['review_pending'] = sum(is_pending(i) for i in load_queue()['issues'])
    data['updating'] = _ingest_lock.locked()
    return data


async def _sections() -> list[tuple[str, int, int]]:
    async with session_scope() as session:
        rows = (
            await session.execute(
                select(
                    Document.section,
                    func.count(func.distinct(Document.id)),
                    func.count(Chunk.id),
                )
                .outerjoin(Chunk, Chunk.document_id == Document.id)
                .group_by(Document.section)
                .order_by(func.count(func.distinct(Document.id)).desc())
            )
        ).all()
    return [(section, docs, chunks) for section, docs, chunks in rows]


# Русские названия месяцев в именах файлов: «Акции_МАЙ_2026»
_MONTH_STEMS: list[tuple[int, str]] = [
    (1, "январ"), (2, "феврал"), (3, "март"), (4, "апрел"),
    (5, "май"), (5, "мая"), (6, "июн"), (7, "июл"), (8, "август"),
    (9, "сентябр"), (10, "октябр"), (11, "ноябр"), (12, "декабр"),
]
_YEAR_RE = re.compile(r"20\d{2}")


def _month_from_title(title: str) -> int | None:
    low = title.lower()
    for month, stem in _MONTH_STEMS:
        if stem in low:
            return month - 1
    return None


def _stale_period_titles(titles: list[str]) -> list[str]:
    """Названия документов с прошедшим месяцем/годом прямо в имени.

    «Акции май 2026» в сентябре — почти наверняка неактуально. Бот по
    промпту называет период, а админка подсказывает, что пора обновить.
    """
    today = date.today()
    stale: list[str] = []
    for title in titles:
        years = [int(y) for y in _YEAR_RE.findall(title)]
        month = _month_from_title(title)
        if years and max(years) < today.year:
            stale.append(title)
        elif month is not None and (not years or max(years) == today.year):
            # Месяц без года или при текущем годе: «май» в сентябре
            # устарел, «декабрь» — это план на будущее, не трогаем
            if month < today.month - 1:
                stale.append(title)
    return stale


async def _load_stale_period_titles() -> list[str]:
    async with session_scope() as session:
        titles = (
            await session.execute(select(Document.title).order_by(Document.title))
        ).scalars().all()
    return _stale_period_titles(list(titles))


async def _health_checks() -> list[tuple[str, bool, str]]:
    from app.generation import llm

    settings = get_settings()
    checks: list[tuple[str, bool, str]] = []

    ok, detail = await llm.health_check()
    checks.append(("Языковая модель", ok, detail if ok else f"недоступна ({detail})"))

    index = get_vector_index()
    if not index.is_loaded:
        await index.load()
    checks.append(
        (
            "Векторный индекс",
            index.size > 0,
            f"{index.size} фрагм." if index.size else "пуст, нужна индексация",
        )
    )

    profile = settings.profile()
    checks.append(
        (
            "Эмбеддинги",
            True,
            f"{settings.embedding_provider}, {profile.name}, dim {profile.dim}",
        )
    )

    knowledge = settings.knowledge_path
    files = (
        [p for p in knowledge.rglob("*") if p.suffix.lower() in SUPPORTED_EXTENSIONS]
        if knowledge.exists()
        else []
    )
    checks.append(("Папка knowledge", bool(files), f"{len(files)} файлов"))
    return checks


async def _render(callback: CallbackQuery, text: str, markup) -> None:
    """Перерисовать дашборд. Одинаковый текст Telegram отвергает."""
    from app.channels.telegram.panels import render
    await render(callback, text, markup)


@router.message(Command("admin"))
@admin_only
async def cmd_admin(message: Message, user: User, is_admin: bool) -> None:
    await _home_panel(message)


async def _home_panel(event) -> None:
    from app.channels.telegram.panels import render
    data = await _dashboard_data()
    text, markup = views.home(data)
    await render(event, text, markup)


@router.message(Command("stats"))
@admin_only
async def cmd_stats(message: Message, user: User, is_admin: bool) -> None:
    text, markup = views.stats(await _dashboard_data())
    await message.answer(text, reply_markup=markup)


@router.message(Command("reindex"))
@admin_only
async def cmd_reindex(message: Message, user: User, is_admin: bool) -> None:
    status = await message.answer("Запускаю индексацию…")
    await _run_ingest(message.bot, status, full=False)


@router.callback_query(F.data == "adm:home")
@admin_only
async def cb_home(callback: CallbackQuery, user: User, is_admin: bool) -> None:
    await _home_panel(callback)


@router.callback_query(F.data == "adm:knowledge")
@admin_only
async def cb_knowledge(callback: CallbackQuery, user: User, is_admin: bool) -> None:
    from app.channels.telegram.library import root_screen
    await root_screen(callback)


@router.callback_query(F.data == "adm:unanswered")
@admin_only
async def cb_unanswered(callback: CallbackQuery, user: User, is_admin: bool) -> None:
    async with session_scope() as session:
        items = await repo.list_unanswered(session, limit=8)
    text, markup = views.unanswered(items)
    await _render(callback, text, markup)


@router.callback_query(F.data == "adm:quality")
@admin_only
async def cb_quality(callback: CallbackQuery, user: User, is_admin: bool) -> None:
    async with session_scope() as session:
        cases = await repo.feedback_cases(session, limit=3)
    lines = ["<b>Качество ответов</b>", ""]
    rows = []
    if not cases:
        lines.append("Новых замечаний к ответам нет.")
    for index, (query, feedback) in enumerate(cases, 1):
        lines.extend([f"<b>{index}. {html.escape(feedback.comment or 'Не помогло')}</b>",
                      html.escape(query.question[:300]),
                      '<i>Ответ:</i> ' + html.escape((query.answer or '')[:550]), ''])
        rows.append([(f"Разобрано {index}", f"adm:quality_done:{query.id}")])
    lines.append("Отметка «Разобрано» закрывает замечание. Исправление фактов выполняется после проверки данных.")
    rows.append([("Исправить базу", "lib:search"), ("Добавить материал", "lib:add")])
    rows.append([("Вопросы без ответа", "adm:unanswered"), ("‹ Назад", "adm:home")])
    await _render(callback, '\n'.join(lines), views._kb(rows))


@router.callback_query(F.data.startswith("adm:quality_done:"))
@admin_only
async def cb_quality_done(callback: CallbackQuery, user: User, is_admin: bool) -> None:
    owner = get_bot_config().get("knowledge_review.owner_id")
    if int(user.external_id) != owner:
        await callback.answer("Разбор ведёт ответственный за базу", show_alert=True)
        return
    raw = (callback.data or '').rsplit(':', 1)[1]
    if not raw.isdigit():
        await callback.answer()
        return
    async with session_scope() as session:
        await repo.resolve_feedback_case(session, int(raw), user.id)
    await cb_quality(callback, user=user, is_admin=is_admin)


@router.callback_query(F.data.startswith("adm:review:"))
@admin_only
async def cb_review(callback: CallbackQuery, user: User, is_admin: bool) -> None:
    from app.core.knowledge_review import load_queue, csv_export, is_pending
    queue = load_queue()
    if int(user.external_id) != queue['owner_id']:
        await callback.answer("Подтверждение данных ведёт ответственный за базу", show_alert=True)
        return
    raw = (callback.data or '').rsplit(':', 1)[1]
    if raw == "export":
        await callback.message.answer_document(BufferedInputFile(csv_export(), filename="компании-данные-на-проверку.csv"),
            caption="Заполни подтверждённые значения и дату проверки. Перед применением сверим изменения с исходными материалами.")
        await callback.answer()
        return
    if not raw.isdigit():
        await callback.answer()
        return
    page = int(raw)
    issues = [i for i in queue['issues'] if is_pending(i)]
    page = min(page, max(0, (len(issues)-1)//3))
    lines = ["<b>Уточнить данные</b>", f"Ожидают подтверждения: {len(issues)}", '']
    for item in issues[page*3:page*3+3]:
        lines.extend([f"<b>{html.escape(item['id'])} · {html.escape(item['city'])}</b>",
                      html.escape(item['issue'][:500]),
                      html.escape(item.get('current_value', '')[:300]),
                      '<i>' + html.escape(Path(item['file']).name) + '</i>', ''])
    lines.append("Значения не исправляются догадкой. Пришли подтверждённые сведения или заполни список.")
    nav = []
    if page: nav.append(("‹", f"adm:review:{page-1}"))
    if (page+1)*3 < len(issues): nav.append(("›", f"adm:review:{page+1}"))
    edit_rows = [[(f"Исправить {n+1}", f"lib:issue:{item['id']}")] for n,item in enumerate(issues[page*3:page*3+3])]
    rows = edit_rows + ([nav] if nav else []) + [[("Скачать список", "adm:review:export")], [("‹ Назад", "adm:home")]]
    await _render(callback, '\n'.join(lines), views._kb(rows))


@router.callback_query(F.data.startswith("adm:resolve:"))
@admin_only
async def cb_resolve(callback: CallbackQuery, user: User, is_admin: bool) -> None:
    record_id = int(callback.data.split(":")[-1])
    async with session_scope() as session:
        await repo.resolve_unanswered(session, record_id, user.id)
        items = await repo.list_unanswered(session, limit=8)
    text, markup = views.unanswered(items)
    with contextlib.suppress(Exception):
        await callback.message.edit_text(text, reply_markup=markup)
    await callback.answer("Отмечено решённым")


@router.callback_query(F.data.startswith("adm:stats"))
@admin_only
async def cb_stats(callback: CallbackQuery, user: User, is_admin: bool) -> None:
    parts = callback.data.split(":")
    days = int(parts[2]) if len(parts) > 2 else 7
    text, markup = views.stats(await _dashboard_data(days))
    await _render(callback, text, markup)


@router.callback_query(F.data == "adm:users")
@admin_only
async def cb_users(callback: CallbackQuery, user: User, is_admin: bool) -> None:
    async with session_scope() as session:
        items = await repo.list_users(session)
    text, markup = views.users(items)
    await _render(callback, text, markup)


async def _load_user(user_id: int) -> User | None:
    async with session_scope() as session:
        return await session.get(User, user_id)


async def _queries_today(user_id: int) -> int:
    async with session_scope() as session:
        return await repo.count_recent_queries(session, user_id)


@router.callback_query(F.data.startswith("adm:user:"))
@admin_only
async def cb_user_card(callback: CallbackQuery, user: User, is_admin: bool) -> None:
    target = await _load_user(int(callback.data.split(":")[2]))
    if target is None:
        await callback.answer("Пользователь не найден", show_alert=True)
        return
    used = await _queries_today(target.id)
    text, markup = views.user_card(target, used)
    await _render(callback, text, markup)


@router.callback_query(F.data.startswith("adm:urole:"))
@admin_only
async def cb_user_role(callback: CallbackQuery, user: User, is_admin: bool) -> None:
    _, _, target_id, role = callback.data.split(":")
    async with session_scope() as session:
        target = await session.get(User, int(target_id))
        if target:
            target.role = role
            target.is_active = True
    await callback.answer("Роль обновлена")
    target = await _load_user(int(target_id))
    text, markup = views.user_card(target, await _queries_today(target.id))
    await _render(callback, text, markup)


@router.callback_query(F.data.startswith("adm:uactive:"))
@admin_only
async def cb_user_active(callback: CallbackQuery, user: User, is_admin: bool) -> None:
    _, _, target_id, flag = callback.data.split(":")
    async with session_scope() as session:
        target = await session.get(User, int(target_id))
        if target:
            target.is_active = flag == "1"
    await callback.answer("Доступ обновлён")
    target = await _load_user(int(target_id))
    text, markup = views.user_card(target, await _queries_today(target.id))
    await _render(callback, text, markup)


@router.callback_query(F.data.startswith("adm:ulimit:"))
@admin_only
async def cb_user_limit(callback: CallbackQuery, user: User, is_admin: bool) -> None:
    parts = callback.data.split(":")
    target_id = int(parts[2])
    if len(parts) < 4:
        # Открывается меню лимита
        target = await _load_user(target_id)
        if target is None:
            await callback.answer("Пользователь не найден", show_alert=True)
            return
        text, markup = views.user_limit_menu(target)
        await _render(callback, text, markup)
        return
    value = parts[3]
    if value == "input":
        # «Свой вариант»: ждём число следующим сообщением админа
        _limit_input_targets[(callback.message.chat.id, callback.from_user.id)] = target_id
        await callback.message.answer(
            "Пришли число — сколько вопросов в сутки этому пользователю. "
            "0 — без лимита. Любой другой текст отменит настройку."
        )
        await callback.answer()
        return
    target = await _load_user(target_id)
    if target is None:
        await callback.answer("Пользователь не найден", show_alert=True)
        return
    # Сохранение: cfg -> NULL, иначе число
    async with session_scope() as session:
        row = await session.get(User, target_id)
        row.daily_limit = None if value == "cfg" else max(0, int(value))
    await callback.answer("Лимит сохранён")
    target = await _load_user(target_id)
    text, markup = views.user_card(target, await _queries_today(target_id))
    await _render(callback, text, markup)


@router.message(UserLimitInput(), F.text.regexp(r"^\d{1,4}$"))
@admin_only
async def on_user_limit_value(message: Message, user: User, is_admin: bool) -> None:
    """Число после «Свой вариант» — личный лимит для выбранного пользователя."""
    limit = max(0, int(message.text.strip()))
    target_id = _limit_input_targets.pop(_limit_input_key(message), None)
    target = await _load_user(target_id) if target_id is not None else None
    if target is None:
        await message.answer("Пользователь не найден, начни заново из админки.")
        return
    async with session_scope() as session:
        row = await session.get(User, target.id)
        row.daily_limit = limit
    target = await _load_user(target.id)
    text, markup = views.user_card(target, await _queries_today(target.id))
    await message.answer(
        f"Личный лимит <b>{limit}</b> установлен.\n\n{text}",
        reply_markup=markup,
    )


@router.message(UserLimitInput())
@admin_only
async def on_user_limit_cancel(message: Message, user: User, is_admin: bool) -> None:
    """Другой текст в режиме ввода лимита — отмена (вопрос надо прислать заново)."""
    _limit_input_targets.pop(_limit_input_key(message), None)
    await message.answer("Настройка лимита отменена.")


@router.callback_query(F.data == "adm:health")
@admin_only
async def cb_health(callback: CallbackQuery, user: User, is_admin: bool) -> None:
    await callback.answer("Проверяю…")
    text, markup = views.health(await _health_checks())
    with contextlib.suppress(Exception):
        await callback.message.edit_text(text, reply_markup=markup)


@router.callback_query(F.data == "adm:reindex_confirm")
@admin_only
async def cb_reindex_confirm(callback: CallbackQuery, user: User, is_admin: bool) -> None:
    text, markup = views.reindex_confirm()
    await _render(callback, text, markup)


@router.callback_query(F.data.in_({"adm:ingest", "adm:reindex_full"}))
@admin_only
async def cb_ingest(callback: CallbackQuery, user: User, is_admin: bool) -> None:
    full = callback.data == "adm:reindex_full"
    await callback.answer()
    status = await callback.message.answer(
        "Полная переиндексация…" if full else "Проверяю изменения…"
    )
    await _run_ingest(callback.bot, status, full=full)


async def _run_ingest(bot: Bot, status: Message, full: bool, *, already_locked: bool = False) -> None:
    """Индексация с живым прогрессом в одном сообщении."""
    if _ingest_lock.locked() and not already_locked:
        await status.edit_text("Индексация уже идёт. Дождитесь завершения.")
        return

    async with contextlib.AsyncExitStack() as stack:
        if not already_locked:
            await stack.enter_async_context(_ingest_lock)
        last_edit = 0.0
        from app.core.library_editor import store, write_json
        job = {'status': 'running', 'stage': 'Подготовка', 'current': 0, 'total': 0,
               'started_at': datetime.now(timezone.utc).isoformat()}
        write_json(store() / 'last-update.json', job)

        async def on_progress(progress: IngestProgress) -> None:
            nonlocal last_edit
            now = time.monotonic()
            # Telegram ограничивает частоту правок
            if now - last_edit < 1.5 and progress.stage != "Готово":
                return
            last_edit = now
            job.update(stage=progress.stage, current=progress.current, total=progress.total, percent=progress.percent)
            write_json(store() / 'last-update.json', job)
            bar = progress_bar(progress.percent)
            name = html.escape(progress.file_name[:60]) if progress.file_name else ""
            body = (
                f"<b>{progress.stage}</b>\n\n"
                f"<code>{bar}</code> {progress.percent}%\n"
                f"{progress.current} из {progress.total}"
            )
            if name:
                body += f"\n<i>{name}</i>"
            with contextlib.suppress(Exception):
                await status.edit_text(body)

        try:
            ingestor = Ingestor()
            result = await ingestor.run(full=full, on_progress=on_progress)
        except Exception as exc:  # noqa: BLE001 — показать причину, а не молчать
            log.exception("Индексация упала")
            job.update(status='error', stage='Обновление прервано', finished_at=datetime.now(timezone.utc).isoformat())
            write_json(store() / 'last-update.json', job)
            await status.edit_text(
                f"<b>⚠ Индексация не завершена</b>\n\n<code>{html.escape(str(exc)[:400])}</code>"
            )
            return

        if result.changed:
            await get_vector_index().invalidate()
            from app.retrieval.search import reset_sections_cache
            from app.ingest.stemming import reset_sections_cache as reset_stemming_cache
            reset_sections_cache()
            reset_stemming_cache()

        from app.core.library_editor import update_statuses
        await update_statuses()
        job.update(status='ok' if result.ok else 'error', stage='Завершено' if result.ok else 'Завершено с ошибками',
                   finished_at=datetime.now(timezone.utc).isoformat(), errors=result.errors[:5],
                   changed=result.files_changed, duration_seconds=result.duration_seconds,
                   embeddings_computed=result.embeddings_computed, embeddings_from_cache=result.embeddings_from_cache)
        write_json(store() / 'last-update.json', job)
        await status.edit_text(_ingest_report(result), reply_markup=keyboards.back("lib:root"))


def _ingest_report(result: IngestResult) -> str:
    mark = views.OK if result.ok else views.WARN
    lines = [
        f"{mark} <b>Индексация завершена</b>",
        "",
        f"файлов: {result.files_total} "
        f"(изменено {result.files_changed}, без изменений {result.files_skipped})",
        f"фрагментов: {result.chunks_created}",
    ]
    if result.files_deleted:
        lines.append(f"удалено документов: {result.files_deleted}")
    lines.append(
        f"токенов в самом крупном: {result.max_token_count} из {result.token_limit}"
    )
    if result.embeddings_computed or result.embeddings_from_cache:
        lines.append(
            f"векторов: {result.embeddings_computed} новых, "
            f"{result.embeddings_from_cache} из кэша"
        )
    lines.append(f"время: {result.duration_seconds:.1f} с")

    if result.warnings:
        lines += ["", f"<b>Предупреждения ({len(result.warnings)})</b>"]
        lines.extend(f"• {html.escape(w[:200])}" for w in result.warnings[:5])
    if result.errors:
        lines += ["", f"<b>Ошибки ({len(result.errors)})</b>"]
        lines.extend(f"• {html.escape(e[:200])}" for e in result.errors[:5])

    return "\n".join(lines)


@router.message(F.document)
@admin_only
async def on_document(message: Message, user: User, is_admin: bool) -> None:
    """Приём документа файлом: сохранить в knowledge/ и проиндексировать."""
    config = get_bot_config()
    settings = get_settings()
    document = message.document

    suffix = Path(document.file_name or "").suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        await message.answer(
            f"Формат {suffix or '—'} не поддерживается.\n"
            f"Можно: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )
        return

    max_mb = float(config.get("ingest.max_file_mb", 20))
    if document.file_size and document.file_size / 1_048_576 > max_mb:
        await message.answer(f"Файл больше лимита {max_mb:.0f} МБ.")
        return

    section = str(config.get("ingest.upload_section", "inbox"))
    try:
        target = safe_upload_path(settings.knowledge_path, section, document.file_name)
    except ValueError:
        await message.answer("Недопустимое имя файла или раздела.")
        return
    target.parent.mkdir(parents=True, exist_ok=True)

    status = await message.answer(f"Сохраняю <code>{html.escape(document.file_name)}</code>…")
    temporary = None
    try:
        file = await message.bot.get_file(document.file_id)
        if not file.file_path:
            raise ValueError("Telegram не вернул файл")
        with tempfile.NamedTemporaryFile(dir=target.parent, prefix=".upload-", suffix=".tmp", delete=False) as fh:
            temporary = Path(fh.name)
        await message.bot.download_file(file.file_path, destination=temporary)
        if temporary.stat().st_size > max_mb * 1_048_576:
            raise ValueError("Файл превышает лимит размера")
        temporary.replace(target)
    except Exception as exc:  # noqa: BLE001
        log.exception("Не удалось скачать документ")
        await status.edit_text(f"⚠ Не удалось сохранить файл: {html.escape(str(exc)[:200])}")
        return
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)

    await status.edit_text(
        f"Сохранён в <code>{section}/{html.escape(document.file_name)}</code>\n"
        "Индексирую…"
    )
    await _run_ingest(message.bot, status, full=False)


def safe_upload_path(root: Path, section: str, filename: str | None) -> Path:
    if not filename or filename in (".", "..") or any(c in filename for c in '/\\:\x00'):
        raise ValueError("Недопустимое имя")
    if any(ord(c) < 32 for c in filename):
        raise ValueError("Недопустимое имя")
    root = root.resolve()
    target = (root / section / filename).resolve()
    if not target.is_relative_to(root) or target == root:
        raise ValueError("Файл за пределами базы знаний")
    return target


@router.message(Command("invite"))
@admin_only
async def cmd_invite(message: Message, user: User, is_admin: bool) -> None:
    async with session_scope() as session:
        invite = await repo.create_invite(session, user.id)
        code = invite.code
    await message.answer(f"Одноразовое приглашение: <code>{code}</code>\nОтправьте этот код сотруднику.")
