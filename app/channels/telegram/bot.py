"""Точка входа Telegram-бота."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.types import BotCommand

from app.channels.telegram import admin, handlers, library
from app.channels.telegram.access import AccessMiddleware
from app.channels.telegram.notifications import notify_ingest_done, notify_startup
from app.core.config import get_bot_config, get_settings
from app.ingest.loaders.registry import SUPPORTED_EXTENSIONS
from app.retrieval.vector import get_vector_index
from app.storage.db import EmbeddingModelMismatch, check_embedding_model, init_db

log = logging.getLogger(__name__)

# Нативное меню Telegram вместо клавиатуры в чате
STAFF_COMMANDS = [
    BotCommand(command="help", description="Что я знаю"),
    BotCommand(command="new", description="Начать заново"),
    BotCommand(command="profile", description="Мой город, филиал и должность"),
]


async def _startup_checks() -> list[tuple[str, bool, str]]:
    from app.generation import llm

    settings = get_settings()
    checks: list[tuple[str, bool, str]] = []

    ok, detail = await llm.health_check()
    checks.append(
        ("Языковая модель", ok, settings.llm_model if ok else f"недоступна ({detail})")
    )

    index = get_vector_index()
    await index.load()
    checks.append(
        (
            "База знаний",
            index.size > 0,
            f"{index.size} фрагм." if index.size else "пуста, нужна индексация",
        )
    )

    profile = settings.profile()
    checks.append(
        (
            "Эмбеддинги",
            True,
            f"{settings.embedding_provider}, dim {profile.dim}",
        )
    )
    return checks


async def _watch_index_version(bot: Bot) -> None:
    """Следить за версией индекса и перечитывать его после чужой индексации."""
    from app.retrieval.search import reset_sections_cache
    from app.retrieval.vector import get_vector_index
    from app.storage.db import get_meta, session_scope

    last_seen = None
    while True:
        try:
            async with session_scope() as session:
                version = await get_meta(session, "index_version")
            current = int(version or 0)
            if last_seen is not None and current != last_seen:
                log.info("Индекс обновлён (версия %s), перечитываю", current)
                await get_vector_index().invalidate()
                reset_sections_cache()
            last_seen = current
        except Exception:  # noqa: BLE001
            log.exception("Не удалось проверить версию индекса")
        await asyncio.sleep(60)


async def _watch_knowledge(bot: Bot) -> None:
    """Периодическая проверка папки knowledge/, если включена в конфиге."""
    config = get_bot_config()
    interval = int(config.get("ingest.watch_interval_minutes", 0))
    if interval <= 0:
        return

    from app.ingest.pipeline import Ingestor
    from app.ingest.stemming import reset_sections_cache
    from app.retrieval.search import reset_sections_cache as reset_search_cache

    log.info("Автопроверка папки knowledge каждые %s мин.", interval)
    while True:
        await asyncio.sleep(interval * 60)
        try:
            async with admin._ingest_lock:
                result = await Ingestor().run()
            if result.changed:
                await get_vector_index().invalidate()
                reset_sections_cache()
                reset_search_cache()
            await notify_ingest_done(bot, result)
        except Exception:  # noqa: BLE001
            log.exception("Автоиндексация не удалась")


async def main() -> None:
    settings = get_settings()
    from app.observability.logging import setup_logging

    setup_logging("bot", settings.log_level)
    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )

    if not settings.telegram_bot_token:
        raise SystemExit(
            "Не задан TELEGRAM_BOT_TOKEN.\n"
            "Скопируйте .env.example в .env и заполните значения."
        )
    if not settings.admin_ids:
        log.warning(
            "ADMIN_IDS пуст: панель управления будет недоступна. "
            "Узнайте свой ID у @userinfobot и добавьте в .env"
        )

    lock = acquire_instance_lock()
    if lock is None:
        raise SystemExit(
            "Бот уже запущен в другом процессе.\n"
            "Второй экземпляр приведёт к потере сообщений, запуск отменён.\n"
            "Если это ошибка и прошлый процесс завершён некорректно — "
            "удалите файл data/bot.lock"
        )

    try:
        await _run_bot(settings)
    finally:
        release_instance_lock(lock)
        from app.storage import db
        if db._engine is not None:
            await db._engine.dispose()


async def _run_bot(settings) -> None:
    await init_db()
    try:
        await check_embedding_model()
    except EmbeddingModelMismatch as exc:
        raise SystemExit(str(exc)) from exc

    bot = Bot(
        token=settings.telegram_bot_token,
        default=DefaultBotProperties(parse_mode=ParseMode.HTML),
    )
    try:
        await _poll_bot(bot)
    finally:
        await bot.session.close()


async def _poll_bot(bot: Bot) -> None:
    dispatcher = Dispatcher()

    dispatcher.include_router(library.router)
    dispatcher.include_router(admin.router)
    dispatcher.include_router(handlers.router)

    middleware = AccessMiddleware()
    dispatcher.message.middleware(middleware)
    dispatcher.callback_query.middleware(middleware)

    await bot.set_my_commands(STAFF_COMMANDS)

    checks = await _startup_checks()
    for label, ok, detail in checks:
        log.info("%s %s: %s", "OK " if ok else "!! ", label, detail)
    await notify_startup(bot, checks)

    watcher = asyncio.create_task(_watch_index_version(bot))
    knowledge_watcher = asyncio.create_task(_watch_knowledge(bot))
    try:
        formats = ", ".join(sorted(e.lstrip(".") for e in SUPPORTED_EXTENSIONS))
        log.info("Бот запущен. Поддерживаемые форматы: %s", formats)
        await dispatcher.start_polling(bot)
    finally:
        for task in (watcher, knowledge_watcher):
            task.cancel()
        for task in (watcher, knowledge_watcher):
            try:
                await task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass


def acquire_instance_lock() -> object | None:
    """Файловая блокировка: защита от второго экземпляра бота."""
    import sys
    from app.core.config import get_settings

    lock_path = get_settings().db_path.parent / "bot.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = lock_path.open("a")
    handle.seek(0)
    try:
        if sys.platform == "win32":
            import msvcrt
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        handle.close()
        return None
    handle.seek(0)
    handle.truncate(0)
    handle.write(str(Path(__file__).resolve()))
    handle.truncate()
    handle.flush()
    return handle


def release_instance_lock(handle: object | None) -> None:
    import sys

    if handle is None or handle.closed:
        return
    try:
        if sys.platform == "win32":
            import msvcrt
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    except OSError:  # noqa: BLE001
        pass
    finally:
        handle.close()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit) as exc:
        if isinstance(exc, SystemExit) and exc.code:
            print(f"\n{exc}")
