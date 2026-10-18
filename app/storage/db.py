"""Подключение к базе и инициализация схемы.

FTS5-таблица создаётся прямым SQL: SQLAlchemy не умеет объявлять виртуальные
таблицы. Синхронизация с chunks идёт из кода индексации, а не триггерами —
так понятнее, где что происходит, и легче отладить.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from sqlalchemy import text
from sqlalchemy.engine import make_url
from sqlalchemy.exc import OperationalError
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.core.config import get_settings
from app.storage.models import Base

log = logging.getLogger(__name__)

SCHEMA_VERSION = "2"

_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


def get_engine() -> AsyncEngine:
    global _engine
    if _engine is None:
        settings = get_settings()
        url = make_url(settings.database_url)
        if url.get_backend_name() != "sqlite":
            raise ValueError("Этот бот поддерживает только SQLite")
        if url.database != ":memory:":
            settings.db_path.parent.mkdir(parents=True, exist_ok=True)
            url = url.set(database=str(settings.db_path))
        _engine = create_async_engine(
            url,
            echo=False,
            # ждать блокировку до 30 с: соседний процесс не должен ронять ingest
            connect_args={"timeout": 30},
        )
    return _engine


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    global _session_factory
    if _session_factory is None:
        _session_factory = async_sessionmaker(
            get_engine(), expire_on_commit=False, class_=AsyncSession
        )
    return _session_factory


@asynccontextmanager
async def session_scope() -> AsyncIterator[AsyncSession]:
    """Сессия с автоматическим commit или rollback."""
    factory = get_session_factory()
    async with factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


FTS_TABLE_SQL = """
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    stemmed,
    chunk_id UNINDEXED,
    tokenize = "unicode61 remove_diacritics 2"
)
"""


async def init_db() -> None:
    """Создать таблицы, FTS-индекс и записать метаданные модели."""
    engine = get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        await conn.execute(text(FTS_TABLE_SQL))
        await conn.execute(text("PRAGMA journal_mode=WAL"))
        # Мягкая миграция: у давно созданных баз колонки нет
        cols = await conn.execute(text("PRAGMA table_info(users)"))
        if "daily_limit" not in {row[1] for row in cols}:
            await conn.execute(
                text("ALTER TABLE users ADD COLUMN daily_limit INTEGER")
            )

    settings = get_settings()
    profile = settings.profile()
    try:
        async with session_scope() as session:
            previous_version = await get_meta(session, "schema_version")
            if previous_version != SCHEMA_VERSION:
                await rebuild_fts_index(session)
            await _set_meta(session, "schema_version", SCHEMA_VERSION)
            # Запуск не меняет паспорт уже существующего индекса.
            has_chunks = (await session.execute(text("SELECT 1 FROM chunks LIMIT 1"))).first()
            if not has_chunks:
                await _set_meta(session, "embedding_model", profile.name)
                await _set_meta(session, "embedding_dim", str(profile.dim))
    except OperationalError as exc:
        # Индексация держит запись минутами. Схема уже создана, метаданные
        # не критичны для чтения — бот должен подниматься и отвечать.
        if "locked" not in str(exc).lower():
            raise
        log.warning("База занята индексацией, метаданные обновим позже")


async def rebuild_fts_index(session: AsyncSession) -> None:
    """Восстановить текстовый индекс из актуальных фрагментов, без API."""
    from app.ingest.stemming import stem_text
    rows = (await session.execute(text(
        "SELECT c.id, c.breadcrumbs, d.title, c.text FROM chunks c "
        "JOIN documents d ON d.id=c.document_id ORDER BY c.id"
    ))).all()
    await session.execute(text("DELETE FROM chunks_fts"))
    if rows:
        await session.execute(text(
            "INSERT INTO chunks_fts (stemmed, chunk_id) VALUES (:stemmed, :id)"
        ), [{"id": cid, "stemmed": stem_text(f"{crumbs or ''} {title} {body}")}
            for cid, crumbs, title, body in rows])
    log.info("Текстовый индекс восстановлен: %s фрагментов", len(rows))


async def _set_meta(session: AsyncSession, key: str, value: str) -> None:
    await session.execute(
        text(
            "INSERT INTO meta (key, value) VALUES (:k, :v) "
            "ON CONFLICT(key) DO UPDATE SET value = :v"
        ),
        {"k": key, "v": value},
    )


async def get_meta(session: AsyncSession, key: str) -> str | None:
    row = await session.execute(
        text("SELECT value FROM meta WHERE key = :k"), {"k": key}
    )
    found = row.first()
    return found[0] if found else None


class EmbeddingModelMismatch(RuntimeError):
    """База проиндексирована другой моделью — векторы несовместимы."""


async def check_embedding_model() -> None:
    """Сверить модель в базе с текущей настройкой.

    Векторные пространства разных моделей несовместимы: без этой проверки
    бот выдавал бы правдоподобный мусор вместо ошибки.
    """
    settings = get_settings()
    profile = settings.profile()
    async with session_scope() as session:
        stored_model = await get_meta(session, "embedding_model")
        stored_dim = await get_meta(session, "embedding_dim")
        has_chunks = (
            await session.execute(text("SELECT 1 FROM chunks LIMIT 1"))
        ).first()
        incompatible = (await session.execute(text(
            "SELECT 1 FROM chunks WHERE embedding_model IS NULL OR embedding_dim IS NULL "
            "OR embedding_model != :model OR embedding_dim != :dim LIMIT 1"
        ), {"model": profile.name, "dim": profile.dim})).first()

    if not has_chunks:
        return

    if incompatible or stored_model != profile.name or stored_dim != str(profile.dim):
        raise EmbeddingModelMismatch(
            f"База проиндексирована моделью {stored_model} "
            f"(размерность {stored_dim}), а сейчас настроена {profile.name} "
            f"(размерность {profile.dim}).\n"
            "Векторы несовместимы. Нужна полная переиндексация:\n"
            "    python -m app.cli reindex --full"
        )
