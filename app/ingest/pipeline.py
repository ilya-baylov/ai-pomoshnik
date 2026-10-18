"""Пайплайн индексации.

Инкрементальность: файл с неизменившимся SHA-256 пропускается, эмбеддинги
берутся из кэша по хешу фрагмента. Правка абзаца пересчитывает один фрагмент,
а не базу целиком.

Прогресс отдаётся через колбэк — на нём построен живой индикатор в админке.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Awaitable, Callable

import numpy as np
from sqlalchemy import delete, select, text as sql_text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_bot_config, get_settings
from app.embeddings.base import EmbeddingProvider, content_hash, pack
from app.embeddings.cache import get_cached, put_cached
from app.embeddings.factory import get_embedding_provider
from app.ingest.chunker import Chunker
from app.ingest.loaders.registry import SUPPORTED_EXTENSIONS, load_document
from app.ingest.metadata import build_meta, file_sha256
from app.ingest.stemming import stem_text
from app.storage.db import session_scope
from app.storage.models import Chunk, Document, IngestRun, utcnow

log = logging.getLogger(__name__)

ProgressCallback = Callable[["IngestProgress"], Awaitable[None]]


@dataclass
class IngestProgress:
    stage: str
    current: int = 0
    total: int = 0
    file_name: str = ""

    @property
    def percent(self) -> int:
        return int(self.current / self.total * 100) if self.total else 0


@dataclass
class IngestResult:
    files_total: int = 0
    files_changed: int = 0
    files_skipped: int = 0
    files_deleted: int = 0
    chunks_created: int = 0
    chunks_deleted: int = 0
    max_token_count: int = 0
    token_limit: int = 0
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    duration_seconds: float = 0.0
    embeddings_from_cache: int = 0
    embeddings_computed: int = 0

    @property
    def ok(self) -> bool:
        return not self.errors

    @property
    def changed(self) -> bool:
        return bool(self.files_changed or self.files_deleted)


class Ingestor:
    def __init__(self, provider: EmbeddingProvider | None = None):
        self.settings = get_settings()
        self.config = get_bot_config()
        self.provider = provider or get_embedding_provider()
        profile = self.settings.profile()

        # Размеры берутся из профиля провайдера: у моделей разные лимиты,
        # и подставлять их вручную — верный способ обрезать фрагменты
        target = self.config.get("chunking.target_tokens") or profile.target_chunk_tokens
        overlap = self.config.get("chunking.overlap_tokens")
        if overlap is None:
            overlap = profile.overlap_tokens

        self.token_limit = profile.max_input_tokens
        self.chunker = Chunker(
            count_tokens=self.provider.count_tokens,
            target_tokens=int(target),
            overlap_tokens=int(overlap),
            max_tokens=profile.max_input_tokens,
            min_tokens=int(self.config.get("chunking.min_tokens", 40)),
        )

    async def run(
        self, full: bool = False, on_progress: ProgressCallback | None = None
    ) -> IngestResult:
        started = utcnow()
        result = IngestResult(token_limit=self.token_limit)
        root = self.settings.knowledge_path
        root.mkdir(parents=True, exist_ok=True)
        from app.storage.db import check_embedding_model
        if not full:
            await check_embedding_model()

        files = sorted(
            p
            for p in root.rglob("*")
            if p.is_file()
            and p.suffix.lower() in SUPPORTED_EXTENSIONS
            and not p.name.startswith("~$")
        )
        result.files_total = len(files)

        async with session_scope() as session:
            run = IngestRun(started_at=started, files_total=len(files))
            session.add(run)
            await session.flush()
            run_id = run.id

        await self._report(on_progress, IngestProgress("Сканирование", 0, len(files)))

        max_mb = float(self.config.get("ingest.max_file_mb", 20))
        for number, path in enumerate(files, start=1):
            relative = str(path.relative_to(root)).replace("\\", "/")
            await self._report(
                on_progress,
                IngestProgress("Индексация", number, len(files), relative),
            )
            size_mb = path.stat().st_size / 1_048_576
            if size_mb > max_mb:
                result.warnings.append(
                    f"{relative}: {size_mb:.1f} МБ больше лимита {max_mb:.0f} МБ, пропущен"
                )
                continue
            try:
                # Пер-файловый таймаут: застрявшая сеть или гигантский PDF
                # не должны морозить остальные 150 файлов
                async with asyncio.timeout(1800):
                    await self._process_file(path, relative, root, full, result)
            except TimeoutError:
                log.warning("Файл %s не обработан за 30 минут, пропуск", relative)
                result.warnings.append(
                    f"{relative}: не обработан за 30 минут (сеть зависла?), запустите ingest повторно"
                )
            except Exception as exc:  # noqa: BLE001 — один файл не должен рушить прогон
                log.exception("Ошибка индексации %s", relative)
                result.errors.append(f"{relative}: {exc}")

        async with session_scope() as session:
            result.files_deleted, deleted_chunks = await self._remove_missing(
                session, {str(p.relative_to(root)).replace("\\", "/") for p in files}
            )
            result.chunks_deleted += deleted_chunks

        async with session_scope() as session:
            run = await session.get(IngestRun, run_id)
            if run is not None:
                run.finished_at = utcnow()
                run.files_changed = result.files_changed
                run.chunks_created = result.chunks_created
                run.chunks_deleted = result.chunks_deleted
                run.max_token_count = result.max_token_count
                run.warnings = (result.errors + result.warnings)[:50]
                run.status = "ok" if result.ok else "error"

        # Поднять версию индекса: работающий бот заметит и перечитает
        # векторы без перезапуска. Только когда данные действительно менялись
        if result.ok:
            # Метаданные обновляются только после проверки ВСЕХ векторов.
            async with session_scope() as session:
                mismatch = (await session.execute(sql_text(
                    "SELECT 1 FROM chunks WHERE embedding_model IS NULL OR embedding_dim IS NULL "
                    "OR embedding_model != :model OR embedding_dim != :dim LIMIT 1"
                ), {"model": self.provider.name, "dim": self.provider.dim})).first()
                if mismatch:
                    result.errors.append("В индексе остались несовместимые векторы; завершите полную переиндексацию")
                else:
                    from app.storage.db import _set_meta
                    await _set_meta(session, "embedding_model", self.provider.name)
                    await _set_meta(session, "embedding_dim", str(self.provider.dim))
        if result.changed:
            await self._bump_index_version()

        result.duration_seconds = (utcnow() - started).total_seconds()
        await self._report(on_progress, IngestProgress("Готово", len(files), len(files)))
        return result

    async def _bump_index_version(self) -> None:
        from app.storage.db import session_scope as _scope

        async with _scope() as session:
            await session.execute(
                sql_text(
                    "INSERT INTO meta (key, value) VALUES ('index_version', '1') "
                    "ON CONFLICT(key) DO UPDATE SET "
                    "value = CAST(CAST(value AS INTEGER) + 1 AS TEXT)"
                )
            )
        log.info("Версия индекса увеличена: бот перечитает базу")

    async def _process_file(
        self,
        path: Path,
        relative: str,
        root: Path,
        full: bool,
        result: IngestResult,
    ) -> None:
        digest = file_sha256(path)
        sidecar = path.with_suffix(path.suffix + ".meta.yaml")
        if sidecar.exists():
            import hashlib
            digest = hashlib.sha256((digest + file_sha256(sidecar)).encode()).hexdigest()

        # Короткая проверка: нужно ли вообще трогать файл
        async with session_scope() as session:
            existing_id, existing_sha = None, None
            row = (
                await session.execute(
                    select(Document.id, Document.sha256).where(Document.path == relative)
                )
            ).first()
            if row:
                existing_id, existing_sha = row

        if existing_id is not None and existing_sha == digest and not full:
            result.files_skipped += 1
            return

        # Разбор и нарезка — без открытой транзакции: SQLite пускает
        # одного писателя, а бот в это время должен отвечать людям
        loaded = load_document(path)
        result.warnings.extend(loaded.warnings)
        if not loaded.blocks:
            if existing_id is not None:
                await self._drop_empty(relative, result)
            return

        meta = build_meta(path, root, loaded.title, loaded.meta)
        drafts = self.chunker.split(loaded.blocks)
        if not drafts:
            if existing_id is not None:
                await self._drop_empty(relative, result)
            return

        # В вектор попадает контекст: название документа и раздел несут
        # город и тему, которых часто нет в теле фрагмента — без них
        # «Расписание мастеров» разных городов неразличимы.
        # Лимит проверяется по полному входу векторизации.
        embed_texts = [
            "\n".join(
                part
                for part in (meta.title, meta.section or "", d.breadcrumbs or "")
                if part
            )
            + "\n"
            + d.text
            for d in drafts
        ]
        embed_counts = [self.provider.count_tokens(t) for t in embed_texts]
        oversized = [c for c in embed_counts if c > self.token_limit]
        if oversized:
            result.errors.append(
                f"{relative}: {len(oversized)} фрагм. превышают лимит "
                f"{self.token_limit} токенов вместе с крошками "
                f"(максимум {max(oversized)})"
            )
            return

        result.max_token_count = max(
            result.max_token_count, max(d.token_count for d in drafts)
        )

        hashes = [content_hash(text) for text in embed_texts]

        # Кэш читаем коротким запросом, векторы считаем БЕЗ транзакции:
        # SQLite пускает одного писателя, и бот в это время отвечает людям
        async with session_scope() as session:
            cached = await get_cached(session, hashes, self.provider.name, self.provider.dim)
        result.embeddings_from_cache += sum(1 for h in hashes if h in cached)

        missing_idx = [i for i, h in enumerate(hashes) if h not in cached]
        unique_missing = list({hashes[i]: i for i in missing_idx}.values())
        fresh: dict[str, np.ndarray] = {}
        if unique_missing:
            matrix = await self.provider.embed_documents(
                [embed_texts[i] for i in unique_missing]
            )
            result.embeddings_computed += len(unique_missing)
            for position, index in enumerate(unique_missing):
                fresh[hashes[index]] = matrix[position]

        vectors = [cached[h] if h in cached else fresh[h] for h in hashes]

        # Запись — одной короткой транзакцией
        async with session_scope() as session:
            await put_cached(session, fresh, self.provider.name)
            existing = (
                await session.execute(
                    select(Document).where(Document.path == relative)
                )
            ).scalar_one_or_none()

            if existing:
                await session.execute(
                    sql_text(
                        "DELETE FROM chunks_fts WHERE chunk_id IN "
                        "(SELECT id FROM chunks WHERE document_id = :doc)"
                    ),
                    {"doc": existing.id},
                )
                await session.execute(delete(Chunk).where(Chunk.document_id == existing.id))
                document = existing
                document.sha256 = digest
                document.title = meta.title
                document.section = meta.section
                document.tags = meta.tags
                document.priority = meta.priority
                document.valid_until = meta.valid_until
                document.indexed_at = utcnow()
                document.format = path.suffix.lower().lstrip(".")
            else:
                document = Document(
                    path=relative,
                    format=path.suffix.lower().lstrip("."),
                    title=meta.title,
                    section=meta.section,
                    tags=meta.tags,
                    priority=meta.priority,
                    valid_until=meta.valid_until,
                    sha256=digest,
                )
                session.add(document)
                await session.flush()

            for ordinal, (draft, digest_text, vector) in enumerate(
                zip(drafts, hashes, vectors)
            ):
                chunk = Chunk(
                    document_id=document.id,
                    ordinal=ordinal,
                    text=draft.text,
                    breadcrumbs=draft.breadcrumbs or None,
                    page=draft.page,
                    token_count=draft.token_count,
                    content_hash=digest_text,
                    embedding=pack(vector),
                    embedding_model=self.provider.name,
                    embedding_dim=self.provider.dim,
                )
                session.add(chunk)
                await session.flush()
                # Крошки в поисковый текст: запрос «отпуск согласование»
                # должен находить раздел, даже если слова только в заголовке
                searchable = f"{draft.breadcrumbs or ''} {document.title} {draft.text}"
                await session.execute(
                    sql_text(
                        "INSERT INTO chunks_fts (stemmed, chunk_id) VALUES (:s, :c)"
                    ),
                    {"s": stem_text(searchable), "c": chunk.id},
                )

            document.chunk_count = len(drafts)
            result.files_changed += 1
            result.chunks_created += len(drafts)

    async def _drop_empty(self, relative: str, result: IngestResult) -> None:
        async with session_scope() as session:
            doc_id = (await session.execute(select(Document.id).where(Document.path == relative))).scalar_one_or_none()
            if doc_id is None:
                return
            ids = (await session.execute(select(Chunk.id).where(Chunk.document_id == doc_id))).scalars().all()
            await session.execute(sql_text("DELETE FROM chunks_fts WHERE chunk_id IN (SELECT id FROM chunks WHERE document_id=:doc)"), {"doc":doc_id})
            await session.execute(delete(Chunk).where(Chunk.document_id == doc_id))
            await session.execute(delete(Document).where(Document.id == doc_id))
            result.chunks_deleted += len(ids)
            result.files_changed += 1
            result.warnings.append(f"{relative}: документ пуст, прежний текст исключён из поиска")

    async def _remove_missing(
        self, session: AsyncSession, present: set[str]
    ) -> tuple[int, int]:
        """Убрать из индекса файлы, которых больше нет в папке."""
        rows = (await session.execute(select(Document.id, Document.path))).all()
        stale = [(doc_id, path) for doc_id, path in rows if path not in present]
        if not stale:
            return 0, 0

        chunks_removed = 0
        for doc_id, _ in stale:
            count = (
                await session.execute(
                    select(Chunk.id).where(Chunk.document_id == doc_id)
                )
            ).all()
            chunks_removed += len(count)
            await session.execute(
                sql_text(
                    "DELETE FROM chunks_fts WHERE chunk_id IN "
                    "(SELECT id FROM chunks WHERE document_id = :doc)"
                ),
                {"doc": doc_id},
            )
            await session.execute(delete(Document).where(Document.id == doc_id))
            await session.execute(delete(Chunk).where(Chunk.document_id == doc_id))
        return len(stale), chunks_removed

    @staticmethod
    async def _report(
        callback: ProgressCallback | None, progress: IngestProgress
    ) -> None:
        if callback is not None:
            await callback(progress)
