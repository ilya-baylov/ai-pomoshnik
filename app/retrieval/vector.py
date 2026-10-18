"""Векторный поиск: полный перебор через numpy.

Для базы одной компании (до нескольких десятков тысяч фрагментов) индекс
целиком лежит в памяти: 2000 фрагментов при размерности 1536 — это 12 МБ,
а поиск занимает единицы миллисекунд. Внешняя векторная БД тут ничего не
ускорит, только добавит инфраструктуру.
"""

from __future__ import annotations

import asyncio
import logging

import numpy as np
from sqlalchemy import select

from app.embeddings.base import unpack
from app.storage.db import session_scope
from app.storage.models import Chunk

log = logging.getLogger(__name__)


class VectorIndex:
    """Матрица векторов в памяти, перезагружается после индексации."""

    def __init__(self) -> None:
        self._matrix: np.ndarray | None = None
        self._chunk_ids: list[int] = []
        self._lock = asyncio.Lock()

    @property
    def size(self) -> int:
        return len(self._chunk_ids)

    @property
    def is_loaded(self) -> bool:
        return self._matrix is not None

    async def load(self) -> None:
        async with self._lock:
            await self._load_unlocked()

    async def _load_unlocked(self) -> None:
        from app.storage.db import check_embedding_model
        await check_embedding_model()
        async with session_scope() as session:
            rows = (
                await session.execute(
                    select(Chunk.id, Chunk.embedding)
                    .where(Chunk.embedding.is_not(None))
                    .order_by(Chunk.id)
                )
            ).all()

        if not rows:
            self._matrix = None
            self._chunk_ids = []
            log.info("Векторный индекс пуст")
            return

        self._chunk_ids = [row[0] for row in rows]
        self._matrix = np.vstack([unpack(row[1]) for row in rows])
        log.info(
            "Векторный индекс: %s фрагм., %.1f МБ",
            len(self._chunk_ids),
            self._matrix.nbytes / 1_048_576,
        )

    async def invalidate(self) -> None:
        """Перечитать индекс — вызывается после индексации."""
        await self.load()

    async def search(
        self,
        query: np.ndarray,
        top_k: int,
        allowed: set[int] | None = None,
    ) -> list[tuple[int, float]]:
        """Ближайшие фрагменты. allowed ограничивает поиск подмножеством.

        Ограничение нужно, когда в вопросе назван город: тогда поиск идёт
        внутри его раздела и находит релевантное там, а не «что-нибудь».
        """
        if self._matrix is None:
            await self.load()
        if self._matrix is None:
            return []

        matrix = self._matrix
        vector = np.asarray(query, dtype=np.float32)

        if vector.shape[0] != matrix.shape[1]:
            log.error(
                "Размерность запроса %s не совпадает с индексом %s",
                vector.shape[0],
                matrix.shape[1],
            )
            return []

        # Векторы уже нормализованы, поэтому скалярное произведение
        # равно косинусной близости
        scores = matrix @ vector

        if allowed is not None:
            if not allowed:
                return []
            mask = np.fromiter(
                (cid in allowed for cid in self._chunk_ids),
                dtype=bool,
                count=len(self._chunk_ids),
            )
            scores = np.where(mask, scores, -1.0)
        limit = min(top_k, len(scores))
        top = np.argpartition(-scores, limit - 1)[:limit]
        top = top[np.argsort(-scores[top])]
        return [(self._chunk_ids[i], float(scores[i])) for i in top]


_index: VectorIndex | None = None


def get_vector_index() -> VectorIndex:
    global _index
    if _index is None:
        _index = VectorIndex()
    return _index
