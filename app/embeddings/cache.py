"""Кэш эмбеддингов по хешу текста.

Правка одного абзаца пересчитывает один фрагмент, а не всю базу: на Gemini
это экономит запросы, на локальной модели — минуты работы CPU.
"""

from __future__ import annotations

import numpy as np
from sqlalchemy import select
from sqlalchemy.dialects.sqlite import insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.embeddings.base import pack, unpack
from app.storage.models import EmbeddingCache


async def get_cached(
    session: AsyncSession, hashes: list[str], model: str, dim: int | None = None
) -> dict[str, np.ndarray]:
    if not hashes:
        return {}
    rows = await session.execute(
        select(EmbeddingCache.content_hash, EmbeddingCache.embedding).where(
            EmbeddingCache.content_hash.in_(hashes),
            EmbeddingCache.model == model,
        )
    )
    vectors = {h: unpack(blob) for h, blob in rows.all()}
    return {h: vec for h, vec in vectors.items() if dim is None or len(vec) == dim}


async def put_cached(
    session: AsyncSession, items: dict[str, np.ndarray], model: str
) -> None:
    if not items:
        return
    rows = [
        {"content_hash": h, "model": model, "embedding": pack(vec)}
        for h, vec in items.items()
    ]
    stmt = insert(EmbeddingCache).values(rows)
    # Один и тот же текст мог прийти из двух документов
    await session.execute(
        stmt.on_conflict_do_update(index_elements=["content_hash", "model"],
                                   set_={"embedding": stmt.excluded.embedding})
    )
