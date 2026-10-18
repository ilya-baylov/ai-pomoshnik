"""Полнотекстовый поиск через FTS5.

Ловит то, что векторный поиск пропускает: номера регламентов, артикулы,
аббревиатуры, редкие названия.
"""

from __future__ import annotations

import logging

from sqlalchemy import text as sql_text

from app.ingest.stemming import build_fts_query
from app.storage.db import session_scope

log = logging.getLogger(__name__)

# bm25() возвращает отрицательные значения: чем меньше, тем релевантнее
SEARCH_SQL = """
SELECT chunk_id, bm25(chunks_fts) AS rank
FROM chunks_fts
WHERE chunks_fts MATCH :query
ORDER BY rank
LIMIT :limit
"""


async def search_lexical(query: str, top_k: int) -> list[tuple[int, float]]:
    match_query = build_fts_query(query)
    if not match_query:
        return []

    try:
        async with session_scope() as session:
            rows = (
                await session.execute(
                    sql_text(SEARCH_SQL), {"query": match_query, "limit": top_k}
                )
            ).all()
    except Exception as exc:  # noqa: BLE001 — синтаксис FTS не должен ронять поиск
        log.warning("Полнотекстовый поиск не сработал: %s", exc)
        return []

    # Инвертируем знак, чтобы больше означало релевантнее
    return [(int(chunk_id), -float(rank)) for chunk_id, rank in rows]
