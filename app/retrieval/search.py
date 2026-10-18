"""Гибридный поиск: вектор + FTS5, объединение через RRF.

Оценки двух поисков несопоставимы по шкале (косинус 0..1 против BM25),
поэтому объединяем по позициям в списках, а не по значениям.
"""

from __future__ import annotations

import logging
import re
from datetime import date

from sqlalchemy import select

from app.core.config import get_bot_config
from app.core.contracts import RetrievedChunk
from app.embeddings.factory import get_embedding_provider
from app.ingest.stemming import stem_word
from app.retrieval.lexical import search_lexical
from app.retrieval.vector import get_vector_index
from app.storage.db import session_scope
from app.storage.models import Chunk, Document

log = logging.getLogger(__name__)


def expand_synonyms(query: str, synonyms: dict[str, list[str]]) -> str:
    """Дописать к запросу синонимы компании.

    Сравнение по основам слов, чтобы «зп» и «командировки» тоже срабатывали.
    """
    if not synonyms:
        return query
    stems = {stem_word(w) for w in query.split()}
    extra: list[str] = []
    for key, values in synonyms.items():
        if stem_word(key) in stems or key.lower() in query.lower():
            extra.extend(values)
    return f"{query} {' '.join(extra)}" if extra else query


def _rrf(ranked: list[tuple[int, float]], k: int) -> dict[int, float]:
    return {
        chunk_id: 1.0 / (k + position)
        for position, (chunk_id, _) in enumerate(ranked, start=1)
    }


# Разделы базы (города) — читаются один раз на процесс
_sections_cache: dict[str, str] | None = None


def reset_sections_cache() -> None:
    """Сбросить кэш разделов: после индексации состав мог измениться."""
    global _sections_cache
    _sections_cache = None


async def _known_sections() -> dict[str, str]:
    """Слово раздела (в нижнем регистре) -> имя раздела.

    Дефисные названия разбираются на части: «Санкт-Петербург» находится
    и по «Петербурге».
    """
    global _sections_cache
    if _sections_cache is not None:
        return _sections_cache
    async with session_scope() as session:
        rows = (
            await session.execute(
                select(Document.section).where(Document.section.is_not(None)).distinct()
            )
        ).all()
    cache: dict[str, str] = {}
    for (section,) in rows:
        name = str(section)
        for word in name.replace("-", " ").split():
            if len(word) > 2:
                cache[word.lower()] = name
    _sections_cache = cache
    return cache


async def section_from_query(query: str) -> str | None:
    """Раздел, названный в вопросе: «расписание в Ростове» -> Ростов.

    В мультигородской базе одноимённые документы («Расписание мастеров»)
    есть у каждого города, а тело фрагмента города не содержит — поэтому
    названный город решает, где искать.

    Сопоставление по префиксу, а не по стемму: русский стеммер режет
    «Ростов» до «рост», а «Ростове» — до «ростов», и точное совпадение
    не срабатывает.
    """
    matched = await sections_from_query(query)
    return next(iter(matched)) if len(matched) == 1 else None


def normalize_query(query: str) -> str:
    """Городские сокращения и частая опечатка без изменения самой базы."""
    aliases = {
        r"питер(?:а|е|у|ом)?|спб|петербург(?:а|е|у|ом)?": "Санкт-Петербург",
        r"новосиб(?:а|е|у|ом)?|нск": "Новосибирск",
        r"екб": "Екатеринбург",
        r"мск": "Москва",
        r"крск": "Красноярск",
    }
    for pattern, city in aliases.items():
        query = re.sub(rf"(?<![\w-])(?:{pattern})(?![\w-])", city, query, flags=re.I)
    return re.sub(r"\bаккц", "акц", query, flags=re.I)


async def sections_from_query(query: str) -> set[str]:
    sections = await _known_sections()
    query = normalize_query(query)
    words: list[str] = []
    for raw in query.split():
        cleaned = raw.lower().strip(",.!?;:()«»\"'")
        words.append(cleaned)
        if "-" in cleaned:
            words.extend(part for part in cleaned.split("-") if part)
    matched: set[str] = set()
    for word in words:
        if len(word) < 3:
            continue
        if word in sections:
            matched.add(sections[word])
            continue
        for name, section in sections.items():
            # Только полная основа города + падежное окончание.
            # Стемм «нов» ошибочно связывал «нового» с двумя городами.
            bases = {name}
            if name.endswith(("а", "я", "ь", "й")):
                bases.add(name[:-1])
            endings = {"", "а", "я", "у", "ю", "е", "и", "ы", "ом", "ем", "ой", "ою", "ами", "ям", "ях", "ах"}
            if any(len(base) >= 3 and word.startswith(base) and word[len(base):] in endings for base in bases):
                matched.add(section)
    return matched


async def search(query: str, limit: int | None = None, *, advice: bool = False) -> list[RetrievedChunk]:
    config = get_bot_config()
    query = normalize_query(query)
    # Полный обзор акций нельзя составить из восьми случайных строк:
    # выбираем таблицы по городу и актуальности, без вызова эмбеддингов.
    from app.retrieval.promotion_catalog import promotion_catalog
    from app.retrieval.staff_catalog import staff_catalog
    from app.retrieval.career_catalog import career_catalog
    from app.retrieval.document_catalog import document_catalog
    from app.retrieval.book_catalog import book_catalog, named_books, book_label
    from app.retrieval.knowledge_inventory import knowledge_inventory

    sections = await sections_from_query(query)
    requested_books = named_books(query)
    catalog = await knowledge_inventory(query)
    if catalog is None:
        catalog = await book_catalog(query)
    # Questions about promotion, grades and career advancement must reach the
    # policy source before the generic staff directory interprets "top" as a
    # request for every top specialist in the city.
    if catalog is None and not requested_books:
        catalog = await career_catalog(query)
    if catalog is None and not requested_books:
        catalog = await promotion_catalog(query, sections)
    if catalog is None and not requested_books:
        catalog = await staff_catalog(query, sections)
    if catalog is None and not requested_books:
        catalog = await document_catalog(query, sections)
    if catalog is not None:
        return catalog
    provider = get_embedding_provider()

    top_k_vector = int(config.get("retrieval.top_k_vector", 20))
    top_k_lexical = int(config.get("retrieval.top_k_lexical", 20))
    top_k_final = limit or int(config.get("retrieval.top_k_final", 5))
    if advice:
        top_k_final = min(top_k_final, 4)
    rrf_k = int(config.get("retrieval.rrf_k", 60))
    priority_weight = float(config.get("retrieval.priority_weight", 0.02))
    context_bonus = float(config.get("retrieval.context_bonus", 0.004))

    expanded = expand_synonyms(query, config.synonyms)

    wanted_section = await section_from_query(query)
    section_ids: set[int] | None = None
    requested_books = named_books(query)
    if requested_books:
        async with session_scope() as session:
            section_ids = set((await session.execute(select(Chunk.id).join(Document)
                .where(Document.section == 'Продажи', Document.title.in_(requested_books)))).scalars().all())
    elif wanted_section:
        async with session_scope() as session:
            aliases = (await session.execute(select(Document.section).distinct())).scalars().all()
            aliases = [s for s in aliases if s and s.casefold() == wanted_section.casefold()]
            rows_ids = (
                await session.execute(
                    select(Chunk.id)
                    .join(Document, Chunk.document_id == Document.id)
                    .where(Document.section.in_(aliases))
                )
            ).all()
        section_ids = {row[0] for row in rows_ids}

    vector = await provider.embed_query(expanded)
    index = get_vector_index()
    vector_hits = await index.search(vector, top_k_vector)
    section_hits: list[tuple[int, float]] = []
    if section_ids:
        # Отдельный проход внутри названного города: там фрагменты получают
        # настоящую косинусную близость, а не остатки общего рейтинга
        section_hits = await index.search(vector, top_k_vector, allowed=section_ids)
        best: dict[int, float] = {cid: score for cid, score in vector_hits}
        for cid, score in section_hits:
            best[cid] = max(best.get(cid, -1.0), score)
        vector_hits = sorted(best.items(), key=lambda kv: kv[1], reverse=True)
    lexical_hits = await search_lexical(expanded, top_k_lexical)

    if not vector_hits and not lexical_hits:
        return []

    fused = _rrf(vector_hits, rrf_k)
    for chunk_id, score in _rrf(lexical_hits, rrf_k).items():
        fused[chunk_id] = fused.get(chunk_id, 0.0) + score

    # Косинусная близость лучшего вектора — понятная метрика для порога
    # уверенности, в отличие от суммы RRF, у которой нет смысла в абсолюте
    cosine_by_id = dict(vector_hits)

    candidate_ids = sorted(fused, key=lambda cid: fused[cid], reverse=True)[
        # Пул кандидатов с запасом: городской документ может стоять
        # ниже по RRF, а приоритет раздела поднимет его уже здесь
        : max(top_k_final * 6, 60)
    ]
    if section_hits:
        # Городские фрагменты включаем принудительно: по общему рейтингу
        # их выдавливают документы других городов
        forced = [cid for cid, _ in section_hits[: top_k_final * 2]]
        candidate_ids = list(dict.fromkeys(candidate_ids + forced))

    async with session_scope() as session:
        rows = (
            await session.execute(
                select(
                    Chunk.id,
                    Chunk.text,
                    Chunk.breadcrumbs,
                    Chunk.page,
                    Document.title,
                    Document.section,
                    Document.priority,
                    Document.valid_until,
                    Document.tags,
                )
                .join(Document, Chunk.document_id == Document.id)
                .where(Chunk.id.in_(candidate_ids))
            )
        ).all()

        if requested_books:
            rows = [r for r in rows if r[4] in requested_books and r[5] == 'Продажи']
        # Город назван в вопросе — оставляем его раздел и общие документы
        # (корпоративные правила лежат вне городских папок)
        if wanted_section:
            filtered = [r for r in rows if r[5] is None or r[5].casefold() == wanted_section.casefold()]
            rows = filtered
            log.info("Вопрос про раздел «%s»: сузил поиск", wanted_section)


    # Основы слов запроса: совпадение с названием документа или разделом
    # поднимает нужный город. Без этого «расписание в Ростове» вытягивает
    # расписания других городов — в теле фрагмента города обычно нет
    query_stems = {stem_word(word) for word in query.split() if len(word) > 2}

    today = date.today()
    ranks: dict[int, float] = {}
    results: list[RetrievedChunk] = []
    for row in rows:
        (chunk_id, text, crumbs, page, title, section, priority, valid_until, tags) = row
        context = f"{title} {section or ''} {crumbs or ''}"
        context_stems = {stem_word(word) for word in context.split() if len(word) > 2}
        overlap = len(query_stems & context_stems)
        rank_score = (
            fused.get(chunk_id, 0.0)
            + priority * priority_weight
            + overlap * context_bonus
        )
        # Город назван прямо — его документы важнее общих: RRF даёт
        # максимум ~0.016, поэтому 0.05 решает исход
        if wanted_section and section and section.casefold() == wanted_section.casefold():
            rank_score += 0.05
        # Приоритет раздела из конфига: книги по продажам не должны
        # перебивать телефоны и расписания
        rank_score += float(
            config.get("retrieval.section_priority", {}).get(section or "", 0)
        ) if not advice or section != 'Продажи' else 0.0
        if advice and any(tag in (tags or []) for tag in ('promotions', 'services', 'schedule', 'contacts', 'staff_update')):
            continue
        ranks[chunk_id] = rank_score
        stale = valid_until is not None and valid_until < today
        results.append(
            RetrievedChunk(
                chunk_id=chunk_id,
                text=text[:6000] if advice else text,
                title=book_label(title) if section == 'Продажи' else title,
                breadcrumbs=crumbs,
                page=page,
                # score — косинус, если фрагмент нашёл векторный поиск;
                # иначе только лексический, и уверенность считаем по RRF
                score=cosine_by_id.get(chunk_id, rank_score),
                section=section,
                is_stale=stale,
                valid_until=valid_until.isoformat() if valid_until else None,
                is_current="current" in (tags or []),
            )
        )

    results.sort(key=lambda c: ranks.get(c.chunk_id, 0.0), reverse=True)
    return results[:top_k_final]
