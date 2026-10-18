"""Выбор явно названной рабочей памятки без условного порога косинуса.

Только небольшие корпоративные документы и точное сочетание темы и роли.
Книги, цены, мастера и акции продолжают искать своими маршрутами.
"""
import re
from datetime import date
from sqlalchemy import select
from app.ingest.stemming import stem_word
from app.core.contracts import RetrievedChunk
from app.storage.db import session_scope
from app.storage.models import Chunk, Document

STOP = {'для', 'по', 'как', 'что', 'какой', 'какие', 'правило', 'теория',
        'информация', 'company', 'компания', 'мск', 'екб', 'рнд', 'спб', 'нск',
        'а', 'ди', 'и', 'в', 'на', 'должностной', 'инструкция', 'мастер'}
STOP_STEMS = {stem_word(w) for w in STOP}
SPECIFIC = {stem_word(w) for w in ('жалоба','дезинфекция','коммуникация','скрипт','лояльность')}
ROLE_PATTERNS = ('администратор', 'маникюр|педикюр|ногт', 'депиляц', 'ресниц|л[еэ]ш', 'парикмахер|волос', 'массаж', 'косметолог')

def requested_roles(text: str) -> set[int]:
    return {i for i, pattern in enumerate(ROLE_PATTERNS) if re.search(pattern, text, re.I)}

def topic_words(text: str) -> set[str]:
    return {stem_word(w) for w in re.findall(r'[^\W\d_]+', text.casefold())
            if len(w) > 2} - STOP_STEMS

def title_matches(query: str, title: str) -> bool:
    if re.search(r'\b(?:акци\w*|телефон\w*|расписани\w*|график\w*|цен[аыуе]\w*)',query.casefold()):
        return False
    q=topic_words(query); t=topic_words(title)
    overlap=q&t
    if re.search(r'\bди\b',title.casefold().replace('_',' ')):
        if re.search(r'маникюр\w*|педикюр\w*',query.casefold()):
            overlap |= topic_words('ногтевой сервис') & t
        if re.search(r'л[еэ]шм[еэ]йкер\w*',query.casefold()):
            overlap |= topic_words('ресницы') & t
        # Название должности само по себе не означает вопрос об обязанностях.
        return bool(requested_roles(query) & requested_roles(title)) and bool(re.search(r'обязан\w*|должност\w*|должен|инструкц\w*',query.casefold()))
    return len(overlap)>=2 or bool(overlap & SPECIFIC) or (len(t)==1 and bool(overlap))

async def document_catalog(query: str, sections: set[str]) -> list[RetrievedChunk] | None:
    async with session_scope() as session:
        docs=(await session.execute(select(Document))).scalars().all()
        selected=[]
        for doc in docs:
            if doc.section == 'Продажи' or any(tag in (doc.tags or []) for tag in ('promotions','services','schedule','contacts','staff_update')):
                continue
            if doc.section is not None and doc.section not in sections:
                continue
            if title_matches(query,doc.title):selected.append(doc)
        if not selected:return None
        exact = [doc for doc in selected if doc.title.casefold() in query.casefold() and doc.title != doc.section]
        selected = exact or selected
        result=[]
        for doc in selected:
            rows=(await session.execute(select(Chunk).where(Chunk.document_id==doc.id).order_by(Chunk.ordinal))).scalars().all()
            # Большие учебники выбираются обычным поиском по конкретной теме.
            is_duty = bool(re.search(r'\bди\b', doc.title.casefold().replace('_', ' ')))
            if not rows or (sum(len(r.text) for r in rows)>16000 and doc not in exact and not is_duty):continue
            parts=[(f'Страница {r.page}:\n' if r.page else '')+r.text for r in rows]
            content = '\n\n'.join(parts)
            if len(content) > 24000:
                content = content[:24000] + '\nПодборка неполная: уточни нужный раздел документа.'
            result.append(RetrievedChunk(rows[0].id,content,doc.title,None,None,
                1.,section=doc.section,is_stale=doc.valid_until is not None and doc.valid_until<date.today(),
                valid_until=doc.valid_until.isoformat() if doc.valid_until else None))
        # Без явного совпадения ничего не утверждаем.
        return result[:4] or None
