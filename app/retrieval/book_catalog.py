"""Verified book inventory and source-bounded excerpts."""
import re
from sqlalchemy import select
from app.storage.db import session_scope
from app.storage.models import Document, Chunk
from app.core.contracts import RetrievedChunk
from app.ingest.stemming import stem_word

BOOKS = {
 'batyrev-45': ('Максим Батырев (управление командой)', r'батырев|татуиров'),
 'kennedy': ('Гэвин Кеннеди (переговоры)', r'кеннеди'),
 'kuznetsov': ('Михаил Кузнецов, Елена Иванова (управление салоном красоты)', r'кузнецов'),
 'owen': ('Ричард Оуэн, Лаура Брукс (клиентский опыт и Net Promoter)', r'оуэн|брукс'),
 'reiheld': ('Фред Райхельд (лояльность и NPS)', r'райхельд|рейхельд'),
 'sewell': ('Карл Сьюэлл, Пол Браун — Клиенты на всю жизнь', r'сью[эе]лл|клиент\w* на всю жизнь'),
 'shoul-1': ('Джон Шоул — Первоклассный сервис как конкурентное преимущество', r'шоул'),
 'shoul-2': ('Джон Шоул — Лояльный клиент: Как превратить разгневанного покупателя в счастливого за 60 секунд', r'шоул|лояльный клиент'),
 'vavilov': ('Владислав Вавилов (реклама салона красоты)', r'вавилов'),
 'zhigiliy': ('Евгений Жигилий — Мастер звонка', r'жигил|мастер звонка'),
}

def book_label(title):
    return BOOKS.get(title, (title, ''))[0]

def named_books(query):
    return {key for key, (_, pattern) in BOOKS.items() if re.search(pattern, query, re.I)}

def book_request(query):
    return bool(re.search(r'книг|цитат|из базы.*продаж|продаж.*из базы', query, re.I) or named_books(query))

async def book_catalog(query):
    if not book_request(query):
        return None
    inventory = bool(re.search(r'(?:какие|каких|список|перечень|все|вся).*книг|книг\w*.*(?:есть|знаешь|доступн)', query, re.I))
    quoting = bool(re.search(r'цитат', query, re.I))
    requested = named_books(query)
    if not inventory and not quoting and not requested:
        return None
    async with session_scope() as session:
        docs = (await session.execute(select(Document).where(Document.section == 'Продажи'))).scalars().all()
        requested = named_books(query)
        docs = [d for d in docs if not requested or d.title in requested]
        if not docs:
            return None
        rows = (await session.execute(select(Chunk).where(Chunk.document_id.in_([d.id for d in docs]))
                                      .order_by(Chunk.ordinal, Chunk.id))).scalars().all()
    if not rows:
        return None
    by_id = {d.id: d for d in docs}
    if inventory:
        answer = f'Доступно **{len(docs)} книг** по работе с клиентами, продажам и управлению:\n\n'
        answer += '\n'.join('• ' + book_label(d.title) for d in docs)
        answer += '\n\nМогу разобрать нужную тему, найти короткую цитату или помочь применить идею в работе салона.'
        return [RetrievedChunk(rows[0].id, answer, 'Учебные книги', None, None, 1.0,
                               section='Продажи', direct_answer=answer)]
    stops = {'цитата','книга','база','дать','дай','короткий','продажа','какой','наш','рассказать','советует','что'}
    stops.update(w for key in requested for w in re.findall(r'[а-яё]+',BOOKS[key][0].split('—')[0].split('(')[0].casefold()))
    q = {stem_word(w) for w in re.findall(r'[а-яё]+', query.casefold()) if len(w)>3}
    q -= {stem_word(w) for w in stops}
    q = q or {stem_word(w) for w in ('клиент','сервис','продажа','лояльность')}
    scored = []
    for row in rows:
        if row.ordinal < 2 or len(row.text) < 200:
            continue
        words = {stem_word(w) for w in re.findall(r'[а-яё]+', row.text.casefold()) if len(w)>3}
        stems=[stem_word(w) for w in re.findall(r'[а-яё]+',row.text.casefold()) if len(w)>3]
        score = sum(min(stems.count(w),8) for w in q & words)
        if score:
            scored.append((score, row))
    scored.sort(key=lambda item: (-item[0], item[1].id))
    result, seen = [], set()
    for _, row in scored:
        if row.document_id in seen:
            continue
        seen.add(row.document_id)
        doc = by_id[row.document_id]
        result.append(RetrievedChunk(row.id, row.text[:3000], book_label(doc.title), row.breadcrumbs,
                                      row.page, 1.0, section='Продажи'))
        if len(result) == 3:
            break
    if not quoting:
        # Several excerpts from one explicitly named book may be needed.
        return [RetrievedChunk(row.id,row.text,book_label(by_id[row.document_id].title),row.breadcrumbs,
                    row.page,1.0,section='Продажи') for _,row in scored[:4]] or None
    # Explicit quote requests use only complete sentences copied verbatim.
    # A descriptive label in parentheses is a topic, not an invented book title.
    quoted = []
    for chunk in result:
        candidates = []
        for sentence in re.split(r'(?<=[.!?])\s+', chunk.text):
            sentence = sentence.strip()
            if not 7 <= len(sentence.split()) <= 25 or not re.search(r'[.!?]$', sentence):
                continue
            if not sentence[:1].isupper() or re.search(r'\b(?:Глава|Оглавление|ISBN|notes|видеофильм\w*|видеоурок\w*)\b', sentence, re.I):
                continue
            overlap = len(q & {stem_word(w) for w in re.findall(r'[а-яё]+', sentence.casefold())})
            if overlap >= min(2,len(q)):
                candidates.append((overlap, sentence))
        if candidates:
            sentence = max(candidates, key=lambda x: x[0])[1]
            chunk.direct_answer = f'**{chunk.title}**\n«{sentence}»'
            quoted.append(chunk)
    if quoted:
        return quoted if re.search(r'цитаты|цитат[ыу].*(?:несколько|из разных)|несколько.*цитат',query,re.I) else quoted[:1]
    return result or None
