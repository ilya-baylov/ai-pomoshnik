"""Прямой поиск строк контактов и мастеров по городу и имени."""
from __future__ import annotations
import re
from datetime import date
from collections import defaultdict

from sqlalchemy import select
from app.core.contracts import RetrievedChunk
from app.ingest.stemming import stem_word
from app.storage.db import session_scope
from app.storage.models import Chunk, Document

NAME = re.compile(r"(?:^|; )(?:(?:Имя )?мастера?|Сотрудник \(наименование\)|дл):\s*([^;\n]+)", re.I)
BRANCH = re.compile(r"(?:^|; )(?:Название салона[^:;]*|Название|Салон|Филиал):\s*([^;\n]+)", re.I)


def name_match_count(name: str, words: set[str]) -> int:
    matched = set()
    for word in re.findall(r"[а-яёa-z0-9]+", name.casefold()):
        if len(word) < 3 or word in {"сотрудник", "сотрудника", "мастер", "мастера", "обновление", "копия"}:
            continue
        if word in words:
            matched.add(word)
            continue
        bases = {word}
        if word.endswith(('а', 'я', 'ь', 'й')):
            bases.add(word[:-1])
        endings = {'', 'а', 'я', 'у', 'ю', 'е', 'и', 'ы', 'ом', 'ем', 'ой', 'ою', 'ами', 'ями'}
        if any(len(base) >= 3 and w.startswith(base) and w[len(base):] in endings
               for base in bases for w in words):
            matched.add(word)
            continue
        stem = stem_word(word)
        if len(stem) >= 3 and any(stem_word(w) == stem for w in words if len(w) >= 4):
            matched.add(word)
    return len(matched)


def name_matches(name: str, words: set[str]) -> bool:
    return name_match_count(name, words) > 0


def _one_edit(a: str, b: str) -> bool:
    """One typo only; short names are never guessed."""
    if min(len(a), len(b)) < 5 or abs(len(a) - len(b)) > 1:
        return False
    if len(a) == len(b):
        return sum(x != y for x, y in zip(a, b)) == 1
    short, long = sorted((a, b), key=len)
    return any(long[:i] + long[i + 1:] == short for i in range(len(long)))


def person_match_count(name: str, words: set[str]) -> int:
    exact = name_match_count(name, words)
    if exact != 1:
        return exact
    parts = re.findall(r'[а-яёa-z0-9]+', name.casefold())
    if any(len(part) == 1 and part in words for part in parts):
        return 2
    # A typo is accepted only alongside another correctly matched name part.
    if any(not name_matches(part, words) and any(_one_edit(part, w) for w in words)
           for part in parts):
        return 2
    return exact


async def staff_catalog(query: str, sections: set[str]) -> list[RetrievedChunk] | None:
    if len(sections) > 1:
        return None
    folded = query.casefold()
    if re.search(r'обязан|должност|должен|инструкц', folded):
        return None
    if re.search(r"\b(?:телефон\w*|контакт\w*|адрес\w*|мессенджер\w*|ватсап\w*|вотсап\w*)", folded):
        category = "contacts"
    elif re.search(r"\b(?:график\w*|расписан\w*|работает|работают|выходн\w*)", folded) and not (
            re.search(r'депиляц|маникюр|педикюр|лазер|массаж|наращ|ламинир|процедур|услуг', folded)
            and not re.search(r'график|расписан|завтра|сегодня|выходн|понедельник|вторник|сред[ау]|четверг|пятниц|суббот|воскресен', folded)):
        category = "schedule"
    elif re.search(r"\b(?:мастер\w*|делает|делают|услуг\w*|процедур\w*|наращ\w*|ламинир\w*)", folded) or re.match(r'\s*кто\s+(?:такой|такая|такие)\b', folded) or (re.search(r'\bкто\b', folded) and re.search(r'депиляц|маникюр|педикюр|массаж|ресниц|бров|лазер', folded)) or re.search(r'\b(?:где\s+(?:принимает|работает)|в\s+каком\s+(?:филиале|салоне))\b',folded):
        category = "services"
    else:
        return None
    city = next(iter(sections), '')
    if not city and re.search(r'\b(?:в|из|город(?:е)?\s*:?)\s+[а-яё]', folded):
        # An explicitly named unknown location must not be silently replaced.
        return None
    city_stems = {stem_word(w) for w in re.findall(r"[а-яёa-z]+", city.casefold())}
    words = {w for w in re.findall(r"[а-яёa-z0-9]+", folded) if stem_word(w) not in city_stems}
    async with session_scope() as session:
        docs = (await session.execute(select(Document))).scalars().all()
        chosen = [d for d in docs if d.section and (not city or d.section.casefold() == city.casefold()) and category in (d.tags or [])]
        current_sections = {d.section.casefold() for d in chosen if 'current' in (d.tags or [])}
        chosen = [d for d in chosen if d.section.casefold() not in current_sections or 'current' in (d.tags or [])]
        # Увольнения — отдельные записи, не теряются при запросе о мастере.
        updates = [d for d in docs if d.section and (not city or d.section.casefold() == city.casefold())
                   and "staff_update" in (d.tags or []) and person_match_count(re.sub(re.escape(d.section), "", d.title, flags=re.I), words) >= (1 if city else 2)]
        chosen += updates
        supplemental_ids = set()
        if category == 'schedule':
            # Some service tables contain the only published schedule for a person.
            # Use those rows only when both an employee name and a schedule field match.
            supplemental = [d for d in docs if d.section and (not city or d.section.casefold() == city.casefold())
                            and 'services' in (d.tags or []) and d not in chosen]
            supplemental_ids = {d.id for d in supplemental}
            chosen += supplemental
        if not chosen:
            return None
        rows = (await session.execute(select(Chunk.document_id, Chunk.id, Chunk.text, Chunk.breadcrumbs)
                    .where(Chunk.document_id.in_([d.id for d in chosen])).order_by(Chunk.ordinal, Chunk.id))).all()
        grouped = defaultdict(list)
        for row in rows:
            if row[0] in supplemental_ids and not re.search(
                    r'(?:^|; )(?:(?:График|Расписание)(?: работы)?):\s*[^;\n]+', row[2], re.I):
                continue
            grouped[row[0]].append(row)

    if category != "contacts":
        named = [(doc, row, score) for doc in chosen for row in grouped[doc.id]
                 if (score := max((person_match_count(m[1], words) for m in NAME.finditer(row[2])), default=0)) >= (1 if city else 2)]
        if named:
            best_match = max(score for _, _, score in named)
            named = [item for item in named if item[2] == best_match]
            primary = [item for item in named if item[0].id not in supplemental_ids]
            named = primary or named
            allowed = {row[1] for _, row, _ in named}
            for doc in chosen:
                if doc not in updates:
                    grouped[doc.id] = [r for r in grouped[doc.id] if r[1] in allowed]
            # One unambiguous service row is safer to render than to paraphrase.
            if category == 'services' and not updates and len(named)==1 and len(list(NAME.finditer(named[0][1][2])))==1 and (
                    named[0][0].valid_until is None or named[0][0].valid_until >= date.today()):
                from app.retrieval.staff_filters import render_services, SPECIAL
                doc,row,_=named[0]
                if re.search(r'где\s+(?:принимает|работает)|в\s+каком\s+(?:филиале|салоне)',folded):
                    from app.retrieval.staff_filters import value
                    name=value(row[2],'мастер','имя мастера')
                    branch=value(row[2],'филиал','салон')
                    if name and branch:
                        reply=f'**{name}** указана в графике филиала **«{branch}»**.' if 'schedule' in (doc.tags or []) else f'**{name}** указана в базе в филиале **«{branch}»**.'
                        return [RetrievedChunk(row[1],row[2],doc.title,doc.section,None,1.0,
                             section=doc.section,is_current='current' in (doc.tags or []),direct_answer=reply)]
                skill=next((s for trigger,role,s in SPECIAL if re.search(trigger,folded)),None)
                reply=render_services(doc.section,[row],skill)
                if reply:
                    return [RetrievedChunk(row[1],row[2],doc.title,doc.section,None,1.0,
                         section=doc.section,is_current='current' in (doc.tags or []),direct_answer=reply)]
        elif category == 'services' and city and not updates:
            from app.retrieval.staff_filters import filter_services, render_services
            candidates = [(doc, row) for doc in chosen for row in grouped[doc.id]
                          if len(list(NAME.finditer(row[2]))) == 1]
            filtered = filter_services(query, [r for _, r in candidates])
            if filtered is None:
                return None
            selected, skill = filtered
            reply = render_services(city, selected, skill, full=bool(re.search(r'\bвсе\b|полный|подроб', folded)))
            if not reply:
                return None
            first = selected[0]
            doc = next(d for d, r in candidates if r[1] == first[1])
            return [RetrievedChunk(first[1], '\n'.join(r[2] for r in selected), doc.title,
                    city, None, 1.0, section=city, is_current='current' in (doc.tags or []),
                    direct_answer=reply)]
        elif category == 'schedule' and city and not updates:
            from app.retrieval.staff_filters import filter_services
            candidates=[r for doc in chosen if doc.id not in supplemental_ids for r in grouped[doc.id]
                        if len(list(NAME.finditer(r[2])))==1]
            schedule_query=query+' мастера'
            if re.search(r'\bтоп\b|технолог',query,re.I):
                service_rows=[r for r in rows if r[0] in supplemental_ids]
                source_levels=filter_services(query,service_rows)
                if source_levels is None or not source_levels[0]:return None
                from app.retrieval.staff_filters import value
                allowed_names={value(r[2],'мастер','имя мастера').casefold() for r in source_levels[0]}
                candidates=[r for r in candidates if value(r[2],'мастер','имя мастера').casefold() in allowed_names]
                schedule_query=re.sub(r'\bтоп\b|технолог\w*','',schedule_query,flags=re.I)
            filtered=filter_services(schedule_query,candidates)
            if filtered is None or not filtered[0]:
                return None
            allowed={r[1] for r in filtered[0]}
            for doc in chosen:
                grouped[doc.id]=[r for r in grouped[doc.id] if r[1] in allowed]
            from app.retrieval.staff_filters import render_day_schedule
            if all(d.valid_until is None or d.valid_until>=date.today() for d in chosen if grouped[d.id]):
                direct=render_day_schedule(query,filtered[0])
                if direct:
                    first=filtered[0][0]
                    doc=next(d for d in chosen if d.id==first[0])
                    return [RetrievedChunk(first[1],'\n'.join(r[2] for r in filtered[0]),doc.title,city,None,1.0,
                            section=city,direct_answer=direct)]
        elif not updates and (not city or not re.search(r'какие\s+услуги.*мастер|все.*мастер|список.*мастер|график.*мастер', folded)):
            # Общие списки процедур остаются гибридному поиску; он учитывает
            # название нужной процедуры и исключения из строки.
            return None
        else:
            # General schedule lists do not need unrelated service rows.
            for doc_id in supplemental_ids:
                grouped[doc_id] = []
    if category == "contacts":
        if not city:
            return None
        matched = {row[1] for doc in chosen for row in grouped[doc.id]
                   if (m := BRANCH.search(row[2])) and name_matches(m[1], words)}
        if matched:
            for doc in chosen:
                grouped[doc.id] = [r for r in grouped[doc.id] if r[1] in matched]

    results = []
    today = date.today()
    for doc in chosen:
        candidates = grouped[doc.id]
        if not candidates:
            continue
        selected = []; used = 0
        for _, cid, text, crumbs in candidates:
            if used + len(text) > 24_000:
                selected.append("Подборка неполная. Уточни филиал или мастера; не утверждай, что это полный список.")
                break
            typo_names = [m[1] for m in NAME.finditer(text)
                          if person_match_count(m[1], words) == 2 and name_match_count(m[1], words) == 1
                          and any(not name_matches(part, words) and any(_one_edit(part, w) for w in words)
                                  for part in re.findall(r'[а-яёa-z0-9]+', m[1].casefold()))]
            note = ''
            if typo_names:
                note = ('Вероятная опечатка в вопросе; найдена запись: ' + ', '.join(typo_names) + '. '
                        'Используй написание имени из источника, при необходимости начни «Если речь о …». '
                        'Не заявляй об отсутствии данных только из-за этой опечатки.\n')
            selected.append(note + (f"Раздел: {crumbs}\n" if crumbs else "") + text)
            used += len(text)
        expired = doc.valid_until is not None and doc.valid_until < today
        results.append(RetrievedChunk(chunk_id=candidates[0][1], text=f"Город: {doc.section}.\n" + "\n\n".join(selected),
                    title=doc.title, breadcrumbs=doc.section, page=None, score=1.0, section=doc.section,
                    is_stale=expired, valid_until=doc.valid_until.isoformat() if doc.valid_until else None,
                    is_current="current" in (doc.tags or []) and not expired))
    if not city and len({r.section for r in results}) > 1:
        for result in results:
            result.text = ('Имя найдено в нескольких городах. Уточни город; '
                           'не объединяй услуги разных сотрудников.\n' + result.text)
    return results or None
