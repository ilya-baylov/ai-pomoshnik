"""Сбор таблиц акций по городам для обзоров, без случайной выборки строк."""
from __future__ import annotations

import re
from collections import defaultdict
from datetime import date

from sqlalchemy import select

from app.core.contracts import RetrievedChunk
from app.storage.db import session_scope
from app.storage.models import Chunk, Document


def _client_message(query: str) -> bool:
    folded = query.casefold()
    folded = re.sub(r'\bне\s+(?:готов|составл|пиш|подготавлива|отправл)\w*[^.!?;]*', '', folded)
    folded = re.sub(r'\bбез\s+(?:подготовки\s+)?(?:сообщения|текста)\s+клиент\w*', '', folded)
    return bool(re.search(r"\bклиент\w*", folded) and
                (re.search(r"\b(?:напиш|состав|подготов|отправ|сообщени|текст)\w*", folded)
                 or re.search(r'\b(?:расскажи|расскажите|озвучь|озвучьте)\s+клиент',folded)))


def _public_row(text: str, today: date) -> bool:
    """Передавать клиентскому тексту только явно разрешённые, проверяемые строки."""
    folded = text.casefold()
    if not re.search(r"разглашение клиентам:\s*(?:да\b|можно\b)", folded):
        return False
    if any(marker in folded for marker in ("не подтвержд", "не подтверждё", "нужна проверка")):
        return False
    period = re.search(r"срок:\s*([^;]+)", folded)
    if not period:
        return bool(re.search(r"тип:\s*постоянная\b", folded) or "бессроч" in folded)
    if "бессроч" in period[1]:
        return True
    months = {name: index for index, name in enumerate(
        ["января", "февраля", "марта", "апреля", "мая", "июня", "июля", "августа", "сентября", "октября", "ноября", "декабря"], 1)}
    matches = re.findall(r"\b(\d{1,2})\s+(\w+)\s+(20\d{2})\b", period[1])
    try:
        dates = [date(int(y), months[m], int(d)) for d, m, y in matches]
    except (KeyError, ValueError):
        return False
    if len(dates) == 2:
        return dates[0] <= today <= dates[1]
    if len(dates) == 1 and re.search(r"\bдо\b", period[1]):
        return today <= dates[0]
    return False


def _price_fact(text: str) -> str:
    """Явно отделить фиксированную цену из названия от размера скидки."""
    if 'Проверенный смысл цены:' in text:
        return text
    title = re.search(r'Акция:\s*([^;]+)', text, re.I)
    if title:
        if re.search(r'сертификат', title[1], re.I) and re.search(r'подари|подарочн',title[1],re.I):
            amount=re.search(r'(\d+(?:[.,]\d+)?\s*(?:₽|р\b|руб\w*|tl\b|aed\b))',title[1],re.I)
            if amount:
                return text+f'\nПроверенный смысл цены: {amount[1]} — номинал подарочного сертификата, не цена его покупки; не говори «сертификат за {amount[1]}».'
        price = re.search(r'\bза\s+(\d+(?:[.,]\d+)?\s*(?:₽|р\b|руб\w*|tl\b|aed\b))', title[1], re.I)
        if price:
            return text + f'\nПроверенный смысл цены: итоговая цена процедуры {price[1]}, а не уменьшение цены на эту сумму.'
    return text


def _normalize_row(text: str) -> str:
    """Одинаковые имена полей для разных форм исходных таблиц."""
    replacements = {
        r'\bАКЦИЯ\s+(?:СЕНТЯБРЬ|ОКТЯБРЬ):': 'Акция:',
        r'\bМОЖНО ЛИ РАССКАЗЫВАТЬ:': 'Разглашение клиентам:',
        r'\bГДЕ ДЕЙСТВУЕТ:': 'Филиал:',
        r'\bУСЛОВИЯ:': 'Условия:',
        r'\bОПЛАТА МАСТЕРУ:': 'Оплата мастеру:',
    }
    for pattern, replacement in replacements.items():
        text = re.sub(pattern, replacement, text, flags=re.I)
    return text


def _creative_promotion(query: str) -> bool:
    # «Предложения» — существительное, не просьба придумать новую акцию.
    return bool(re.search(r'\b(?:придум\w*|созда\w*|разработ\w*|предлож(?:и|ить|ите|им))\b', query, re.I))


async def clarify_promotion_city(query: str) -> str | None:
    from app.retrieval.book_catalog import named_books
    if named_books(query):
        return None
    if not re.search(r'\bакци\w*', query, re.I) or _creative_promotion(query):
        return None
    from app.retrieval.search import sections_from_query
    if await sections_from_query(query):
        return None
    if re.search(r'\b(?:все|всем|всех|всеми|весь)\b', query, re.I):
        return None
    common = {'акции','акциях','акциям','нашим','наших','всем','всех','сети','салоне','салонах','этом','нас','месяце','company','компания'}
    if any(w.casefold() not in common for w in re.findall(r'\b(?:в|во|по)\s+(\w+)', query, re.I)):
        return None  # Явно назван неизвестный город: его нельзя игнорировать.
    return 'По какому городу рассказать акции? Могу также дать обзор по всем городам.'


async def promotion_catalog(query: str, sections: set[str]) -> list[RetrievedChunk] | None:
    """None — обычный поиск; список — прямое совпадение с каталогом акций.

    Векторная близость здесь не используется. score=1 означает подтверждённый
    выбор таблицы по типу и городу; действительность самих акций задаёт источник.
    """
    folded = query.casefold()
    client_message = _client_message(query)
    if not re.search(r"\bакци\w*", folded):
        return None
    if _creative_promotion(folded) and not client_message:
        return None
    overview = bool(re.search(r"\b(?:какие|все|всем|всех|нового|актуаль\w*|действу\w*|список|обзор)\b", folded))
    all_cities = bool(re.search(r"\b(?:все|всем|всех)\s+город\w*|\bвс[яей]+\s+сет\w*", folded))
    if not sections and not overview:
        return None
    common = {"акции", "акциям", "акциях", "все", "всем", "всех", "городам", "городах", "городе", "салонах", "салоне", "этом", "нас", "сети", "месяце", "октябре"}
    locations = re.findall(r"\b(?:в|во|по)\s+(\w+)", folded)
    if not sections and not all_cities and any(word not in common for word in locations):
        # Не подменять неизвестный город сводкой по всей сети.
        return None

    wanted = set() if all_cities else {s.casefold() for s in sections}
    today = date.today()
    async with session_scope() as session:
        documents = (await session.execute(select(Document))).scalars().all()
        grouped: dict[str, list[Document]] = defaultdict(list)
        for doc in documents:
            if doc.chunk_count and doc.section and "акци" in doc.title.casefold():
                if not wanted or doc.section.casefold() in wanted:
                    grouped[doc.section.casefold()].append(doc)

        selected: list[Document] = []
        for docs in grouped.values():
            current = [d for d in docs if "current" in (d.tags or [])]
            selected.extend(current or docs)
        selected.sort(key=lambda d: (d.section.casefold(), d.title.casefold()))
        if not selected:
            return None

        rows = (await session.execute(
            select(Chunk.document_id, Chunk.id, Chunk.text)
            .where(Chunk.document_id.in_([d.id for d in selected]))
            .order_by(Chunk.document_id, Chunk.ordinal, Chunk.id)
        )).all()
        by_document: dict[int, list] = defaultdict(list)
        for row in rows:
            by_document[row[0]].append(row)

        results: list[RetrievedChunk] = []
        # Ограничение контекста применяется целыми строками, без обрыва условий.
        # При росте таблиц модель видит явное предупреждение о неполной подборке.
        remaining = 120_000
        for doc in selected:
            doc_rows = by_document[doc.id]
            if not doc_rows:
                continue
            current = "current" in (doc.tags or [])
            expired = doc.valid_until is not None and doc.valid_until < today
            confirmed = current and not expired and "period:unconfirmed" not in (doc.tags or [])
            notice = f"Город: {doc.section}.\n"
            if not confirmed:
                notice += (
                    f"Имеется список «{doc.title}». Действительность акций на "
                    f"{today.isoformat()} не подтверждена. Не называй их действующими.\n"
                )
            texts: list[str] = []
            included_rows: list[str] = []
            # В общем обзоре старые города нужны как честная отметка пробела,
            # а подробности старой таблицы доступны при вопросе о самом городе.
            if current or wanted:
                city_budget = min(24_000, remaining)
                used = 0
                for _, _, text in doc_rows:
                    text = _normalize_row(text)
                    if client_message and (not confirmed or not _public_row(text, today)):
                        continue
                    if used + len(text) + 2 > city_budget:
                        texts.append("Подборка неполная: оставшиеся строки не включены. Не утверждай, что перечислены все акции.")
                        break
                    texts.append(_price_fact(text))
                    if re.search(r'\bАкция:', text, re.I):
                        included_rows.append(text)
                    used += len(text) + 2
                remaining = max(0, remaining - used)
            if client_message:
                notice += "Для клиентского текста отобраны только явно разрешённые предложения с проверяемыми сроками. Сохрани сроки и ограничения каждой строки. Остальные предложения требуют уточнения у администратора.\n"
            results.append(RetrievedChunk(
                chunk_id=doc_rows[0][1], text=notice + "\n\n".join(texts),
                title=doc.title, breadcrumbs=doc.section, page=None,
                score=1.0, section=doc.section, is_stale=expired,
                valid_until=doc.valid_until.isoformat() if doc.valid_until else None,
                is_current=current and not expired,
                promotion_rows=included_rows,
            ))
    return results
