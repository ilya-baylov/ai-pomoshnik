"""Describe the actual editable document inventory without guessing contents."""
import re
from collections import Counter
from sqlalchemy import select
from app.storage.db import session_scope
from app.storage.models import Document, Chunk
from app.core.contracts import RetrievedChunk

async def knowledge_inventory(query):
    company = bool(re.search(
        r'(?:расскажи(?:\s+(?:мне|нам|кратко|подробно))*\s+(?:про|о|об)\s+(?:нашу\s+)?компани[июе](?:\s+компании)?(?=[.!?]|$))|'
        r'(?:что(?:\s+ты)?\s+знаешь\s+(?:о|об)\s+(?:нашей\s+)?компании(?:\s+компании)?[.?!]*$)', query, re.I))
    if not company and not re.search(r'(?:что|че|чё|какие|обзор|расскажи).*\b(?:баз\w*|бащ\w*|документ\w*|материал\w*)', query, re.I):
        return None
    if not company and re.search(r'\b(?:по|про|о)\s+', query, re.I) and not re.search(r'\b(?:по|про|о)\s+(?:баз|бащ|документ|материал)', query, re.I):
        return None
    async with session_scope() as session:
        docs = (await session.execute(select(Document).order_by(Document.title))).scalars().all()
        first = (await session.execute(select(Chunk.id).limit(1))).scalar()
    if not docs or first is None:
        return None
    counts = Counter(tag for d in docs for tag in (d.tags or []))
    cities = sorted({d.section for d in docs if d.section and any(t in (d.tags or [])
                    for t in ('services', 'schedule', 'promotions', 'contacts'))})
    books = sum(d.section == 'Продажи' for d in docs)
    other = [d.title for d in docs if d.section is None and not any(t in (d.tags or [])
             for t in ('services', 'schedule', 'promotions', 'contacts'))]
    labels = [('services','услуги мастеров'),('schedule','графики'),
              ('promotions','акции'),('contacts','контакты салонов')]
    text = ('**База знаний компании.** Здесь собраны документы и рабочие инструкции.\n\n' if company else '')
    text += f'Доступно **{len(docs)} документов**. Основные разделы:\n\n'
    text += '\n'.join(f'• **{label.capitalize()}** — {counts[tag]} документов.'
                      for tag, label in labels if counts[tag])
    text += f'\n• **Книги по сервису, продажам и управлению** — {books}.'
    if other:
        text += '\n• **Обучение и рабочие правила:** ' + '; '.join(other[:12]) + '.'
    if cities:
        text += '\n\nГородские сведения: ' + ', '.join(cities) + '.'
    text += '\n\nУкажите тему — помогу найти нужные условия. Полнота сведений различается по городам; спорные сроки и незаполненные поля требуют подтверждения.'
    return [RetrievedChunk(first,text,'Состав знаний',None,None,1.0,direct_answer=text)]
