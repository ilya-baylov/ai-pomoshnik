"""Прямой выбор корпоративных материалов о разрядах и карьерной лестнице."""
import re
from datetime import date
from sqlalchemy import select
from app.core.contracts import RetrievedChunk
from app.storage.db import session_scope
from app.storage.models import Chunk, Document


async def career_catalog(query: str) -> list[RetrievedChunk] | None:
    folded = query.casefold()
    if not (re.search(r'\b(?:карьер\w*|профсоюз\w*|разряд\w*|льгот\w*|prof)\b', folded)
            or re.search(r'(?:вырасти|дорасти|стать|перейти|переход\w*|повысить).*?(?:топ\b|топ-мастер|технолог|категори|управляющ)', folded)
            or re.search(r'(?:что|каки\w*).*?(?:нужн\w*|требован\w*|конкретн\w*).*?топ',folded)):
        return None
    async with session_scope() as session:
        docs = (await session.execute(select(Document).where(Document.section.is_(None)))).scalars().all()
        docs = [d for d in docs if d.title.casefold() in {
            'информация профсоюз', 'профсоюз company — карьерная лестница и льготы'}]
        if not docs:
            return None
        docs.sort(key=lambda d: 'карьерная лестница' not in d.title.casefold())
        result = []
        for doc in docs:
            rows = (await session.execute(select(Chunk).where(Chunk.document_id == doc.id).order_by(Chunk.ordinal))).scalars().all()
            if not rows:
                continue
            texts = []; used = 0
            for row in rows:
                part = (f'Страница {row.page}:\n' if row.page else '') + row.text
                if used + len(part) > 24000:
                    texts.append('Подборка неполная; не утверждай полноту льгот.')
                    break
                texts.append(part); used += len(part)
            expired = doc.valid_until is not None and doc.valid_until < date.today()
            direct=None
            if not expired and re.search(r'(?:вырасти|дорасти|стать|перейти).*?\bтоп',folded):
                candidates=[t for t in texts if re.search(r'ВОЗМОЖНОСТЬ\s+ПЕРЕЙТИ\s+В\s+ТОПЫ',t,re.I)]
                if len(candidates)==1:
                    part=candidates[0]
                    period=re.search(r'(\d+)\s*[-–]\s*(\d+)\s*ГОД',part,re.I)
                    grade=re.search(r'(\d+)\s*РАЗРЯДА',part,re.I)
                    if period and grade:
                        direct=(f'В профсоюзной карьерной лестнице возможность перейти в топы салонов указана для '
                                f'**{grade[1]} разряда ({period[1]}–{period[2]} лет в компании)**. '
                                'Это возможность, а не автоматический перевод.\n\n'
                                'Критерии оценки, порядок перехода и доступные места в документе не указаны — '
                                'их нужно уточнить у управляющего.')
            result.append(RetrievedChunk(rows[0].id, '\n\n'.join(texts), doc.title, None, None,
                                         1.0, is_stale=expired,
                                         valid_until=doc.valid_until.isoformat() if doc.valid_until else None,
                                         direct_answer=direct))
            if direct:
                return [result[-1]]
        return result or None
