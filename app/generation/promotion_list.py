"""Полный внутренний список одного города без потерь при пересказе моделью."""
import re
from datetime import date
from app.core.contracts import RetrievedChunk
from app.retrieval.promotion_catalog import _client_message, _public_row, _price_fact


def _readable(text: str) -> str:
    corrections = {'слелдующее':'следующее', 'иньекций':'инъекций',
                   'инькций':'инъекций', 'предьявить':'предъявить',
                   'действет':'действует', 'технлогв':'технологов',
                   'клиееитиу':'клиенту', 'штапм':'штамп', 'клиентпервый':'клиент первый'}
    for before, after in corrections.items():
        text = re.sub(r'\b'+before+r'\b', after, text, flags=re.I)
    return text


def _format_details(detail: str) -> str:
    """Выделить поля и предупреждения, сохранив все условия источника."""
    detail = re.sub(r'(?m)^([А-ЯЁA-Z][^:\n]{0,60}:)[ \t]*',
                    lambda match: f'**{match[1]}** ', detail)
    detail = re.sub(r'\bНе раскрывать клиентам\b', '**Не раскрывать клиентам**', detail, flags=re.I)
    detail = re.sub(r'(?m)^\*\*(Срок(?:и)?):\*\* ([^\n]+)$',
                    lambda match: f'**{match[1]}: {match[2]}**', detail)
    return detail


def repair_staff_omission(text: str, question: str, chunks: list[RetrievedChunk]) -> str:
    """Сотруднику нельзя отказывать из-за пометки о рекламе клиентам."""
    if _client_message(question) or len(chunks)!=1 or chunks[0].promotion_rows is None:
        return text
    sentences=re.split(r'(?<=[.!?])\s+|\n',text)
    wrongful=any(re.search(r'клиент|партн[её]р|закрыт',s,re.I)
                 and re.search(r'не включ(?:ены|аю|аем)|не раскрываю|не перечисляю|не могу (?:раскры|расска)',s,re.I)
                 for s in sentences)
    if not wrongful:return text
    return complete_promotion_list(question+' Все акции.',chunks) or text


def complete_promotion_list(question: str, chunks: list[RetrievedChunk]) -> str | None:
    public = _client_message(question)
    if not re.search(r'\bакци\w*', question, re.I):
        return None
    if not re.search(r'\b(?:все|всё|всех|всеми|полный|полностью|остальные|другие)\b', question, re.I):
        return None
    if len(chunks) != 1 or chunks[0].promotion_rows is None:
        return None
    chunk = chunks[0]
    rows = chunk.promotion_rows
    # Не выдавать ограниченную бюджетом выборку за полный список.
    if 'Подборка неполная' in chunk.text:
        return None
    if public:
        rows = [] if 'Действительность акций' in chunk.text else [r for r in rows if _public_row(r, date.today())]
        if not rows:
            return 'Нет подтверждённых предложений для клиентского сообщения по этому городу. Уточните сроки и разрешение на публикацию у администратора.'
        parts = [f'Здравствуйте! В компании ({chunk.section}) доступны следующие предложения.']
    else:
        if not rows:
            return None
        parts = [f'**{chunk.section}: полный внутренний список — {len(rows)} предложений.**']
    if 'Действительность акций' in chunk.text:
        parts.append('**Нужно уточнить: актуальность этого списка не подтверждена.** Ниже приведены условия из имеющихся сведений; перед применением уточните сроки у администратора.')
    if not public:
        parts.append('Пометка «не раскрывать клиентам» относится к общению с клиентами. Вам как сотруднику доступны условия этих предложений. Спорные сроки отмечены отдельно.')
    for index, row in enumerate(rows, 1):
        match = re.search(r'\bАкция:\s*([^;]+)(?:;\s*|$)', row, re.I)
        if not match:
            return None
        title = ' '.join(match[1].split())
        detail = row[match.end():].strip()
        price_text = _price_fact(row)
        if price_text != row:
            detail = re.sub(r'\bскидка действует\b', 'предложение действует', detail, flags=re.I)
            if 'номинал подарочного сертификата' in price_text:
                detail += '; Указанная сумма — номинал подарочного сертификата, а не цена покупки.'
        if public:
            detail = re.sub(r'(?:^|;\s*)(?:Оплата мастеру|Разглашение клиентам|Тип):[^;]*(?=;|$)', '', detail, flags=re.I)
            detail = detail.strip('; ')
        detail = re.sub(r';\s*(?=[А-ЯЁA-Z][^;\n]{0,60}:)', '\n', detail)
        detail = re.sub(r'Разглашение клиентам:\s*нет\b', 'Не раскрывать клиентам', detail, flags=re.I)
        parts.append(f'{index}. **{_readable(title)}**\n{_format_details(_readable(detail))}')
    return '\n\n'.join(parts)


def named_promotion_answer(question: str, chunks: list[RetrievedChunk]) -> str | None:
    """One explicitly named promotion, with conditions copied from its row."""
    from dataclasses import replace
    from app.ingest.stemming import stem_word
    from app.retrieval.promotion_catalog import _creative_promotion
    if len(chunks)!=1 or chunks[0].promotion_rows is None or not re.search(r'\bакци\w*',question,re.I) or _creative_promotion(question):
        return None
    common={'акция','скидка','клиент','новый','новому','посещение','процедура','депиляция',
            'наращивание','бикини','любой','сертификат','подарок','салон','день','рождение',
            'карта','компания','company','весна','осень','месяц','промокод'}
    ignored={stem_word(w) for w in common}
    q={stem_word(w) for w in re.findall(r'[а-яёa-z]+',question.casefold())}
    matching=[]
    for row in chunks[0].promotion_rows:
        title=re.search(r'\bАкция:\s*([^;]+)',row,re.I)
        if not title:continue
        anchors={stem_word(w) for w in re.findall(r'[а-яёa-z]+',title[1].casefold()) if len(w)>=5}-ignored
        if anchors & q:matching.append(row)
    if len(matching)!=1:return None
    row=matching[0]
    if re.search(r'показать|предъявить|предъявл|что\s+нужно\s+показ|какой\s+документ',question,re.I):
        condition=re.search(r'\bУсловия:\s*([^;]+)',row,re.I)
        if condition and re.search(r'показ|предъяв',condition[1],re.I):
            detail=re.split(r',\s*не\s+действует\b',condition[1],maxsplit=1,flags=re.I)[0].strip()
            restricted=bool(re.search(r'Разглашение клиентам:\s*(?:нет|нельзя)',row,re.I))
            reply=f'В условиях акции указано: **{detail}**.'
            if restricted:
                reply+=' Эта акция помечена **«не раскрывать клиентам»** — сообщать эти условия клиенту не следует.'
            if re.search(r'\bна\s+1\s*[РP]\b',condition[1],re.I):
                reply+=' Значение пометки «1Р» в источнике не объяснено.'
            return reply
    if re.search(r'можно.*клиент\w*.*(?:рассказ|озвуч|сообщ)|(?:рассказ|озвуч|сообщ).*клиент',question,re.I):
        if re.search(r'Разглашение клиентам:\s*(?:нет|нельзя)',row,re.I):
            return '**Нет.** В условиях этой акции прямо указано: **не раскрывать клиентам**. Вам как сотруднику доступны её условия.'
    subset=replace(chunks[0],promotion_rows=matching)
    answer=complete_promotion_list(question+' Все акции.',[subset])
    if answer:
        answer=answer.replace(f'**{chunks[0].section}: полный внутренний список — 1 предложений.**',
                              f'**{chunks[0].section}: условия указанной акции.**')
        if re.search(r'сертификат[^.;\n]{0,60}\bна\s+1\s*[РP]\b',matching[0],re.I):
            answer += '\n\n**Нужно уточнить:** значение пометки «1Р» в источнике не объяснено; условие приведено дословно.'
    return answer


def promotion_overview(question: str, chunks: list[RetrievedChunk]) -> str | None:
    """An ordinary broad question gets a source-backed menu, not guessed terms."""
    if len(chunks)!=1 or chunks[0].promotion_rows is None:
        return None
    folded=question.casefold().split('город сотрудника:')[0].strip()
    if not re.search(r'(?:что.*(?:по|про).*акци|какие.*акци|расскажи.*(?:наши|про)\s+акци)',folded):
        return None
    if _client_message(question):
        return complete_promotion_list(question+' Все акции.',chunks)
    rows=chunks[0].promotion_rows
    parts=[f'**{chunks[0].section}: предложения из базы.**']
    if 'Действительность акций' in chunks[0].text:
        parts.append('**Актуальность списка не подтверждена.** Перед применением нужно уточнить сроки.')
    for row in rows:
        match=re.search(r'\bАкция:\s*([^;]+)',row,re.I)
        if not match:continue
        restricted=bool(re.search(r'Разглашение клиентам:\s*(?:нет|нельзя)',row,re.I))
        parts.append('• '+_readable(match[1].strip())+(' — **не раскрывать клиентам**' if restricted else ''))
    if len(parts)<2:return None
    if 'Подборка неполная' in chunks[0].text:
        parts.append('Это часть списка.')
    parts.append('Назовите предложение — расскажу условия и ограничения. Или напишите «все подробно».')
    return '\n\n'.join(parts)
