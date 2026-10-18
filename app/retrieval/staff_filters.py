"""Filter service rows using explicit columns; never infer a missing skill."""
import re
from datetime import date, timedelta
from app.ingest.stemming import stem_word

def render_day_schedule(query, rows, today=None):
    """Copy weekday cells and branch notes without asking a model to infer shifts."""
    if not re.search(r'\b(?:сегодня|завтра)\b',query,re.I):
        return None
    target=(today or date.today())+timedelta(days=1 if re.search(r'\bзавтра\b',query,re.I) else 0)
    weekday=('понедельник','вторник','среда','четверг','пятница','суббота','воскресенье')[target.weekday()]
    working=[];unknown=[];multi=False
    for row in rows:
        name=value(row[2],'мастер','имя мастера');branch=value(row[2],'филиал','салон')
        cell=value(row[2],weekday)
        if re.search(r'не\s+работ|выходн',cell,re.I):continue
        if not re.fullmatch(r'работает|да',cell.strip(),re.I):
            unknown.append(f'• **{name}** — {branch}: конкретный день не подтверждён.')
            continue
        note=value(row[2],'дополнительная информация')
        working.append(f'• **{name}** — {branch}.'+(f'\n  Примечание графика: {note}.' if note else ''))
        multi=multi or bool(re.search(r'[/\\]|\sи\s',branch,re.I))
    if not working and not unknown:return None
    answer=f'**По графику на {target:%d.%m.%Y} ({weekday}):**\n\n'
    answer+='Рабочий день указан у:\n'+'\n'.join(working) if working else 'В выбранных записях работа в этот день не указана.'
    if unknown:answer+='\n\n**Нужно уточнить день:**\n'+'\n'.join(unknown)
    if multi:answer+='\n\nЕсли у мастера несколько филиалов, учитывайте примечание графика. Без явной привязки дня нужно уточнить, в каком филиале смена.'
    answer+='\n\nФактический выход и свободные записи нужно подтвердить у администратора.'
    return answer

def fields(text):
    headers=list(re.finditer(r'(?:^|;\s*)([^:;\n]+):\s*',text))
    return {m[1].strip().casefold():text[m.end():headers[i+1].start() if i+1<len(headers) else len(text)].strip()
            for i,m in enumerate(headers)}

def value(row, *keys):
    f = fields(row)
    for key in keys:
        if f.get(key):
            return f[key]
        for label, content in f.items():
            if content and re.match(re.escape(key) + r'\s*[,(/]', label):
                return content
    return ''

def tokens(text):
    return {stem_word(w) for w in re.findall(r'[а-яёa-z0-9]+', text.casefold()) if len(w) > 2}

# A narrow procedure uses the broad specialty only to identify rows to inspect.
SPECIAL = [
    (r'\bled\b|лед[- ]?наращ', r'л[аеэ]ш|ресниц', r'\bled\b|лед'),
    (r'ламинир', r'л[аеэ]ш|ресниц|бров|депиляц', r'ламинир'),
    (r'наращ\w*\s+ногт|ногт\w*\s+наращ', r'маникюр|ногт', r'наращ\w*\s+ногт|ногт\w*\s+наращ'),
]
ROLES = [r'депиляц', r'лазер', r'маникюр|педикюр|ногт',
         r'л[аеэ]ш|ресниц', r'бров', r'массаж', r'парикмах|стриж', r'косметолог']

def filter_services(query, rows):
    """Return selected rows and an optional exact procedure predicate."""
    folded = query.casefold()
    special = next(((role, skill) for trigger, role, skill in SPECIAL
                    if re.search(trigger, folded)), None)
    role = special[0] if special else next((p for p in ROLES if re.search(p, folded)), None)
    if not role and not re.search(r'мастер|сотрудник', folded):
        return None
    # Only service/additional columns count as positive evidence.
    if role:
        rows = [r for r in rows if re.search(role, value(r[2], 'процедура', 'специальность') + ' ' +
                          value(r[2], 'дополнительные процедуры'), re.I) or
                (special and re.search(special[1], value(r[2], 'не делает'), re.I))]
    if re.search(r'\bтоп\w*', folded):
        rows = [r for r in rows if re.search(r'\bтоп\b', value(r[2], 'категория сотрудника', 'категория'), re.I)]
    elif re.search(r'технолог', folded):
        rows = [r for r in rows if re.search(r'технолог', value(r[2], 'категория сотрудника', 'категория'), re.I)]
    # Branch names must match multiple words, an address number, or a distinctive
    # single word. Common city/role words are not a branch match.
    q = tokens(query)
    ignored = tokens('салон филиал город улица мастер сотрудник Казань Москва Новосибирск депиляция маникюр')
    scores = []
    for r in rows:
        branch = value(r[2], 'филиал', 'салон')
        overlap = (tokens(branch) - ignored) & q
        nums = set(re.findall(r'\b\d+[а-я]?\b', branch.casefold())) & set(re.findall(r'\b\d+[а-я]?\b', folded))
        scores.append((r, len(overlap) + 2 * bool(nums)))
    best = max((n for _, n in scores), default=0)
    if best:
        rows = [r for r, n in scores if n == best]
    elif re.search(r'\b(?:филиал\w*|салон\w*)\s+(?:на|в|«)|\bна\s+(?!(?:ламинир|наращ|депиляц|маникюр|педикюр|массаж|процедур|ресниц|бров))[а-яё]', folded):
        # Explicit unmatched branch: do not recommend another branch.
        return None
    return rows, special[1] if special else None

def render_services(city, rows, skill=None, full=False):
    yes, no, unknown = [], [], []
    seen = set()
    for r in rows:
        text = r[2]
        name = value(text, 'мастер', 'имя мастера')
        branch = value(text, 'филиал', 'салон')
        role = value(text, 'процедура', 'специальность')
        level = value(text, 'категория сотрудника', 'категория')
        extra = value(text, 'дополнительные процедуры')
        excluded = value(text, 'не делает')
        if not name or (name, branch, role) in seen:
            continue
        seen.add((name, branch, role))
        label = f'• **{name}** — {branch}' + (f' ({level})' if level else '')
        if skill:
            if re.search(skill, excluded, re.I):
                no.append(label + f': **не делает** — {excluded}.')
            elif re.search(skill, role + ' ' + extra, re.I):
                yes.append(label + ': ' + role + (f'; {extra}' if extra else '') + '.')
            else:
                unknown.append(label + ': выполнение процедуры **не указано**.')
        else:
            yes.append(label + f': {role}' + (f'; также {extra}' if extra else '') +
                       (f'. **Не делает:** {excluded}' if excluded else '') + '.')
    if not seen:
        return None
    if not skill:
        return f'В городе **{city}** в таблице указаны:\n\n' + '\n'.join(yes)
    parts = [f'По этой процедуре в городе **{city}**:']
    if yes:
        parts += ['**Выполнение указано:**\n' + '\n'.join(yes)]
    if no:
        if len(no) > 6 and not full:
            parts += [f'У **{len(no)} мастеров** выполнение этой процедуры явно исключено. Полный список могу показать по запросу; удобнее уточнить филиал.']
        else:
            parts += ['**Не выполняют:**\n' + '\n'.join(no)]
    if unknown:
        parts += ['**Нужно подтвердить перед записью:**\n' + '\n'.join(unknown)]
    if not yes:
        parts += ['Конкретного мастера для записи пока рекомендовать не могу. Уточните у администратора выполнение именно этой процедуры.']
    return '\n\n'.join(parts)
