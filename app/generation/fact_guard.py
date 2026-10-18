"""Reject critical numeric values absent from the retrieved evidence.

This checks values, not the meaning or ownership of every claim. Structured
tables and the generation rules remain responsible for those distinctions.
"""
import re
from decimal import Decimal
from datetime import date, timedelta

AMOUNT = re.compile(r'(?<![\w.,])(\d+(?:[ \u00a0]\d{3})*(?:[.,]\d+)?)\s*(%|₽|руб\w*|р\b|rub\b|rur\b|aed\b|dhs\b|дирхам\w*|тенге\b|kzt\b|₸|€|евро\b|eur\b|usd\b|\$|доллар\w*)',re.I)
PHONE = re.compile(r'(?<!\d)\+?\d[\d() .-]{8,}\d(?!\d)')
MONTHS = {m:i for i,m in enumerate(('январ','феврал','март','апрел','ма','июн','июл','август','сентябр','октябр','ноябр','декабр'),1)}
DATE_WORDS = re.compile(r'\b([0-3]?\d)\s+(январ\w*|феврал\w*|март\w*|апрел\w*|ма[йяе]|июн\w*|июл\w*|август\w*|сентябр\w*|октябр\w*|ноябр\w*|декабр\w*)(?:\s+(20\d\d))?',re.I)

def critical_values(text):
    values=set()
    # Remove only Markdown emphasis, preserving punctuation inside numbers.
    text=text.replace('**','').replace('__','').replace('\u202f',' ').replace('\u2009',' ')
    values.update(('year', year) for year in re.findall(r'\b20\d\d\b', text))
    for number, unit in AMOUNT.findall(text):
        amount=str(Decimal(number.replace(' ','').replace('\u00a0','').replace(',','.')).normalize())
        unit=unit.casefold()
        kind=('percent' if unit=='%' else 'rub' if unit.startswith(('руб','р')) or unit in {'₽','rub','rur'} else
              'aed' if unit.startswith(('aed','дирхам')) or unit=='dhs' else 'kzt' if unit in {'тенге','₸','kzt'} else
              'eur' if unit in {'€','евро','eur'} else 'usd')
        values.add((kind,amount))
    for candidate in PHONE.findall(text):
        if re.fullmatch(r'\d{4}-\d{2}-\d{2}',candidate.strip()):continue
        digits=re.sub(r'\D','',candidate)
        if 10<=len(digits)<=15:
            if len(digits)==11 and digits.startswith('8'):digits='7'+digits[1:]
            values.add(('phone',digits))
    for day,month,year in DATE_WORDS.findall(text):
        month=next(i for prefix,i in MONTHS.items() if month.casefold().startswith(prefix))
        values.add(('day-month',f'{int(day)}-{month}'))
        if year:values.add(('date',f'{year}-{month:02d}-{int(day):02d}'))
    for year,month,day in re.findall(r'\b(20\d\d)-(\d{2})-(\d{2})\b',text):
        values.add(('date',f'{year}-{month}-{day}'));values.add(('day-month',f'{int(day)}-{int(month)}'))
    for day,month,year in re.findall(r'\b(\d{1,2})[./](\d{1,2})[./](20\d\d)\b',text):
        values.add(('date',f'{year}-{int(month):02d}-{int(day):02d}'));values.add(('day-month',f'{int(day)}-{int(month)}'))
    return values

def unsupported_values(answer,chunks,today=None,question=''):
    evidence='\n'.join('\n'.join((c.title,c.breadcrumbs or '',c.text,c.valid_until or '')) for c in chunks)
    allowed=critical_values(evidence)
    current=today or date.today()
    allowed.add(('date',current.isoformat()))
    allowed.add(('day-month',f'{current.day}-{current.month}'))
    allowed.add(('year',str(current.year)))
    if re.search(r'\bзавтра\b',question,re.I):
        tomorrow=current+timedelta(days=1)
        allowed.update({('date',tomorrow.isoformat()),('day-month',f'{tomorrow.day}-{tomorrow.month}'),('year',str(tomorrow.year))})
    # An explicit calculation may produce a new value. Accept only a verified
    # discount calculation with one unambiguous price and one percentage.
    if re.search(r'посчит|рассчит|итогов|со\s+скидк',question,re.I) and re.search(r'скидк',evidence,re.I) and re.search(r'цен[ауы]|стоимост|\bза\s+\d',evidence,re.I):
        prices=[(kind,value) for kind,value in allowed if kind in {'rub','aed','usd','eur','kzt'}]
        percentages=[value for kind,value in allowed if kind=='percent']
        if len(prices)==len(percentages)==1 and 0<=Decimal(percentages[0])<=100:
            kind,price=prices[0]
            calculated=(Decimal(price)*(1-Decimal(percentages[0])/100)).quantize(Decimal('0.01'))
            allowed.add((kind,str(calculated.normalize())))
    missing=critical_values(answer)-allowed
    # An opaque marker in a certificate condition is not an explicit currency.
    if re.search(r'сертификат[^.;\n]{0,60}\bна\s+1\s*[РP]\b',evidence,re.I) and re.search(
            r'сертификат[^.;\n]{0,60}\b1\s*(?:₽|руб\w*)',answer,re.I):
        missing.add(('meaning','Пометка «1Р» не расшифрована: не заменяй её ценой 1 ₽'))
    return missing
