"""Карточки с реальными данными, без запросов к генеративной модели."""
from __future__ import annotations

import io
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from zoneinfo import ZoneInfo

from PIL import Image, ImageDraw, ImageFont, ImageFilter

BG = '#FFF5FA'
INK = '#4E1832'
MUTED = '#805369'
ACCENT = '#D50D67'
GREEN = '#22785C'
AMBER = '#A36514'


@lru_cache(maxsize=1)
def brand_badge():
    path = Path(__file__).resolve().parents[2] / 'assets/company-brand.png'
    if not path.exists(): return None
    with Image.open(path) as original:
        badge = original.convert('RGB').resize((180, 180), Image.Resampling.LANCZOS)
    mask = Image.new('L', (540, 540))
    ImageDraw.Draw(mask).ellipse((3, 3, 537, 537), fill=255)
    mask = mask.resize((180, 180), Image.Resampling.LANCZOS)
    return badge, mask


@lru_cache(maxsize=16)
def font(size, bold=False):
    names = [Path('C:/Windows/Fonts') / ('segoeuib.ttf' if bold else 'segoeui.ttf'),
             Path('/usr/share/fonts/truetype/dejavu') / ('DejaVuSans-Bold.ttf' if bold else 'DejaVuSans.ttf')]
    for path in names:
        if path.exists(): return ImageFont.truetype(str(path), size)
    return ImageFont.load_default(size=size)


def card(kind: str, data: dict) -> bytes:
    canvas = Image.new('RGB', (960, 790), BG)
    d = ImageDraw.Draw(canvas)
    # Светлая розовая подложка; цифры и текст остаются контрастными.
    for y in range(790):
        t = y / 789
        color = tuple(round(a + (b-a)*t) for a,b in zip((255,247,251),(251,218,235)))
        d.line((0,y,959,y),fill=color)
    d.text((48, 30), 'Помощник', font=font(38, True), fill=ACCENT)
    now = datetime.now(ZoneInfo('Asia/Qyzylorda'))
    timestamp = now.strftime('%d.%m · %H:%M')
    d.text((960-48-d.textlength(timestamp, font=font(34)), 34), timestamp, font=font(34), fill=MUTED)
    if kind == 'home':
        heading, subtitle = 'Панель управления', 'База · задачи · сотрудники'
        tasks = data.get('review_pending',0)+data.get('unanswered_new',0)+data.get('negative',0)
        tiles = [(data['documents'], 'Документы\nв поиске', ACCENT),
                 (tasks, 'Открытые\nзадачи', AMBER if tasks else GREEN),
                 (data['queries'], f"Вопросы\nза {data['days']} дней", ACCENT)]
        state = 'База обновляется' if data.get('updating') else 'Бот на связи'
        state_color = AMBER if data.get('updating') else GREEN
        foot = f"База обновлена: {data.get('last_ingest') or '—'}"
    elif kind == 'knowledge':
        heading, subtitle = 'База знаний', 'Документы и правки'
        tiles = [(data['documents'], 'Документы', ACCENT), (data['ready'], 'Версии\nв поиске', GREEN),
                 (data['pending'], 'Ждут\nобновления', AMBER if data['pending'] else ACCENT)]
        state, state_color = ('Есть изменения' if data['pending'] else 'Поиск обновлён'), (AMBER if data['pending'] else GREEN)
        foot = 'Условия проверяй отдельно от версии'
    else:
        heading, subtitle = 'Задачи', 'Что нужно проверить'
        tiles = [(data['unanswered_new'], 'Вопросы\nбез ответа', ACCENT), (data['negative'], 'Замечания\nк ответам', ACCENT),
                 (data['review_pending'], 'Данные\nна проверку', AMBER if data['review_pending'] else GREEN)]
        total=sum(t[0] for t in tiles)
        state, state_color = ('Нужна проверка' if total else 'Очереди пусты'), (AMBER if total else GREEN)
        foot = 'Исправь материал → закрой обращение'
    d.text((46, 112), heading, font=font(60, True), fill=INK)
    d.text((48, 198), subtitle, font=font(38), fill=MUTED)
    badge = brand_badge()
    if badge:
        canvas.paste(badge[0], (732, 108), badge[1])
    pill_width = min(620, int(d.textlength(state,font=font(40,True)))+94)
    d.rounded_rectangle((44,264,44+pill_width,323),radius=28,fill='#FFF9FC',outline='#F3BDD6',width=2)
    d.ellipse((50, 283, 74, 307), fill=state_color)
    d.text((94, 265), state, font=font(40, True), fill=state_color)
    width=272
    shadows = Image.new('RGBA', canvas.size)
    shadow_draw = ImageDraw.Draw(shadows)
    for n in range(3):
        x=48+n*296
        shadow_draw.rounded_rectangle((x+1,359,x+width+1,618),radius=30,fill=(178,31,103,35))
    canvas = Image.alpha_composite(canvas.convert('RGBA'),shadows.filter(ImageFilter.GaussianBlur(8))).convert('RGB')
    d = ImageDraw.Draw(canvas)
    for n,(value,label,color) in enumerate(tiles):
        x=48+n*296
        d.rounded_rectangle((x, 350, x+width, 608), radius=30, fill='#FFF9FC',outline='#F4BCD5',width=2)
        number=str(value)
        size=88
        while size>32 and d.textlength(number,font=font(size,True))>width-40:
            size-=2
        d.text((x+20, 367), number, font=font(size,True), fill=color)
        d.multiline_text((x+20, 494), label, font=font(36), fill=MUTED, spacing=8)
    d.line((48, 655, 912, 655), fill='#EFB0CD', width=2)
    d.text((48, 677), foot, font=font(36), fill=MUTED)
    d.text((48, 730), 'Локально · обновление по кнопке', font=font(34), fill=MUTED)
    output=io.BytesIO()
    canvas.save(output, format='PNG', optimize=True)
    return output.getvalue()
