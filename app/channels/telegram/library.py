"""Редактор базы в Telegram: каталог, пункты, черновики, публикация, версии."""
from __future__ import annotations

import asyncio
import contextlib
import html
import json
import logging
import re
import tempfile
import uuid
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from aiogram import F, Router
from aiogram.filters import BaseFilter
from aiogram.types import BufferedInputFile, CallbackQuery, Message, User as TelegramUser

from app.channels.telegram import admin, views
from app.core import library_editor as editor
from app.core.config import get_bot_config
from app.core.knowledge_review import load_queue, is_pending
from app.ingest.loaders.registry import SUPPORTED_EXTENSIONS
from app.storage.models import User

router = Router(name='library')
log = logging.getLogger(__name__)
_inputs: dict[tuple[int, int], dict] = {}
_started = datetime.now(ZoneInfo('Asia/Qyzylorda'))
PAGE = 6


def input_key(event) -> tuple[int, int]:
    message = event.message if isinstance(event, CallbackQuery) else event
    return message.chat.id, event.from_user.id


def kb(rows):
    return views._kb(rows)


def back(target='lib:root'):
    return kb([[('‹ Назад', target)]])


def local_time(value):
    if not value: return '—'
    dt = datetime.fromisoformat(str(value))
    if dt.tzinfo is None: dt = dt.replace(tzinfo=ZoneInfo('UTC'))
    return dt.astimezone(ZoneInfo('Asia/Qyzylorda')).strftime('%d.%m %H:%M')


def pending_issues(key):
    return [i for i in load_queue()['issues'] if is_pending(i)
            and editor.key_for(i['file'].removeprefix('knowledge/')) == key]


async def show(event, text, markup):
    from app.channels.telegram.panels import render
    return await render(event, text, markup)


async def root_screen(event):
    items = await editor.catalog()
    active = [d for d in items if not d.get('archived')]
    cities = sorted({d['city'] or 'Общие материалы' for d in active})
    city_buttons = [(f"{c} · {sum((d['city'] or 'Общие материалы') == c for d in active)}", f'lib:city:{n}:0')
                    for n, c in enumerate(cities)]
    rows = [city_buttons[n:n+2] for n in range(0, len(city_buttons), 2)]
    rows += [[('🔎 Найти документ', 'lib:search'), ('＋ Добавить', 'lib:add')],
             [('Черновики', 'lib:drafts'), ('Архив', 'lib:archive:0')],
             [('Обновить базу', 'adm:ingest'), ('‹ Главная', 'adm:home')]]
    data={'documents':len(active),'ready':sum(d['ready'] for d in active),
          'pending':sum(not d['ready'] for d in active)+sum(d['indexed'] for d in items if d.get('archived'))}
    await show(event, '<b>База знаний</b>\n\n'
               f"Документов: <b>{data['documents']}</b> · в поиске: <b>{data['ready']}</b>\n"
               f"Ожидают обновления: <b>{data['pending']}</b>\n\nВыбери город или найди документ. "
               'Правки сначала сохраняются черновиками.\n\n'
               '🟢 В поиске · 🟡 Ожидает обновления · ✏️ Черновик', kb(rows))


async def list_screen(event, items, page, target, title):
    page = min(max(0, page), max(0, (len(items)-1)//PAGE))
    rows = []
    lines = [f'<b>{html.escape(title)}</b>', f'Документов: {len(items)} · страница {page+1}/{max(1,(len(items)+PAGE-1)//PAGE)}', '']
    for d in items[page*PAGE:(page+1)*PAGE]:
        mark = '⚪' if d.get('archived') else ('🟢' if d['ready'] else '🟡')
        warn = ' ⚠' if pending_issues(d['key']) else ''
        lines.append(f"{mark} {html.escape(d['title'][:200])}{warn}" + (' ✏️' if d['drafts'] else ''))
        rows.append([(f"{mark} {d['title'][:48]}", f"lib:doc:{d['key']}")])
    if not items: lines.append('Здесь пока нет документов.')
    nav = []
    if page: nav.append(('‹ Предыдущие', f'{target}:{page-1}'))
    if (page+1)*PAGE < len(items): nav.append(('Следующие ›', f'{target}:{page+1}'))
    if nav: rows.append(nav)
    rows.append([('‹ К городам', 'lib:root')])
    await show(event, '\n'.join(lines), kb(rows))


async def doc_screen(event, key):
    d = await editor.find(key)
    status = 'В архиве: не используется' if d.get('archived') and not d['indexed'] else (
        'Архивирование ожидает обновления' if d.get('archived') else ('Используется ботом' if d['ready'] else 'Ожидает обновления базы'))
    rows = []
    blocks = 0
    if not d.get('archived'):
        value = editor.payload(editor.safe_path(d['path']))
        blocks = len(value['blocks'])
        rows += [[('Читать / править', f'lib:blocks:{key}:0'), ('Найти пункт', f'lib:bsearch:{key}')],
                 [('＋ Добавить пункт', f'lib:append:{key}'), ('Ещё действия', f'lib:docmore:{key}')]]
    rows += [[('История правок', f'lib:history:{key}'), ('Черновики', 'lib:drafts')]]
    issues = pending_issues(key)
    if issues: rows.append([('⚠ Проверить спорные данные', f"lib:issue:{issues[0]['id']}")])
    if not d['ready']: rows.append([('Обновить базу', 'adm:ingest')])
    rows.append([('‹ К городам', 'lib:root')])
    await show(event, f"<b>{html.escape(d['title'])}</b>\n\n"
        f"Город: {html.escape(d['city'] or 'Общие материалы')}\nСтатус: <b>{status}</b>\n"
        f"Пунктов: {blocks}\n"
        f"Обновлено в поиске: {local_time(d['indexed_at'])}\n"
        f"Неподтверждённых записей: {len(issues)}\n"
        f"Черновиков: {d['drafts']}", kb(rows))


async def blocks_screen(event, key, page=0, term=''):
    d = await editor.find(key)
    value = editor.payload(editor.safe_path(d['path']))
    blocks = [(n, b) for n, b in enumerate(value['blocks']) if not term or term.casefold() in b['text'].casefold()]
    page = min(max(0, page), max(0, (len(blocks)-1)//PAGE))
    lines = [f"<b>{html.escape(d['title'][:180])}</b>", f'Пунктов: {len(blocks)} · страница {page+1}', '']
    rows = []
    for n, block in blocks[page*PAGE:(page+1)*PAGE]:
        snippet = re.sub(r'\s+', ' ', block['text'])[:160]
        lines.append(f'<b>{n+1}.</b> {html.escape(snippet)}')
        rows.append([(f'{n+1}. {snippet[:45]}', f'lib:block:{key}:{n}')])
    if not blocks: lines.append('Совпадений нет. Попробуй другое слово.')
    if not term:
        nav = []
        if page: nav.append(('‹', f'lib:blocks:{key}:{page-1}'))
        if (page+1)*PAGE < len(blocks): nav.append(('›', f'lib:blocks:{key}:{page+1}'))
        if nav: rows.append(nav)
    elif len(blocks) > PAGE:
        lines.append('Показаны первые 6 совпадений. Уточни поиск или открой общий список.')
    rows += [[('🔎 Поиск по тексту', f'lib:bsearch:{key}'), ('Все пункты', f'lib:blocks:{key}:0')],
             [('‹ К документу', f'lib:doc:{key}')]]
    await show(event, '\n'.join(lines), kb(rows))


async def block_screen(event, key, index, issue_id=None):
    d = await editor.find(key)
    value = editor.payload(editor.safe_path(d['path']))
    if not 0 <= index < len(value['blocks']): raise ValueError('Пункт уже изменился. Открой документ заново.')
    text = value['blocks'][index]['text']
    _inputs.pop(input_key(event), None)
    suffix = f':{issue_id}' if issue_id else ''
    rows = [[('✏️ Изменить этот пункт', f'lib:edit:{key}:{index}{suffix}')],
            [('Удалить пункт', f'lib:delete:{key}:{index}')],
            [('‹ К пунктам', f'lib:blocks:{key}:{index//PAGE}'), ('К документу', f'lib:doc:{key}')]]
    note = '\n\nТекст длинный; полностью доступен через «Скачать текст» в карточке документа.' if len(text)>3000 else ''
    await show(event, f"<b>{html.escape(d['title'][:150])} · пункт {index+1}</b>\n\n{html.escape(text[:3000])}{note}", kb(rows))


async def draft_screen(event, token, actor):
    draft = editor.get_draft(token, actor)
    value = draft['value']
    title = (value or {}).get('metadata', {}).get('title', Path(draft['path']).stem)
    old = editor.payload(editor.safe_path(draft['path'])) if editor.safe_path(draft['path']).exists() else None
    if draft['action'] == 'archive':
        preview = 'Документ будет убран из ответов бота. Сохранённую версию можно восстановить.'
    else:
        changes = [b['text'] for n,b in enumerate(value['blocks']) if not old or n >= len(old['blocks']) or b != old['blocks'][n]]
        preview = '\n\n'.join(changes)[:2300] or 'Изменяются настройки документа или восстанавливается его версия.'
    await show(event, f'<b>Проверь правку перед применением</b>\n\n{html.escape(title[:180])}\n'
               f"Действие: {html.escape(draft['label'][:150])}\n\n{html.escape(preview)}\n\n"
               'Это черновик. После применения база обновится; результат будет показан отдельно.',
               kb([[('✅ Применить и обновить базу', f'lib:publish:{token}')],
                   [('Оставить черновиком', 'lib:drafts'), ('Отменить правку', f'lib:discard:{token}')]]))


async def ask(event, state, text, target='lib:root'):
    _inputs[input_key(event)] = state
    await show(event, text + '\n\n/cancel — отменить ввод.', kb([[('Отмена', f'lib:cancel:{target}')]]))


@router.callback_query(F.data.startswith('lib:'))
@admin.admin_only
async def navigate(callback: CallbackQuery, user: User, is_admin: bool):
    parts = (callback.data or '').split(':')
    action = parts[1]
    actor = callback.from_user.id
    try:
        if action == 'cancel':
            _inputs.pop(input_key(callback), None)
            await root_screen(callback); return
        if action in ('root', 'city', 'doc', 'blocks', 'history', 'drafts', 'archive', 'status', 'tasks', 'work'):
            _inputs.pop(input_key(callback), None)
        if action == 'root': await root_screen(callback)
        elif action == 'city':
            items = [d for d in await editor.catalog() if not d.get('archived')]
            cities = sorted({d['city'] or 'Общие материалы' for d in items})
            city = cities[int(parts[2])]
            await list_screen(callback, [d for d in items if (d['city'] or 'Общие материалы') == city], int(parts[3]), f'lib:city:{parts[2]}', city)
        elif action == 'archive':
            await list_screen(callback, [d for d in await editor.catalog() if d.get('archived')], int(parts[2]), 'lib:archive', 'Архив')
        elif action == 'search': await ask(callback, {'mode':'search'}, 'Напиши часть названия документа или город.')
        elif action == 'doc': await doc_screen(callback, parts[2])
        elif action == 'docmore':
            key=parts[2];d=await editor.find(key)
            await show(callback,f"<b>{html.escape(d['title'][:180])}</b>\n\nДополнительные действия с документом.",
                       kb([[('Заменить файлом',f'lib:replace:{key}'),('Скачать текст',f'lib:export:{key}')],
                           [('Название, город и срок',f'lib:meta:{key}')],
                           [('Убрать в архив',f'lib:archiveask:{key}')],[('‹ К документу',f'lib:doc:{key}')]]))
        elif action == 'blocks': await blocks_screen(callback, parts[2], int(parts[3]))
        elif action == 'block': await block_screen(callback, parts[2], int(parts[3]))
        elif action == 'bsearch': await ask(callback, {'mode':'bsearch','key':parts[2]}, 'Какое слово или имя найти в документе?')
        elif action in ('edit', 'append', 'delete'):
            d = await editor.find(parts[2]); value = editor.payload(editor.safe_path(d['path']))
            if not d['editable']: raise ValueError('Для этого формата используй замену файлом.')
            index = int(parts[3]) if action != 'append' else len(value['blocks'])
            if action == 'delete':
                if not 0 <= index < len(value['blocks']): raise ValueError('Пункт не найден')
                removed = value['blocks'].pop(index)
                if not any(b['text'].strip() for b in value['blocks']): raise ValueError('Нельзя удалить последний пункт. Можно убрать документ в архив.')
                token = editor.make_draft(d['path'], value, actor, f"Удаление пункта {index+1}: {removed['text'][:90]}")
                await draft_screen(callback, token, actor)
            else:
                issue_id = parts[4] if len(parts)>4 else None
                if issue_id and actor != load_queue()['owner_id']: raise ValueError('Подтверждать спорные сведения может ответственный за базу')
                await ask(callback, {'mode':action,'key':parts[2],'index':index,'value':value,
                                     'sha':editor.file_sha256(editor.safe_path(d['path'])), 'issue_id':issue_id},
                          'Пришли новый текст пункта целиком. Сохрани условия, исключения, имя мастера или название акции.\n'
                          'Для спорных сведений указывай только проверенные данные.', f"lib:doc:{parts[2]}")
        elif action == 'meta':
            key=parts[2]
            await show(callback, '<b>Настройки документа</b>\n\nСрок — последняя дата действия документа. '
                       'Изменение срока не подтверждает автоматически отдельные акции внутри него.',
                       kb([[('Переименовать', f'lib:title:{key}'), ('Срок действия', f'lib:expiry:{key}')],
                           [('Город', f'lib:move:{key}'), ('Тип материала', f'lib:category:{key}')], [('‹ Назад', f'lib:doc:{key}')]]))
        elif action in ('title','expiry'):
            d=await editor.find(parts[2]); value=editor.payload(editor.safe_path(d['path']))
            await ask(callback, {'mode':action,'key':parts[2],'value':value,'sha':editor.file_sha256(editor.safe_path(d['path']))},
                      'Напиши новое название (до 200 символов).' if action=='title' else 'Введи дату ГГГГ-ММ-ДД, например 2026-10-20. Или «без срока».')
        elif action in ('move','add'):
            from app.core.staff_context import CITIES
            rows=[[(city,f"lib:choose:{parts[2] if action=='move' else 'new'}:{n}")] for n,city in enumerate((*CITIES,'Общие материалы'))]
            rows.append([('‹ Назад','lib:root')])
            await show(callback, '<b>Выбери город документа</b>', kb(rows))
        elif action == 'choose':
            from app.core.staff_context import CITIES
            city=(*CITIES,'')[int(parts[3])]
            if parts[2]=='new':
                await ask(callback, {'mode':'new','city':city}, 'Пришли файл PDF, Word, Excel или текстовый документ.\n'
                          'Либо напиши материал сообщением: первая строка — название, дальше — содержание.')
            else:
                d=await editor.find(parts[2]);value=editor.payload(editor.safe_path(d['path']))
                value['metadata']['section']=city or None
                token=editor.make_draft(d['path'],value,actor,'Изменение города')
                await draft_screen(callback,token,actor)
        elif action == 'category':
            rows=[[(label,f'lib:catset:{parts[2]}:{kind}')] for kind,label in
                  [('other','Правила / обучение'),('promotions','Акции'),('contacts','Адреса / контакты'),('schedule','Графики'),('services','Услуги мастеров')]]
            rows.append([('‹ Назад',f'lib:doc:{parts[2]}')])
            await show(callback,'<b>Тип материала</b>\n\nОпределяет, как бот ищет этот документ.',kb(rows))
        elif action == 'catset':
            if parts[3] not in ('other','promotions','contacts','schedule','services'): raise ValueError('Неизвестный тип')
            d=await editor.find(parts[2]);value=editor.payload(editor.safe_path(d['path']))
            meta=value['metadata']; meta['category']=parts[3]
            meta['tags']=[t for t in meta.get('tags',[]) if t not in ('other','promotions','contacts','schedule','services')]+[parts[3]]
            token=editor.make_draft(d['path'],value,actor,'Изменение типа материала')
            await draft_screen(callback,token,actor)
        elif action == 'replace':
            d=await editor.find(parts[2])
            await ask(callback, {'mode':'replace','key':parts[2],'sha':editor.file_sha256(editor.safe_path(d['path']))},
                      'Пришли новый файл. Он заменит содержание целиком; название, город и тип материала сохранятся.')
        elif action == 'export':
            d=await editor.find(parts[2]);value=editor.payload(editor.safe_path(d['path']))
            text=value['metadata']['title']+'\n\n'+'\n\n'.join(f"[{n+1}] {b['text']}" for n,b in enumerate(value['blocks']))
            await callback.message.answer_document(BufferedInputFile(text.encode('utf8'),filename='компании-документ.txt'),caption='Полный текст. Для изменения одного пункта используй кнопку редактора.')
            await callback.answer()
        elif action == 'archiveask':
            d=await editor.find(parts[2])
            token=editor.make_draft(d['path'],None,actor,'Убрать документ в архив',action='archive')
            await draft_screen(callback,token,actor)
        elif action == 'draft': await draft_screen(callback,parts[2],actor)
        elif action == 'discard':
            editor.cancel_draft(parts[2],actor); await root_screen(callback)
        elif action == 'drafts':
            drafts=[(token,d) for token,d in editor.read_state()['drafts'].items() if d['actor']==actor and d['status']=='draft']
            rows=[[(f"{d['label'][:45]} · {local_time(d['created_at'])}",f'lib:draft:{token}')] for token,d in drafts[-15:]]
            rows.append([('‹ Назад','lib:root')])
            await show(callback, f'<b>Мои черновики · {len(drafts)}</b>\n\nОткрой правку, чтобы проверить и применить её. '
                       'Черновики не меняют ответы бота. Показаны последние 15.',kb(rows))
        elif action == 'publish':
            if admin._ingest_lock.locked(): raise ValueError('База сейчас обновляется. Дождись завершения и примени черновик снова.')
            async with admin._ingest_lock:
                record=editor.publish(parts[2],actor)
                await callback.answer('Правка сохранена. Обновляю базу…')
                status=await callback.message.answer('<b>Обновляю базу после правки…</b>')
                await admin._run_ingest(callback.bot,status,full=False,already_locked=True)
            await doc_screen(callback,record['key'])
        elif action == 'history':
            d=await editor.find(parts[2]);history=[h for h in editor.read_state()['history'] if h['key']==parts[2]][-8:]
            lines=[f"<b>История: {html.escape(d['title'][:150])}</b>",'']
            rows=[]
            labels={'active':'Применено','archived':'В архиве','pending_index':'Ожидает обновления','superseded':'Заменено более поздней правкой'}
            for h in reversed(history):
                lines.append(f"{local_time(h['at'])} · {html.escape(h['label'][:160])}\n{labels.get(h['status'],h['status'])}")
                if h.get('backup'): rows.append([(f"Вернуть версию до {local_time(h['at'])}",f"lib:restore:{h['id']}")])
            if not history: lines.append('Правок из админки пока нет. Первая правка сохранит текущую версию.')
            rows.append([('‹ Назад',f'lib:doc:{parts[2]}')])
            await show(callback,'\n\n'.join(lines),kb(rows))
        elif action == 'restore':
            token=editor.restore_draft(parts[2],actor); await draft_screen(callback,token,actor)
        elif action == 'issue':
            queue=load_queue()
            if actor!=queue['owner_id']: raise ValueError('Подтверждать спорные сведения может ответственный за базу')
            issue=next((i for i in queue['issues'] if i['id']==parts[2]),None)
            if not issue: raise ValueError('Запись не найдена')
            if not is_pending(issue): raise ValueError('Этот пункт уже проверен. Открой актуальную очередь задач.')
            key=editor.key_for(issue['file'].removeprefix('knowledge/'))
            if 'block' in issue:
                await callback.message.answer('<b>Что нужно проверить</b>\n\n'+html.escape(issue['issue'][:1500]))
                await block_screen(callback,key,issue['block'],issue['id'])
            else:
                await show(callback, '<b>Нужно уточнить</b>\n\n'+html.escape(issue['issue'])+
                           '\n\nПроверь документ и внеси подтверждённые условия в соответствующие пункты. Одна смена названия не подтверждает условия.',
                           kb([[('Открыть документ',f'lib:doc:{key}')],[('‹ Назад','adm:review:0')]]))
        elif action == 'tasks':
            data=await admin._dashboard_data()
            text=('<b>Задачи</b>\n\n'
                  f"<b>Без ответа · {data['unanswered_new']}</b> — чего не хватает в базе.\n"
                  f"<b>Замечания · {data['negative']}</b> — ответы, на которые пожаловались.\n"
                  f"<b>Проверить данные · {data['review_pending']}</b> — спорные условия и сроки.\n\n"
                  'Открой очередь → исправь материал → закрой обращение.')
            await show(callback,text,kb([[('Без ответа','adm:unanswered'),('Замечания','adm:quality')],
                                        [('Проверить данные','adm:review:0')],[('‹ Главная','adm:home')]]))
        elif action == 'work':
            await show(callback,'<b>Работа бота</b>\n\nСтатусы, история обновлений и расходы на обработку базы.\n'
                       'Модель проверяется только по кнопке — открытие меню не расходует токены.',
                       kb([[('Состояние и прогресс','lib:status')],
                           [('Статистика ответов','adm:stats'),('История обновлений','lib:runs')],
                           [('Как пользоваться','lib:guide')],[('‹ Главная','adm:home')]]))
        elif action == 'status':
            data=await admin._dashboard_data();items=await editor.catalog()
            pending=sum(not d['ready'] and not d.get('archived') for d in items)
            state=editor.read_state()
            job_path=editor.store()/'last-update.json'
            job=json.loads(job_path.read_text('utf8')) if job_path.exists() else {}
            job_text=''
            if job:
                job_status={'ok':'🟢 Завершено','error':'🔴 Ошибка','running':'🟡 Выполняется' if admin._ingest_lock.locked() else '⚠ Прервано'}.get(job.get('status'),'—')
                job_text=f'\nПоследняя операция: {job_status}\nЭтап: {html.escape(job.get("stage","—"))}\n'
                if admin._ingest_lock.locked(): job_text+=f'Прогресс: {job.get("current",0)} / {job.get("total",0)} · {job.get("percent",0)}%\n'
                if job.get('finished_at'): job_text+=f'Завершена: {local_time(job["finished_at"])}\n'
                if job.get('errors'): job_text+='Причина: '+html.escape(str(job['errors'][0])[:250])+'\n'
            await show(callback,'<b>Состояние бота</b>\n\n'
                       f'🟢 Бот отвечает на команды\nЗапущен: {_started.strftime("%d.%m %H:%M")}\n'
                       f'База: {"🟡 Обновляется" if admin._ingest_lock.locked() else "🟢 Обновление не выполняется"}\n'
                       f'Документов в поиске: {data["documents"]}\nОжидают обновления: {pending}\n'
                       f'Черновиков: {sum(d["status"]=="draft" for d in state["drafts"].values())}\n'
                       f'Последнее обновление: {data["last_ingest"] or "—"}\n'+job_text+'\n'
                       'Модель и соединения проверяются кнопкой ниже. Работа локальная: компьютер должен быть включён.',
                       kb([[('Проверить модель и базу','adm:health'),('Обновить статусы','lib:status')],
                           [('Обновить базу','adm:ingest'),('История обновлений','lib:runs')],
                           [('Обслуживание базы','lib:maintenance')],[('‹ Главная','adm:home')]]))
        elif action == 'maintenance':
            await show(callback,'<b>Обслуживание базы</b>\n\nОбычное обновление обрабатывает только изменённые документы и использует кэш. '
                       'Полная пересборка нужна при смене настроек поиска или восстановлении; она расходует больше запросов.',
                       kb([[('Обновить изменения','adm:ingest')],[('Полная пересборка…','adm:reindex_confirm')],[('‹ Назад','lib:status')]]))
        elif action == 'guide':
            await show(callback,'<b>Как пользоваться админкой</b>\n\n'
                       '<b>Исправить текст:</b> База → город → документ → Читать и редактировать → пункт → Изменить. '
                       'Отправь новый текст, проверь предпросмотр и нажми «Применить».\n\n'
                       '<b>Добавить материал:</b> «Добавить документ» → город → файл или сообщение с названием в первой строке.\n\n'
                       '<b>Заменить документ:</b> открой его карточку → «Заменить файлом». Предыдущая версия сохранится.\n\n'
                       '<b>Вернуть старую версию:</b> карточка → «История / восстановить» → нужная версия → Применить.\n\n'
                       '<b>Убрать устаревшее:</b> «Убрать в архив». Документ перестанет использоваться после обновления базы.\n\n'
                       '<b>Замечания к ответам:</b> сначала внеси правку в базу, затем закрой замечание. '
                       '«Данные на проверку» — отдельный список неподтверждённых фактов.\n\n'
                       '🟢 Документ используется · 🟡 Ожидает обновления · ⚪ В архиве · ✏️ Есть черновик.\n'
                       '/cancel отменяет текущий ввод. Автоматические технические сообщения получает только ответственный.',back('adm:home'))
        elif action == 'runs':
            from sqlalchemy import select
            from app.storage.db import session_scope
            from app.storage.models import IngestRun
            async with session_scope() as session:
                runs=(await session.execute(select(IngestRun).order_by(IngestRun.id.desc()).limit(6))).scalars().all()
            lines=['<b>История обновлений базы</b>','']
            for run in runs:
                status={'ok':'🟢 Завершено','error':'🔴 Завершено с ошибками','running':'🟡 Выполняется' if admin._ingest_lock.locked() else '⚠ Не завершено / прервано'}.get(run.status,run.status)
                lines.append(f'{local_time(run.started_at)} · {status}\nФайлов: {run.files_total}, изменено: {run.files_changed}\n'+html.escape(str(run.warnings or 'Предупреждений нет')[:350]))
            await show(callback,'\n\n'.join(lines)[:3800],back('lib:status'))
    except (ValueError, IndexError, KeyError) as exc:
        await callback.answer(str(exc)[:190],show_alert=True)
    except Exception:
        log.exception('Ошибка редактора базы')
        await callback.answer('Не удалось выполнить действие. Черновики и версии сохранены; попробуй снова.',show_alert=True)


class EditorInput(BaseFilter):
    async def __call__(self,message: Message):
        return bool(message.from_user and input_key(message) in _inputs)


@router.message(EditorInput())
@admin.admin_only
async def on_input(message: Message, user: User, is_admin: bool):
    state=_inputs.get(input_key(message));actor=message.from_user.id
    if message.text and message.text.startswith('/'):
        _inputs.pop(input_key(message),None)
        await message.answer('Ввод отменён.',reply_markup=back())
        if message.text.split('@')[0]=='/admin':
            await admin.cmd_admin(message,user=user,is_admin=is_admin)
        return
    try:
        mode=state['mode'];text=(message.text or '').strip()
        if mode=='search':
            if not text: raise ValueError('Напиши название текстом')
            items=[d for d in await editor.catalog() if not d.get('archived') and text.casefold() in (d['title']+' '+(d['city'] or '')).casefold()]
            # Результат хранится только в контексте текущего сотрудника.
            _inputs.pop(input_key(message),None)
            if len(items)>PAGE:
                await message.answer(f'Найдено {len(items)} документов. Показаны первые 6; уточни название для остальных.')
            await list_screen(message,items[:PAGE],0,'lib:root','Результаты поиска');return
        if mode=='bsearch':
            if not text: raise ValueError('Напиши слово для поиска')
            _inputs.pop(input_key(message),None);await blocks_screen(message,state['key'],term=text);return
        if mode in ('new','replace') and message.document:
            await receive_file(message,state);return
        if not text: raise ValueError('Пришли текст или подходящий документ')
        if mode=='replace': raise ValueError('Здесь ожидается файл. Для текста выбери редактирование пункта.')
        if mode=='new':
            title,separator,body=text.partition('\n')
            if not separator or not body.strip(): raise ValueError('Первая строка — название, со второй строки — содержание')
            if len(title)>200: raise ValueError('Название должно быть короче 200 символов')
            value={'schema':'company.knowledge.v1','metadata':{'title':title,'section':state['city'] or None,'tags':['other'],'category':'other'},
                   'blocks':[{'kind':'paragraph','text':b.strip()} for b in re.split(r'\n\s*\n',body) if b.strip()]}
            relative=f"{state['city'] or 'Общее'}/Добавлено_{uuid.uuid4().hex[:10]}.kdoc.json"
        else:
            d=await editor.find(state['key']);relative=d['path'];value=state['value']
            if editor.file_sha256(editor.safe_path(relative))!=state['sha']: raise ValueError('Документ изменился. Открой пункт заново.')
            if mode=='edit': value['blocks'][state['index']]['text']=text
            elif mode=='append': value['blocks'].append({'kind':'paragraph','text':text})
            elif mode=='title':
                if len(text)>200: raise ValueError('Название должно быть короче 200 символов')
                value['metadata']['title']=text
            elif mode=='expiry': value['metadata']['valid_until']=None if text.casefold()=='без срока' else date.fromisoformat(text).isoformat()
        label={'edit':'Правка пункта','append':'Новый пункт','title':'Новое название','expiry':'Срок действия','new':'Новый документ'}[mode]
        token=editor.make_draft(relative,value,actor,label,issue_id=state.get('issue_id'))
        _inputs.pop(input_key(message),None)
        await draft_screen(message,token,actor)
    except (ValueError,KeyError,IndexError) as exc:
        await message.answer(html.escape(str(exc))+'\nПопробуй снова или /cancel.')
    except Exception:
        log.exception('Не удалось подготовить правку')
        await message.answer('Не удалось подготовить правку. Исходный документ не изменён. Попробуй снова или /cancel.')


async def receive_file(message,state):
    doc=message.document;suffix=Path(doc.file_name or '').suffix.lower()
    limit=float(get_bot_config().get('ingest.max_file_mb',20))*1048576
    if suffix not in SUPPORTED_EXTENSIONS: raise ValueError('Формат не поддерживается. Подойдут PDF, Word, Excel, TXT, Markdown или JSON.')
    if doc.file_size and doc.file_size>limit: raise ValueError('Файл превышает лимит размера')
    status=await message.answer('Читаю документ. Он пока не используется ботом…')
    with tempfile.TemporaryDirectory(dir=editor.store()) as directory:
        path=Path(directory)/('source'+suffix)
        remote=await message.bot.get_file(doc.file_id)
        await message.bot.download_file(remote.file_path,destination=path)
        if path.stat().st_size>limit: raise ValueError('Файл превышает лимит размера')
        previous=None
        if state['mode']=='replace':
            d=await editor.find(state['key']);original=editor.safe_path(d['path'])
            if editor.file_sha256(original)!=state['sha']: raise ValueError('Документ изменился. Открой замену заново.')
            if not d['editable']: raise ValueError('Редактор заменяет подготовленные документы. Добавь новый документ и убери старый в архив.')
            previous=editor.payload(original);relative=d['path'];city=previous['metadata'].get('section')
        else:
            city=state['city'];relative=f"{city or 'Общее'}/Добавлено_{uuid.uuid4().hex[:10]}.kdoc.json"
        value=await asyncio.to_thread(editor.payload,path,city,previous)
        if not previous:
            value['metadata']['title']=Path(doc.file_name).stem[:200]
            value['metadata'].setdefault('tags',['other'])
            value['metadata'].setdefault('category','other')
        original_dir=editor.store()/'uploads';original_dir.mkdir(exist_ok=True)
        source=original_dir/(uuid.uuid4().hex+suffix)
        source.write_bytes(path.read_bytes())
        value['metadata']['editor_upload']={'filename':doc.file_name,'sha256':editor.file_sha256(source),
                                             'stored_file':source.name,'actor':message.from_user.id,
                                             'received_at':datetime.now(ZoneInfo('UTC')).isoformat()}
        # Проверить нормализованный формат до создания черновика.
        check=Path(directory)/'check.kdoc.json';editor.write_json(check,value)
        from app.ingest.loaders.registry import load_document
        await asyncio.to_thread(load_document,check)
        token=editor.make_draft(relative,value,message.from_user.id,'Замена документа' if previous else 'Загрузка документа')
    _inputs.pop(input_key(message),None)
    await status.edit_text(f'Извлечено пунктов: {len(value["blocks"])}. Проверь текст перед применением. '
                           'Изображения и сложные таблицы могут требовать ручной правки.')
    await draft_screen(message,token,message.from_user.id)


@router.message(F.document)
@admin.admin_only
async def unsolicited_upload(message: Message,user: User,is_admin: bool):
    await message.answer('Сначала выбери город в «База → Добавить» или открой документ и нажми «Заменить файлом». '
                         'Так файл попадёт в нужный раздел, а прежняя версия сохранится.',
                         reply_markup=kb([[('＋ Добавить документ','lib:add'),('Открыть базу','lib:root')]]))
