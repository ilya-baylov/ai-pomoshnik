"""Версии документов и черновики. Источники меняются только при публикации."""
from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import select
from app.core.config import ROOT, get_settings
from app.ingest.loaders.registry import SUPPORTED_EXTENSIONS, load_document
from app.ingest.metadata import file_sha256
from app.storage.db import session_scope
from app.storage.models import Document


def store() -> Path:
    path = ROOT / 'data/library-editor'
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_state() -> dict:
    path = store() / 'state.json'
    return json.loads(path.read_text('utf8')) if path.exists() else {'drafts': {}, 'history': []}


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), 'utf8')
    temporary.replace(path)


def safe_path(relative: str) -> Path:
    root = get_settings().knowledge_path.resolve()
    path = (root / relative).resolve()
    if path == root or not path.is_relative_to(root):
        raise ValueError('Документ находится вне базы знаний')
    return path


def key_for(relative: str) -> str:
    return hashlib.sha256(relative.encode()).hexdigest()[:12]


def payload(path: Path, section: str | None = None, previous: dict | None = None) -> dict:
    if path.name.endswith('.kdoc.json'):
        result = json.loads(path.read_text('utf8'))
        if not isinstance(result.get('blocks'), list) or not isinstance(result.get('metadata'), dict):
            raise ValueError('Файл не содержит документ базы')
    else:
        loaded = load_document(path)
        result = {'schema': 'company.knowledge.v1', 'metadata': {**loaded.meta, 'title': loaded.title, 'section': section},
                  'blocks': [{'text': b.text, 'kind': b.kind.value, 'level': b.level, 'page': b.page}
                             for b in loaded.blocks], 'warnings': loaded.warnings}
    if previous:
        # Замена файла сохраняет привязку к городу и маршрутам поиска.
        result['metadata'] = {**result['metadata'], **previous['metadata']}
    if section is not None:
        result['metadata']['section'] = section or None
    if not any(str(b.get('text', '')).strip() for b in result['blocks']):
        raise ValueError('Не удалось извлечь текст. Файл не добавлен: скану может требоваться распознавание.')
    return result


async def catalog() -> list[dict]:
    root = get_settings().knowledge_path
    async with session_scope() as session:
        indexed = {d.path: d for d in (await session.execute(select(Document))).scalars().all()}
    state = read_state()
    result = []
    for p in sorted(root.rglob('*')):
        if not p.is_file() or p.suffix.lower() not in SUPPORTED_EXTENSIONS or p.name.startswith('~$'):
            continue
        relative = p.relative_to(root).as_posix()
        doc = indexed.get(relative)
        data = None
        if p.name.endswith('.kdoc.json'):
            try: data = json.loads(p.read_text('utf8'))
            except (ValueError, OSError): pass
        meta = (data or {}).get('metadata', {})
        key = key_for(relative)
        result.append({'key': key, 'path': relative, 'title': meta.get('title') or (doc.title if doc else p.stem),
                       'city': meta.get('section', doc.section if doc else (p.parent.name if p.parent != root else None)),
                       'indexed': bool(doc), 'ready': bool(doc and doc.sha256 == file_sha256(p)),
                       'chunks': doc.chunk_count if doc else 0, 'indexed_at': str(doc.indexed_at) if doc else None,
                       'editable': p.name.endswith('.kdoc.json'),
                       'drafts': sum(d['key'] == key and d.get('status') == 'draft' for d in state['drafts'].values())})
    # Архив доступен для восстановления, но не попадает в поиск бота.
    for h in state['history']:
        if h.get('action') == 'archive' and not safe_path(h['path']).exists() and not any(d['key'] == h['key'] for d in result):
            result.append({'key': h['key'], 'path': h['path'], 'title': h['title'], 'city': h['city'],
                           'indexed': h['path'] in indexed, 'ready': False, 'chunks': 0, 'archived': True,
                           'editable': True, 'drafts': 0, 'indexed_at': None})
    return result


async def find(key: str) -> dict:
    item = next((d for d in await catalog() if d['key'] == key), None)
    if item is None: raise ValueError('Документ не найден. Открой список заново.')
    return item


def make_draft(relative: str, value: dict | None, actor: int, label: str, *, action='edit', issue_id=None) -> str:
    path = safe_path(relative)
    if not path.name.endswith('.kdoc.json'):
        raise ValueError('Для исходного файла добавь подготовленный документ и затем убери старый материал вручную.')
    confirmed_value = None
    if issue_id:
        from app.core.knowledge_review import load_queue, is_pending
        queue = load_queue()
        if actor != queue['owner_id']:
            raise ValueError('Подтверждать сведения может ответственный за базу')
        issue = next((i for i in queue['issues'] if i['id'] == issue_id), None)
        if issue and not is_pending(issue):
            raise ValueError('Этот пункт уже проверен. Открой актуальную очередь задач.')
        if not issue or issue['file'].removeprefix('knowledge/') != relative or 'block' not in issue:
            raise ValueError('Запись на проверку не соответствует документу')
        original = payload(path)
        index = issue['block']
        if original['blocks'][index]['text'] != issue.get('current_value'):
            raise ValueError('Спорный пункт уже изменён. Сначала проверь новую версию документа.')
        confirmed_value = value['blocks'][index]['text']
    token = uuid.uuid4().hex[:12]
    state = read_state()
    state['drafts'][token] = {'key': key_for(relative), 'path': relative, 'actor': actor, 'label': label,
                            'base_sha': file_sha256(path) if path.exists() else None,
                            'value': value, 'action': action, 'issue_id': issue_id,
                            'created_at': datetime.now(timezone.utc).isoformat(), 'status': 'draft',
                            'confirmed_value': confirmed_value}
    write_json(store() / 'state.json', state)
    return token


def get_draft(token: str, actor: int) -> dict:
    draft = read_state()['drafts'].get(token)
    if not draft or draft['actor'] != actor or draft['status'] != 'draft':
        raise ValueError('Черновик недоступен или уже применён')
    return draft


def cancel_draft(token: str, actor: int) -> None:
    get_draft(token, actor)
    state = read_state(); state['drafts'][token]['status'] = 'cancelled'
    write_json(store() / 'state.json', state)


def publish(token: str, actor: int) -> dict:
    draft = get_draft(token, actor)
    path = safe_path(draft['path'])
    current_sha = file_sha256(path) if path.exists() else None
    if current_sha != draft['base_sha']:
        raise ValueError('Документ уже изменился. Открой его заново: устаревшая правка не применена.')
    state = read_state()
    if draft['action'] != 'archive':
        value = draft['value']
        if not value or not any(str(b.get('text', '')).strip() for b in value.get('blocks', [])):
            raise ValueError('Нельзя опубликовать пустой документ')
        # Проверка структуры не расходует запросы к модели.
        check = store() / f'validate-{token}.kdoc.json'
        try:
            write_json(check, value)
            load_document(check)
        finally:
            check.unlink(missing_ok=True)
    backup = None
    if path.exists():
        backup = f'{uuid.uuid4().hex}.json'
        (store() / backup).write_bytes(path.read_bytes())
    if draft['action'] == 'archive':
        old = payload(path)
        path.unlink()
        meta = old['metadata']
    else:
        value = draft['value']
        if not value or not any(str(b.get('text', '')).strip() for b in value.get('blocks', [])):
            raise ValueError('Нельзя опубликовать пустой документ')
        meta = value['metadata']
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json(path, value)
    record = {'id': token, 'key': draft['key'], 'path': draft['path'], 'actor': actor,
              'title': meta.get('title', path.stem), 'city': meta.get('section'),
              'action': draft['action'], 'label': draft['label'], 'backup': backup,
              'at': datetime.now(timezone.utc).isoformat(), 'status': 'pending_index',
              'published_sha': file_sha256(path) if path.exists() else None, 'issue_id': draft.get('issue_id'),
              'confirmed_value': draft.get('confirmed_value')}
    state['history'].append(record)
    state['drafts'][token]['status'] = 'published'
    write_json(store() / 'state.json', state)
    return record


async def update_statuses() -> None:
    items = {d['key']: d for d in await catalog()}
    state = read_state()
    for h in state['history']:
        if h['status'] != 'pending_index': continue
        p = safe_path(h['path']); item = items.get(h['key'])
        if (h['action'] == 'archive' and p.exists()) or (
                h['action'] != 'archive' and (not p.exists() or file_sha256(p) != h.get('published_sha'))):
            h['status'] = 'superseded'
            continue
        ready = (not p.exists() and not (item or {}).get('indexed')) if h['action'] == 'archive' else bool(
            p.exists() and file_sha256(p) == h.get('published_sha') and (item or {}).get('ready'))
        if ready:
            h['status'] = 'active' if h['action'] != 'archive' else 'archived'
            if h.get('issue_id'):
                from app.core.knowledge_review import queue_path, load_queue
                queue = load_queue()
                issue = next((i for i in queue['issues'] if i['id'] == h['issue_id']), None)
                if issue:
                    issue.update(status='confirmed', confirmed_at=h['at'], confirmed_by=h['actor'],
                                 confirmed_value=h.get('confirmed_value') or h['label'], source_sha256=h['published_sha'])
                    write_json(queue_path(), queue)
    write_json(store() / 'state.json', state)
    from app.core.knowledge_review import queue_path, load_queue
    queue = load_queue()
    changed = False
    for issue in queue['issues']:
        if issue.get('status') not in ('confirmed', 'resolved_extraction') or 'block' not in issue:
            continue
        path = safe_path(issue['file'].removeprefix('knowledge/'))
        try:
            current = payload(path)['blocks'][issue['block']]['text']
        except (OSError, ValueError, IndexError, KeyError):
            current = None
        expected = issue.get('resolved_value') if issue.get('status') == 'resolved_extraction' else issue.get('confirmed_value')
        if current != expected:
            issue['status'] = 'pending'
            issue['current_value'] = current
            issue['reopened_at'] = datetime.now(timezone.utc).isoformat()
            changed = True
    if changed: write_json(queue_path(), queue)


def restore_draft(revision_id: str, actor: int) -> str:
    h = next((h for h in read_state()['history'] if h['id'] == revision_id), None)
    if not h or not h.get('backup'): raise ValueError('У этой версии нет сохранённого предыдущего документа')
    value = json.loads((store() / h['backup']).read_text('utf8'))
    return make_draft(h['path'], value, actor, 'Восстановление сохранённой версии', action='restore')
