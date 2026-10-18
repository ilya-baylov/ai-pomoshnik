"""Список вопросов к владельцу данных. Подтверждение не выдумывается."""
import csv
import io
import json
from pathlib import Path
from app.core.config import ROOT, get_bot_config

def queue_path() -> Path:
    return ROOT / "BD_UNIFIED/review-queue.json"

def load_queue() -> dict:
    path = queue_path()
    if not path.exists():
        return {"issues": [], "owner_id": get_bot_config().get("knowledge_review.owner_id")}
    return json.loads(path.read_text('utf8'))


def is_pending(item: dict) -> bool:
    return item.get('status') not in ('confirmed', 'resolved_extraction')

def csv_export() -> bytes:
    stream = io.StringIO(newline='')
    writer = csv.writer(stream, delimiter=';')
    writer.writerow(["ID", "Город", "Вопрос", "Источник", "Строка/блок", "Текущее значение", "Подтверждённое значение", "Дата проверки", "Комментарий"])
    for item in load_queue()['issues']:
        if not is_pending(item):
            continue
        values = [item['id'], item['city'], item['issue'], item['file'], item.get('block'),
                  item.get('current_value', ''), '', '', '']
        writer.writerow(["'"+str(v) if str(v).startswith(('=', '+', '-', '@')) else v for v in values])
    return ('\ufeff' + stream.getvalue()).encode('utf8')
