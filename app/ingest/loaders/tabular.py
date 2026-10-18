"""Загрузчики csv, xlsx, json.

Табличные данные режутся построчно с подстановкой названий колонок: иначе
артикул или цена в отрыве от заголовка таблицы поиском не находятся.
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path

from app.ingest.loaders.base import Block, BlockKind, LoadedDoc, Loader

log = logging.getLogger(__name__)


def _row_to_text(header: list[str], row: list) -> str | None:
    pairs = []
    for h, v in zip(header, row):
        if v is None or not str(v).strip():
            continue
        label = str(h).strip()
        value = str(v).strip()
        # В исходнике колонка названа «ИСКЛЮЧИТЬ ПРОЦЕДУРЫ», но семантика
        # разная по файлам: в Новосибирске это услуги в меню мастера
        # («СЛОЖНЫЙ ДИЗАЙН» = делает), в Ростове — настоящие исключения
        # («не делает мужское бикини»). Отрицание уже в тексте значения —
        # оставляем его как отдельную фразу; нейтральные значения
        # переименовываем в положительные, иначе модель советует
        # мастеров наоборот.
        if label.upper().startswith("ИСКЛЮЧИТЬ"):
            if value[:2].lower() == "не":
                pairs.append(value)
            else:
                pairs.append(f"УСЛУГИ МАСТЕРА: {value}")
            continue
        pairs.append(f"{label}: {value}")
    return "; ".join(pairs) if pairs else None


class CsvLoader(Loader):
    extensions = (".csv",)

    def load(self, path: Path) -> LoadedDoc:
        blocks: list[Block] = []
        with path.open(encoding="utf-8-sig", errors="replace", newline="") as fh:
            sample = fh.read(8192)
            fh.seek(0)
            try:
                dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
            except csv.Error:
                dialect = csv.excel  # type: ignore[assignment]
            reader = csv.reader(fh, dialect)
            header: list[str] | None = None
            for row in reader:
                if not any(cell.strip() for cell in row):
                    continue
                if header is None:
                    header = [c.strip() for c in row]
                    continue
                text = _row_to_text(header, row)
                if text:
                    blocks.append(Block(text=text, kind=BlockKind.TABLE_ROW))
        return LoadedDoc(title=path.stem, blocks=blocks)


class XlsxLoader(Loader):
    extensions = (".xlsx", ".xlsm")

    def load(self, path: Path) -> LoadedDoc:
        import yaml
        from app.ingest.loaders.xlsx_xml import read_xlsx_rows
        metadata_path = path.with_suffix(path.suffix + ".meta.yaml")
        metadata = yaml.safe_load(metadata_path.read_text(encoding="utf8")) if metadata_path.exists() else {}
        metadata = metadata or {}
        if metadata.get("source_sha256"):
            from app.ingest.metadata import file_sha256
            if metadata["source_sha256"] != file_sha256(path):
                metadata.pop("skip_rows", None)
                metadata.setdefault("source_warnings", []).append("Источник изменился; прежние исключения строк отменены, требуется проверка данных")
        sheets, warnings = read_xlsx_rows(path)
        blocks = []
        city = metadata.get("source_city")
        for warning in metadata.get("source_warnings", []):
            warnings.append(f"{path.name}: {warning}")
            blocks.append(Block(text=f"ВНИМАНИЕ К ИСТОЧНИКУ: {warning}"))
        for sheet in sheets:
            excluded = set(metadata.get("skip_rows", {}).get(sheet["name"], []))
            rows = [(number,cells) for number,cells in sheet["rows"] if number not in excluded]
            if not rows:
                continue
            heading = metadata.get("table_title") or sheet["name"]
            blocks.append(Block(text=f"{city}: {heading}" if city else heading, kind=BlockKind.HEADING, level=1))
            first = rows[:5]
            header_number, raw_header = max(first, key=lambda r: sum(bool(c) for c in r[1]))
            if max(sum(bool(c) for c in r[1]) for r in first) == 1:
                blocks.extend(Block(text=" ".join(c for c in cells if c)) for _,cells in rows)
                continue
            header = raw_header[:]
            for column,label in metadata.get("header_overrides", {}).items():
                index=int(column)-1
                if 0 <= index < len(header):
                    header[index]=str(label)
            if city:
                for i,label in enumerate(header):
                    lower = label.casefold()
                    if "фамилия" in lower and "имя" in lower:
                        header[i] = "Мастер"
                    elif "филиал" in lower or "наименование салона" in lower:
                        header[i] = "Салон / филиал"
                    elif "название акции" in lower:
                        header[i] = "Акция"
            for number,cells in rows:
                if number == header_number:
                    continue
                if number < header_number:
                    if any(c and not c.isdigit() for c in cells):
                        blocks.append(Block(text=" ".join(c for c in cells if c)))
                    continue
                if not any(v and not v.isdigit() for v in cells):
                    continue  # Empty numbered template rows.
                pairs = []
                for i,value in enumerate(cells):
                    if not value:
                        continue
                    label = header[i] if i < len(header) else ""
                    if label.casefold().strip() in ("№", "номер", "n"):
                        continue
                    pairs.append(f"{label or ('Колонка ' + str(i+1))}: {value}")
                if pairs:
                    prefix = f"Город: {city}; " if city else ""
                    if metadata.get("source_warnings"):
                        prefix += "Примечание к источнику: " + " ".join(metadata["source_warnings"]) + "; "
                    blocks.append(Block(text=prefix + "; ".join(pairs), kind=BlockKind.TABLE_ROW))
        return LoadedDoc(title=metadata.get("title") or path.stem, blocks=blocks,
                         meta=metadata, warnings=warnings)


class JsonLoader(Loader):
    """Массив пар «вопрос-ответ».

    Ожидается [{"question": "...", "answer": "..."}, ...].
    Каждая пара — отдельный фрагмент, дробление запрещено.
    """

    extensions = (".json",)

    def load(self, path: Path) -> LoadedDoc:
        data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
        # Подготовленная общая база сохраняет структуру, страницы и происхождение.
        # Обычные пользовательские JSON с вопросами/ответами остаются совместимы.
        if isinstance(data, dict) and data.get("schema") == "company.knowledge.v1":
            meta = data.get("metadata", {})
            blocks = []
            for item in data.get("blocks", []):
                if not isinstance(item, dict) or not isinstance(item.get("text"), str):
                    raise ValueError("Некорректный блок подготовленной базы")
                text = item["text"].strip()
                if not text:
                    continue
                kind = BlockKind(item.get("kind", "paragraph"))
                level = int(item.get("level", 0))
                page = item.get("page")
                if not 0 <= level <= 6 or (page is not None and (not isinstance(page, int) or page < 1)):
                    raise ValueError("Некорректный заголовок или страница подготовленной базы")
                blocks.append(Block(text=text, kind=kind, level=level, page=page))
            return LoadedDoc(title=str(meta.get("title") or path.stem), blocks=blocks,
                             meta=meta, warnings=data.get("warnings", []))
        blocks: list[Block] = []
        warnings: list[str] = []

        items = data if isinstance(data, list) else data.get("items", [])
        for item in items:
            if not isinstance(item, dict):
                continue
            question = str(
                item.get("question") or item.get("q") or item.get("вопрос") or ""
            ).strip()
            answer = str(
                item.get("answer") or item.get("a") or item.get("ответ") or ""
            ).strip()
            if question and answer:
                blocks.append(
                    Block(text=f"Вопрос: {question}\nОтвет: {answer}", kind=BlockKind.QA)
                )

        if not blocks:
            warnings.append(
                f"{path.name}: не найдено пар вопрос-ответ. Ожидается массив "
                'объектов с полями question и answer.'
            )
        return LoadedDoc(title=path.stem, blocks=blocks, warnings=warnings)
