"""Реестр загрузчиков по расширению файла.

Добавление формата = новый класс Loader и одна строка здесь.
"""

from __future__ import annotations

from pathlib import Path

from app.ingest.loaders.base import Loader, LoadedDoc
from app.ingest.loaders.office import DocxLoader, HtmlLoader, PdfLoader
from app.ingest.loaders.tabular import CsvLoader, JsonLoader, XlsxLoader
from app.ingest.loaders.text import MarkdownLoader, PlainTextLoader

_LOADERS: list[Loader] = [
    MarkdownLoader(),
    PlainTextLoader(),
    DocxLoader(),
    PdfLoader(),
    HtmlLoader(),
    CsvLoader(),
    XlsxLoader(),
    JsonLoader(),
]

_BY_EXTENSION: dict[str, Loader] = {
    ext: loader for loader in _LOADERS for ext in loader.extensions
}

SUPPORTED_EXTENSIONS = frozenset(_BY_EXTENSION)


def get_loader(path: Path) -> Loader | None:
    return _BY_EXTENSION.get(path.suffix.lower())


def load_document(path: Path) -> LoadedDoc:
    loader = get_loader(path)
    if loader is None:
        raise ValueError(f"Формат {path.suffix} не поддерживается")
    return loader.load(path)
