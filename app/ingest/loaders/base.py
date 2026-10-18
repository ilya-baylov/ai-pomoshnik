"""Общий вид документа после загрузки.

Каждый загрузчик приводит свой формат к списку блоков. Дальше нарезка работает
только с блоками и не знает, откуда они пришли, — поэтому добавление формата
не задевает логику нарезки.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class BlockKind(str, Enum):
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    # Строка таблицы: не склеивается с соседями, иначе теряется смысл
    TABLE_ROW = "table_row"
    # Готовая пара «вопрос-ответ»: дробить нельзя
    QA = "qa"


@dataclass
class Block:
    text: str
    kind: BlockKind = BlockKind.PARAGRAPH
    # Уровень заголовка 1-6, только для HEADING
    level: int = 0
    page: int | None = None


@dataclass
class LoadedDoc:
    title: str
    blocks: list[Block]
    # Метаданные из front-matter, переопределяют выведенные из пути
    meta: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


class Loader:
    """Базовый загрузчик. Регистрируется по расширению файла."""

    extensions: tuple[str, ...] = ()

    def load(self, path: Path) -> LoadedDoc:  # pragma: no cover - интерфейс
        raise NotImplementedError
