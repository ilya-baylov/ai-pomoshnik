"""Форматирование сообщений для Telegram.

Конвертация Markdown → HTML живёт в markdown_render.py. Здесь — разрезание
длинных ответов: оно выполняется по границам блоков Markdown и ДО
конвертации, поэтому HTML-тег физически не может оказаться разрезан.
Используется HTML, а не MarkdownV2: последний требует экранировать полтора
десятка символов, и любой пропущенный роняет отправку целого ответа.
"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass

from app.channels.telegram.markdown_render import render_markdown
from app.core.contracts import Source

TELEGRAM_LIMIT = 4096
# С запасом под блок источников и служебные теги
SAFE_LIMIT = 3400
# Лимит для частей исходного Markdown: после конвертации часть
# подрастает на теги, поэтому исходник держим заметно короче
RAW_LIMIT = 2800

_SENTENCE = re.compile(r"(?<=[.!?;…])\s+")


def to_html(text: str) -> str:
    """Markdown от модели в HTML для Telegram."""
    return render_markdown(text)


def format_sources(sources: list[Source]) -> str:
    """Источники в раскрывающемся блоке — не занимают место в чате."""
    if not sources:
        return ""
    lines = [
        f"{source.index}. {html.escape(source.label())}" for source in sources
    ]
    body = "\n".join(lines)
    return (
        f"\n\n<blockquote expandable><b>Источники</b>\n{body}</blockquote>"
    )


@dataclass
class MarkdownBlock:
    """Блок исходного Markdown: разрезание идёт только по таким границам."""

    text: str
    kind: str  # heading | paragraph | list_item | fence | blockquote | table | hr | blank


def _split_blocks(markdown: str) -> list[MarkdownBlock]:
    """Разобрать Markdown на блоки верхнего уровня.

    Наивное разбиение по пустым строкам достаточно: границы блоков Markdown
    совпадают с ними, а внутри кодовых блоков пустая строка не рвёт блок —
    он распознаётся парсером дальше по ограждению ``` и склеивается.
    """
    blocks: list[MarkdownBlock] = []
    pending_code: list[str] = []

    for raw_line in markdown.split("\n"):
        stripped = raw_line.strip()

        if pending_code:
            pending_code.append(raw_line)
            if stripped.startswith("```") and len(pending_code) > 1:
                blocks.append(
                    MarkdownBlock("\n".join(pending_code), kind="fence")
                )
                pending_code.clear()
            continue
        if stripped.startswith("```"):
            pending_code.append(raw_line)
            continue

        if not stripped:
            blocks.append(MarkdownBlock("", kind="blank"))
        elif stripped.startswith("#"):
            blocks.append(MarkdownBlock(raw_line, kind="heading"))
        elif stripped.startswith(">"):
            blocks.append(MarkdownBlock(raw_line, kind="blockquote"))
        elif stripped.startswith("|"):
            blocks.append(MarkdownBlock(raw_line, kind="table"))
        elif stripped.startswith(("-", "*", "+")) or (
            len(stripped) > 1 and stripped[0].isdigit() and stripped[1] in ".)"
        ):
            blocks.append(MarkdownBlock(raw_line, kind="list_item"))
        elif set(stripped) <= {"-", "_", "*"} and len(stripped) >= 3:
            blocks.append(MarkdownBlock(raw_line, kind="hr"))
        else:
            blocks.append(MarkdownBlock(raw_line, kind="paragraph"))

    if pending_code:  # незакрытый кодовый блок — отдаём как есть
        blocks.append(MarkdownBlock("\n".join(pending_code), kind="fence"))
    return blocks


def _group_units(markdown: str) -> list[str]:
    """Собрать блоки в неделимые единицы: абзац = все его строки вместе.

    Вложенные списки оставляем как отдельные пункты — их перенос на новую
    часть допустим. Таблицу склеиваем в одну единицу, чтобы сетка не рвалась.
    """
    units: list[str] = []
    current: list[str] = []
    current_kind = ""

    def flush() -> None:
        if current:
            units.append("\n".join(current).strip("\n"))
            current.clear()

    for block in _split_blocks(markdown):
        if block.kind == "blank":
            flush()
            continue
        same_run = block.kind in ("paragraph", "list_item", "table") and block.kind == current_kind
        if current and not same_run:
            flush()
        current.append(block.text)
        current_kind = block.kind
    flush()
    return [u for u in units if u.strip()]


def split_markdown(text: str, limit: int = RAW_LIMIT) -> list[str]:
    """Разрезать длинный Markdown на части по границам блоков.

    Лимит действует на исходный Markdown: при конвертации часть подрастает
    на теги и остаётся в пределах Telegram. Каждый кусок затем конвертируется
    независимо, поэтому тег не может оказаться разрезан посередине.
    """
    if len(text) <= limit:
        return [text]

    parts: list[str] = []
    current: list[str] = []
    current_len = 0

    for unit in _group_units(text):
        unit_len = len(unit) + (2 if current else 0)  # \n\n между блоками

        # Сам по себе гигантский абзац или таблица — режем по строкам,
        # а строку длиннее лимита — по предложениям и словам
        if len(unit) > limit:
            if current:
                parts.append("\n\n".join(current))
                current, current_len = [], 0
            parts.extend(_hard_wrap(unit, limit))
            continue

        if current and current_len + unit_len > limit:
            parts.append("\n\n".join(current))
            current, current_len = [], 0
            unit_len = len(unit)

        current.append(unit)
        current_len += unit_len

    if current:
        parts.append("\n\n".join(current))
    return parts


def _hard_wrap(unit: str, limit: int) -> list[str]:
    """Блок длиннее лимита: по строкам, внутри строки — по предложениям,
    совсем длинное предложение — по словам.

    Разрез идёт по сырому Markdown: незакрытая пара ** останется литеральной
    звёздочкой, но HTML-теги при этом не порвутся.
    """
    parts: list[str] = []
    current = ""

    def push(piece: str) -> None:
        nonlocal current
        if not piece:
            return
        candidate = f"{current}\n{piece}" if current else piece
        if len(candidate) <= limit:
            current = candidate
            return
        if current:
            parts.append(current)
            current = ""
        # Кусок сам длиннее лимита — режем по словам
        while len(piece) > limit:
            cut = piece.rfind(" ", 0, limit)
            cut = cut if cut > limit // 2 else limit
            parts.append(piece[:cut])
            piece = piece[cut:].lstrip()
        current = piece

    for line in unit.split("\n"):
        if not line.strip():
            continue
        if len(line) <= limit:
            push(line)
            continue
        buffer = ""
        for sentence in _SENTENCE.split(line):
            sentence = sentence.strip()
            if not sentence:
                continue
            candidate = f"{buffer} {sentence}" if buffer else sentence
            if len(candidate) <= limit:
                buffer = candidate
                continue
            if buffer:
                push(buffer)
                buffer = ""
            push(sentence)
        if buffer:
            push(buffer)

    if current:
        parts.append(current)
    return parts


def split_message(text: str, limit: int = RAW_LIMIT) -> list[str]:
    """Совместимость со старым именем: режем Markdown до конвертации."""
    return split_markdown(text, limit)


def render_parts(markdown: str, limit: int = RAW_LIMIT) -> list[str]:
    """Полный путь: Markdown → части → готовые HTML-куски для Telegram."""
    return [render_markdown(part) for part in split_markdown(markdown, limit)]


def progress_bar(percent: int, width: int = 10) -> str:
    """Полоса прогресса для живого индикатора индексации."""
    filled = round(percent / 100 * width)
    return "▰" * filled + "▱" * (width - filled)
