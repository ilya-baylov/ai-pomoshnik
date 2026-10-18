"""Markdown → Telegram-HTML.

Модель отвечает в Markdown, адаптер канала превращает его в подмножество
HTML, которое понимает Telegram. Парсер — markdown-it-py (уже в
requirements): вложенные списки и таблицы он разбирает надёжнее регэкспов.

Чего в Telegram нет и чем заменяем:
- настоящие заголовки → жирная строка;
- таблицы → выровненная сетка моноширинным шрифтом;
- картинки в тексте → подпись-ссылка;
- подчёркивание (в Markdown его нет) → соглашение ++текст++ из промпта.

Сырой HTML из ответа модели не пропускается: у парсера выключен html,
поэтому теги становятся обычным текстом. Telegram отвергает сообщения с
неизвестными тегами, так что рисковать нельзя.
"""

from __future__ import annotations

import html
import re
from urllib.parse import urlsplit

from markdown_it import MarkdownIt
from markdown_it.token import Token

# strict-пресет без сырого HTML и автосвязывания: всё постороннее
# превращается в обычный текст и дальше просто экранируется
_md = MarkdownIt("commonmark", {"html": False, "linkify": False})
_md.enable("table")
_md.enable("strikethrough")

# Дедлайны: ++текст++ → <u>. Первый и последний символ внутри должны быть
# непробельными, иначе «C++ и C++» превратилось бы в подчёркивание.
_UNDERLINE = re.compile(r"\+\+(\S(?:.*?\S)?)\+\+", re.DOTALL)

_ALLOWED_SCHEMES = frozenset({"http", "https"})


def _esc(text: str) -> str:
    return html.escape(text, quote=False)


def _esc_attr(text: str) -> str:
    return html.escape(text, quote=True)


def render_markdown(text: str) -> str:
    """Markdown из ответа модели в HTML для Telegram."""
    if not text or not text.strip():
        return ""
    blocks: list[str] = []
    _render_tokens(_md.parse(text), blocks)
    return "\n\n".join(block for block in (b.strip() for b in blocks) if block)


def _matching_close(tokens: list[Token], open_idx: int) -> int:
    """Индекс парного *_close с учётом вложенности того же типа."""
    open_token = tokens[open_idx]
    close_type = open_token.type[: -len("_open")] + "_close"
    for idx in range(open_idx + 1, len(tokens)):
        token = tokens[idx]
        if token.type == close_type and token.level == open_token.level:
            return idx
    return len(tokens) - 1


def _render_tokens(tokens: list[Token], out: list[str]) -> None:
    i = 0
    while i < len(tokens):
        token = tokens[i]

        if token.type in ("paragraph_open", "heading_open"):
            # Структура всегда open → inline → close
            rendered = _inline(tokens[i + 1])
            if token.type == "heading_open":
                # Настоящих заголовков в Telegram нет — жирная строка
                rendered = f"<b>{rendered}</b>"
            out.append(rendered)
            i += 3
            continue

        if token.type in ("bullet_list_open", "ordered_list_open"):
            close = _matching_close(tokens, i)
            _render_list(tokens, i, close, out)
            i = close + 1
            continue

        if token.type == "blockquote_open":
            close = _matching_close(tokens, i)
            inner: list[str] = []
            _render_tokens(tokens[i + 1 : close], inner)
            body = "\n".join(line for line in inner if line)
            tag = "blockquote expandable" if body.count("\n") >= 2 or len(body) > 200 else "blockquote"
            out.append(f"<{tag}>{body}</blockquote>")
            i = close + 1
            continue

        if token.type in ("fence", "code_block"):
            content = token.content.rstrip("\n")
            out.append(f"<pre>{_esc(content)}</pre>")
            i += 1
            continue

        if token.type == "table_open":
            close = _matching_close(tokens, i)
            out.append(_render_table(tokens[i + 1 : close]))
            i = close + 1
            continue

        if token.type == "hr":
            # Разделитель: пустая строка, отфильтруется при сборке
            out.append("")
            i += 1
            continue

        i += 1


def _render_list(tokens: list[Token], open_idx: int, close_idx: int, out: list[str]) -> None:
    open_token = tokens[open_idx]
    ordered = open_token.type == "ordered_list_open"
    try:
        number = int(open_token.attrGet("start") or 1)
    except (TypeError, ValueError):
        number = 1

    # Продолжение пункта выравнивается под маркер
    continuation = "  "
    items: list[str] = []
    i = open_idx + 1
    while i < close_idx:
        token = tokens[i]
        if token.type == "list_item_open":
            close = _matching_close(tokens, i)
            inner: list[str] = []
            _render_tokens(tokens[i + 1 : close], inner)
            marker = f"{number}. " if ordered else "• "
            if ordered:
                number += 1
            if inner:
                items.append(marker + inner[0])
                items.extend(
                    continuation + line if line else line for line in inner[1:]
                )
            else:
                items.append(marker.rstrip())
            i = close + 1
        else:
            i += 1

    if items:
        out.append("\n".join(items))


def _inline(token: Token) -> str:
    """Inline-токен в HTML: жирный, курсив, код, ссылки, ++подчёркивание++."""
    if token.type != "inline":
        return _esc(token.content)
    if not token.children:
        return _underline(_esc(token.content))
    return _render_inline_children(token.children)


def _render_inline_children(children: list[Token]) -> str:
    parts: list[str] = []
    i = 0
    skip_close = False  # запрещённая ссылка: её </a> тоже пропускаем
    while i < len(children):
        token = children[i]
        kind = token.type

        if kind in ("text", "text_special"):
            parts.append(_underline(_esc(token.content)))
        elif kind == "code_inline":
            parts.append(f"<code>{_esc(token.content)}</code>")
        elif kind == "strong_open":
            parts.append("<b>")
        elif kind == "strong_close":
            parts.append("</b>")
        elif kind == "em_open":
            parts.append("<i>")
        elif kind == "em_close":
            parts.append("</i>")
        elif kind == "s_open":
            parts.append("<s>")
        elif kind == "s_close":
            parts.append("</s>")
        elif kind == "link_open":
            href = token.attrGet("href") or ""
            if _safe_scheme(href):
                parts.append(f'<a href="{_esc_attr(href)}">')
            else:
                # Схема не из разрешённых: текст остаётся, ссылка исчезает
                skip_close = True
        elif kind == "link_close":
            if skip_close:
                skip_close = False
            else:
                parts.append("</a>")
        elif kind == "image":
            alt = token.content or "изображение"
            href = token.attrGet("src") or ""
            parts.append(_link(alt, href))
        elif kind in ("softbreak", "hardbreak"):
            parts.append("\n")
        elif token.content:
            parts.append(_esc(token.content))

        i += 1
    return "".join(parts)


def _safe_scheme(href: str) -> bool:
    scheme = (urlsplit(href).scheme or "").lower()
    return scheme in _ALLOWED_SCHEMES


def _link(text: str, href: str) -> str:
    if _safe_scheme(href):
        return f'<a href="{_esc_attr(href)}">{_esc(text)}</a>'
    return _esc(text)


def _underline(escaped_text: str) -> str:
    if "++" not in escaped_text:
        return escaped_text
    return _UNDERLINE.sub(lambda m: f"<u>{m.group(1)}</u>", escaped_text)


def _render_table(tokens: list[Token]) -> str:
    """Таблица в выровненную сетку моноширинным шрифтом."""
    rows: list[list[str]] = []
    header_flags: list[bool] = []
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if token.type != "tr_open":
            i += 1
            continue
        cells: list[str] = []
        from_header = False
        i += 1
        while i < len(tokens) and tokens[i].type != "tr_close":
            cell_token = tokens[i]
            if cell_token.type in ("th_open", "td_open"):
                from_header = from_header or cell_token.type == "th_open"
                cells.append(_plain(tokens[i + 1]))
                i += 3
                continue
            i += 1
        rows.append(cells)
        header_flags.append(from_header)
        i += 1

    if not rows:
        return ""

    width = max(len(row) for row in rows)
    for row in rows:
        row.extend([""] * (width - len(row)))
    widths = [max(len(row[col]) for row in rows) for col in range(width)]

    lines: list[str] = []
    for index, row in enumerate(rows):
        lines.append(
            " │ ".join(
                cell.ljust(widths[col]) for col, cell in enumerate(row)
            ).rstrip()
        )
        if header_flags[index]:
            lines.append("─┼─".join("─" * w for w in widths).rstrip())

    return "<pre>" + _esc("\n".join(lines)) + "</pre>"


def _plain(inline: Token) -> str:
    """Ячейка таблицы в чистый текст: без тегов, переносы — пробелом."""
    if inline.type != "inline":
        return inline.content.replace("\n", " ")
    rendered = _render_inline_children(inline.children or [])
    text = re.sub(r"<[^>]+>", "", rendered)
    return html.unescape(text).replace("\n", " ").strip()
