"""Тесты разрезания Markdown и служебного форматирования.

Главное свойство: части режутся только по границам блоков, и каждая часть
после рендера даёт сбалансированный HTML — иначе Telegram отвергнет ответ.
"""

from __future__ import annotations

import re

from app.channels.telegram.formatting import (
    RAW_LIMIT,
    format_sources,
    progress_bar,
    render_parts,
    split_markdown,
)
from app.channels.telegram.markdown_render import render_markdown
from app.core.contracts import Source

TAGS = ("b", "i", "u", "s", "code", "pre", "blockquote")


def balanced(html_text: str) -> bool:
    for tag in TAGS:
        opened = len(re.findall(rf"<{tag}[\s>]", html_text))
        closed = html_text.count(f"</{tag}>")
        if opened != closed:
            return False
    return True


# ── разрезание Markdown ──────────────────────────────────────────


def test_short_text_not_split():
    assert split_markdown("короткий ответ") == ["короткий ответ"]


def test_long_text_split_by_paragraphs():
    text = "\n\n".join(f"Абзац номер {i} с содержанием." for i in range(400))
    parts = split_markdown(text)

    assert len(parts) > 1
    assert all(len(part) <= RAW_LIMIT for part in parts)
    # Блоки не потерялись
    assert sum(part.count("Абзац номер") for part in parts) == 400


def test_split_parts_render_to_balanced_html():
    """Ключевая гарантия: ни в одной части не рван тег."""
    text = "\n\n".join(
        f"Абзац {i} с **жирным** и `кодом` и ++сроком++." for i in range(400)
    )
    for part in render_parts(text):
        assert balanced(part), part[:200]


def test_list_stays_intact():
    """Список не должен рваться посередине пункта."""
    items = "\n".join(f"- пункт {i} с описанием" for i in range(200))
    text = "Вступление.\n\n" + items
    parts = split_markdown(text)

    for part in parts:
        for line in part.split("\n"):
            # Каждый пункт списка начинается с маркера
            if line.strip() and not line.startswith(("Вступление", "- ", "• ")):
                assert line.startswith("  "), line  # продолжение вложенности


def test_table_not_split():
    """Таблица — одна единица: сетка не рвётся посередине."""
    rows = "\n".join(f"| Код{i} | Наименование {i} |" for i in range(300))
    text = "| Код | Наименование |\n|---|---|\n" + rows
    parts = split_markdown(text)

    for part in parts:
        rendered = render_markdown(part)
        assert balanced(rendered)
        # Строки таблицы не делятся между частями по одной ячейке
        assert rendered.count("<pre>") == rendered.count("</pre>")


def test_code_fence_not_split():
    fence = "```\n" + "\n".join(f"строка кода {i}" for i in range(400)) + "\n```"
    parts = split_markdown(fence)

    for part in render_parts(fence):
        assert balanced(part)


def test_split_prefers_block_boundary():
    head = "а" * (RAW_LIMIT - 200) + " конец абзаца."
    text = head + "\n\n" + "b" * 500
    parts = split_markdown(text)

    assert parts[0].endswith("конец абзаца.")
    assert parts[1].startswith("b")


def test_hard_cut_without_boundary():
    parts = split_markdown("z" * (RAW_LIMIT * 2))
    assert len(parts) >= 2
    assert all(len(part) <= RAW_LIMIT for part in parts)


def test_render_parts_full_pipeline():
    """Сквозной путь из обработчика: Markdown → части → HTML."""
    markdown = (
        "Короткий ответ: **24 дня**.\n\n"
        "Детали:\n- пункт `А-100`\n- ещё пункт\n\n"
        "Сдать ++до пятницы++."
    )
    parts = render_parts(markdown)
    assert len(parts) == 1
    assert "<b>24 дня</b>" in parts[0]
    assert "<code>А-100</code>" in parts[0]
    assert "<u>до пятницы</u>" in parts[0]
    assert "• пункт" in parts[0]


# ── источники ────────────────────────────────────────────────────


def test_sources_in_expandable_block():
    sources = [Source(index=1, title="Регламент отпусков", breadcrumbs="Порядок")]
    result = format_sources(sources)

    assert "blockquote expandable" in result
    assert "Регламент отпусков → Порядок" in result


def test_sources_escape_html():
    sources = [Source(index=1, title="Договор <ООО>")]
    assert "&lt;ООО&gt;" in format_sources(sources)


def test_no_sources_gives_empty_string():
    assert format_sources([]) == ""


def test_stale_source_marked():
    sources = [Source(index=1, title="Старый приказ", is_stale=True)]
    assert "устарел" in format_sources(sources)


# ── прочее ───────────────────────────────────────────────────────


def test_progress_bar_bounds():
    assert progress_bar(0).count("▰") == 0
    assert progress_bar(100).count("▱") == 0
    assert len(progress_bar(50)) == 10


# ── стили кнопок админки ─────────────────────────────────────────


def test_admin_button_styles():
    from app.channels.telegram.keyboards import back, confirm
    from app.channels.telegram.views import _kb, home, reindex_confirm

    kb = _kb(
        [
            [("Кнопка Primary", "data:1", "primary"), ("Кнопка Danger", "data:2", "danger")],
            [("Обычная", "data:3")],
        ]
    )
    assert kb.inline_keyboard[0][0].style == "primary"
    assert kb.inline_keyboard[0][1].style == "danger"
    assert kb.inline_keyboard[1][0].style is None

    assert back().inline_keyboard[0][0].style == "primary"
    assert confirm("adm:test", style="danger").inline_keyboard[0][0].style == "danger"

    data = {
        "unanswered_new": 2,
        "negative": 0,
        "stale_docs": 0,
        "stale_period": None,
        "chunks": 5,
        "documents": 2,
        "days": 7,
        "queries": 10,
        "users": 2,
        "fallback_rate": 0,
        "avg_latency_ms": 100,
    }
    _, home_kb = home(data)
    # The compact home screen contains navigation, not destructive actions.
    buttons = [button for row in home_kb.inline_keyboard for button in row]
    assert [b.callback_data for b in buttons] == [
        'lib:root', 'lib:tasks', 'lib:work', 'adm:users', 'adm:home']
    assert all(b.style is None for b in buttons)

    _, reindex_kb = reindex_confirm()
    assert reindex_kb.inline_keyboard[0][0].style == "danger"


def test_expandable_blockquote_rendering():
    long_quote = "> строка 1\n> строка 2\n> строка 3"
    rendered = render_markdown(long_quote)
    assert "<blockquote expandable>" in rendered
    assert "</blockquote>" in rendered


def test_table_box_divider():
    markdown = "| Услуга | Цена |\n|---|---|\n| Маникюр | 1500 |"
    result = render_markdown(markdown)
    assert "<pre>" in result
    assert " │ " in result
    assert "─┼─" in result
