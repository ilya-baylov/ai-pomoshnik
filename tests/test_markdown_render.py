"""Golden-тесты конвертера Markdown → Telegram-HTML.

Проверяют и внешний вид (что увидит сотрудник), и безопасность: сырой
HTML и запрещённые схемы ссылок не должны проходить в сообщение.
"""

from __future__ import annotations

import re

from app.channels.telegram.markdown_render import render_markdown

TAGS = ("b", "i", "u", "s", "code", "pre", "blockquote")


def balanced(html_text: str) -> bool:
    """Все парные теги закрыты — иначе Telegram отвергнет сообщение."""
    for tag in TAGS:
        opened = len(re.findall(rf"<{tag}[\s>]", html_text))
        closed = html_text.count(f"</{tag}>")
        if opened != closed:
            return False
    return True


# ── базовые конструкции ──────────────────────────────────────────


def test_plain_text_untouched():
    assert render_markdown("просто текст") == "просто текст"


def test_bold_and_italic():
    result = render_markdown("**24 календарных дня** и *не больше*")
    assert result == "<b>24 календарных дня</b> и <i>не больше</i>"


def test_strikethrough():
    assert render_markdown("~~отменено~~") == "<s>отменено</s>"


def test_heading_becomes_bold_line():
    result = render_markdown("## Порядок согласования\n\nТекст абзаца.")
    assert result == "<b>Порядок согласования</b>\n\nТекст абзаца."


def test_inline_code():
    assert render_markdown("заявление `А-100` подать") == (
        "заявление <code>А-100</code> подать"
    )


def test_code_block_escaped_and_preserved():
    result = render_markdown("```\nSELECT * FROM <table>\n```")
    assert result == "<pre>SELECT * FROM &lt;table&gt;</pre>"


def test_paragraphs_separated_by_blank_line():
    result = render_markdown("первый\n\nвторой")
    assert result == "первый\n\nвторой"


# ── списки ───────────────────────────────────────────────────────


def test_bullet_list():
    result = render_markdown("- пункт один\n- пункт два")
    assert result == "• пункт один\n• пункт два"


def test_ordered_list_keeps_numbers():
    result = render_markdown("3. согласовать\n4. подать")
    assert result == "3. согласовать\n4. подать"


def test_nested_list_indented():
    result = render_markdown("- верхний\n  - вложенный")
    assert result == "• верхний\n  • вложенный"


def test_list_with_bold():
    result = render_markdown("- срок **14 дней**")
    assert result == "• срок <b>14 дней</b>"


# ── ссылки и картинки ────────────────────────────────────────────


def test_https_link_clickable():
    result = render_markdown("[регламент](https://intranet.example/docs)")
    assert result == '<a href="https://intranet.example/docs">регламент</a>'


def test_javascript_link_neutralized():
    """Схема javascript: не должна попасть в сообщение."""
    result = render_markdown("[жми](javascript:alert(1))")
    assert "<a" not in result
    assert "жми" in result


def test_relative_link_neutralized():
    result = render_markdown("[док](/docs/page)")
    assert "<a" not in result
    assert "док" in result


def test_image_becomes_link():
    result = render_markdown("![схема](https://example.com/plan.png)")
    assert result == '<a href="https://example.com/plan.png">схема</a>'


# ── цитаты и подчёркивание ───────────────────────────────────────


def test_blockquote():
    result = render_markdown("> цитата из приказа")
    assert result == "<blockquote>цитата из приказа</blockquote>"


def test_underline_convention():
    assert render_markdown("сдать ++до 14 дней++") == "сдать <u>до 14 дней</u>"


def test_underline_needs_content():
    """Пустые ++ не должны превращаться в пустой тег."""
    result = render_markdown("++++")
    assert "<u>" not in result


def test_cpp_not_underlined():
    """Типографский подвох: C++ не должен ломаться в подчёркивание."""
    result = render_markdown("написано на C++ и C++")
    assert "<u>" not in result


# ── таблицы ──────────────────────────────────────────────────────


def test_table_becomes_monospace_grid():
    markdown = "| Документ | Срок |\n|---|---|\n| Заявление | 14 дней |"
    result = render_markdown(markdown)

    assert result.startswith("<pre>")
    assert "Документ" in result and "Срок" in result
    assert "Заявление" in result
    assert "|" not in result.split("</pre>")[0].split("<pre>")[1]
    assert balanced(result)


def test_table_cells_escaped():
    markdown = "| Код | Знак |\n|---|---|\n| X | <b> |"
    result = render_markdown(markdown)
    assert "&lt;b&gt;" in result


# ── безопасность ─────────────────────────────────────────────────


def test_raw_html_escaped_not_executed():
    """Telegram отвергает неизвестные теги — сырой HTML обязан экранироваться."""
    result = render_markdown("до <script>alert(1)</script> после")
    assert "<script>" not in result
    assert "&lt;script&gt;" in result


def test_html_in_bold_escaped():
    result = render_markdown("**важно <b>очень**</b>")
    assert balanced(result)
    assert "&lt;b&gt;" in result


def test_special_chars_escaped():
    result = render_markdown("условие: a < b & c")
    assert "&lt;" in result and "&amp;" in result


def test_every_result_balanced():
    samples = [
        "# Заголовок\n\n**текст** с [ссылкой](https://x.y)\n",
        "- раз\n  - два\n- три",
        "> цитата\n\n```\nкод\n```",
        "| а | б |\n|---|---|\n| 1 | 2 |",
        "смешанный **жирный** и ++срок++ и `код`",
    ]
    for sample in samples:
        assert balanced(render_markdown(sample)), sample


def test_empty_input():
    assert render_markdown("") == ""
    assert render_markdown("   \n  ") == ""
