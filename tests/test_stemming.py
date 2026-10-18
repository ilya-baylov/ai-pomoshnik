"""Тесты стемминга: без него FTS5 не находит русские словоформы."""

from __future__ import annotations

from app.ingest.stemming import build_fts_query, stem_text, stem_word


def test_russian_word_forms_share_stem():
    """Главная причина существования стеммера."""
    assert stem_word("отпуска") == stem_word("отпуск")
    assert stem_word("командировки") == stem_word("командировка")
    assert stem_word("документами") == stem_word("документ")


def test_digits_preserved():
    """Номера регламентов и артикулы менять нельзя."""
    assert stem_word("1500") == "1500"
    assert "2024" in stem_text("приказ от 2024 года")


def test_latin_handled():
    assert stem_word("Documents") == stem_word("document")


def test_stem_text_keeps_all_words():
    result = stem_text("Заявление подаётся за 14 дней")
    assert len(result.split()) == 5


def test_fts_query_joins_with_or():
    query = build_fts_query("суточные командировка")
    assert " OR " in query
    assert query.count('"') == 4


def test_fts_query_quotes_terms():
    """Кавычки защищают от служебных слов FTS5 вроде AND и NOT."""
    query = build_fts_query("отпуск AND справка")
    assert '"and"' in query.lower() or "and" not in query.lower().replace('"', "")


def test_fts_query_skips_single_chars():
    query = build_fts_query("я в отпуске")
    assert '"я"' not in query
    assert '"в"' not in query


def test_empty_query_returns_empty():
    assert build_fts_query("") == ""
    assert build_fts_query("!!!") == ""


def test_matching_forms_produce_same_query():
    assert build_fts_query("отпуска") == build_fts_query("отпуск")
