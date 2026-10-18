"""Стемминг для полнотекстового поиска.

FTS5 не знает морфологии русского, поэтому «отпуска» не находит «отпуск».
Решение: в поисковую колонку кладём стеммированный текст, и запрос стеммим
тем же способом.
"""

from __future__ import annotations

import functools
import re

WORD = re.compile(r"[^\W\d_]+|\d+", re.UNICODE)


@functools.lru_cache(maxsize=2)
def _stemmers() -> tuple:
    import snowballstemmer

    return (
        snowballstemmer.stemmer("russian"),
        snowballstemmer.stemmer("english"),
    )


@functools.lru_cache(maxsize=200_000)
def stem_word(word: str) -> str:
    russian, english = _stemmers()
    lowered = word.lower()
    if lowered.isdigit():
        return lowered
    # Кириллица — русский стеммер, остальное — английский
    if any("а" <= ch <= "я" or ch == "ё" for ch in lowered):
        return russian.stemWord(lowered)
    return english.stemWord(lowered)


def stem_text(text: str) -> str:
    """Текст в набор основ слов, разделённых пробелом."""
    return " ".join(stem_word(match.group()) for match in WORD.finditer(text))


def build_fts_query(text: str) -> str:
    """Запрос для FTS5 MATCH: основы через OR.

    Кавычки обязательны — иначе слова вроде AND или NOT ломают синтаксис.
    """
    stems = {stem_word(m.group()) for m in WORD.finditer(text)}
    terms = [f'"{s}"' for s in stems if len(s) > 1]
    return " OR ".join(terms)
