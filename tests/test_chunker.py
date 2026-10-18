"""Тесты нарезки. Главное — соблюдение лимита токенов модели."""

from __future__ import annotations

from app.ingest.chunker import Chunker
from app.ingest.loaders.base import Block, BlockKind


def fake_count(text: str) -> int:
    """Один токен на слово — предсказуемо для тестов."""
    return max(1, len(text.split()))


def make_chunker(target: int = 20, overlap: int = 4, max_tokens: int = 30) -> Chunker:
    return Chunker(
        count_tokens=fake_count,
        target_tokens=target,
        overlap_tokens=overlap,
        max_tokens=max_tokens,
        min_tokens=3,
    )


def test_breadcrumbs_follow_heading_hierarchy():
    blocks = [
        Block("Регламент отпусков", BlockKind.HEADING, level=1),
        Block("Порядок согласования", BlockKind.HEADING, level=2),
        Block("Заявление подаётся за 14 дней до начала отпуска.", BlockKind.PARAGRAPH),
    ]
    chunks = make_chunker().split(blocks)

    assert len(chunks) == 1
    assert chunks[0].breadcrumbs == "Регламент отпусков → Порядок согласования"


def test_heading_closes_nested_levels():
    """Заголовок первого уровня должен сбросить вложенные."""
    blocks = [
        Block("Раздел А", BlockKind.HEADING, level=1),
        Block("Подраздел", BlockKind.HEADING, level=2),
        Block("Текст один.", BlockKind.PARAGRAPH),
        Block("Раздел Б", BlockKind.HEADING, level=1),
        Block("Текст два.", BlockKind.PARAGRAPH),
    ]
    chunks = make_chunker().split(blocks)

    assert chunks[-1].breadcrumbs == "Раздел Б"


def test_never_exceeds_model_limit():
    """Ключевая гарантия: модель молча обрезает вход длиннее лимита."""
    long_text = " ".join(f"слово{i}" for i in range(500))
    blocks = [Block(long_text, BlockKind.PARAGRAPH)]

    chunker = make_chunker(target=20, overlap=4, max_tokens=30)
    chunks = chunker.split(blocks)

    assert len(chunks) > 1
    assert all(c.token_count <= 30 for c in chunks), [c.token_count for c in chunks]


def test_table_rows_stay_separate():
    """Строки таблицы не склеиваются: иначе теряется соответствие колонкам."""
    blocks = [
        Block("Артикул: А-1; Цена: 100", BlockKind.TABLE_ROW),
        Block("Артикул: А-2; Цена: 200", BlockKind.TABLE_ROW),
        Block("Артикул: А-3; Цена: 300", BlockKind.TABLE_ROW),
    ]
    chunks = make_chunker().split(blocks)

    assert len(chunks) == 3
    assert "А-2" in chunks[1].text


def test_qa_pair_not_split():
    blocks = [
        Block("Вопрос: как получить справку?\nОтвет: написать в отдел кадров", BlockKind.QA)
    ]
    chunks = make_chunker().split(blocks)

    assert len(chunks) == 1
    assert "Ответ:" in chunks[0].text


def test_tiny_chunks_merged():
    """Фрагмент из двух слов бесполезен — приклеивается к соседнему."""
    blocks = [
        Block(" ".join(f"текст{i}" for i in range(18)), BlockKind.PARAGRAPH),
        Block("Да.", BlockKind.PARAGRAPH),
    ]
    chunks = make_chunker(target=20, overlap=0, max_tokens=30).split(blocks)

    assert len(chunks) == 1


def test_overlap_carries_context():
    blocks = [
        Block(f"предложение{i} слово слово слово." , BlockKind.PARAGRAPH)
        for i in range(6)
    ]
    chunks = make_chunker(target=10, overlap=4, max_tokens=30).split(blocks)

    assert len(chunks) >= 2
    # Хвост предыдущего фрагмента должен встречаться в следующем
    assert any(
        part in chunks[1].text
        for part in chunks[0].text.split("\n")[-1:]
    )


def test_empty_blocks_produce_nothing():
    assert make_chunker().split([]) == []


def test_overlap_page_follows_retained_text():
    blocks = [Block(f"страница{i} слово слово слово.", BlockKind.PARAGRAPH, page=i)
              for i in range(1, 12)]
    chunks = make_chunker(target=10, overlap=4).split(blocks)
    for chunk in chunks:
        first_page = int(chunk.text.split()[0].removeprefix("страница"))
        assert chunk.page == first_page
    assert chunks[-1].page >= 9


def test_no_overlap_uses_new_page():
    blocks = [Block("первый текст три слова", BlockKind.PARAGRAPH, page=2),
              Block("второй текст три слова", BlockKind.PARAGRAPH, page=80)]
    chunks = make_chunker(target=4, overlap=0).split(blocks)
    assert [c.page for c in chunks] == [2, 80]
