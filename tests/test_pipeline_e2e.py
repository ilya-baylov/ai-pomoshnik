"""Сквозной тест: индексация демо-базы и поиск по ней.

Эмбеддинги подменены заглушкой — сеть в тестах не нужна. Проверяется, что
пайплайн собирается целиком: загрузчики, нарезка, запись в БД, FTS5,
векторный индекс, объединение результатов.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from app.core.profiles import EmbeddingProfile
from app.embeddings.base import EmbeddingProvider

DIM = 64


class FakeEmbeddings(EmbeddingProvider):
    """Детерминированные векторы из хеша слов.

    Совпадающие слова дают близкие векторы, поэтому поиск работает
    осмысленно, хотя настоящей модели нет.
    """

    def __init__(self) -> None:
        super().__init__(
            EmbeddingProfile(
                name="fake-test-model",
                dim=DIM,
                max_input_tokens=512,
                target_chunk_tokens=180,
                overlap_tokens=30,
                needs_normalization=True,
            )
        )

    def _vector(self, text: str) -> np.ndarray:
        vector = np.zeros(DIM, dtype=np.float32)
        for word in text.lower().split():
            digest = hashlib.md5(word.encode("utf-8")).digest()
            vector[digest[0] % DIM] += 1.0
        norm = np.linalg.norm(vector)
        return vector / norm if norm else vector

    async def embed_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, DIM), dtype=np.float32)
        return np.vstack([self._vector(t) for t in texts])

    async def embed_query(self, text: str) -> np.ndarray:
        return self._vector(text)

    def count_tokens(self, text: str) -> int:
        return max(1, len(text.split()))


@pytest.fixture
async def sandbox(tmp_path, monkeypatch):
    """Изолированная база и папка knowledge с демо-документами.

    Документы создаются на месте, чтобы тест не зависел от реальной
    базы проекта: md с YAML-шапкой (проверка метаданных), csv (поиск
    по значению колонки), второй md (крошки, стемминг), json (удаление).
    """
    from app.core import config as config_module
    from app.embeddings import factory
    from app.storage import db as db_module

    knowledge = tmp_path / "knowledge"
    example = knowledge / "example"
    example.mkdir(parents=True)

    (example / "otpuska.md").write_text(
        "---\n"
        "title: Регламент предоставления отпусков\n"
        "section: hr\n"
        "tags: [отпуск, заявление]\n"
        "priority: 10\n"
        "---\n"
        "# Отпуска\n\n"
        "## Порядок оформления\n\n"
        "Заявление на отпуск подаётся за 14 дней до начала.\n\n"
        "## Продолжительность\n\n"
        "Ежегодный оплачиваемый отпуск составляет 28 календарных дней.\n",
        encoding="utf-8",
    )
    (example / "komandirovki.md").write_text(
        "# Командировки\n\n"
        "## Оформление\n\n"
        "Командировка оформляется приказом, суточные — 700 рублей.\n",
        encoding="utf-8",
    )
    (example / "kontakty.csv").write_text(
        "Отдел,Телефон,Этаж\n"
        "Бухгалтерия,1201,3\n"
        "HR,1202,2\n",
        encoding="utf-8",
    )
    (example / "faq.json").write_text(
        '[{"question": "Где парковка?", "answer": "Парковка за зданием"}]',
        encoding="utf-8",
    )

    db_file = tmp_path / "test.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{db_file.as_posix()}")
    monkeypatch.setenv("KNOWLEDGE_DIR", str(knowledge))
    monkeypatch.setenv("EMBEDDING_PROVIDER", "gemini")
    monkeypatch.setenv("EMBEDDING_MODEL", "fake-test-model")
    monkeypatch.setenv("EMBEDDING_DIM", str(DIM))
    monkeypatch.setenv("LLM_API_KEY", "test-key")

    # Сбрасываем кэши: настройки и движок БД кэшируются на процесс
    config_module.get_settings.cache_clear()
    config_module.get_bot_config.cache_clear()
    factory.get_embedding_provider.cache_clear()
    db_module._engine = None
    db_module._session_factory = None

    provider = FakeEmbeddings()
    monkeypatch.setattr(factory, "get_embedding_provider", lambda: provider)
    monkeypatch.setattr(
        "app.retrieval.search.get_embedding_provider", lambda: provider
    )

    await db_module.init_db()
    yield {"knowledge": knowledge, "provider": provider}

    # Освобождаем движок до сброса: иначе поток aiosqlite доживает за
    # закрытым циклом событий и оставляет предупреждение в тестах
    if db_module._engine is not None:
        await db_module._engine.dispose()
    config_module.get_settings.cache_clear()
    config_module.get_bot_config.cache_clear()
    db_module._engine = None
    db_module._session_factory = None


async def _ingest(provider):
    from app.ingest.pipeline import Ingestor
    from app.retrieval.vector import VectorIndex

    result = await Ingestor(provider=provider).run()
    index = VectorIndex()
    await index.load()
    return result, index


@pytest.mark.asyncio
async def test_demo_base_indexes_without_errors(sandbox):
    result, index = await _ingest(sandbox["provider"])

    assert result.ok, result.errors
    assert result.files_total >= 4, "демо-база должна содержать 4 документа"
    assert result.chunks_created > 0
    assert result.max_token_count <= result.token_limit
    assert index.size == result.chunks_created


@pytest.mark.asyncio
async def test_second_run_is_incremental(sandbox):
    """Повторный прогон не должен делать лишнюю работу."""
    from app.ingest.pipeline import Ingestor

    first = await Ingestor(provider=sandbox["provider"]).run()
    second = await Ingestor(provider=sandbox["provider"]).run()

    assert second.files_changed == 0
    assert second.files_skipped == first.files_total
    assert second.chunks_created == 0


@pytest.mark.asyncio
async def test_edit_reindexes_only_changed_file(sandbox):
    from app.ingest.pipeline import Ingestor

    await Ingestor(provider=sandbox["provider"]).run()

    target = sandbox["knowledge"] / "example" / "otpuska.md"
    target.write_text(
        target.read_text(encoding="utf-8") + "\n\n## Новый раздел\n\nДополнение.\n",
        encoding="utf-8",
    )

    result = await Ingestor(provider=sandbox["provider"]).run()
    assert result.files_changed == 1
    assert result.chunks_created > 0


@pytest.mark.asyncio
async def test_deleted_file_removed_from_index(sandbox):
    from app.ingest.pipeline import Ingestor

    await Ingestor(provider=sandbox["provider"]).run()
    (sandbox["knowledge"] / "example" / "faq.json").unlink()

    result = await Ingestor(provider=sandbox["provider"]).run()
    assert result.files_deleted == 1
    assert result.chunks_deleted > 0


@pytest.mark.asyncio
async def test_lexical_search_finds_russian_word_forms(sandbox):
    """Запрос в другой словоформе должен находить документ."""
    from app.retrieval.lexical import search_lexical

    await _ingest(sandbox["provider"])
    hits = await search_lexical("отпуска", top_k=10)
    assert hits, "стемминг должен связать «отпуска» и «отпуск»"


@pytest.mark.asyncio
async def test_table_row_searchable_by_column_value(sandbox):
    """Подстановка имён колонок: поиск по телефону из CSV."""
    from app.retrieval.lexical import search_lexical

    await _ingest(sandbox["provider"])
    hits = await search_lexical("1201 бухгалтерия", top_k=10)
    assert hits


@pytest.mark.asyncio
async def test_hybrid_search_returns_relevant_document(sandbox):
    from app.retrieval.search import search
    from app.retrieval.vector import get_vector_index

    await _ingest(sandbox["provider"])
    await get_vector_index().invalidate()

    results = await search("сколько дней отпуска положено")
    assert results
    titles = " ".join(r.title.lower() for r in results)
    assert "отпуск" in titles


@pytest.mark.asyncio
async def test_chunks_carry_breadcrumbs(sandbox):
    from sqlalchemy import select

    from app.storage.db import session_scope
    from app.storage.models import Chunk

    await _ingest(sandbox["provider"])
    async with session_scope() as session:
        crumbs = (
            await session.execute(
                select(Chunk.breadcrumbs).where(Chunk.breadcrumbs.is_not(None)).limit(5)
            )
        ).scalars().all()

    assert crumbs, "фрагменты markdown должны нести путь заголовков"


@pytest.mark.asyncio
async def test_metadata_from_front_matter(sandbox):
    """priority и section из YAML-блока должны попасть в базу."""
    from sqlalchemy import select

    from app.storage.db import session_scope
    from app.storage.models import Document

    await _ingest(sandbox["provider"])
    async with session_scope() as session:
        document = (
            await session.execute(
                select(Document).where(Document.path.like("%otpuska.md"))
            )
        ).scalar_one()

    assert document.title == "Регламент предоставления отпусков"
    assert document.section == "hr"
    assert document.priority == 10
    assert "отпуск" in (document.tags or [])
