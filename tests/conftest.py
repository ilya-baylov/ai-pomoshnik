"""Tests never load server credentials, use its database, or reach the network."""
import pytest
import pytest_asyncio


@pytest_asyncio.fixture(autouse=True)
async def isolated_settings(tmp_path, monkeypatch):
    from app.core import config
    from app.storage import db
    from app.embeddings import factory
    from app.retrieval import search, vector
    from app import service
    clear_provider_cache = factory.get_embedding_provider.cache_clear
    for name, field in config.Settings.model_fields.items():
        monkeypatch.delenv(name.upper(), raising=False)
        if isinstance(field.validation_alias, str):
            monkeypatch.delenv(field.validation_alias, raising=False)
    monkeypatch.setitem(config.Settings.model_config, "env_file", None)
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{(tmp_path / 'test.db').as_posix()}")
    monkeypatch.setenv("KNOWLEDGE_DIR", str(tmp_path / "knowledge"))
    config.get_settings.cache_clear()
    config.get_bot_config.cache_clear()
    clear_provider_cache()
    db._engine = db._session_factory = None
    search.reset_sections_cache()
    vector._index = None
    service._service = None
    import httpx
    import openai
    async def blocked(*args, **kwargs):
        raise AssertionError("Network access is forbidden in tests")
    monkeypatch.setattr(httpx.AsyncClient, "send", blocked)
    # SDK 3 uses httpx2; block at its own transport as well.
    try:
        import httpx2
        monkeypatch.setattr(httpx2.AsyncClient, "send", blocked)
    except ImportError:
        pass
    yield
    if db._engine is not None:
        await db._engine.dispose()
    db._engine = db._session_factory = None
    config.get_settings.cache_clear()
    config.get_bot_config.cache_clear()
    clear_provider_cache()
    search.reset_sections_cache()
    service._service = None
