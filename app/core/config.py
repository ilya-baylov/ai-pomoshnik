"""Загрузка настроек: секреты из .env, поведение из config/bot.yaml."""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.core.profiles import EmbeddingProfile, get_profile

ROOT = Path(__file__).resolve().parents[2]


def _parse_ids(value: str) -> list[int]:
    """Разбор `123,456` в список чисел; пустое и мусорное — в пустой список."""
    ids: list[int] = []
    for chunk in (value or "").replace(" ", "").split(","):
        if not chunk:
            continue
        try:
            ids.append(int(chunk))
        except ValueError:
            continue
    return ids


class Settings(BaseSettings):
    """Секреты и выбор провайдеров. Источник — .env."""

    model_config = SettingsConfigDict(
        env_file=ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    telegram_bot_token: str = ""

    llm_api_key: str = ""
    openai_api_key: str = ""
    llm_base_url: str = "https://api.openai.com/v1"
    llm_model: str = "gpt-6-luna"
    llm_reasoning_effort: str = "none"

    # Резервная модель: срабатывает при отказе или лимите основной.
    # Включается явно; ключ Google не передаётся другим провайдерам.
    llm_fallback_enabled: bool = False
    llm_fallback_base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai/"
    llm_fallback_model: str = "gemini-3.6-flash"
    llm_fallback_api_key: str = ""

    embedding_provider: str = "openai"
    embedding_model: str = "text-embedding-3-small"
    embedding_dim: int = 1536
    embedding_api_key: str = ""
    # Несколько ключей через запятую для преодоления лимитов: квота
    # у Google на проект, поэтому ключи из разных проектов складываются
    embedding_api_keys_raw: str = Field(default="", validation_alias="EMBEDDING_API_KEYS")
    # Сервисный аккаунт Google (JSON): квота проекта Vertex AI, приоритетнее ключей
    embedding_service_account: str = ""
    # Для EMBEDDING_PROVIDER=openai: адрес /v1 и модель (EMBEDDING_MODEL)
    embedding_base_url: str = "https://api.openai.com/v1"

    # ID через запятую. Строкой, а не list[int]: pydantic-settings пытается
    # распарсить сложные поля как JSON и падает на пустом значении ещё до
    # валидаторов — при ADMIN_IDS= из .env бот вообще не стартовал
    admin_ids_raw: str = Field(default="", validation_alias="ADMIN_IDS")
    staff_ids_raw: str = Field(default="", validation_alias="STAFF_IDS")

    database_url: str = "sqlite+aiosqlite:///data/bot.db"
    knowledge_dir: str = "knowledge"
    log_level: str = "INFO"

    # Распознавание речи через Groq Whisper
    groq_api_key: str = ""
    groq_whisper_model: str = "whisper-large-v3"
    stt_provider: str = "openai"
    stt_model: str = "gpt-4o-mini-transcribe"
    stt_api_key: str = ""

    @property
    def admin_ids(self) -> list[int]:
        return _parse_ids(self.admin_ids_raw)

    @property
    def staff_ids(self) -> list[int]:
        return _parse_ids(self.staff_ids_raw)

    @property
    def embedding_keys(self) -> list[str]:
        """Все ключи для эмбеддингов: EMBEDDING_API_KEYS + одиночный.

        Порядок сохраняется, дубликаты убираются. Квота Google на проект,
        поэтому ключи из разных проектов складываются в общий бюджет.
        Когда явных ключей нет, а LLM — тоже Gemini, используется LLM_API_KEY:
        дефолтная связка живёт на одном ключе. Если LLM на OpenRouter или
        DeepSeek, подставлять их ключ нельзя — эмбеддинги падали бы с 401.
        """
        keys: list[str] = []
        for chunk in self.embedding_api_keys_raw.replace(" ", "").split(","):
            if chunk and chunk not in keys:
                keys.append(chunk)
        if self.embedding_api_key and self.embedding_api_key not in keys:
            keys.append(self.embedding_api_key)
        if not keys and self.llm_api_key and (
            urlparse(self.llm_base_url).hostname == "generativelanguage.googleapis.com"
        ):
            keys.append(self.llm_api_key)
        return keys

    @property
    def service_account_path(self) -> Path | None:
        """Путь к JSON сервисного аккаунта, если настроен."""
        if not self.embedding_service_account:
            return None
        path = Path(self.embedding_service_account)
        return path if path.is_absolute() else ROOT / path

    @property
    def embeddings_key(self) -> str:
        """Первый ключ эмбеддингов (используется резервной LLM)."""
        keys = self.embedding_keys
        return keys[0] if keys else ""

    @property
    def knowledge_path(self) -> Path:
        path = Path(self.knowledge_dir)
        return path if path.is_absolute() else ROOT / path

    @property
    def db_path(self) -> Path:
        """Путь к файлу SQLite, выведенный из database_url."""
        from sqlalchemy.engine import make_url
        raw = make_url(self.database_url).database or ":memory:"
        path = Path(raw)
        return path if path.is_absolute() else ROOT / path

    def profile(self) -> EmbeddingProfile:
        if self.embedding_provider == "local":
            return get_profile(self.embedding_model)
        return get_profile(self.embedding_model, self.embedding_dim)


class BotConfig:
    """Поведение бота из config/bot.yaml.

    Доступ через точечный путь: cfg.get("retrieval.min_score", 0.4).
    Так добавление параметра в YAML не требует правки этого класса.
    """

    def __init__(self, data: dict[str, Any], synonyms: dict[str, list[str]]):
        self._data = data
        self.synonyms = synonyms

    def get(self, path: str, default: Any = None) -> Any:
        node: Any = self._data
        for key in path.split("."):
            if not isinstance(node, dict) or key not in node:
                return default
            node = node[key]
        return node if node is not None else default

    def message(self, key: str) -> str:
        """Текст из блока messages с подстановкой имени и компании."""
        template = str(self.get(f"messages.{key}", "")).strip()
        return template.format(
            name=self.get("persona.name", "Ассистент"),
            company=self.get("persona.company", "компании"),
        )

    def persona(self) -> dict[str, str]:
        return {
            "name": str(self.get("persona.name", "Ассистент")),
            "company": str(self.get("persona.company", "компании")),
            "style": str(self.get("persona.style", "")).strip(),
        }


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


@functools.lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


@functools.lru_cache(maxsize=1)
def get_bot_config() -> BotConfig:
    return BotConfig(
        data=_read_yaml(ROOT / "config" / "bot.yaml"),
        synonyms=_read_yaml(ROOT / "config" / "synonyms.yaml"),
    )
