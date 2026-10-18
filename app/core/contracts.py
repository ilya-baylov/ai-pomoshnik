"""Контракты между слоями.

Ядро принимает IncomingMessage и возвращает OutgoingMessage, ничего не зная
о транспорте. Всё, что специфично для мессенджера, живёт в адаптере канала.
Благодаря этому подключение нового мессенджера не задевает поиск и генерацию.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Role(str, Enum):
    ADMIN = "admin"
    STAFF = "staff"
    PENDING = "pending"


@dataclass
class Attachment:
    file_name: str
    file_id: str
    size_bytes: int
    mime_type: str | None = None


@dataclass
class IncomingMessage:
    channel: str
    external_user_id: str
    chat_id: str
    text: str
    attachments: list[Attachment] = field(default_factory=list)


@dataclass
class Source:
    """Источник ответа — то, что сотрудник может проверить сам."""

    index: int
    title: str
    breadcrumbs: str | None = None
    page: int | None = None
    is_stale: bool = False

    def label(self) -> str:
        # Крошки часто уже начинаются с названия документа — не дублируем
        text = self.title
        if self.breadcrumbs and not self.breadcrumbs.startswith(self.title):
            text += f" → {self.breadcrumbs}"
        if self.page:
            text += f", стр. {self.page}"
        return f"{text} (устарел)" if self.is_stale else text


@dataclass
class OutgoingMessage:
    text: str
    sources: list[Source] = field(default_factory=list)
    is_fallback: bool = False
    # id записи в queries — нужен, чтобы привязать оценку к ответу
    query_id: int | None = None


@dataclass
class RetrievedChunk:
    """Фрагмент, найденный поиском, с данными для промпта и ссылки."""

    chunk_id: int
    text: str
    title: str
    breadcrumbs: str | None
    page: int | None
    score: float
    section: str | None = None
    is_stale: bool = False
    valid_until: str | None = None
    is_current: bool = False
    promotion_rows: list[str] | None = None
    direct_answer: str | None = None
