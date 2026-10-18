"""Метаданные документа: из пути к файлу и из front-matter."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path


@dataclass
class DocMeta:
    title: str
    section: str | None
    tags: list[str] = field(default_factory=list)
    priority: int = 0
    valid_until: date | None = None

    @property
    def is_stale(self) -> bool:
        return self.valid_until is not None and self.valid_until < date.today()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


def _parse_date(value: object) -> date | None:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value.strip())
        except ValueError:
            return None
    return None


def build_meta(
    path: Path, knowledge_root: Path, doc_title: str, front_matter: dict
) -> DocMeta:
    """Метаданные: путь задаёт основу, front-matter переопределяет."""
    relative = path.relative_to(knowledge_root)
    # Первая папка внутри knowledge/ — раздел
    section = relative.parts[0] if len(relative.parts) > 1 else None

    tags = front_matter.get("tags") or []
    if isinstance(tags, str):
        tags = [t.strip() for t in tags.split(",") if t.strip()]

    try:
        priority = int(front_matter.get("priority") or 0)
    except (TypeError, ValueError):
        priority = 0

    return DocMeta(
        title=str(front_matter.get("title") or doc_title).strip(),
        # Явное null означает общий документ, даже если он лежит в «Общее/».
        section=str(front_matter.get("section", section)) if front_matter.get("section", section) else None,
        tags=[str(t) for t in tags],
        priority=priority,
        valid_until=_parse_date(front_matter.get("valid_until")),
    )
