"""Загрузчики markdown и обычного текста."""

from __future__ import annotations

import re
from pathlib import Path

import yaml

from app.ingest.loaders.base import Block, BlockKind, LoadedDoc, Loader

FRONT_MATTER = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)
HEADING = re.compile(r"^(#{1,6})\s+(.*)$")
# Строка таблицы markdown: | a | b |
TABLE_ROW = re.compile(r"^\s*\|.*\|\s*$")
TABLE_SEPARATOR = re.compile(r"^\s*\|[\s:|-]+\|\s*$")


class MarkdownLoader(Loader):
    extensions = (".md", ".markdown")

    def load(self, path: Path) -> LoadedDoc:
        raw = path.read_text(encoding="utf-8", errors="replace")
        meta: dict = {}

        match = FRONT_MATTER.match(raw)
        if match:
            try:
                meta = yaml.safe_load(match.group(1)) or {}
            except yaml.YAMLError:
                meta = {}
            raw = raw[match.end() :]

        blocks: list[Block] = []
        table_header: list[str] | None = None
        buffer: list[str] = []

        def flush() -> None:
            if buffer:
                text = "\n".join(buffer).strip()
                if text:
                    blocks.append(Block(text=text, kind=BlockKind.PARAGRAPH))
                buffer.clear()

        for line in raw.splitlines():
            heading = HEADING.match(line)
            if heading:
                flush()
                table_header = None
                blocks.append(
                    Block(
                        text=heading.group(2).strip(),
                        kind=BlockKind.HEADING,
                        level=len(heading.group(1)),
                    )
                )
                continue

            if TABLE_ROW.match(line):
                if TABLE_SEPARATOR.match(line):
                    continue
                cells = [c.strip() for c in line.strip().strip("|").split("|")]
                if table_header is None:
                    # Первая строка таблицы — заголовок колонок
                    flush()
                    table_header = cells
                    continue
                # Подставляем названия колонок: без них цифры в отрыве
                # от заголовка не находятся поиском
                pairs = [
                    f"{head}: {value}"
                    for head, value in zip(table_header, cells)
                    if value
                ]
                if pairs:
                    blocks.append(
                        Block(text="; ".join(pairs), kind=BlockKind.TABLE_ROW)
                    )
                continue

            table_header = None
            if not line.strip():
                flush()
            else:
                buffer.append(line)

        flush()
        title = str(meta.get("title") or _first_heading(blocks) or path.stem)
        return LoadedDoc(title=title, blocks=blocks, meta=meta)


class PlainTextLoader(Loader):
    extensions = (".txt",)

    def load(self, path: Path) -> LoadedDoc:
        raw = path.read_text(encoding="utf-8", errors="replace")
        blocks = [
            Block(text=part.strip(), kind=BlockKind.PARAGRAPH)
            for part in re.split(r"\n\s*\n", raw)
            if part.strip()
        ]
        return LoadedDoc(title=path.stem, blocks=blocks)


def _first_heading(blocks: list[Block]) -> str | None:
    for block in blocks:
        if block.kind is BlockKind.HEADING:
            return block.text
    return None
