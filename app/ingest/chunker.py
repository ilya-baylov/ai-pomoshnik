"""Нарезка документа на фрагменты.

Режется по структуре, а не по длине: сначала границы задают заголовки, внутри
разделов текст добивается до целевого размера с перекрытием. Каждый фрагмент
несёт «хлебные крошки» — путь заголовков, — чтобы модель понимала контекст
текста, вырванного из середины документа.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

from app.ingest.loaders.base import Block, BlockKind

SENTENCE_END = re.compile(r"(?<=[.!?;])\s+")
# Буквы любого алфавита: критерий «во фрагменте есть человеческий текст»
LETTERS = re.compile(r"[^\W\d_]", re.UNICODE)
# Служебная пунктуация таблиц: двоеточия, точки с запятой, тире, скобки
NOISE = re.compile(r"[:;—\-–()\[\]|•·]")
# Пустая метка перед двоеточием: «: ПН» вместо «Имя: ПН». Так распадаются
# строки таблиц, где имена были в объединённых ячейках
EMPTY_LABEL = re.compile(r"(?:^|[;|])\s*:")


def is_garbage(text: str) -> bool:
    """Мусорный фрагмент: развалившиеся таблицы и чистый шум.

    Отсекается: «: ПН; : ВТ; ВИЗАЖ/РЕСНИЦЫ: СР» (строки расписаний, где
    имена потерялись при нарезке), «— — —», «::: ;;;». НЕ отсекаются
    короткие, но содержательные строки: «Цена: 900», телефоны,
    «Депиляция подмышек».
    """
    stripped = text.strip()
    if not stripped:
        return True

    # Много цифр — контакт или сумма: это данные, даже если букв мало
    if sum(ch.isdigit() for ch in stripped) >= 6:
        return False

    letters = len(LETTERS.findall(stripped))
    # Пустая метка перед двоеточием («: ВТ») — таблица, в которой имена
    # были в объединённых ячейках и потерялись при нарезке. В коротком
    # фрагменте достаточно одной такой метки; в длинном — двух.
    threshold = 1 if letters < 30 else 2
    if len(EMPTY_LABEL.findall(stripped)) >= threshold:
        return True
    # Совсем без букв — чистый шум («— — —», «::: ;;;»)
    if letters == 0:
        return True
    # Пунктуации больше половины длины — данные развалились
    if len(NOISE.findall(stripped)) > len(stripped) / 2:
        return True
    return False

@dataclass
class ChunkDraft:
    text: str
    breadcrumbs: str
    page: int | None
    token_count: int


class Chunker:
    def __init__(
        self,
        count_tokens: Callable[[str], int],
        target_tokens: int,
        overlap_tokens: int,
        max_tokens: int,
        min_tokens: int = 40,
    ):
        self._count = count_tokens
        self.target = target_tokens
        self.overlap = overlap_tokens
        self.max_tokens = max_tokens
        self.min_tokens = min_tokens

    def split(self, blocks: list[Block]) -> list[ChunkDraft]:
        chunks: list[ChunkDraft] = []
        headings: dict[int, str] = {}
        buffer: list[Block] = []

        def breadcrumbs() -> str:
            return " → ".join(headings[lvl] for lvl in sorted(headings))

        def flush() -> None:
            if not buffer:
                return
            chunks.extend(self._pack(buffer, breadcrumbs()))
            buffer.clear()

        for block in blocks:
            if block.kind is BlockKind.HEADING:
                flush()
                # Заголовок закрывает все вложенные уровни
                for level in [lvl for lvl in headings if lvl >= block.level]:
                    del headings[level]
                headings[block.level] = block.text
                continue

            # Табличные строки и пары «вопрос-ответ» атомарны
            if block.kind in (BlockKind.TABLE_ROW, BlockKind.QA):
                flush()
                chunks.extend(self._atomic(block, breadcrumbs()))
                continue

            buffer.append(block)

        flush()
        # Мусорные фрагменты не индексируем: они мешают поиску
        return [c for c in chunks if c.text.strip() and not is_garbage(c.text)]

    def _atomic(self, block: Block, crumbs: str) -> list[ChunkDraft]:
        """Блок, который нельзя склеивать с соседями."""
        tokens = self._count(block.text)
        if tokens <= self.max_tokens:
            return [
                ChunkDraft(
                    text=block.text,
                    breadcrumbs=crumbs,
                    page=block.page,
                    token_count=tokens,
                )
            ]
        # Аномально длинная строка таблицы — режем по предложениям
        return self._pack([block], crumbs)

    def _pack(self, blocks: list[Block], crumbs: str) -> list[ChunkDraft]:
        """Собрать абзацы в фрагменты целевого размера."""
        pieces: list[tuple[str, int | None]] = []
        for block in blocks:
            for sentence in self._to_pieces(block.text):
                pieces.append((sentence, block.page))

        chunks: list[ChunkDraft] = []
        current: list[str] = []
        current_pages: list[int | None] = []
        current_tokens = 0
        page: int | None = None

        # Целевой размер — ориентир, лимит модели — жёсткая граница
        ceiling = min(self.target, self.max_tokens)
        for text, piece_page in pieces:
            tokens = self._count(text)
            if current and current_tokens + tokens > ceiling:
                chunks.append(self._make(current, crumbs, page, current_tokens))
                tail = self._tail(current)
                current_pages = current_pages[-len(tail):] if tail else []
                current = tail[:]
                current_tokens = sum(self._count(t) for t in current)
                page = current_pages[0] if current_pages else piece_page
            if not current:
                page = piece_page
            current.append(text)
            current_pages.append(piece_page)
            current_tokens += tokens

        if current:
            chunks.append(self._make(current, crumbs, page, current_tokens))

        return self._merge_tiny(chunks, crumbs)

    def _to_pieces(self, text: str) -> list[str]:
        """Абзац целиком, либо по предложениям, если он больше цели."""
        if self._count(text) <= self.target:
            return [text]
        budget = self._piece_budget()
        sentences = [s.strip() for s in SENTENCE_END.split(text) if s.strip()]
        result: list[str] = []
        for sentence in sentences:
            if self._count(sentence) <= budget:
                result.append(sentence)
                continue
            # Предложение не влезает в бюджет куска — рубим по словам
            result.extend(self._by_words(sentence))
        return result or [text]

    def _piece_budget(self) -> int:
        """Максимальный размер одного куска.

        Кусок обязан оставлять место под перекрытие: иначе хвост предыдущего
        фрагмента, сложенный со следующим куском, пробивает лимит модели.
        """
        return max(1, min(self.target, self.max_tokens - self.overlap))

    def _by_words(self, text: str) -> list[str]:
        """Резать по словам, не превышая бюджет одного куска."""
        budget = self._piece_budget()
        words = text.split()
        parts: list[str] = []
        current: list[str] = []
        for word in words:
            candidate = current + [word]
            if current and self._count(" ".join(candidate)) > budget:
                parts.append(" ".join(current))
                current = [word]
            else:
                current = candidate
        if current:
            parts.append(" ".join(current))
        return parts

    def _tail(self, texts: list[str]) -> list[str]:
        """Хвост предыдущего фрагмента для перекрытия.

        Кусок, который сам не влезает в бюджет перекрытия, не берётся вовсе:
        иначе он складывается со следующим и превышает лимит модели.
        """
        if self.overlap <= 0:
            return []
        tail: list[str] = []
        total = 0
        for text in reversed(texts):
            tokens = self._count(text)
            if total + tokens > self.overlap:
                break
            tail.insert(0, text)
            total += tokens
        return tail

    def _make(
        self, texts: list[str], crumbs: str, page: int | None, tokens: int
    ) -> ChunkDraft:
        return ChunkDraft(
            text="\n".join(texts).strip(),
            breadcrumbs=crumbs,
            page=page,
            token_count=tokens,
        )

    def _merge_tiny(self, chunks: list[ChunkDraft], crumbs: str) -> list[ChunkDraft]:
        """Приклеить огрызки к соседям: отдельный фрагмент из двух слов бесполезен."""
        if len(chunks) < 2:
            return chunks
        result: list[ChunkDraft] = []
        for chunk in chunks:
            if (
                result
                and chunk.token_count < self.min_tokens
                and result[-1].token_count + chunk.token_count <= self.max_tokens
            ):
                previous = result[-1]
                result[-1] = ChunkDraft(
                    text=f"{previous.text}\n{chunk.text}",
                    breadcrumbs=crumbs,
                    page=previous.page,
                    token_count=previous.token_count + chunk.token_count,
                )
            else:
                result.append(chunk)
        return result
