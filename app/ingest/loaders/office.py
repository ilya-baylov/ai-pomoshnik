"""Загрузчики docx, pdf, html."""

from __future__ import annotations

import logging
import re
from pathlib import Path

from app.ingest.loaders.base import Block, BlockKind, LoadedDoc, Loader
from app.ingest.loaders.tabular import _row_to_text

log = logging.getLogger(__name__)

# Стиль заголовка в docx: "Heading 1" или "Заголовок 1"
HEADING_STYLE = re.compile(r"(?:heading|заголовок)\s*(\d)", re.IGNORECASE)


class DocxLoader(Loader):
    extensions = (".docx",)

    def load(self, path: Path) -> LoadedDoc:
        try:
            return self._load_with_docx(path)
        except Exception as exc:  # noqa: BLE001 — битые docx спасаем сырым XML
            log.warning("python-docx не смог %s (%s), читаю сырой XML", path.name, exc)
            return self._load_raw_xml(path)

    def _load_with_docx(self, path: Path) -> LoadedDoc:
        import docx

        document = docx.Document(str(path))
        blocks: list[Block] = []
        from docx.text.paragraph import Paragraph
        # Порядок важен: подпись филиала перед таблицей относится именно к ней.
        for element in document.iter_inner_content():
            if isinstance(element, Paragraph):
                text = element.text.strip()
                if not text:
                    continue
                match = HEADING_STYLE.search(element.style.name or "")
                blocks.append(Block(text=text, kind=BlockKind.HEADING if match else BlockKind.PARAGRAPH,
                                    level=int(match.group(1)) if match else 0))
                continue
            rows = [[cell.text.strip() for cell in row.cells] for row in element.rows]
            if not rows:
                continue
            if len(rows) == 1:
                # Однострочная таблица без шапки: вся строка — контент
                text = "; ".join(v for v in rows[0] if v)
                if text:
                    blocks.append(Block(text=text, kind=BlockKind.TABLE_ROW))
                continue
            header = rows[0]
            for row in rows[1:]:
                text = _row_to_text(header, row)
                if text:
                    blocks.append(Block(text=text, kind=BlockKind.TABLE_ROW))

        title = _first_heading(blocks) or path.stem
        return LoadedDoc(title=title, blocks=blocks)

    def _load_raw_xml(self, path: Path) -> LoadedDoc:
        """Битые docx: читаем word/document.xml напрямую из zip.

        python-docx падает на кривых relationships (например, часть с
        именем NULL), но сам текст документа при этом лежит в целости.
        Заголовки теряются — стилей без python-docx не видно, — зато
        текст сохраняется полностью, включая таблицы: без их разбора
        строки таблицы сливаются в плоский текст, где «ИСКЛЮЧИТЬ
        ПРОЦЕДУРЫ: X» неотличимо от «делает X».
        """
        import zipfile
        import xml.etree.ElementTree as ET

        from app.ingest.loaders.tabular import _row_to_text

        ns = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
        text_tags = {f"{ns}t", f"{ns}tab"}
        blocks: list[Block] = []

        def para_text(elem) -> str:
            return "".join(
                (node.text or "").replace("\t", " ")
                for node in elem.iter()
                if node.tag in text_tags
            ).strip()

        def cell_text(cell) -> str:
            return " ".join(para_text(p) for p in cell.iter(f"{ns}p")).strip()

        def add_table(tbl) -> None:
            rows = [
                [cell_text(c) for c in tr.findall(f"{ns}tc")]
                for tr in tbl.findall(f"{ns}tr")
            ]
            rows = [r for r in rows if any(v for v in r)]
            if len(rows) < 2:
                return
            header = max(rows[:3], key=lambda r: sum(1 for v in r if v))
            idx = rows.index(header)
            for row in rows[:idx]:
                text = " ".join(v for v in row if v)
                if text:
                    blocks.append(Block(text=text, kind=BlockKind.HEADING, level=2))
            for row in rows[idx + 1 :]:
                # Строка из одной ячейки — подзаголовок раздела таблицы
                # («ДЕПИЛЯЦИЯ», «МАНИКЮР»), а не данные
                filled = [v for v in row if v]
                if len(filled) == 1:
                    blocks.append(Block(text=filled[0], kind=BlockKind.HEADING, level=2))
                    continue
                text = _row_to_text(header, row)
                if text:
                    blocks.append(Block(text=text, kind=BlockKind.TABLE_ROW))

        def walk(elem) -> None:
            for child in elem:
                if child.tag == f"{ns}tbl":
                    add_table(child)
                elif child.tag == f"{ns}p":
                    text = para_text(child)
                    if text:
                        blocks.append(Block(text=text, kind=BlockKind.PARAGRAPH))
                else:
                    walk(child)

        with zipfile.ZipFile(path) as zf:
            xml = zf.read("word/document.xml")

        root = ET.fromstring(xml)
        walk(root)

        if not blocks:
            return LoadedDoc(title=path.stem, blocks=[], warnings=[
                f"{path.name}: docx не читается и текста в document.xml нет"
            ])

        title = _first_heading(blocks) or path.stem
        return LoadedDoc(
            title=title,
            blocks=blocks,
            warnings=[f"{path.name}: файл битый, загружен в обход python-docx"],
        )


class PdfLoader(Loader):
    extensions = (".pdf",)

    def load(self, path: Path) -> LoadedDoc:
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        blocks: list[Block] = []
        warnings: list[str] = []
        empty_pages = 0

        for number, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if not text:
                empty_pages += 1
                continue
            for part in re.split(r"\n\s*\n", text):
                cleaned = " ".join(part.split())
                if cleaned:
                    blocks.append(
                        Block(text=cleaned, kind=BlockKind.PARAGRAPH, page=number)
                    )

        # Скан без текстового слоя: молча пропускать нельзя, иначе документ
        # считается проиндексированным, а искать в нём нечего
        if empty_pages:
            total = len(reader.pages)
            if not blocks:
                warnings.append(
                    f"{path.name}: текстового слоя нет ни на одной из {total} стр. "
                    "Похоже на скан — нужен OCR, документ не проиндексирован."
                )
            else:
                warnings.append(
                    f"{path.name}: {empty_pages} из {total} стр. без текста"
                )

        return LoadedDoc(title=path.stem, blocks=blocks, warnings=warnings)


class HtmlLoader(Loader):
    extensions = (".html", ".htm")

    def load(self, path: Path) -> LoadedDoc:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(
            path.read_text(encoding="utf-8", errors="replace"), "html.parser"
        )
        for tag in soup(["script", "style", "nav", "footer"]):
            tag.decompose()

        blocks: list[Block] = []
        for element in soup.find_all(
            ["h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "td"]
        ):
            text = " ".join(element.get_text(" ").split())
            if not text:
                continue
            name = element.name
            if name.startswith("h") and len(name) == 2 and name[1].isdigit():
                blocks.append(
                    Block(text=text, kind=BlockKind.HEADING, level=int(name[1]))
                )
            else:
                blocks.append(Block(text=text, kind=BlockKind.PARAGRAPH))

        page_title = soup.title.string.strip() if soup.title and soup.title.string else None
        return LoadedDoc(title=page_title or _first_heading(blocks) or path.stem, blocks=blocks)


def _first_heading(blocks: list[Block]) -> str | None:
    for block in blocks:
        if block.kind is BlockKind.HEADING:
            return block.text
    return None
