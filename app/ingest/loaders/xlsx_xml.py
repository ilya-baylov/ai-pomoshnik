"""Read XLSX cells by coordinate, including files with invalid border styles."""
from __future__ import annotations

from pathlib import Path
import posixpath
import re
import zipfile
import xml.etree.ElementTree as ET

NS = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


def _date_text(value, number_format: str) -> str:
    """Preserve an explicit day/month display, including Excel-coerced shifts.

    A cell displayed as 5/2 must not become an invented full calendar date.
    Other date/time formats retain the existing unambiguous ISO representation.
    """
    fmt = number_format.split(';', 1)[0].strip().lower()
    match = re.fullmatch(r'(d{1,2}|m{1,2})/(d{1,2}|m{1,2})', fmt)
    if match and match[1][0] != match[2][0] and hasattr(value, 'day'):
        parts = []
        for token in match.groups():
            number = value.day if token[0] == 'd' else value.month
            parts.append(str(number).zfill(len(token)))
        return '/'.join(parts)
    return value.isoformat(sep=" ") if hasattr(value, "hour") and hasattr(value, "date") else value.isoformat()


def _position(address: str) -> tuple[int, int]:
    from openpyxl.utils.cell import coordinate_from_string, column_index_from_string
    col, row = coordinate_from_string(address)
    return row, column_index_from_string(col)


def read_xlsx_rows(path: Path) -> tuple[list[dict], list[str]]:
    """Sparse values, original row numbers, and warnings; never calculate formulas."""
    warnings: list[str] = []
    sheets: list[dict] = []
    with zipfile.ZipFile(path) as archive:
        strings = []
        if "xl/sharedStrings.xml" in archive.namelist():
            strings = ["".join(n.text or "" for n in si.findall(".//s:t", NS))
                       for si in ET.fromstring(archive.read("xl/sharedStrings.xml")).findall("s:si", NS)]
        rel_ns = {"r": "http://schemas.openxmlformats.org/package/2006/relationships"}
        relationships = {n.attrib["Id"]: n.attrib["Target"] for n in
                         ET.fromstring(archive.read("xl/_rels/workbook.xml.rels")).findall("r:Relationship", rel_ns)}
        book = ET.fromstring(archive.read("xl/workbook.xml"))
        from openpyxl.styles.numbers import BUILTIN_FORMATS, is_date_format
        from openpyxl.utils.datetime import from_excel, MAC_EPOCH, WINDOWS_EPOCH
        formats = {}
        style_ids = []
        if "xl/styles.xml" in archive.namelist():
            styles = ET.fromstring(archive.read("xl/styles.xml"))
            formats = {int(n.attrib["numFmtId"]): n.attrib["formatCode"] for n in styles.findall("s:numFmts/s:numFmt", NS)}
            style_ids = [int(n.attrib.get("numFmtId", 0)) for n in styles.findall("s:cellXfs/s:xf", NS)]
        properties = book.find("s:workbookPr", NS)
        epoch = MAC_EPOCH if properties is not None and properties.attrib.get("date1904") in ("1", "true") else WINDOWS_EPOCH
        for sheet in book.findall("s:sheets/s:sheet", NS):
            name = sheet.attrib["name"]
            if sheet.attrib.get("state", "visible") != "visible":
                warnings.append(f"{path.name}: скрытый лист {name} пропущен")
                continue
            rid = sheet.attrib["{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"]
            target = relationships[rid]
            target = posixpath.normpath(target.lstrip("/") if target.startswith("/") else "xl/" + target)
            if not target.startswith("xl/worksheets/"):
                continue
            xml = ET.fromstring(archive.read(target))
            values: dict[tuple[int, int], str] = {}
            for cell in xml.findall("s:sheetData/s:row/s:c", NS):
                address = cell.attrib["r"]
                position = _position(address)
                value_node = cell.find("s:v", NS)
                value = value_node.text or "" if value_node is not None else ""
                kind = cell.attrib.get("t")
                style = int(cell.attrib.get("s", 0))
                fmt_id = style_ids[style] if style < len(style_ids) else 0
                fmt = formats.get(fmt_id, BUILTIN_FORMATS.get(fmt_id, ""))
                if kind == "s":
                    value = strings[int(value)] if value else ""
                elif kind == "inlineStr":
                    value = "".join(n.text or "" for n in cell.findall(".//s:t", NS))
                elif kind == "b":
                    value = "да" if value == "1" else "нет"
                elif value and kind == "d" and re.fullmatch(r'(?:d{1,2}/m{1,2}|m{1,2}/d{1,2})', fmt.lower().strip()):
                    from datetime import datetime
                    try:
                        value = _date_text(datetime.fromisoformat(value), fmt)
                    except ValueError:
                        warnings.append(f"{path.name}/{name}!{address}: некорректная дата")
                elif kind == "e":
                    warnings.append(f"{path.name}/{name}!{address}: ошибка формулы {value}, ячейка пропущена")
                    value = ""
                elif value and kind not in ("str", "d"):
                    try:
                        if is_date_format(fmt):
                            converted = from_excel(float(value), epoch)
                            value = _date_text(converted, fmt)
                        elif "%" in fmt:
                            value = f"{float(value)*100:g}%"
                    except (ValueError, OverflowError):
                        warnings.append(f"{path.name}/{name}!{address}: некорректный формат значения")
                if cell.find("s:f", NS) is not None and not value:
                    warnings.append(f"{path.name}/{name}!{address}: формула без сохранённого результата")
                if value.strip():
                    values[position] = value.strip()
            # Only explicit vertical merges inherit a value. Ordinary blank cells stay blank.
            for merged in xml.findall("s:mergeCells/s:mergeCell", NS):
                first, last = merged.attrib["ref"].split(":") if ":" in merged.attrib["ref"] else (merged.attrib["ref"], merged.attrib["ref"])
                r1, c1 = _position(first)
                r2, c2 = _position(last)
                if r2-r1 > 10000:
                    raise ValueError("Объединённая область XLSX слишком велика")
                anchor = values.get((r1,c1))
                if anchor and r2 > r1:
                    for row in range(r1+1,r2+1):
                        values.setdefault((row,c1),anchor)
            row_numbers = sorted({r for r,c in values})
            width = max((c for r,c in values), default=0)
            if width > 256:
                raise ValueError("Таблица XLSX слишком широка для индексации")
            rows = [(r,[values.get((r,c),"") for c in range(1,width+1)]) for r in row_numbers]
            sheets.append({"name":name,"rows":rows})
    return sheets, warnings
