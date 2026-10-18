"""OCR сканов через Gemini Vision.

Сканы (PDF без текстового слоя) не читаются pypdf. Вместо тяжёлого
tesseract или torch-моделей используем то, что уже подключено: Gemini
многомодальный и хорошо читает русский текст с картинок.

Результат кладётся рядом с PDF отдельным .ocr.md — его индексирует
обычный markdown-загрузчик. Так распознавание делается один раз, файл
можно прочитать глазами и поправить руками.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

from app.core.config import get_settings

log = logging.getLogger(__name__)

PROMPT = (
    "Извлеки весь полезный текст с изображения. Это слайд презентации "
    "или скан документа.\n"
    "ВАЖНО:\n"
    "- Декоративный и фоновый текст игнорируй: логотипы, водяные знаки, "
    "текст на картинках-украшениях. Если буквы стоят вертикально в "
    "столбик — это логотип, напиши его название одним словом или "
    "пропусти.\n"
    "- Читай текст блоками как он расположен: заголовок, абзацы, "
    "списки. Не разбивай слова на отдельные буквы.\n"
    "- Таблицы сохраняй строками «поле: значение».\n"
    "Не добавляй комментариев и не пересказывай — только текст со слайда."
)

# Модель зрения: flash достаточно и дешевле pro
VISION_MODEL = "gemini-3.6-flash"
SUFFIX = ".ocr.md"


def _vision_client():
    """Клиент Vertex AI под сервисным аккаунтом (у него есть Vision)."""
    from google import genai
    from google.genai import types
    from google.oauth2 import service_account

    settings = get_settings()
    path = settings.service_account_path
    if path is None:
        raise RuntimeError(
            "OCR требует сервисный аккаунт: заполните EMBEDDING_SERVICE_ACCOUNT"
        )
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    creds = service_account.Credentials.from_service_account_file(
        str(path), scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )
    return genai.Client(
        vertexai=True,
        project=raw.get("project_id", ""),
        credentials=creds,
        http_options=types.HttpOptions(timeout=180 * 1000),
    )


def find_scanned_pdfs(root: Path) -> list[Path]:
    """PDF без текстового слоя, для которых ещё нет распознанного файла."""
    from pypdf import PdfReader

    scans: list[Path] = []
    for path in sorted(root.rglob("*.pdf")):
        sidecar = path.with_suffix(path.suffix + SUFFIX)
        if sidecar.exists():
            continue
        try:
            reader = PdfReader(str(path))
            has_text = any(
                (page.extract_text() or "").strip() for page in reader.pages[:5]
            )
            if not has_text:
                scans.append(path)
        except Exception as exc:  # noqa: BLE001
            log.warning("Не удалось проверить %s: %s", path.name, exc)
    return scans


async def ocr_pdf(path: Path, client=None) -> Path | None:
    """Распознать PDF постранично и записать <имя>.pdf.ocr.md."""
    from google.genai import types
    from pypdf import PdfReader

    client = client or _vision_client()
    reader = PdfReader(str(path))
    pages_text: list[str] = []

    for number, page in enumerate(reader.pages, start=1):
        images = list(page.images)
        if not images:
            continue
        # Берём самую крупную картинку страницы: это и есть скан
        image = max(images, key=lambda item: len(item.data))
        mime = "image/png" if image.name.lower().endswith(".png") else "image/jpeg"
        try:
            response = await asyncio.to_thread(
                client.models.generate_content,
                model=VISION_MODEL,
                contents=[
                    types.Part.from_bytes(data=image.data, mime_type=mime),
                    PROMPT,
                ],
            )
            text = (response.text or "").strip()
        except Exception as exc:  # noqa: BLE001
            log.warning("Страница %s из %s не распознана: %s", number, path.name, exc)
            continue
        if text:
            pages_text.append(f"## Страница {number}\n\n{text}")
        log.info("%s: страница %s из %s", path.name, number, len(reader.pages))

    if not pages_text:
        log.warning("%s: не удалось распознать ни одной страницы", path.name)
        return None

    sidecar = path.with_suffix(path.suffix + SUFFIX)
    header = "\n".join(
        [
            "---",
            f"title: {path.stem}",
            "tags: [распознано, скан]",
            "---",
            "",
            f"# {path.stem}",
            "",
            f"Текст распознан автоматически с изображений файла {path.name}.",
            "",
        ]
    )
    sidecar.write_text(header + "\n\n".join(pages_text) + "\n", encoding="utf-8")
    return sidecar


async def ocr_all(root: Path) -> list[Path]:
    """Распознать все сканы в папке. Возвращает созданные файлы."""
    scans = find_scanned_pdfs(root)
    if not scans:
        return []
    client = _vision_client()
    created: list[Path] = []
    for path in scans:
        result = await ocr_pdf(path, client)
        if result is not None:
            created.append(result)
    return created
