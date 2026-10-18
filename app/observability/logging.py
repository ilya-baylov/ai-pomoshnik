"""Настройка логирования: файлы с ротацией + консоль.

Продакшен-требование: терминал закрывается, а история инцидентов должна
оставаться. Каждый процесс (бот, индексация) пишет в свой файл, чтобы
гонки за запись не перемешивали строки.
"""

from __future__ import annotations

import logging
import logging.handlers
import sys
from pathlib import Path

from app.core.config import ROOT

_LOG_DIR = ROOT / "data" / "logs"
_FORMAT = "%(asctime)s %(levelname)-8s %(name)s: %(message)s"
_MAX_BYTES = 5 * 1024 * 1024
_BACKUPS = 5

_configured: set[str] = set()


def setup_logging(name: str, level: str = "INFO") -> None:
    """Логгер `<name>` пишет в data/logs/<name>.log и в консоль.

    Повторный вызов с тем же именем ничего не меняет — важно для тестов
    и для повторного запуска индексации в одном процессе.
    """
    if name in _configured:
        return

    _LOG_DIR.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    root.setLevel(level.upper())

    file_handler = logging.handlers.RotatingFileHandler(
        _LOG_DIR / f"{name}.log",
        maxBytes=_MAX_BYTES,
        backupCount=_BACKUPS,
        encoding="utf-8",
    )
    file_handler.setFormatter(logging.Formatter(_FORMAT))
    root.addHandler(file_handler)

    if not any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler) for h in root.handlers):
        console = logging.StreamHandler(sys.stderr)
        console.setFormatter(logging.Formatter(_FORMAT))
        root.addHandler(console)

    _configured.add(name)


def log_dir() -> Path:
    return _LOG_DIR
