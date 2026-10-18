"""Интерфейс провайдера эмбеддингов.

Асимметрия «запрос — документ» инкапсулирована здесь: наружу торчат только
embed_documents и embed_query, поэтому пометить запрос как документ (частая
ошибка, роняющая качество поиска) технически невозможно.
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod

import numpy as np

from app.core.profiles import EmbeddingProfile


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def pack(vector: np.ndarray) -> bytes:
    """Вектор в BLOB для хранения в SQLite."""
    return np.asarray(vector, dtype=np.float32).tobytes()


def unpack(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32)


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    """Нормализация по L2.

    Обязательна для Gemini при размерности не 3072: усечённые MRL-векторы
    приходят ненормализованными, и косинусная близость на них считается
    неверно.
    """
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.ndim == 1:
        norm = np.linalg.norm(matrix)
        return matrix / norm if norm > 0 else matrix
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


class EmbeddingProvider(ABC):
    """Провайдер эмбеддингов. Реализации: Gemini API и локальная модель."""

    def __init__(self, profile: EmbeddingProfile):
        self.profile = profile

    @property
    def name(self) -> str:
        return self.profile.name

    @property
    def dim(self) -> int:
        return self.profile.dim

    @abstractmethod
    async def embed_documents(self, texts: list[str]) -> np.ndarray:
        """Векторы фрагментов базы. Возвращает матрицу (n, dim)."""

    @abstractmethod
    async def embed_query(self, text: str) -> np.ndarray:
        """Вектор поискового запроса. Возвращает (dim,)."""

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """Оценка числа токенов — нужна при нарезке."""

    async def warmup(self) -> None:
        """Прогрев: скачать модель или проверить доступность API."""
        return None
