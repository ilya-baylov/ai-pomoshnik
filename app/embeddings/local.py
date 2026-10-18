"""Локальные эмбеддинги через fastembed (ONNX, без PyTorch).

База не покидает машину. Модель ~2.24 ГБ скачивается один раз.
"""

from __future__ import annotations

import asyncio
import logging
import os

import numpy as np

from app.core.profiles import EmbeddingProfile
from app.embeddings.base import EmbeddingProvider, l2_normalize

log = logging.getLogger(__name__)

# Rust-ускоритель выгрузки HF (hf-xet) падает по памяти на больших файлах;
# обычная HTTP-загрузка медленнее, но надёжна
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")


class LocalEmbeddings(EmbeddingProvider):
    def __init__(self, profile: EmbeddingProfile, cache_dir: str | None = None):
        super().__init__(profile)
        self._cache_dir = cache_dir
        self._model = None
        self._tokenizer = None

    # Модели, которых нет в списке fastembed — регистрируются как кастомные
    _CUSTOM_MODELS = {"intfloat/multilingual-e5-small"}
    _registered_custom: set[str] = set()

    def _ensure_model(self):
        """Ленивая загрузка: не тянуть модель при импорте модуля."""
        if self._model is not None:
            return self._model
        try:
            from fastembed import TextEmbedding
        except ImportError as exc:
            raise RuntimeError(
                "Для EMBEDDING_PROVIDER=local установите зависимости:\n"
                "    pip install -r requirements-local.txt"
            ) from exc

        if self.profile.name in self._CUSTOM_MODELS and (
            self.profile.name not in self._registered_custom
        ):
            self._register_custom_model()
            self._registered_custom.add(self.profile.name)

        log.info("Загружаю модель %s", self.profile.name)
        try:
            self._model = TextEmbedding(
                model_name=self.profile.name, cache_dir=self._cache_dir
            )
        except (MemoryError, OSError) as exc:
            # e5-large требует ~3 ГБ RAM; на машинах с 4 ГБ физически не влезает
            log.error("Не удалось загрузить %s: %s", self.profile.name, exc)
            return self._fallback_small_model()
        return self._model

    def _register_custom_model(self) -> None:
        """Регистрация моделей вне встроенного списка fastembed.

        e5-small: 0.13 ГБ вместо 2.24, без нормализации выходов справляется
        сам (normalization=True), а наша l2-нормализация поверх безопасна.
        """
        import fastembed
        from fastembed.common.model_description import ModelSource, PoolingType

        fastembed.TextEmbedding.add_custom_model(
            model=self.profile.name,
            pooling=PoolingType.MEAN,
            normalization=True,
            sources=ModelSource(hf=self.profile.name),
            dim=self.profile.dim,
            model_file="onnx/model.onnx",
        )

    def _fallback_small_model(self):
        """Резерв для слабых машин: e5-small (0.13 ГБ вместо 2.24)."""
        name = "intfloat/multilingual-e5-small"
        if name not in self._registered_custom:
            self._register_custom_model()
            self._registered_custom.add(name)
        log.warning(
            "Использую %s вместо %s: слабая машина или ошибка загрузки large",
            name,
            self.profile.name,
        )
        from fastembed import TextEmbedding

        self._model = TextEmbedding(model_name=name, cache_dir=self._cache_dir)
        return self._model

    async def embed_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        prefixed = [self.profile.passage_prefix + t for t in texts]
        return await asyncio.to_thread(self._embed_sync, prefixed)

    async def embed_query(self, text: str) -> np.ndarray:
        prefixed = self.profile.query_prefix + text
        matrix = await asyncio.to_thread(self._embed_sync, [prefixed])
        return matrix[0]

    def _embed_sync(self, texts: list[str]) -> np.ndarray:
        model = self._ensure_model()
        # batch_size=8: дефолтная пачка 256 с паддингом до самой длинной
        # последовательности съедает гигабайты — на машине с 4 ГБ RAM
        # это гарантированный MemoryError. Пачка поменьше почти не
        # влияет на скорость на CPU.
        raw = np.array(list(model.embed(texts, batch_size=8)), dtype=np.float32)
        # fastembed 0.8 (mean pooling) отдаёт НЕнормализованные векторы:
        # нормы доходят до ~30, а наш поиск считает косинус через
        # скалярное произведение и требует норму 1
        return l2_normalize(raw)

    def count_tokens(self, text: str) -> int:
        """Подсчёт настоящим токенизатором модели.

        Лимит 512 токенов у E5 жёсткий, поэтому здесь нужна точность,
        а не оценка: превышение молча обрезает хвост фрагмента.
        """
        tokenizer = self._ensure_tokenizer()
        if tokenizer is None:
            return max(1, int(len(text) / 2.2))
        return len(tokenizer.encode(text).ids)

    def _ensure_tokenizer(self):
        if self._tokenizer is not None:
            return self._tokenizer
        try:
            model = self._ensure_model()
            # fastembed держит токенизатор внутри модели
            self._tokenizer = getattr(model.model, "tokenizer", None)
        except Exception:  # noqa: BLE001 — при сбое переходим на оценку
            log.warning("Токенизатор недоступен, использую оценку по символам")
            self._tokenizer = None
        return self._tokenizer

    async def warmup(self) -> None:
        await self.embed_query("проверка")
        log.info("Локальная модель готова: %s, dim=%s", self.profile.name, self.dim)
