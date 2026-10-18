"""Эмбеддинги через любой OpenAI-совместимый API.

Провайдер-заплата на будущее: OpenAI, OpenRouter, локальная Ollama,
Azure — любой сервис с /v1/embeddings. Векторизация и поиск в проекте
также умеют Gemini и локальные модели; провайдер выбирается в .env.
"""

from __future__ import annotations

import logging
from collections import OrderedDict

import numpy as np

from app.core.profiles import EmbeddingProfile
from app.embeddings.base import EmbeddingProvider, l2_normalize

log = logging.getLogger(__name__)


class OpenAIEmbeddings(EmbeddingProvider):
    def __init__(
        self,
        profile: EmbeddingProfile,
        base_url: str,
        api_key: str,
    ):
        super().__init__(profile)
        if not api_key:
            raise ValueError("EMBEDDING_API_KEY пуст для провайдера openai")
        from openai import AsyncOpenAI

        # max_retries оставляем SDK: он умеет в 429 с Retry-After
        self._client = AsyncOpenAI(
            api_key=api_key, base_url=base_url, timeout=60.0, max_retries=3
        )
        self.input_tokens = 0
        self.requests = 0
        self._query_cache: OrderedDict[str, np.ndarray] = OrderedDict()

    async def embed_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)

        vectors: list[list[float]] = []
        batches: list[list[str]] = []
        batch: list[str] = []
        tokens = 0
        for value in texts:
            count = self.count_tokens(value)
            if count > self.profile.max_input_tokens or not value.strip():
                raise ValueError("Пустой или слишком длинный текст для эмбеддингов")
            if batch and (len(batch) >= 64 or tokens + count > 200_000):
                batches.append(batch)
                batch, tokens = [], 0
            batch.append(value)
            tokens += count
        if batch:
            batches.append(batch)
        for batch in batches:
            response = await self._client.embeddings.create(
                model=self.profile.name, input=batch,
                **({"dimensions": self.dim} if self.profile.name.startswith("text-embedding-3-") else {}),
            )
            self.requests += 1
            self.input_tokens += int(getattr(response.usage, "prompt_tokens", 0) or 0) if getattr(response, "usage", None) else 0
            items = sorted(response.data, key=lambda item: item.index)
            if [item.index for item in items] != list(range(len(batch))):
                raise ValueError("Провайдер вернул неполный набор эмбеддингов")
            if any(len(item.embedding) != self.dim for item in items):
                raise ValueError("Размерность эмбеддингов не совпадает с настройкой")
            vectors.extend(item.embedding for item in items)
        if not np.isfinite(np.array(vectors)).all():
            raise ValueError("Провайдер вернул некорректные числа в эмбеддингах")
        return l2_normalize(np.array(vectors, dtype=np.float32))

    async def embed_query(self, text: str) -> np.ndarray:
        # Вектор зависит только от текста и модели данного экземпляра.
        # Кэш не хранит ответы, поэтому не возвращает устаревшие факты компании.
        key = " ".join(text.split())
        if key in self._query_cache:
            self._query_cache.move_to_end(key)
            return self._query_cache[key].copy()
        vector = (await self.embed_documents([key]))[0]
        self._query_cache[key] = vector.copy()
        if len(self._query_cache) > 256:
            self._query_cache.popitem(last=False)
        return vector

    def count_tokens(self, text: str) -> int:
        if self.profile.name.startswith("text-embedding-"):
            import tiktoken
            return max(1,len(tiktoken.get_encoding("cl100k_base").encode(text, disallowed_special=())))
        return max(1, int(len(text) / 2.2))

    async def warmup(self) -> None:
        await self.embed_query("проверка доступности")
        log.info("OpenAI-совместимые эмбеддинги готовы: %s", self.profile.name)
