"""Профили провайдеров эмбеддингов.

У моделей разные лимиты входа, поэтому размер фрагментов нельзя задавать
константой: при переключении провайдера он должен меняться сам. Иначе на
локальной модели с лимитом 512 токенов фрагменты по 1200 обрежутся молча,
и хвосты документов просто не попадут в поиск.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EmbeddingProfile:
    """Характеристики модели эмбеддингов и вытекающие параметры нарезки."""

    name: str
    dim: int
    max_input_tokens: int
    target_chunk_tokens: int
    overlap_tokens: int
    # Нужна ли ручная L2-нормализация после получения вектора
    needs_normalization: bool
    # Как помечается асимметрия «запрос — документ»
    query_prefix: str = ""
    passage_prefix: str = ""


PROFILES: dict[str, EmbeddingProfile] = {
    "text-embedding-3-small": EmbeddingProfile(
        name="text-embedding-3-small", dim=1536, max_input_tokens=8192,
        target_chunk_tokens=1200, overlap_tokens=150, needs_normalization=True,
    ),
    "text-embedding-3-large": EmbeddingProfile(
        name="text-embedding-3-large", dim=3072, max_input_tokens=8192,
        target_chunk_tokens=1200, overlap_tokens=150, needs_normalization=True,
    ),
    # Google, стабильная. Лимит 2048 токенов, MRL-усечение до 1536.
    # При любой размерности кроме 3072 нормализовать обязательно.
    "gemini-embedding-001": EmbeddingProfile(
        name="gemini-embedding-001",
        dim=1536,
        max_input_tokens=2048,
        target_chunk_tokens=1200,
        overlap_tokens=150,
        needs_normalization=True,
    ),
    # Локальная, XLM-RoBERTa. Жёсткий лимит 512 токенов.
    # Префиксы обязательны: без них качество поиска заметно падает.
    # needs_normalization=True: fastembed 0.8 с mean pooling отдаёт
    # ненормализованные векторы (нормы до ~30)
    "intfloat/multilingual-e5-large": EmbeddingProfile(
        name="intfloat/multilingual-e5-large",
        dim=1024,
        max_input_tokens=512,
        target_chunk_tokens=400,
        overlap_tokens=80,
        needs_normalization=True,
        query_prefix="query: ",
        passage_prefix="passage: ",
    ),
    # Облегчённая E5: 0.13 ГБ вместо 2.24, для машин с 4 ГБ RAM.
    # В списке fastembed её нет, поэтому local.py регистрирует её
    # через add_custom_model перед загрузкой.
    "intfloat/multilingual-e5-small": EmbeddingProfile(
        name="intfloat/multilingual-e5-small",
        dim=384,
        max_input_tokens=512,
        target_chunk_tokens=380,
        overlap_tokens=80,
        needs_normalization=True,
        query_prefix="query: ",
        passage_prefix="passage: ",
    ),
}

# Резервный профиль для неизвестной модели: консервативные значения,
# чтобы не обрезать фрагменты, если кто-то указал свою модель.
FALLBACK_PROFILE = EmbeddingProfile(
    name="unknown",
    dim=768,
    max_input_tokens=512,
    target_chunk_tokens=380,
    overlap_tokens=80,
    needs_normalization=True,
)


def get_profile(model: str, dim_override: int | None = None) -> EmbeddingProfile:
    """Профиль по имени модели.

    dim_override позволяет задать размерность из .env — у Gemini она
    настраивается через MRL-усечение.
    """
    if dim_override is not None and dim_override <= 0:
        raise ValueError("Размерность эмбеддингов должна быть положительной")
    profile = PROFILES.get(model)
    if profile is None:
        profile = EmbeddingProfile(**{**FALLBACK_PROFILE.__dict__, "name": model})
    if model.startswith("text-embedding-3-") and dim_override and dim_override > profile.dim:
        raise ValueError("Размерность превышает размерность модели OpenAI")
    if dim_override and dim_override != profile.dim:
        profile = EmbeddingProfile(**{**profile.__dict__, "dim": dim_override})
    return profile
