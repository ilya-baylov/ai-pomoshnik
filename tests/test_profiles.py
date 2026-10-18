"""Тесты профилей провайдеров и нормализации векторов."""

from __future__ import annotations

import numpy as np

from app.core.profiles import FALLBACK_PROFILE, get_profile
from app.embeddings.base import content_hash, l2_normalize, pack, unpack


def test_gemini_profile_limits():
    profile = get_profile("gemini-embedding-001")
    assert profile.max_input_tokens == 2048
    assert profile.target_chunk_tokens < profile.max_input_tokens
    assert profile.needs_normalization is True


def test_local_profile_has_e5_prefixes():
    """Без префиксов качество поиска у E5 заметно падает."""
    profile = get_profile("intfloat/multilingual-e5-large")
    assert profile.query_prefix == "query: "
    assert profile.passage_prefix == "passage: "
    assert profile.max_input_tokens == 512
    # fastembed 0.8 (mean pooling) отдаёт ненормализованные векторы
    assert profile.needs_normalization is True


def test_e5_small_profile_is_light():
    """Облегчённая модель для машин с 4 ГБ RAM."""
    profile = get_profile("intfloat/multilingual-e5-small")
    assert profile.dim == 384
    assert profile.max_input_tokens == 512
    assert profile.query_prefix == "query: "


def test_chunk_size_fits_model_limit():
    """Целевой размер плюс перекрытие не должны превышать лимит модели."""
    for name in ("gemini-embedding-001", "intfloat/multilingual-e5-large"):
        profile = get_profile(name)
        assert profile.target_chunk_tokens + profile.overlap_tokens <= profile.max_input_tokens


def test_unknown_model_gets_conservative_fallback():
    profile = get_profile("какая-то-своя-модель")
    assert profile.max_input_tokens == FALLBACK_PROFILE.max_input_tokens


def test_dim_override_applies():
    profile = get_profile("gemini-embedding-001", dim_override=768)
    assert profile.dim == 768
    assert profile.max_input_tokens == 2048


def test_l2_normalize_matrix():
    matrix = np.array([[3.0, 4.0], [1.0, 0.0]], dtype=np.float32)
    result = l2_normalize(matrix)
    norms = np.linalg.norm(result, axis=1)
    assert np.allclose(norms, 1.0)


def test_l2_normalize_single_vector():
    result = l2_normalize(np.array([3.0, 4.0], dtype=np.float32))
    assert np.isclose(np.linalg.norm(result), 1.0)


def test_l2_normalize_zero_vector_safe():
    """Нулевой вектор не должен давать деление на ноль."""
    result = l2_normalize(np.zeros(4, dtype=np.float32))
    assert not np.isnan(result).any()


def test_pack_unpack_roundtrip():
    original = np.array([0.1, -0.2, 0.3], dtype=np.float32)
    assert np.allclose(unpack(pack(original)), original)


def test_content_hash_stable_and_distinct():
    assert content_hash("текст") == content_hash("текст")
    assert content_hash("текст") != content_hash("текст ")
