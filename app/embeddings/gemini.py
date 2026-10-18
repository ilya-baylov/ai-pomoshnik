"""Эмбеддинги через Gemini API с адаптивной ротацией нескольких ключей.

Ключей может быть несколько (EMBEDDING_API_KEYS через запятую): у Google
квота на проект, поэтому ключи из разных проектов складываются в общий
бюджет — причём у разных проектов лимиты могут различаться, поэтому
система адаптируется: пер-минутный 429 уменьшает размер пачки и охлаждает
ключ на минуту, навсегда вычёркиваются только дневные лимиты и 403.

Используется нативный google-genai, а не OpenAI-совместимый слой: последний
не принимает task_type, без которого теряется асимметрия «запрос — документ».
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from pathlib import Path

import numpy as np
from google.genai import types as genai_types
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from app.core.profiles import EmbeddingProfile
from app.embeddings.base import EmbeddingProvider, l2_normalize

log = logging.getLogger(__name__)

# Базовый темп для API-ключей: ~85 текстов/мин, пачка не больше 10.
# Сервисный аккаунт идёт по своему темпу (квота Vertex AI выше), при
# минутных 429 адаптив сам уменьшает пачку и охлаждает.
_MIN_REQUEST_INTERVAL = 0.7
_SA_REQUEST_INTERVAL = 0.25
_SA_BATCH_CEILING = 15
_BATCH_CEILING = 10
# Параллельных запросов к API одновременно
_CONCURRENCY = 6
# Максимум символов в одном запросе: крупные фрагменты книг
# в больших пачках сервер обрабатывает минутами
_MAX_BATCH_CHARS = 8000

# После минутного 429 ключ охлаждается; окно меньше минуты обычно
# сгорает из-за всплеска, поэтому ждём чуть дольше окна
_COOLDOWN_SECONDS = 70
# Десятки минутных отказов подряд вопреки кулдаунам — ключ сломан
_MAX_MINUTE_FAILS = 15


class KeyPool:
    """Круговая очередь ключей: кулдауны вместо долбёжки.

    pick() возвращает (ключ, 0), когда есть готовый, либо (None, секунды
    до ближайшего пробуждения), когда все в кулдауне, либо (None, 0), когда
    живых не осталось. Перманентно вычёркиваются только дневные лимиты и 403.
    """

    def __init__(self, keys: list[str]):
        if not keys:
            raise ValueError(
                "Не задан ключ для эмбеддингов. Заполните EMBEDDING_API_KEYS "
                "(или EMBEDDING_API_KEY) в .env"
            )
        self._keys = list(dict.fromkeys(keys))
        self._minute_fails = {k: 0 for k in self._keys}
        self._cooldown_until = {k: 0.0 for k in self._keys}
        self._dead: set[str] = set()
        self._cursor = 0
        self._lock = threading.Lock()

    @property
    def total(self) -> int:
        return len(self._keys)

    @property
    def alive(self) -> int:
        return sum(
            1
            for k in self._keys
            if k not in self._dead and self._minute_fails[k] < _MAX_MINUTE_FAILS
        )

    def pick(self) -> tuple[str | None, float]:
        now = time.monotonic()
        with self._lock:
            candidates = [
                k
                for k in self._keys
                if k not in self._dead
                and self._minute_fails[k] < _MAX_MINUTE_FAILS
            ]
            if not candidates:
                return None, 0.0
            ready = [k for k in candidates if self._cooldown_until[k] <= now]
            if not ready:
                soonest = min(self._cooldown_until[k] for k in candidates)
                return None, max(0.5, soonest - now)
            key = ready[self._cursor % len(ready)]
            self._cursor += 1
            return key, 0.0

    def success(self, key: str) -> None:
        with self._lock:
            self._minute_fails[key] = 0

    def minute_fail(self, key: str) -> None:
        """Пер-минутный 429: кулдаун, счётчик подряд не увечит ключ навсегда."""
        with self._lock:
            self._minute_fails[key] += 1
            self._cooldown_until[key] = time.monotonic() + _COOLDOWN_SECONDS
            fails = self._minute_fails[key]
        log.info(
            "Ключ %s...%s упёрся в минутный лимит, охлаждаю %s с (подряд: %s)",
            key[:6], key[-4:], _COOLDOWN_SECONDS, fails,
        )
        if fails >= _MAX_MINUTE_FAILS:
            log.warning(
                "Ключ %s...%s отбивается от минутных лимитов %s раз подряд, "
                "вычёркиваю до перезапуска",
                key[:6], key[-4:], fails,
            )

    def hard_fail(self, key: str, reason: str) -> None:
        """Дневной лимит или 403: ключ непригоден до завтра/навсегда."""
        with self._lock:
            if key not in self._dead:
                self._dead.add(key)
        log.warning(
            "Ключ %s...%s вычеркнут (%s), живых ключей: %s/%s",
            key[:6], key[-4:], reason, self.alive, self.total,
        )


class GeminiEmbeddings(EmbeddingProvider):
    def __init__(
        self,
        profile: EmbeddingProfile,
        api_keys: list[str],
        service_account_path: str | None = None,
    ):
        super().__init__(profile)
        self._sa_path = service_account_path
        self._client: object | None = None
        # Пул ключей живёт только когда сервисный аккаунт не задан
        self._pool = None if service_account_path else KeyPool(api_keys)
        self._clients: dict[str, object] = {}
        self._clients_lock = threading.Lock()
        # Адаптивный размер пачки: при минутных 429 уменьшается, при
        # успехе медленно растёт обратно. У сервисного аккаунта стартовый
        # потолок выше — квота Vertex AI обычно больше ключевой.
        if service_account_path:
            self._batch_size = _SA_BATCH_CEILING
            self._batch_ceiling = _SA_BATCH_CEILING
        else:
            self._batch_size = _BATCH_CEILING
            self._batch_ceiling = _BATCH_CEILING

    def _service_account_client(self):
        """Клиент Vertex AI под сервисным аккаунтом.

        Квота считается на проект аккаунта — независимый от API-ключей
        бакет, обычно заметно больше бесплатного ключевого.
        """
        import json

        from google import genai
        from google.oauth2 import service_account

        raw = json.loads(Path(self._sa_path).read_text(encoding="utf-8"))
        project = raw.get("project_id", "")
        creds = service_account.Credentials.from_service_account_file(
            self._sa_path,
            scopes=["https://www.googleapis.com/auth/cloud-platform"],
        )
        client = genai.Client(
            vertexai=True,
            project=project,
            credentials=creds,
            http_options=genai_types.HttpOptions(timeout=60 * 1000),
        )
        log.info("Эмбеддинги через сервисный аккаунт, проект %s", project)
        return client

    async def embed_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)

        # Пачки ограничены и по количеству, и по объёму: у книг фрагменты
        # по 1200 токенов, и запрос из 15 таких сервер жуёт минутами.
        # Лимит символов держит каждый запрос лёгким.
        batches: list[list[str]] = []
        current: list[str] = []
        current_chars = 0
        for text in texts:
            too_many = len(current) >= self._batch_size
            too_long = bool(current) and current_chars + len(text) > _MAX_BATCH_CHARS
            if too_many or too_long:
                batches.append(current)
                current, current_chars = [], 0
            current.append(text)
            current_chars += len(text)
        if current:
            batches.append(current)

        # Параллельные запросы: API отвечает волнами (иногда 40+ с на
        # запрос), и последовательная очередь превращает индексацию в часы.
        # Темп всё равно ограничен _pace, поэтому в лимиты не упираемся.
        semaphore = asyncio.Semaphore(_CONCURRENCY)

        async def run(batch: list[str]) -> np.ndarray:
            async with semaphore:
                return await self._embed(batch, "RETRIEVAL_DOCUMENT")

        vectors = await asyncio.gather(*(run(batch) for batch in batches))
        return np.vstack(vectors)

    async def embed_query(self, text: str) -> np.ndarray:
        matrix = await self._embed([text], "RETRIEVAL_QUERY")
        return matrix[0]

    async def _embed(self, texts: list[str], task_type: str) -> np.ndarray:
        """Один запрос к модели. Роутинг: сервисный аккаунт либо пул ключей."""
        if self._client is None and self._sa_path is not None:
            self._client = await asyncio.to_thread(self._service_account_client)
        if self._client is not None:
            return await self._embed_sa(texts, task_type)
        return await self._embed_pooled(texts, task_type)

    async def _embed_sa(self, texts: list[str], task_type: str) -> np.ndarray:
        """Сервисный аккаунт: паузы на минутных лимитах, без вычёркиваний."""
        last_exc: Exception | None = None
        connection_retries = 0
        for _ in range(24):
            try:
                matrix = await self._call(self._client, texts, task_type, "sa")
                # Успех: пачка медленно растёт обратно к потолку,
                # иначе после первой серии 429 навсегда остаёмся на 2
                self._batch_size = min(
                    self._batch_ceiling, max(self._batch_size, 2) + 1
                )
                return matrix
            except (TimeoutError, asyncio.TimeoutError):
                connection_retries += 1
                if connection_retries > 15:
                    raise RuntimeError("Сеть стабильно недоступна") from last_exc
                log.warning("Пачка не уложилась в таймаут, повтор (%s/15)", connection_retries)
                await asyncio.sleep(3)
                continue
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if _is_connection_error(exc):
                    # Сеть мерцает (IPv4/VPN): короткая пауза, попытка
                    # не съедает основной счётчик
                    connection_retries += 1
                    if connection_retries > 10:
                        raise
                    log.warning("Сбой сети, повтор через 5 с (%s/10)", connection_retries)
                    await asyncio.sleep(5)
                    continue
                if _is_denied(exc):
                    raise RuntimeError(
                        f"Сервисный аккаунт отклонён (проверь, включён ли "
                        f"Vertex AI API на проекте): {str(exc)[:200]}"
                    ) from exc
                if _is_daily_quota(exc):
                    raise RuntimeError(
                        f"Дневная квота сервисного аккаунта исчерпана: "
                        f"{str(exc)[:200]}"
                    ) from exc
                if _is_minute_quota(exc):
                    # Живой вопрос сотрудника ждёт человек — ему короткая
                    # пауза и много попыток. Фоновой индексации не горит:
                    # она ждёт полное окно, освобождая квоту для запросов.
                    is_query = task_type == "RETRIEVAL_QUERY"
                    pause = 4 if is_query else _COOLDOWN_SECONDS
                    if not is_query:
                        self._batch_size = max(2, self._batch_size // 2)
                    log.info(
                        "Минутный лимит, пауза %s с (%s)",
                        pause,
                        "вопрос" if is_query else f"пачка -> {self._batch_size}",
                    )
                    await asyncio.sleep(pause)
                    continue
                raise
        raise RuntimeError(f"Не удалось после пауз: {last_exc}")

    async def _embed_pooled(self, texts: list[str], task_type: str) -> np.ndarray:
        """Ротация API-ключей. Все ключи в кулдауне — ждём, не долбим."""
        last_exc: Exception | None = None
        for _ in range(40):  # страховка от вечного цикла
            key, wait = self._pool.pick()
            if key is None:  # noqa: SIM102
                if wait <= 0:
                    raise RuntimeError(
                        "Все ключи Gemini вычеркнуты по дневным лимитам или "
                        "отклонены. Добавьте EMBEDDING_API_KEYS или ждите "
                        "восстановления квот."
                    )
                await asyncio.sleep(min(wait, 75))
                continue
            try:
                client = self._client_for(key)
                matrix = await self._call(client, texts, task_type, key)
                self._pool.success(key)
                self._batch_size = min(self._batch_ceiling, self._batch_size + 2)
                return matrix
            except (TimeoutError, asyncio.TimeoutError):
                log.warning("Пачка не уложилась в таймаут, повтор")
                await asyncio.sleep(3)
                continue
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if _is_daily_quota(exc):
                    self._pool.hard_fail(key, "дневной лимит")
                elif _is_denied(exc):
                    self._pool.hard_fail(key, "отказ доступа")
                elif _is_minute_quota(exc):
                    self._pool.minute_fail(key)
                    self._batch_size = max(2, self._batch_size // 2)
                else:
                    raise
        raise RuntimeError(
            f"Не удалось выполнить запрос за отведённые попытки: {last_exc}"
        )

    async def _call(
        self,
        client,
        texts: list[str],
        task_type: str,
        pace_key: str,
    ) -> np.ndarray:
        config = genai_types.EmbedContentConfig(
            task_type=task_type,
            output_dimensionality=self.dim,
        )
        # Темп: 1.35 с НА ТЕКСТ — пачка из 10 занимает 13.5 с окна ключа
        await asyncio.to_thread(_pace, pace_key, len(texts))
        response = await _run_bounded(
            lambda: client.models.embed_content(
                model=self.profile.name,
                contents=texts,
                config=config,
            ),
            timeout=45,
        )
        matrix = np.array(
            [item.values for item in response.embeddings],
            dtype=np.float32,
        )
        # При усечении MRL векторы приходят ненормализованными
        if self.profile.needs_normalization:
            matrix = l2_normalize(matrix)
        return matrix

    def _client_for(self, api_key: str):
        with self._clients_lock:
            if api_key not in self._clients:
                from google import genai

                # timeout обязателен: без него HTTP-запрос может висеть
                # вечно (один раз залип на 10+ минут без единого байта)
                self._clients[api_key] = genai.Client(
                    api_key=api_key,
                    http_options=genai_types.HttpOptions(timeout=60 * 1000),
                )
            return self._clients[api_key]

    def count_tokens(self, text: str) -> int:
        """Оценка по символам с запасом.

        Точный подсчёт потребовал бы запроса к API на каждый фрагмент.
        Делитель 2.2 занижает результат сознательно: для русского реальное
        соотношение около 2.5-3 символов на токен, поэтому оценка выходит
        завышенной, а фрагменты — чуть мельче лимита. Это безопасная сторона.
        """
        return max(1, int(len(text) / 2.2))

    async def warmup(self) -> None:
        await self.embed_query("проверка доступности")
        source = "сервисный аккаунт" if self._sa_path else f"{self._pool.total} ключей"
        log.info(
            "Gemini embeddings готовы: %s, dim=%s, источник: %s",
            self.profile.name,
            self.dim,
            source,
        )


_last_by_key: dict[str, float] = {}
_pace_lock = threading.Lock()


def _pace(api_key: str, count: int) -> None:
    """Ограничитель частоты на ключ с учётом размера пачки."""
    interval = _SA_REQUEST_INTERVAL if api_key == "sa" else _MIN_REQUEST_INTERVAL
    with _pace_lock:
        now = time.monotonic()
        wait = (
            interval * count
            - (now - _last_by_key.get(api_key, 0.0))
        )
        if wait > 0:
            time.sleep(wait)
        _last_by_key[api_key] = time.monotonic()


async def _run_bounded(func, timeout: float):
    """Блокирующий вызов с жёстким таймаутом.

    asyncio.timeout/wait_for не способны прервать to_thread на
    невозвратном вызове (DNS-зависание): отмена доносится только после
    завершения потока. asyncio.wait(timeout) возвращает управление всегда —
    застрявший поток бросаем в фоне, он не занимает событийный цикл.
    """
    loop = asyncio.get_running_loop()
    future = loop.run_in_executor(None, func)
    done, pending = await asyncio.wait({future}, timeout=timeout)
    if pending:
        raise TimeoutError(f"вызов не завершился за {timeout:.0f} с")
    return future.result()


def _is_daily_quota(exc: Exception) -> bool:
    """Дневное окно: ключ бесполезен до завтра."""
    text = str(exc).lower()
    return "resource_exhausted" in text and (
        "perday" in text.replace("_", "").replace(" ", "")
        or "per_day" in text
    )


def _is_minute_quota(exc: Exception) -> bool:
    text = str(exc).lower()
    return "429" in text or "resource_exhausted" in text


def _is_denied(exc: Exception) -> bool:
    text = str(exc).lower()
    return "403" in text or "permission_denied" in text


def _is_connection_error(exc: Exception) -> bool:
    """Сетевые сбои: TCP-отказ, таймаут, обрыв. Временные по природе."""
    text = str(exc).lower()
    markers = (
        "connecterror", "connection refused", "10061", "10060",
        "timed out", "timeout", "connect_timeout", "remoteendclosed",
        "10054",
    )
    return any(m in text for m in markers)
