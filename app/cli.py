"""Командная строка: doctor, ingest, search, ask, stats, eval, deps-check.

Позволяет проверять поиск и ответы без Telegram, где ошибка поиска и ошибка
форматирования выглядят одинаково.
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from app.core.config import get_bot_config, get_settings
from app.ingest.loaders.registry import SUPPORTED_EXTENSIONS
from app.storage.db import init_db, session_scope


async def cmd_doctor() -> int:
    """Проверка окружения. Дешевле выяснить ограничения сразу."""
    import sqlite3

    settings = get_settings()
    from app.generation.llm import primary_chain
    problems: list[str] = []
    print("Проверка окружения\n")

    version = sys.version_info
    py_ok = (3, 12) <= (version.major, version.minor) <= (3, 14)
    print(f"  Python           {version.major}.{version.minor}.{version.micro}")
    if not py_ok:
        problems.append(
            f"Python {version.major}.{version.minor} вне диапазона 3.12–3.14 "
            "(numpy требует ≥3.12, aiogram поддерживает <3.15)"
        )

    print(f"  SQLite           {sqlite3.sqlite_version}")
    connection = sqlite3.connect(":memory:")
    try:
        connection.execute("CREATE VIRTUAL TABLE t USING fts5(x)")
        print("  FTS5             доступен")
    except sqlite3.OperationalError as exc:
        problems.append(f"FTS5 недоступен: {exc}. Полнотекстовый поиск не заработает")
    finally:
        connection.close()

    try:
        from app.ingest.stemming import stem_word

        assert stem_word("отпуска") == stem_word("отпуск")
        print("  Стеммер          работает")
    except Exception as exc:  # noqa: BLE001
        problems.append(f"Стеммер не работает: {exc}")

    profile = settings.profile()
    print(f"\n  Эмбеддинги       {settings.embedding_provider}")
    print(f"  Модель           {profile.name}, dim {profile.dim}")
    print(f"  Лимит фрагмента  {profile.target_chunk_tokens} из {profile.max_input_tokens} токенов")
    print(f"  Языковая модель  {settings.llm_model}")

    if not settings.telegram_bot_token:
        problems.append("TELEGRAM_BOT_TOKEN не задан в .env")
    if not primary_chain(settings).api_key:
        problems.append("OPENAI_API_KEY (или LLM_API_KEY) не задан в .env")
    if not settings.admin_ids:
        problems.append("ADMIN_IDS не задан: панель управления будет недоступна")

    knowledge = settings.knowledge_path
    if knowledge.exists():
        files = [p for p in knowledge.rglob("*") if p.suffix.lower() in SUPPORTED_EXTENSIONS]
        print(f"\n  Папка knowledge  {len(files)} файлов")
        others = [
            p
            for p in knowledge.rglob("*")
            if p.is_file() and p.suffix.lower() not in SUPPORTED_EXTENSIONS
        ]
        if others:
            print(f"  Нераспознанных   {len(others)} (будут пропущены)")
    else:
        problems.append(f"Папка {knowledge} не найдена")

    await init_db()
    print("  База данных      схема создана")

    if primary_chain(settings).api_key:
        from app.generation import llm

        ok, detail = await llm.health_check()
        print(f"  Проверка модели  {'ответила' if ok else 'ошибка: ' + detail}")
        if not ok:
            problems.append(f"Языковая модель недоступна: {detail}")

    # Эмбеддинги проверяем отдельно от LLM: они на разных провайдерах,
    # и при OpenRouter/DeepSeek ключ Gemini нужно задать отдельно
    if settings.embedding_provider == "gemini":
        if not settings.embeddings_key and not settings.service_account_path:
            problems.append(
                "Ключ Gemini для эмбеддингов не задан: заполните "
                "EMBEDDING_API_KEY (или LLM_API_KEY, если LLM тоже Gemini)"
            )
        else:
            from app.embeddings.factory import get_embedding_provider

            try:
                provider = get_embedding_provider()
                await provider.warmup()
                print(f"  Эмбеддинги       {provider.name}: доступны")
            except Exception as exc:  # noqa: BLE001
                problems.append(f"Эмбеддинги недоступны: {exc}")
    else:
        from app.embeddings.factory import get_embedding_provider
        try:
            provider = get_embedding_provider()
            await provider.warmup()
            print(f"  Эмбеддинги       {provider.name}: доступны")
        except Exception as exc:
            problems.append(f"Эмбеддинги недоступны: {type(exc).__name__}")

    if problems:
        print(f"\nПроблемы ({len(problems)}):")
        for problem in problems:
            print(f"  — {problem}")
        return 1

    print("\nВсё готово. Дальше: python -m app.cli ingest")
    return 0


async def cmd_deps_check() -> int:
    """Сверить закреплённые версии с актуальными на PyPI."""
    import json
    import urllib.request
    from concurrent.futures import ThreadPoolExecutor
    from pathlib import Path

    from app.core.config import ROOT

    pinned: dict[str, str] = {}
    for name in ("requirements.txt", "requirements-local.txt"):
        path = Path(ROOT / name)
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("#") or "==" not in line or line.startswith("-r"):
                continue
            package, version = line.split("==", 1)
            pinned[package.strip()] = version.strip()

    def latest(package: str) -> tuple[str, str]:
        try:
            url = f"https://pypi.org/pypi/{package}/json"
            with urllib.request.urlopen(url, timeout=20) as response:
                return package, json.load(response)["info"]["version"]
        except Exception as exc:  # noqa: BLE001
            return package, f"ошибка: {type(exc).__name__}"

    print(f"Сверяю {len(pinned)} пакетов с PyPI\n")
    outdated = 0
    with ThreadPoolExecutor(max_workers=8) as pool:
        for package, newest in pool.map(latest, pinned):
            current = pinned[package]
            if newest == current:
                print(f"  = {package:22} {current}")
            else:
                outdated += 1
                print(f"  ! {package:22} {current} -> {newest}")

    print(
        f"\nУстарело: {outdated}. Обновление мажорных версий — отдельная задача "
        "с прогоном тестов."
        if outdated
        else "\nВсе версии актуальны."
    )
    return 0


async def cmd_ingest(full: bool) -> int:
    from app.ingest.pipeline import Ingestor

    await init_db()
    print("Полная переиндексация…" if full else "Проверяю изменения…")

    last_line = ""

    async def on_progress(progress) -> None:
        nonlocal last_line
        line = f"  [{progress.percent:3d}%] {progress.current}/{progress.total} {progress.file_name[:50]}"
        if line != last_line:
            print(line)
            last_line = line

    result = await Ingestor().run(full=full, on_progress=on_progress)

    print(
        f"\nФайлов: {result.files_total} "
        f"(изменено {result.files_changed}, без изменений {result.files_skipped})"
    )
    print(f"Фрагментов создано: {result.chunks_created}")
    if result.files_deleted:
        print(f"Удалено документов: {result.files_deleted}")
    print(f"Максимум токенов: {result.max_token_count} из {result.token_limit}")
    print(
        f"Векторов: {result.embeddings_computed} новых, "
        f"{result.embeddings_from_cache} из кэша"
    )
    print(f"Время: {result.duration_seconds:.1f} с")

    for warning in result.warnings:
        print(f"  ! {warning}")
    for error in result.errors:
        print(f"  ОШИБКА {error}")
    return 0 if result.ok else 1


async def cmd_search(query: str) -> int:
    from app.retrieval.search import search
    from app.retrieval.vector import get_vector_index

    await init_db()
    await get_vector_index().load()
    results = await search(query)

    if not results:
        print("Ничего не найдено.")
        return 0

    min_score = float(get_bot_config().get("retrieval.min_score", 0.42))
    print(f"Найдено {len(results)} фрагм. (порог {min_score})\n")
    for position, chunk in enumerate(results, start=1):
        mark = "+" if chunk.score >= min_score else "-"
        print(f"{mark} [{position}] {chunk.score:.3f}  {chunk.title}")
        if chunk.breadcrumbs:
            print(f"      {chunk.breadcrumbs}")
        print(f"      {chunk.text[:200].replace(chr(10), ' ')}…\n")
    return 0


async def cmd_ask(question: str) -> int:
    from app.generation.answer import build_answer
    from app.retrieval.search import search
    from app.retrieval.vector import get_vector_index

    await init_db()
    await get_vector_index().load()

    chunks = await search(question)
    answer = await build_answer(question, chunks)

    print(f"\n{answer.text}\n")
    if answer.sources:
        print("Источники:")
        for source in answer.sources:
            print(f"  [{source.index}] {source.label()}")
    print(
        f"\n(близость {answer.top_score:.3f}, "
        f"{'фолбэк' if answer.is_fallback else 'ответ по базе'})"
    )
    return 0


async def cmd_stats() -> int:
    from app.storage import repo

    await init_db()
    async with session_scope() as session:
        data = await repo.stats(session)

    print(f"Документов: {data['documents']}, фрагментов: {data['chunks']}")
    print(f"\nЗа {data['days']} дн.:")
    print(f"  вопросов: {data['queries']}, сотрудников: {data['users']}")
    print(f"  без ответа: {data['fallbacks']} ({data['fallback_rate']}%)")
    print(f"  средний отклик: {data['avg_latency_ms']} мс")
    print(f"\nВ очереди без ответа: {data['unanswered_new']}")
    print(f"Оценок «не помогло»: {data['negative']}")
    print(f"Активных пользователей: {data['active_users']}")
    if data["stale_docs"]:
        print(f"Просроченных документов: {data['stale_docs']}")
    return 0


async def cmd_warmup() -> int:
    from app.embeddings.factory import get_embedding_provider

    provider = get_embedding_provider()
    print(f"Прогрев {provider.name}…")
    await provider.warmup()
    print("Готово.")
    return 0


async def cmd_ocr() -> int:
    """Распознать сканы (PDF без текстового слоя) через Gemini Vision."""
    from app.ingest.ocr import find_scanned_pdfs, ocr_all

    settings = get_settings()
    root = settings.knowledge_path
    scans = find_scanned_pdfs(root)
    if not scans:
        print("Сканов без распознанного текста не найдено.")
        return 0

    print(f"Сканов к распознаванию: {len(scans)}")
    for path in scans:
        print(f"  - {path.relative_to(root)}")
    print()
    created = await ocr_all(root)
    print()
    print(f"Распознано файлов: {len(created)}")
    for path in created:
        print(f"  {path.name}: {path.stat().st_size} байт")
    if created:
        print()
        print("Дальше: python -m app.cli ingest — распознанное попадёт в базу")
    return 0


async def cmd_eval(*, answers: bool = False, output: str | None = None) -> int:
    from eval.run import run_eval

    await init_db()
    return await run_eval(answers=answers, output=output)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m app.cli", description="Управление ботом по базе знаний"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("doctor", help="проверить окружение и настройки")
    sub.add_parser("deps-check", help="сверить версии пакетов с PyPI")

    ingest = sub.add_parser("ingest", help="проиндексировать knowledge/")
    ingest.add_argument("--full", action="store_true", help="игнорировать кэш")

    reindex = sub.add_parser("reindex", help="то же, что ingest")
    reindex.add_argument("--full", action="store_true")

    search = sub.add_parser("search", help="проверить поиск")
    search.add_argument("query", nargs="+")

    ask = sub.add_parser("ask", help="задать вопрос из консоли")
    ask.add_argument("question", nargs="+")

    sub.add_parser("stats", help="статистика")
    sub.add_parser("warmup", help="прогреть модель эмбеддингов")
    evaluation = sub.add_parser("eval", help="прогнать проверочные вопросы")
    evaluation.add_argument("--answers", action="store_true", help="проверять также ответы модели; использует API")
    evaluation.add_argument("--output", help="сохранить результаты в JSON")
    sub.add_parser("ocr", help="распознать сканы PDF через Gemini Vision")
    return parser


def main() -> int:
    args = build_parser().parse_args()

    from app.observability.logging import setup_logging

    setup_logging("ingest")

    if args.command == "doctor":
        return asyncio.run(cmd_doctor())
    if args.command == "deps-check":
        return asyncio.run(cmd_deps_check())
    if args.command in ("ingest", "reindex"):
        return asyncio.run(cmd_ingest(full=args.full))
    if args.command == "search":
        return asyncio.run(cmd_search(" ".join(args.query)))
    if args.command == "ask":
        return asyncio.run(cmd_ask(" ".join(args.question)))
    if args.command == "stats":
        return asyncio.run(cmd_stats())
    if args.command == "warmup":
        return asyncio.run(cmd_warmup())
    if args.command == "eval":
        return asyncio.run(cmd_eval(answers=args.answers, output=args.output))
    if args.command == "ocr":
        return asyncio.run(cmd_ocr())
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
