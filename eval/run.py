"""Проверка того, находит ли поиск ожидаемые документы."""
from pathlib import Path
import yaml

QUESTIONS_FILE = Path(__file__).parent / "questions.yaml"


async def run_eval(top_k: int = 5) -> int:
    from app.core.config import get_bot_config
    from app.retrieval.search import search
    from app.retrieval.vector import get_vector_index

    if not QUESTIONS_FILE.exists():
        print(f"Нет файла {QUESTIONS_FILE}")
        return 1
    items = yaml.safe_load(QUESTIONS_FILE.read_text(encoding="utf-8")) or []
    if not items:
        print("Список вопросов пуст.")
        return 1
    await get_vector_index().load()
    hits = 0
    for item in items:
        expected = str(item["expect_source"]).lower()
        results = await search(item["question"], limit=top_k)
        if expected == "none":
            threshold = float(get_bot_config().get("retrieval.min_score", 0.55))
            matched = not results or results[0].score < threshold
        else:
            matched = any(expected in f"{c.title} {c.section or ''} {c.breadcrumbs or ''}".lower() for c in results)
        hits += int(matched)
        if not matched:
            print(f"Не найден ожидаемый источник: {item['question']}")
    recall = hits / len(items) * 100
    print(f"recall@{top_k}: {recall:.0f}% ({hits} из {len(items)})")
    return 0 if recall >= 85 else 1


if __name__ == "__main__":
    import asyncio
    raise SystemExit(asyncio.run(run_eval()))
