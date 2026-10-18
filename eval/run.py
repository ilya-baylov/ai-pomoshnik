"""Проверка поиска и, по запросу, ответов на вымышленной базе."""
from pathlib import Path
import json
from time import perf_counter

import yaml

QUESTIONS_FILE = Path(__file__).parent / "questions.yaml"


def _normalized(text):
    return text.casefold().replace("ё", "е")


async def run_eval(top_k: int = 5, *, answers: bool = False, output: str | None = None) -> int:
    from app.core.config import get_bot_config
    from app.generation.answer import build_answer
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
    records = []
    for item in items:
        started = perf_counter()
        results = await search(item["question"], limit=top_k)
        search_ms = round((perf_counter() - started) * 1000)
        expected = _normalized(str(item["expect_source"]))
        threshold = float(get_bot_config().get("retrieval.min_score", 0.55))
        if expected == "none":
            source_match = not results or max(c.score for c in results) < threshold
        else:
            source_match = any(expected in _normalized(f"{c.title} {c.section or ''} {c.breadcrumbs or ''}") for c in results)
        record = {
            "question": item["question"], "expected_source": item["expect_source"],
            "source_match": source_match, "search_ms": search_ms,
            "retrieved": [{"title": c.title, "section": c.section, "score": c.score} for c in results],
        }
        if answers:
            started = perf_counter()
            answer = await build_answer(item["question"], results, require_company_data=True)
            missing = [term for term in item.get("expected_terms", []) if _normalized(term) not in _normalized(answer.text)]
            record.update(answer=answer.text, answer_ms=round((perf_counter() - started) * 1000),
                          is_fallback=answer.is_fallback, missing_terms=missing,
                          answer_match=(answer.is_fallback if expected == "none" else not answer.is_fallback and not missing))
        record["passed"] = source_match and (record.get("answer_match", True))
        records.append(record)
        print(("OK" if record["passed"] else "FAIL") + ": " + item["question"])
    hits = sum(r["passed"] for r in records)
    report = {"mode": "search-and-answer" if answers else "search", "cases": len(records),
              "passed": hits, "failed": len(records) - hits, "results": records,
              "limitations": "Matching expected sources and text fragments is a regression check, not a semantic accuracy or hallucination-free guarantee."}
    if output:
        target = Path(output)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Пройдено: {hits}/{len(records)}. Режим: {report['mode']}")
    return 0 if hits == len(records) else 1


if __name__ == "__main__":
    import asyncio
    raise SystemExit(asyncio.run(run_eval()))
