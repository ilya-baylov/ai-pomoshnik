"""Сборка ответа: порог уверенности, промпт, фолбэк.

Порог определяет, какие сведения компании можно передать модели.
Свободное общение возможно без найденных документов, но внутренние
сведения компании нельзя подменять общими знаниями.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from dataclasses import replace
from contextlib import aclosing
from datetime import date

from app.core.config import get_bot_config, get_settings
from app.core.contracts import RetrievedChunk, Source
from app.generation import llm
from app.generation.prompts import ModelPrompt, render

log = logging.getLogger(__name__)


@dataclass
class GeneratedAnswer:
    text: str
    sources: list[Source]
    is_fallback: bool
    top_score: float
    used_chunk_ids: list[int]
    # Какая модель сформулировала ответ: основная или резервная
    model: str = ""


def _sources(chunks: list[RetrievedChunk]) -> list[Source]:
    return [
        Source(
            index=position,
            title=chunk.title,
            breadcrumbs=chunk.breadcrumbs,
            page=chunk.page,
            is_stale=chunk.is_stale,
        )
        for position, chunk in enumerate(chunks, start=1)
    ]


def _quote_answer(chunk: RetrievedChunk) -> str:
    """Ответ для чувствительных разделов: цитата без обращения к LLM."""
    header = chunk.title
    if chunk.breadcrumbs:
        header = f"{header} → {chunk.breadcrumbs}"
    return f"Цитата из документа «{header}»:\n\n{chunk.text}"


async def rewrite_query(question: str, history: list[dict]) -> str:
    """Сделать вопрос самодостаточным, если он опирается на диалог."""
    if not history:
        return question
    try:
        rewritten, _ = await llm.complete(
            render("rewrite_query.j2", question=question, history=history),
            temperature=0.0,
            max_tokens=200,
        )
    except llm.LLMUnavailable:
        return question
    # Модель может добавить пояснение — берём первую строку
    first_line = rewritten.splitlines()[0].strip().strip('"') if rewritten else ""
    return first_line or question


def needs_query_rewrite(question: str) -> bool:
    words = question.casefold().split()
    if not words:
        return False
    return len(words) <= 4 or (len(words) <= 12 and (
        words[0] in {"а", "и", "там", "тогда", "ещё", "еще", "он", "она", "они", "это"}
        or any(w.strip(',.?!') in {"там", "его", "её", "нее", "него", "них", "этот", "эта", "эти", "такой", "такая"} for w in words)
    ))


async def rerank(question: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    """Переранжировать топ через LLM. При сбое — исходный порядок."""
    config = get_bot_config()
    output_size = int(config.get("rerank.output_size", 5))
    try:
        raw, _ = await llm.complete(
            render(
                "rerank.j2", question=question, chunks=chunks, output_size=output_size
            ),
            temperature=0.0,
            max_tokens=100,
        )
    except llm.LLMUnavailable:
        return chunks[:output_size]

    order = [int(n) for n in re.findall(r"\d+", raw)]
    picked: list[RetrievedChunk] = []
    for number in order:
        if 1 <= number <= len(chunks) and chunks[number - 1] not in picked:
            picked.append(chunks[number - 1])
    return (picked or chunks)[:output_size]


async def build_answer(
    question: str,
    chunks: list[RetrievedChunk],
    history: list[dict] | None = None,
    on_chunk: Callable[[str], Awaitable[None]] | None = None,
    require_company_data: bool = False,
    profile_context: dict | None = None,
) -> GeneratedAnswer:
    config = get_bot_config()
    min_score = float(config.get("retrieval.min_score", 0.42))
    fallback_text = config.message("fallback")
    primary_model = get_settings().llm_model
    free_chat = bool(config.get("generation.free_chat", False))
    missing_company_reply = config.message("company_data_missing") or "У меня нет подтверждённых данных компании по этому вопросу."

    if not require_company_data:
        # Optional training evidence has a fixed budget and no extra LLM reranker.
        selected = []
        remaining = 9000
        for chunk in chunks:
            if chunk.score < min_score or len(selected) >= 4 or remaining < 200:
                continue
            cap = min(3200, remaining)
            text = chunk.text
            if len(text) > cap:
                text = text[:cap].rsplit(' ', 1)[0] + '\n[Отрывок сокращён.]'
            selected.append(replace(chunk, text=text))
            remaining -= len(text)
        chunks = selected

    top_score = max((c.score for c in chunks), default=0.0)
    # Один сильный результат не делает остальные совпадения достоверными.
    # Порог применяется к каждому фрагменту до передачи источников модели.
    chunks = [chunk for chunk in chunks if chunk.score >= min_score]

    # Complete structured facts can be rendered without a paid model call.
    direct = [c.direct_answer for c in chunks if c.direct_answer and c.score >= min_score]
    if direct and len(direct) == len(chunks):
        text = '\n\n'.join(direct)
        if on_chunk:
            await on_chunk(text)
        return GeneratedAnswer(text, _sources(chunks), False, top_score,
                               [c.chunk_id for c in chunks], '')

    # Не передаём случайные совпадения как достоверные данные компании.
    if not chunks or top_score < min_score:
        if require_company_data:
            text = missing_company_reply + " Уточни нужные сведения у администратора компании."
            if on_chunk:
                await on_chunk(text)
            return GeneratedAnswer(text, [], True, top_score, [], "")
        if not free_chat:
            log.info("Фолбэк: top_score=%.3f < %.3f", top_score, min_score)
            if on_chunk:
                await on_chunk(fallback_text)
            return GeneratedAnswer(fallback_text, [], True, top_score, [], "")
        chunks = []

    sensitive = set(config.get("sensitive_sections", []) or [])
    if sensitive and chunks and chunks[0].section in sensitive:
        best = chunks[0]
        quote_text = _quote_answer(best)
        if on_chunk:
            await on_chunk(quote_text)
        return GeneratedAnswer(
            quote_text,
            _sources([best]),
            False,
            top_score,
            [best.chunk_id],
            "",
        )

    if chunks and require_company_data and config.get("rerank.enabled", False):
        input_size = int(config.get("rerank.input_size", 10))
        chunks = await rerank(question, chunks[:input_size])

    from app.generation.promotion_list import complete_promotion_list, named_promotion_answer, promotion_overview
    complete_list = named_promotion_answer(question, chunks) or complete_promotion_list(question, chunks) or promotion_overview(question,chunks)
    if complete_list is not None:
        if on_chunk:
            await on_chunk(complete_list)
        return GeneratedAnswer(complete_list, _sources(chunks), False, top_score,
                               [c.chunk_id for c in chunks], '')

    domain_template = config.get("generation.domain_rules_template")
    domain_rules = render(domain_template, persona=config.persona()) if domain_template else ""
    instructions = render(
        "answer.j2",
        persona=config.persona(),
        current_date=date.today().isoformat(),
        free_chat=free_chat,
        missing_company_reply=missing_company_reply,
        domain_rules=domain_rules,
    )
    from app.retrieval.promotion_catalog import _price_fact, _client_message
    validate_promotion=any(c.promotion_rows is not None for c in chunks)
    if validate_promotion and not _client_message(question):
        instructions+='\nСейчас нужен внутренний ответ сотруднику. Партнёрские предложения доступны сотруднику. Не исключай их из-за запрета рассказывать клиентам.\n'
    prompt_chunks = [replace(c, text=_price_fact(c.text)) for c in chunks]
    prompt = ModelPrompt(instructions, render("answer_context.j2", chunks=prompt_chunks,
                                            question=question, history=history or [],
                                            profile_context=profile_context or {}))
    budget = int(config.get("generation.max_output_tokens", 1200))
    if free_chat and not chunks:
        budget = min(budget, int(config.get("generation.max_chat_output_tokens", 900)))

    try:
        if on_chunk is not None:
            chunks_text: list[str] = []
            model = primary_model
            async with aclosing(llm.complete_stream(
                prompt,
                temperature=float(config.get("generation.temperature", 0.2)),
                max_tokens=budget,
            )) as stream:
                async for delta, resp_model in stream:
                    chunks_text.append(delta)
                    model = resp_model
                    if not validate_promotion and not require_company_data:
                        await on_chunk(delta)
            text = "".join(chunks_text).strip()
        else:
            text, model = await llm.complete(
                prompt,
                temperature=float(config.get("generation.temperature", 0.2)),
                max_tokens=budget,
            )
    except llm.LLMUnavailable:
        raise

    if not text:
        if on_chunk:
            await on_chunk(fallback_text)
        return GeneratedAnswer(fallback_text, [], True, top_score, [], primary_model)

    if validate_promotion:
        from app.generation.promotion_list import repair_staff_omission
        text=repair_staff_omission(text,question,chunks)

    if require_company_data and chunks:
        from app.generation.fact_guard import unsupported_values
        unverified = unsupported_values(text, chunks, question=question)
        if unverified:
            log.warning('Answer contains %s critical values absent from evidence', len(unverified))
            correction = ModelPrompt(instructions + '\nПроверь ответ: суммы, проценты, даты и телефоны должны присутствовать в документах. '
                'Удали неподтверждённые значения, сохрани полезную подтверждённую часть. '
                'Не выводи новые условия из предыдущих ответов ассистента.',
                prompt.user_input + '\n\nЧерновик для исправления:\n' + text +
                '\nНеподтверждённые значения: ' + repr(sorted(unverified)))
            try:
                text, model = await llm.complete(correction, temperature=0.0, max_tokens=budget)
            except llm.LLMUnavailable:
                text = ''
            if not text or unsupported_values(text, chunks, question=question):
                text = 'Не могу надёжно подтвердить точные условия по этому вопросу. Уточните их у администратора перед применением.'
                if on_chunk:
                    await on_chunk(text)
                return GeneratedAnswer(text, _sources(chunks), True, top_score,
                                       [c.chunk_id for c in chunks], model)
    if on_chunk and (validate_promotion or require_company_data):
        await on_chunk(text)

    return GeneratedAnswer(
        text=text,
        sources=_sources(chunks),
        is_fallback=text.lstrip().startswith(missing_company_reply),
        top_score=top_score,
        used_chunk_ids=[c.chunk_id for c in chunks],
        model=model,
    )
