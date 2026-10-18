"""Ядро: вопрос на входе, ответ на выходе.

Ничего не знает о Telegram. Адаптер канала передаёт IncomingMessage и получает
OutgoingMessage — благодаря этому подключение другого мессенджера не задевает
поиск и генерацию.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Awaitable, Callable

from app.core.config import get_bot_config, get_settings
from app.core.contracts import IncomingMessage, OutgoingMessage, Role
from app.generation import llm
from app.generation.answer import GeneratedAnswer, build_answer, rewrite_query, needs_query_rewrite
from app.generation.scope import route_question
from app.generation.history import trim_history
from app.retrieval.search import search
from app.storage import repo
from app.storage.db import session_scope
from app.storage.models import Query, User

log = logging.getLogger(__name__)


class RateLimited(RuntimeError):
    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class AccessDenied(RuntimeError):
    pass


class QuestionService:
    def __init__(self) -> None:
        self.config = get_bot_config()
        self.settings = get_settings()

    async def resolve_user(self, message: IncomingMessage, username: str | None = None,
                           full_name: str | None = None) -> User:
        """Найти или создать пользователя, выдав роль из .env при первом входе."""
        external = int(message.external_user_id)
        default_role: str | None = None
        if external in self.settings.admin_ids:
            default_role = Role.ADMIN.value
        elif external in self.settings.staff_ids:
            default_role = Role.STAFF.value

        async with session_scope() as session:
            user = await repo.get_or_create_user(
                session,
                channel=message.channel,
                external_id=message.external_user_id,
                username=username,
                full_name=full_name,
                default_role=default_role,
                allow_registration=bool(self.config.get("access.open_registration", False)),
            )
            # Повышение роли, если ID добавили в .env позже
            if default_role == Role.ADMIN.value and user.role != Role.ADMIN.value:
                user.role = Role.ADMIN.value
                user.is_active = True
            return user

    async def check_limits(self, user: User) -> None:
        from app.core.contracts import Role

        if user.role == Role.ADMIN.value:
            return  # администратор без лимитов
        per_minute = int(self.config.get("limits.per_user_per_minute", 8))
        per_day = int(self.config.get("limits.per_user_per_day", 200))
        # Личный лимит из админки: NULL — конфиг, 0 — без лимита
        if user.daily_limit is not None:
            per_day = user.daily_limit

        async with session_scope() as session:
            if per_minute:
                recent = await repo.count_recent_queries(session, user.id, minutes=1)
                if recent >= per_minute:
                    raise RateLimited(
                        "Слишком много вопросов подряд. Подождите минуту."
                    )
            if per_day > 0:
                daily = await repo.count_recent_queries(session, user.id)
                if daily >= per_day:
                    raise RateLimited(self.config.message("daily_limit"))

    async def ask(self, user: User, question: str) -> OutgoingMessage:
        """Полный путь без стриминга."""
        return await self.ask_stream(user, question, on_draft=None)

    async def ask_stream(
        self,
        user: User,
        question: str,
        on_draft: Callable[[str], Awaitable[None]] | None = None,
    ) -> OutgoingMessage:
        """Полный путь: история, переписывание, поиск, генерация, журнал."""
        started = time.perf_counter()
        llm.start_usage()
        if not user.is_active:
            raise AccessDenied("Доступ не выдан")
        question = question.strip()

        history_turns = int(self.config.get("generation.history_turns", 4))
        history: list[dict] = []
        conversation_id: int | None = None
        profile: dict = {}
        try:
            async with session_scope() as session:
                conversation = await repo.active_conversation(session, user.id)
                history = await repo.recent_history(
                    session, conversation.id, history_turns
                )
                conversation_id = conversation.id
                profile = await repo.staff_profile(session, user.id)
        except Exception:  # noqa: BLE001
            # База занята индексацией: отвечаем без истории, это лучше отказа
            log.warning("История диалога недоступна, отвечаю без контекста")

        routing_history = history
        history = trim_history(history, int(self.config.get("generation.history_max_tokens", 2200)))
        direct_answer = None
        effective = question
        use_knowledge = True
        require_company_data = bool(self.config.get('generation.work_only', False))
        if self.config.get("generation.work_only", False):
            route = await route_question(question, routing_history, rewrite=bool(self.config.get("generation.rewrite_query", True)))
            scope = route.scope
            effective = route.question
            use_knowledge = route.use_knowledge
            require_company_data = (use_knowledge if route.require_company_data is None else route.require_company_data)
            if scope in {"OFFTOPIC", "SOCIAL"}:
                key = "off_topic" if scope == "OFFTOPIC" else "work_courtesy"
                direct_answer = GeneratedAnswer(self.config.message(key), [], False, 0.0, [], "")

        if direct_answer is None and use_knowledge:
            from app.core.staff_context import apply_profile_scope
            effective = await apply_profile_scope(question, effective, profile)
            from app.retrieval.promotion_catalog import clarify_promotion_city
            clarification = await clarify_promotion_city(effective)
            if clarification:
                direct_answer = GeneratedAnswer(clarification, [], False, 0.0, [], '')

        if not self.config.get("generation.work_only", False) and direct_answer is None and self.config.get("generation.rewrite_query", True) and history and needs_query_rewrite(question):
            effective = await rewrite_query(question, history)
            if effective != question:
                log.info("Вопрос переписан: %r -> %r", question, effective)

        try:
            if direct_answer is None and use_knowledge:
                chunks = await search(effective, advice=True) if not require_company_data else await search(effective)
            else:
                chunks = []
        except Exception:
            if not self.config.get("generation.free_chat", False):
                raise
            log.warning("Поиск временно недоступен; отвечаю без сведений компании", exc_info=True)
            chunks = []

        cumulative_text: list[str] = []

        async def _on_chunk(delta: str) -> None:
            cumulative_text.append(delta)
            if on_draft is not None:
                await on_draft("".join(cumulative_text))

        answer = direct_answer or await build_answer(
            effective,
            chunks,
            history,
            on_chunk=_on_chunk if on_draft is not None else None,
            require_company_data=require_company_data,
            profile_context=profile,
        )
        latency_ms = int((time.perf_counter() - started) * 1000)
        log.info("Question LLM totals: %s", llm.usage_totals())

        query_id: int | None = None
        try:
            async with session_scope() as session:
                record = Query(
                    user_id=user.id,
                    question=question,
                    rewritten=effective if effective != question else None,
                    top_chunk_ids=answer.used_chunk_ids,
                    top_score=answer.top_score,
                    answer=answer.text,
                    is_fallback=answer.is_fallback,
                    latency_ms=latency_ms,
                    # Какая модель реально отвечала: основная или резервная
                    model=answer.model or None,
                )
                session.add(record)
                await session.flush()
                query_id = record.id

                if conversation_id is not None:
                    await repo.add_message(session, conversation_id, "user", question)
                if conversation_id is not None:
                    await repo.add_message(
                        session, conversation_id, "assistant", answer.text
                    )
                if answer.is_fallback:
                    # Журнал неотвеченных — рабочий список на пополнение базы
                    await repo.add_unanswered(
                        session, question, user.id, answer.top_score
                    )
        except Exception:  # noqa: BLE001
            # Ответ уже готов — журнал не повод его потерять. База может
            # быть занята индексацией (SQLite: один писатель за раз).
            log.warning("Не удалось записать запрос в журнал", exc_info=True)

        return OutgoingMessage(
            text=answer.text,
            sources=answer.sources,
            is_fallback=answer.is_fallback,
            query_id=query_id,
        )

    async def new_conversation(self, user: User) -> None:
        async with session_scope() as session:
            await repo.close_conversations(session, user.id)


_service: QuestionService | None = None


def get_service() -> QuestionService:
    global _service
    if _service is None:
        _service = QuestionService()
    return _service


__all__ = [
    "QuestionService",
    "RateLimited",
    "AccessDenied",
    "get_service",
    "llm",
]
