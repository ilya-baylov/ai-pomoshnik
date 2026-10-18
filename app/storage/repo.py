"""Репозитории: доступ к данным без SQL в бизнес-логике."""

from __future__ import annotations

import secrets
from datetime import timedelta

from sqlalchemy import func, select, update, or_, text as sql_text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.contracts import Role
from app.storage.models import (
    Conversation,
    Document,
    Feedback,
    Invite,
    Message,
    Query,
    Unanswered,
    User,
    StaffProfile,
    FeedbackCase,
    utcnow,
)


async def get_or_create_user(
    session: AsyncSession,
    channel: str,
    external_id: str,
    username: str | None = None,
    full_name: str | None = None,
    default_role: str | None = None,
    allow_registration: bool = False,
) -> User:
    user = (
        await session.execute(
            select(User).where(User.channel == channel, User.external_id == external_id)
        )
    ).scalar_one_or_none()

    if user is not None:
        # Имя могло поменяться — держим свежим для админки
        if username and user.username != username:
            user.username = username
        if full_name and user.full_name != full_name:
            user.full_name = full_name
        return user

    user = User(
        channel=channel,
        external_id=external_id,
        username=username,
        full_name=full_name,
        role=default_role or (Role.STAFF.value if allow_registration else Role.PENDING.value),
        is_active=bool(default_role) or allow_registration,
    )
    session.add(user)
    try:
        await session.flush()
    except IntegrityError:
        # Два сообщения одного пользователя пришли одновременно: запись
        # уже создал параллельный поток — откатываем и читаем её
        await session.rollback()
        user = (
            await session.execute(
                select(User).where(
                    User.channel == channel, User.external_id == external_id
                )
            )
        ).scalar_one()
    return user


async def set_role(
    session: AsyncSession, channel: str, external_id: str, role: str
) -> User:
    user = await get_or_create_user(session, channel, external_id, default_role=role)
    user.role = role
    user.is_active = True
    return user


async def list_users(session: AsyncSession) -> list[User]:
    return list(
        (await session.execute(select(User).order_by(User.created_at.desc()))).scalars()
    )


async def create_invite(session: AsyncSession, created_by: int | None) -> Invite:
    invite = Invite(code=secrets.token_hex(8).upper(), created_by=created_by)
    session.add(invite)
    await session.flush()
    return invite


async def redeem_invite(session: AsyncSession, code: str, user: User) -> bool:
    if user.role != Role.PENDING.value:
        return False  # Код не снимает блокировку, установленную администратором.
    result = await session.execute(
            update(Invite).where(
                Invite.code == code.strip().upper(), Invite.used_by.is_(None)
            ).values(used_by=user.id, used_at=utcnow())
    )
    if result.rowcount != 1:
        return False
    user.is_active = True
    user.role = Role.STAFF.value
    return True


async def active_conversation(session: AsyncSession, user_id: int) -> Conversation:
    conversation = (
        await session.execute(
            select(Conversation)
            .where(Conversation.user_id == user_id, Conversation.closed_at.is_(None))
            .order_by(Conversation.started_at.desc())
            .limit(1)
        )
    ).scalar_one_or_none()
    if conversation is None:
        conversation = Conversation(user_id=user_id)
        session.add(conversation)
        await session.flush()
    return conversation


async def close_conversations(session: AsyncSession, user_id: int) -> None:
    conversations = (
        await session.execute(
            select(Conversation).where(
                Conversation.user_id == user_id, Conversation.closed_at.is_(None)
            )
        )
    ).scalars()
    for conversation in conversations:
        conversation.closed_at = utcnow()


async def add_message(
    session: AsyncSession, conversation_id: int, role: str, text: str
) -> None:
    session.add(Message(conversation_id=conversation_id, role=role, text=text))


async def recent_history(
    session: AsyncSession, conversation_id: int, turns: int
) -> list[dict]:
    if turns <= 0:
        return []
    rows = (
        await session.execute(
            select(Message.role, Message.text)
            .where(Message.conversation_id == conversation_id)
            .order_by(Message.created_at.desc(), Message.id.desc())
            .limit(turns * 2)
        )
    ).all()
    return [
        {"role": "Сотрудник" if role == "user" else "Ассистент", "text": text}
        for role, text in reversed(rows)
    ]


async def count_recent_queries(
    session: AsyncSession, user_id: int, minutes: int | None = None
) -> int:
    stmt = select(func.count(Query.id)).where(Query.user_id == user_id)
    if minutes:
        stmt = stmt.where(Query.created_at >= utcnow() - timedelta(minutes=minutes))
    else:
        stmt = stmt.where(Query.created_at >= utcnow() - timedelta(days=1))
    return int((await session.execute(stmt)).scalar() or 0)


async def add_unanswered(
    session: AsyncSession, question: str, user_id: int | None, top_score: float
) -> Unanswered:
    record = Unanswered(question=question, user_id=user_id, top_score=top_score)
    session.add(record)
    await session.flush()
    return record


async def list_unanswered(
    session: AsyncSession, status: str = "new", limit: int = 10
) -> list[Unanswered]:
    return list(
        (
            await session.execute(
                select(Unanswered)
                .where(Unanswered.status == status)
                .order_by(Unanswered.created_at.desc())
                .limit(limit)
            )
        ).scalars()
    )


async def resolve_unanswered(
    session: AsyncSession, record_id: int, user_id: int
) -> bool:
    record = await session.get(Unanswered, record_id)
    if record is None:
        return False
    record.status = "resolved"
    record.resolved_by = user_id
    record.resolved_at = utcnow()
    return True


async def add_feedback(
    session: AsyncSession, query_id: int, rating: int, comment: str | None = None
) -> Feedback | None:
    existing = (
        await session.execute(select(Feedback).where(Feedback.query_id == query_id))
    ).scalar_one_or_none()
    if existing is not None:
        existing.rating = rating
        if comment is not None:
            existing.comment = comment
        return existing
    feedback = Feedback(query_id=query_id, rating=rating, comment=comment)
    session.add(feedback)
    await session.flush()
    return feedback


async def latest_feedback_query(session: AsyncSession, user_id: int) -> Query | None:
    conversation = (await session.execute(select(Conversation).where(
        Conversation.user_id == user_id, Conversation.closed_at.is_(None))
        .order_by(Conversation.started_at.desc()).limit(1))).scalar_one_or_none()
    if conversation is None:
        return None
    # SQLite returns naive UTC values; freshly created ORM rows may still be aware.
    since = max(conversation.started_at.replace(tzinfo=None),
                (utcnow() - timedelta(hours=6)).replace(tzinfo=None))
    return (await session.execute(select(Query).where(Query.user_id == user_id,
        Query.answer.is_not(None), Query.created_at >= since)
        .order_by(Query.created_at.desc(), Query.id.desc()).limit(1))).scalar_one_or_none()


async def submit_feedback(session: AsyncSession, query_id: int, user_id: int, kind: str, comment: str | None = None) -> bool:
    reasons = {"wrong": "Неверно", "incomplete": "Не хватает информации"}
    if kind not in reasons:
        return False
    query = await session.get(Query, query_id)
    if query is None or query.user_id != user_id:
        return False
    detail = reasons[kind] + (f'\nСотрудник: {comment[:2000]}' if comment else '')
    await add_feedback(session, query_id, -1, comment=detail)
    case = await session.get(FeedbackCase, query_id)
    if case is None:
        session.add(FeedbackCase(query_id=query_id, status="new"))
    else:
        case.status = "new"
        case.resolved_by = case.resolved_at = None
    return True


async def feedback_cases(session: AsyncSession, limit: int = 4, offset: int = 0) -> list:
    return list((await session.execute(select(Query, Feedback)
        .join(Feedback, Feedback.query_id == Query.id)
        .outerjoin(FeedbackCase, FeedbackCase.query_id == Query.id)
        .where(Feedback.rating < 0, or_(FeedbackCase.status.is_(None), FeedbackCase.status == "new"))
        .order_by(Feedback.created_at.desc(), Feedback.id.desc()).limit(limit).offset(offset))).all())


async def resolve_feedback_case(session: AsyncSession, query_id: int, user_id: int) -> bool:
    if await session.get(Query, query_id) is None:
        return False
    case = await session.get(FeedbackCase, query_id)
    if case is None:
        case = FeedbackCase(query_id=query_id)
        session.add(case)
    case.status = "resolved"
    case.resolved_by = user_id
    case.resolved_at = utcnow()
    return True


async def staff_profile(session: AsyncSession, user_id: int) -> dict:
    profile = await session.get(StaffProfile, user_id)
    return {"city": profile.city, "branch": profile.branch, "job_title": profile.job_title} if profile else {}


async def update_staff_profile(session: AsyncSession, user_id: int, field: str, value: str | None) -> dict:
    if field not in {"city", "branch", "job_title"}:
        raise ValueError("Неизвестное поле профиля")
    limits = {"city": 100, "branch": 160, "job_title": 100}
    if value is not None:
        value = " ".join(value.split()).strip()
        if not value or len(value) > limits[field]:
            raise ValueError("Значение пустое или слишком длинное")
    profile = await session.get(StaffProfile, user_id)
    if profile is None:
        profile = StaffProfile(user_id=user_id)
        session.add(profile)
    if field == "city" and profile.city != value:
        profile.branch = None
    setattr(profile, field, value)
    profile.updated_at = utcnow()
    await session.flush()
    return await staff_profile(session, user_id)


async def negative_feedback(session: AsyncSession, limit: int = 10) -> list[tuple]:
    return list(
        (
            await session.execute(
                select(Query.question, Query.answer, Feedback.created_at)
                .join(Feedback, Feedback.query_id == Query.id)
                .where(Feedback.rating < 0)
                .order_by(Feedback.created_at.desc())
                .limit(limit)
            )
        ).all()
    )


async def stats(session: AsyncSession, days: int = 7) -> dict:
    """Сводка для админки."""
    since = utcnow() - timedelta(days=days)

    async def scalar(stmt) -> int:
        return int((await session.execute(stmt)).scalar() or 0)

    total = await scalar(select(func.count(Query.id)).where(Query.created_at >= since))
    fallbacks = await scalar(
        select(func.count(Query.id)).where(
            Query.created_at >= since, Query.is_fallback.is_(True)
        )
    )
    avg_latency = (
        await session.execute(
            select(func.avg(Query.latency_ms)).where(Query.created_at >= since)
        )
    ).scalar()

    return {
        "days": days,
        "queries": total,
        "fallbacks": fallbacks,
        "fallback_rate": round(fallbacks / total * 100) if total else 0,
        "avg_latency_ms": int(avg_latency or 0),
        "users": await scalar(
            select(func.count(func.distinct(Query.user_id))).where(
                Query.created_at >= since
            )
        ),
        "documents": await scalar(select(func.count(Document.id))),
        "chunks": await scalar(sql_text("SELECT COUNT(*) FROM chunks")),
        "unanswered_new": await scalar(
            select(func.count(Unanswered.id)).where(Unanswered.status == "new")
        ),
        "negative": await scalar(
            select(func.count(Feedback.id)).outerjoin(FeedbackCase, FeedbackCase.query_id == Feedback.query_id)
            .where(Feedback.rating < 0, or_(FeedbackCase.status.is_(None), FeedbackCase.status == "new"))
        ),
        "stale_docs": await scalar(
            select(func.count(Document.id)).where(
                Document.valid_until.is_not(None),
                Document.valid_until < utcnow().date(),
            )
        ),
        "active_users": await scalar(
            select(func.count(User.id)).where(User.is_active.is_(True))
        ),
    }
