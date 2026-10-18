"""Экраны админки: тексты и клавиатуры.

Дашборд живёт в одном сообщении и перерисовывается кнопками — чат не
засоряется. Логика в admin.py, здесь только представление.
"""

from __future__ import annotations

import html

from aiogram.types import InlineKeyboardMarkup
from aiogram.utils.keyboard import InlineKeyboardBuilder

from app.core.config import get_bot_config

# Тихие символы статуса вместо кричащих эмодзи
OK = "●"
OFF = "○"
WARN = "⚠"


def _kb(rows: list[list[tuple]]) -> InlineKeyboardMarkup:
    builder = InlineKeyboardBuilder()
    for row in rows:
        for item in row:
            if len(item) >= 3 and item[2] is not None:
                builder.button(text=item[0], callback_data=item[1], style=item[2])
            else:
                builder.button(text=item[0], callback_data=item[1])
    builder.adjust(*[len(row) for row in rows])
    return builder.as_markup()


def home(data: dict) -> tuple[str, InlineKeyboardMarkup]:
    """Главный экран. Сверху то, что требует внимания."""
    tasks = data.get('review_pending', 0) + data['unanswered_new'] + data['negative']
    text = ("<b>Управление компании</b>\n\n"
            f"База: {data['documents']} документов · задачи: {tasks}\n\n"
            "<b>База</b> — читать, добавлять и исправлять.\n"
            "<b>Задачи</b> — вопросы, замечания и спорные данные.\n"
            "<b>Работа бота</b> — статусы, обновления и статистика.\n"
            "<b>Сотрудники</b> — доступ и лимиты.")
    keyboard = _kb([
        [("База знаний", "lib:root"), (f"Задачи · {tasks}", "lib:tasks")],
        [("Работа бота", "lib:work"), ("Сотрудники", "adm:users")],
        [("↻ Обновить", "adm:home")],
    ])
    return text, keyboard


def knowledge(
    data: dict,
    sections: list[tuple[str, int, int]],
    stale_titles: list[str] | None = None,
) -> tuple[str, InlineKeyboardMarkup]:
    lines = [
        "<b>База знаний</b>",
        "",
        f"документов: {data['documents']}",
        f"фрагментов: {data['chunks']}",
    ]
    if data["last_ingest"]:
        lines.append(f"последняя индексация: {data['last_ingest']}")
    if data["stale_docs"]:
        lines.append(f"{WARN} просрочено: {data['stale_docs']}")
    if stale_titles:
        lines += ["", f"{WARN} <b>Данные в названии уже прошли:</b>"]
        lines.extend(f"• {html.escape(t)}" for t in stale_titles[:6])
        if len(stale_titles) > 6:
            lines.append(f"• … ещё {len(stale_titles) - 6}")
        lines.append("Проверьте условия и обновите сведения. Новое имя файла само по себе не подтверждает актуальность.")

    if sections:
        lines += ["", "<b>Разделы</b>"]
        for section, docs, chunks in sections:
            lines.append(f"• {html.escape(section or 'без раздела')} — {docs} док., {chunks} фрагм.")

    lines += [
        "",
        "Чтобы добавить документ, пришлите файл в этот чат.",
    ]

    keyboard = _kb(
        [
            [("Проверить изменения", "adm:ingest", "primary")],
            [("Переиндексировать всё", "adm:reindex_confirm", "danger")],
            [("‹ Назад", "adm:home", "primary")],
        ]
    )
    return "\n".join(lines), keyboard


def reindex_confirm() -> tuple[str, InlineKeyboardMarkup]:
    text = (
        "<b>Полная переиндексация</b>\n\n"
        "Все документы будут разобраны и векторизованы заново, "
        "кэш игнорируется. На платном провайдере это расходует запросы.\n\n"
        "Обычная проверка изменений быстрее и почти всегда достаточна."
    )
    keyboard = _kb([[("Переиндексировать", "adm:reindex_full", "danger"), ("Отмена", "adm:knowledge")]])
    return text, keyboard


def unanswered(items: list) -> tuple[str, InlineKeyboardMarkup]:
    if not items:
        return (
            "<b>Вопросы без ответа</b>\n\nНовых вопросов в списке нет.",
            _kb([[("‹ Назад", "adm:home", "primary")]]),
        )

    lines = ["<b>Вопросы без ответа</b>", ""]
    rows: list[list[tuple]] = []
    for position, item in enumerate(items, start=1):
        when = item.created_at.strftime("%d.%m %H:%M")
        lines.append(f"{position}. {html.escape(item.question[:150])}")
        lines.append(f"    <i>{when}, близость {item.top_score:.2f}</i>")
        rows.append([(f"Решён {position}", f"adm:resolve:{item.id}", "success")])

    lines += ["", "Добавьте ответ в базу и отметьте вопрос решённым."]
    rows.append([("Добавить ответ в базу", "lib:add"), ("Найти документ", "lib:search")])
    rows.append([("‹ Назад", "adm:home", "primary")])
    return "\n".join(lines), _kb(rows)


def stats(data: dict) -> tuple[str, InlineKeyboardMarkup]:
    lines = [
        f"<b>Статистика за {data['days']} дн.</b>",
        "",
        f"вопросов: {data['queries']}",
        f"сотрудников: {data['users']}",
        f"средний отклик: {data['avg_latency_ms']} мс",
        "",
        f"без ответа: {data['fallbacks']} ({data['fallback_rate']}%)",
        f"оценок «не помогло»: {data['negative']}",
    ]
    if data["fallback_rate"] > 30:
        lines += [
            "",
            f"{WARN} Проверьте вопросы без ответа, актуальность данных и выбор источников.",
        ]

    keyboard = _kb(
        [
            [
                ("День", "adm:stats:1", "primary" if data["days"] == 1 else None),
                ("Неделя", "adm:stats:7", "primary" if data["days"] == 7 else None),
                ("Месяц", "adm:stats:30", "primary" if data["days"] == 30 else None),
            ],
            [("‹ Назад", "adm:home", "primary")],
        ]
    )
    return "\n".join(lines), keyboard


def users(items: list) -> tuple[str, InlineKeyboardMarkup]:
    admins = [u for u in items if u.role == "admin"]
    staff = [u for u in items if u.role != "admin" and u.is_active]

    lines = ["<b>Пользователи</b>", ""]
    lines.append(f"<b>Администраторы ({len(admins)})</b> — без лимитов")
    lines.extend(f"• {_name(u)}" for u in admins[:10])
    lines.append("")
    lines.append(f"<b>Пользователи ({len(staff)})</b> — лимит из конфига или личный")
    if staff:
        lines.append("Нажми на пользователя, чтобы настроить доступ:")
        keyboard_rows = [
            [(f"{_name(u)[:28]}", f"adm:user:{u.id}")] for u in staff[:20]
        ]
    else:
        lines.append("— пока никого")
        keyboard_rows = []
    keyboard_rows.append([("‹ Назад", "adm:home", "primary")])
    return "\n".join(lines), _kb(keyboard_rows)


def user_card(user, used_today: int) -> tuple[str, InlineKeyboardMarkup]:
    """Карточка пользователя: доступ, лимит, кнопки управления."""
    config = get_bot_config()
    cfg_limit = int(config.get("limits.per_user_per_day", 200))
    if user.daily_limit is not None:
        limit_text = f"{user.daily_limit}/сутки (личный)" if user.daily_limit else "без лимита"
    else:
        limit_text = f"{cfg_limit}/сутки (из конфига)"

    role_text = "администратор — без лимитов" if user.role == "admin" else limit_text
    lines = [
        "<b>" + _name(user) + "</b>",
        "",
        f"Доступ: {'✅ открыт' if user.is_active else '❌ закрыт'}",
        f"Роль: {role_text}",
        f"Сегодня вопросов: {used_today}",
        f"Telegram ID: <code>{user.external_id}</code>",
    ]
    rows = []
    if user.role != "admin":
        rows.append([("Сделать админом", f"adm:urole:{user.id}:admin", "primary")])
    else:
        rows.append([("Снять админа", f"adm:urole:{user.id}:staff", "danger")])
    if user.is_active:
        rows.append([("Закрыть доступ", f"adm:uactive:{user.id}:0", "danger")])
    else:
        rows.append([("Открыть доступ", f"adm:uactive:{user.id}:1", "success")])
    # Личный лимит: NULL -> конфиг; варианты 0 / 50 / 100 / свой
    if user.daily_limit is None:
        rows.append([("Личный лимит: сейчас из конфига", f"adm:ulimit:{user.id}")])
    else:
        rows.append([
            (f"Личный лимит: {user.daily_limit or '∞'}", f"adm:ulimit:{user.id}"),
        ])
    rows.append([("‹ К списку", "adm:users", "primary")])
    return "\n".join(lines), _kb(rows)


def user_limit_menu(user) -> tuple[str, InlineKeyboardMarkup]:
    """Выбор личного суточного лимита."""
    current = "из конфига" if user.daily_limit is None else (user.daily_limit or "∞")
    lines = [
        "<b>Личный лимит запросов</b>",
        "",
        f"Сейчас: <b>{current}</b>",
        "",
        "0 — без ограничений. «Из конфига» — действует лимит",
        f"Общий лимит сейчас: {get_bot_config().get('limits.per_user_per_day', 200)} вопросов в сутки.",
        "Свой вариант — пришли числом в чат.",
    ]
    rows = [
        [("Из конфига", f"adm:ulimit:{user.id}:cfg"), ("Без лимита", f"adm:ulimit:{user.id}:0")],
        [("50", f"adm:ulimit:{user.id}:50"), ("100", f"adm:ulimit:{user.id}:100")],
        [("✏️ Свой вариант", f"adm:ulimit:{user.id}:input")],
        [("‹ К карточке", f"adm:user:{user.id}", "primary")],
    ]
    return "\n".join(lines), _kb(rows)


def health(checks: list[tuple[str, bool, str]]) -> tuple[str, InlineKeyboardMarkup]:
    lines = ["<b>Проверка системы</b>", ""]
    for label, ok, detail in checks:
        mark = OK if ok else WARN
        lines.append(f"{mark} {label}: {html.escape(detail)}")
    return "\n".join(lines), _kb([[("Проверить снова", "adm:health", "primary")], [("‹ Назад", "adm:home", "primary")]])


def _name(user) -> str:
    parts = []
    if user.full_name:
        parts.append(html.escape(user.full_name))
    if user.username:
        parts.append(f"@{html.escape(user.username)}")
    if not parts:
        parts.append(f"id{user.external_id}")
    return " ".join(parts)
