"""Клавиатуры. Подписи короткие, эмодзи только там, где несут смысл.

Постоянного меню у сотрудника нет: команды /help и /new живут в нативном
меню Telegram. Кнопки оценки появляются только при замечании сотрудника.
"""

from __future__ import annotations

from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup
from aiogram.utils.keyboard import InlineKeyboardBuilder


def back(target: str = "adm:home") -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[[InlineKeyboardButton(text="‹ Назад", callback_data=target, style="primary")]]
    )


def answer_feedback(query_id: int) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[[
        InlineKeyboardButton(text="Неверно", callback_data=f"fb:wrong:{query_id}"),
        InlineKeyboardButton(text="Не хватает информации", callback_data=f"fb:incomplete:{query_id}"),
    ]])


def profile_menu() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="Город", callback_data="profile:edit:city"),
         InlineKeyboardButton(text="Филиал", callback_data="profile:edit:branch")],
        [InlineKeyboardButton(text="Должность", callback_data="profile:edit:job_title")],
        [InlineKeyboardButton(text="Очистить профиль", callback_data="profile:clear")],
    ])


def profile_cities() -> InlineKeyboardMarkup:
    from app.core.staff_context import CITIES
    buttons = [InlineKeyboardButton(text=city, callback_data=f"profile:city:{i}") for i, city in enumerate(CITIES)]
    return InlineKeyboardMarkup(inline_keyboard=[buttons[i:i+2] for i in range(0, len(buttons), 2)])


def confirm(
    action: str, label: str = "Подтвердить", style: str = "danger"
) -> InlineKeyboardMarkup:
    builder = InlineKeyboardBuilder()
    builder.button(text=label, callback_data=action, style=style)
    builder.button(text="Отмена", callback_data="adm:knowledge")
    builder.adjust(2)
    return builder.as_markup()
