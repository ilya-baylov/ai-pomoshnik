"""Явно сохранённый профиль даёт город по умолчанию, не меняет права."""
import re
from app.retrieval.search import sections_from_query

CITIES = ("Воронеж", "Дубай", "Екатеринбург", "Казань", "Красноярск", "Москва",
          "Новокузнецк", "Новосибирск", "Ростов", "Санкт-Петербург", "Сочи", "Стамбул")

async def apply_profile_scope(question: str, effective: str, profile: dict) -> str:
    city = profile.get("city")
    if city not in CITIES or await sections_from_query(effective):
        return effective
    text = question + " " + effective
    from app.retrieval.book_catalog import named_books
    if named_books(effective):
        return effective
    if re.search(r'\b(?:в|во|из)\s+[А-ЯЁ][а-яё]+',effective) or re.search(r'\b(?:в|во)\s+[а-яё]+\s+(?:кто|делает|работает)',effective,re.I):
        return effective
    if re.search(r'\b(?:вс[еёехми]+\s+город|разн\w*\s+город|города\b|друг\w*\s+город)', text, re.I):
        return effective
    # Не подменять явно названный неизвестный город городом профиля.
    if re.search(r'\b(?:в\s+городе|по\s+городу|город\s*:|акци\w*\s+(?:в|по)|'
                 r'мастер\w*\s+в|салон\w*\s+в|цен\w*\s+в|прайс\w*\s+в|телефон\w*\s+в|контакт\w*\s+в|адрес\w*\s+в)',
                 re.sub(r'по\s+(?:указанному\s+охвату|всем\s+филиалам|нашему\s+городу)', '', text, flags=re.I), re.I):
        return effective
    if not re.search(r'акци|мастер|расписан|график|телефон|контакт|адрес|филиал|цен[аыуе]|прайс|материал|оборудован|обязан|должност|ламинир|наращ|депиляц|маникюр|педикюр|массаж|ресниц|бров', effective, re.I):
        return effective
    scoped = f"{effective}\nГород сотрудника: {city}."
    if profile.get("branch") and not re.search(r'филиал|салон|улиц|\bул\b|\bна\s', question, re.I):
        scoped += f" Филиал сотрудника: {profile['branch']}."
    return scoped
