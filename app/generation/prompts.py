"""Рендеринг шаблонов промптов из config/prompts/."""

from __future__ import annotations

import functools
from dataclasses import dataclass

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from app.core.config import ROOT


@dataclass(frozen=True)
class ModelPrompt:
    instructions: str
    user_input: str


@functools.lru_cache(maxsize=1)
def _env() -> Environment:
    return Environment(
        loader=FileSystemLoader(ROOT / "config" / "prompts"),
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
        keep_trailing_newline=False,
    )


def render(template_name: str, **context) -> str:
    return _env().get_template(template_name).render(**context).strip()
