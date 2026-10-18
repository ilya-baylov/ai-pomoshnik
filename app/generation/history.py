"""Ограничение контекста диалога без обращения к модели."""
import tiktoken


def trim_history(history: list[dict], max_tokens: int) -> list[dict]:
    encoding = tiktoken.get_encoding("o200k_base")
    remaining = max(0, max_tokens)
    result = []
    for turn in reversed(history):
        tokens = encoding.encode(str(turn["text"]), disallowed_special=())
        if remaining <= 24:
            break
        if len(tokens) + 12 > remaining:
            size = max(0, remaining - 24)
            text = encoding.decode(tokens[:size]) + "\n[Длинная реплика сокращена]"
            result.append({"role": turn["role"], "text": text})
            break
        result.append(dict(turn))
        remaining -= len(tokens) + 12
    return list(reversed(result))
