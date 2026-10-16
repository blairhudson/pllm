"""Bounded plain-text Responses contexts for the existing benchmark driver."""
import json

BenchmarkContext = str | list[dict[str, str]]

def context_bytes(value: BenchmarkContext) -> bytes:
    if isinstance(value, str):
        return value.encode()
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def validate_context(value) -> BenchmarkContext:
    if isinstance(value, str):
        if not value.strip():
            raise ValueError("prompt must contain 1 to 16384 bytes")
    elif isinstance(value, list) and 1 <= len(value) <= 64:
        for row in value:
            if (not isinstance(row, dict) or set(row) != {"role", "content"}
                    or not isinstance(row["role"], str)
                    or row["role"] not in {"system", "developer", "user", "assistant"}
                    or not isinstance(row["content"], str)):
                raise ValueError("prompt messages require only a text role and content")
        if not any(row["content"].strip() for row in value):
            raise ValueError("prompt messages must contain text")
        value = [dict(row) for row in value]
    else:
        raise ValueError("prompt must be a string or bounded text messages")
    if len(context_bytes(value)) > 16_384:
        raise ValueError("prompt must contain 1 to 16384 bytes")
    return value
