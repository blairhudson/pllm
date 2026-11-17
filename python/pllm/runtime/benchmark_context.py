"""Bounded plain-text Responses contexts for the existing benchmark driver."""
import json

BenchmarkContext = str | list[dict[str, str]]

def context_bytes(value: BenchmarkContext) -> bytes:
    if isinstance(value, str):
        return value.encode()
    if isinstance(value, dict):
        from .prompt_input import _OwnedIds, _token_shape
        _token_shape(value)
        ids = value["ids"]
        if (type(ids) is not _OwnedIds or not 1 <= len(ids) <= 32
                or any(type(item) is not int or not 0 <= item < 2**32 for item in ids)):
            raise ValueError("token context bytes require bounded owned IDs")
        # Invalid UTF-8 prefix cannot collide with ANY legacy text/message bytes.
        return b"\xffpllm.benchmark.token_ids.v1\0" + json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def validate_context(value, *, budget=None, output_tokens=None, requests=1) -> BenchmarkContext:
    from .prompt_input import admit_input, is_token_input
    if is_token_input(value):
        return admit_input(value, budget, output_tokens=output_tokens, requests=requests)
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
