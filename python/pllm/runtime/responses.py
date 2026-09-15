from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, AsyncIterator, Iterable, Iterator

from .types import Response, ResponseEvent, response_events


class ResponsesError(ValueError):
    def __init__(self, message: str, *, code: str = "invalid_request_error", param: str | None = None):
        super().__init__(message)
        self.code = code
        self.param = param

    def to_dict(self) -> dict[str, Any]:
        return {"error": {"message": str(self), "type": self.code, "param": self.param, "code": self.code}}


@dataclass(frozen=True, slots=True)
class NormalizedMessage:
    role: str
    text: str


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        raise ResponsesError("message content must be a string or content-item list", param="input")
    chunks: list[str] = []
    for item in content:
        if not isinstance(item, dict):
            raise ResponsesError("content items must be objects", param="input")
        kind = item.get("type")
        if kind in {"input_text", "output_text", "text"}:
            chunks.append(str(item.get("text", "")))
        elif kind in {"input_image", "input_file"}:
            raise ResponsesError(f"{kind} is not yet supported by the HE text runtime", code="unsupported_feature")
        else:
            raise ResponsesError(f"unsupported content item type: {kind!r}", code="unsupported_feature")
    return "".join(chunks)


def normalize_input(value: Any, *, instructions: Any = None) -> list[NormalizedMessage]:
    rows: list[NormalizedMessage] = []
    if instructions:
        if not isinstance(instructions, str):
            raise ResponsesError("Private runtime currently requires string instructions", param="instructions")
        rows.append(NormalizedMessage("system", instructions))
    if isinstance(value, str):
        rows.append(NormalizedMessage("user", value))
        return rows
    if not isinstance(value, list) or not value:
        raise ResponsesError("input must be a non-empty string or list", param="input")
    for item in value:
        if not isinstance(item, dict):
            raise ResponsesError("input items must be objects", param="input")
        kind = item.get("type")
        if kind in {None, "message"}:
            role = str(item.get("role", "user"))
            if role not in {"system", "developer", "user", "assistant", "tool"}:
                raise ResponsesError(f"unsupported role: {role}", param="input")
            rows.append(NormalizedMessage(role, _content_text(item.get("content", ""))))
        elif kind in {"input_text", "text"}:
            rows.append(NormalizedMessage("user", str(item.get("text", ""))))
        else:
            raise ResponsesError(f"unsupported input item type: {kind!r}", code="unsupported_feature")
    return rows


def prompt_text(messages: Iterable[NormalizedMessage]) -> str:
    return "\n".join(f"{row.role}: {row.text}" for row in messages)


def sse_event(event: ResponseEvent) -> bytes:
    payload = json.dumps(event.to_dict(), separators=(",", ":"), ensure_ascii=False)
    return f"event: {event.type}\ndata: {payload}\n\n".encode()


def sse_done() -> bytes:
    return b"data: [DONE]\n\n"


def iter_sse(response: Response, deltas: Iterable[str]) -> Iterator[bytes]:
    for event in response_events(response, deltas):
        yield sse_event(event)
    yield sse_done()


async def aiter_sse(response: Response, deltas: AsyncIterator[str]) -> AsyncIterator[bytes]:
    collected: list[str] = []
    # lifecycle prefix is generated against the final response shape but with empty output
    prefix = list(response_events(response, []))[:4]
    for event in prefix:
        yield sse_event(event)
    seq = 4
    item = response.output[0]
    for delta in [chunk async for chunk in deltas]:
        collected.append(delta)
        yield sse_event(ResponseEvent(
            "response.output_text.delta",
            seq,
            {
                "item_id": item["id"],
                "output_index": 0,
                "content_index": 0,
                "delta": delta,
                "logprobs": [],
            },
        ))
        seq += 1
    # Generate suffix with correct complete text, adjusting sequence numbers.
    suffix = list(response_events(response, collected))[4 + len(collected):]
    for event in suffix:
        event.sequence_number = seq
        seq += 1
        yield sse_event(event)
    yield sse_done()
