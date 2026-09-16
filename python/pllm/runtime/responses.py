from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, AsyncIterator, Iterable, Iterator

from .types import Response, ResponseEvent, response_events


class ResponsesError(ValueError):
    def __init__(
        self,
        message: str,
        *,
        code: str = "invalid_request_error",
        param: str | None = None,
        status_code: int = 400,
    ):
        super().__init__(message)
        self.code = code
        self.param = param
        self.status_code = status_code

    def to_dict(self) -> dict[str, Any]:
        return {
            "error": {
                "message": str(self),
                "type": self.code,
                "param": self.param,
                "code": self.code,
            }
        }


@dataclass(frozen=True, slots=True)
class NormalizedMessage:
    role: str
    text: str
    tool_calls: tuple[dict[str, Any], ...] = ()
    tool_call_id: str | None = None

    def to_prompt_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = {"role": self.role, "content": self.text}
        if self.tool_calls:
            value["tool_calls"] = list(self.tool_calls)
        if self.tool_call_id is not None:
            value["tool_call_id"] = self.tool_call_id
        return value


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
        elif kind == "input_image":
            raise ResponsesError("input_image is unsupported by this text model")
        elif kind == "input_file":
            raise ResponsesError("input_file is unsupported by this text model")
        else:
            raise ResponsesError(
                f"unsupported content item type: {kind!r}", code="unsupported_feature"
            )
    return "".join(chunks)


def normalize_input(value: Any, *, instructions: Any = None) -> list[NormalizedMessage]:
    rows: list[NormalizedMessage] = []
    if instructions:
        if not isinstance(instructions, str):
            raise ResponsesError(
                "Private runtime currently requires string instructions", param="instructions"
            )
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
        elif kind == "function_call":
            name = item.get("name")
            call_id = item.get("call_id")
            arguments = item.get("arguments")
            if not isinstance(name, str) or not name:
                raise ResponsesError("function_call name is required", param="input")
            if not isinstance(call_id, str) or not call_id:
                raise ResponsesError("function_call call_id is required", param="input")
            if not isinstance(arguments, str):
                raise ResponsesError("function_call arguments must be a JSON string", param="input")
            try:
                parsed_arguments = json.loads(arguments)
            except json.JSONDecodeError as exc:
                raise ResponsesError(
                    "function_call arguments must contain JSON", param="input"
                ) from exc
            call = {
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": parsed_arguments},
            }
            if rows and rows[-1].role == "assistant" and rows[-1].tool_calls:
                previous = rows.pop()
                rows.append(
                    NormalizedMessage(
                        "assistant",
                        previous.text,
                        previous.tool_calls + (call,),
                    )
                )
            else:
                rows.append(NormalizedMessage("assistant", "", (call,)))
        elif kind == "function_call_output":
            call_id = item.get("call_id")
            if not isinstance(call_id, str) or not call_id:
                raise ResponsesError("function_call_output call_id is required", param="input")
            output = item.get("output")
            if not isinstance(output, str):
                output = json.dumps(output, separators=(",", ":"), ensure_ascii=False)
            rows.append(NormalizedMessage("tool", output, tool_call_id=call_id))
        elif kind in {"input_text", "text"}:
            rows.append(NormalizedMessage("user", str(item.get("text", ""))))
        elif kind == "compaction":
            continue
        else:
            raise ResponsesError(
                f"unsupported input item type: {kind!r}", code="unsupported_feature"
            )
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
        yield sse_event(
            ResponseEvent(
                "response.output_text.delta",
                seq,
                {
                    "item_id": item["id"],
                    "output_index": 0,
                    "content_index": 0,
                    "delta": delta,
                    "logprobs": [],
                },
            )
        )
        seq += 1
    # Generate suffix with correct complete text, adjusting sequence numbers.
    suffix = list(response_events(response, collected))[4 + len(collected) :]
    for event in suffix:
        event.sequence_number = seq
        seq += 1
        yield sse_event(event)
    yield sse_done()
