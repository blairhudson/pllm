from __future__ import annotations

import json
import copy
import threading
from contextvars import ContextVar
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


def _render_prompt(messages, descriptor, *, template=None,
                   add_generation_prompt=True, tools=None):
    """Weights-free ordinary admission renderer; a broken template refuses."""
    rows = []
    for item in messages:
        row = dict(item.to_prompt_dict()) if hasattr(item, "to_prompt_dict") else (
            dict(item) if isinstance(item, dict) else {"role": "user", "content": str(item)})
        row["role"] = "system" if row.get("role") == "developer" else str(row.get("role"))
        row["content"] = str(row.get("content", ""))
        rows.append(row)
    selected = descriptor.get("chat_template")
    if selected is None:
        selected = template
    if selected is None:
        suffix = "\nassistant:" if add_generation_prompt else ""
        return "\n".join(f"{row['role']}: {row['content']}" for row in rows) + suffix
    try:
        from jinja2 import StrictUndefined
        from jinja2.sandbox import SandboxedEnvironment

        environment = SandboxedEnvironment(
            undefined=StrictUndefined, trim_blocks=True, lstrip_blocks=True)

        def raise_exception(message):
            raise ResponsesError(str(message), param="input")

        environment.globals["raise_exception"] = raise_exception
        return str(environment.from_string(str(selected)).render(
            messages=rows, add_generation_prompt=bool(add_generation_prompt),
            bos_token=descriptor.get("bos_token", ""),
            eos_token=descriptor.get("eos_token", ""), tools=tools))
    except Exception as error:
        raise ResponsesError("source chat template failed", param="input") from error


def _prompt_snapshot(value):
    """Retain immutable values, including original tokenizer bytes, not a digest."""
    if type(value) is dict:
        return ("dict", tuple((key, _prompt_snapshot(item)) for key, item in sorted(value.items())))
    if type(value) in (list, tuple):
        return (type(value).__name__, tuple(_prompt_snapshot(item) for item in value))
    if value is None or type(value) in (str, bytes, bool, int, float):
        return (type(value).__name__, value)
    raise ResponsesError("unsupported prompt association metadata")


def _prompt_request(body):
    from .http_gateway import validate_response_body

    # Match existing delegated fixed-greedy defaults, without modifying caller data.
    body = json.loads(json.dumps(body, allow_nan=False))
    body = validate_response_body(body)
    for name, default in (("temperature", 0), ("top_p", 1)):
        if body.get(name) is None:
            body[name] = default
    return _prompt_snapshot(body)


def _prompt_source(model, config, descriptor, template=None, source_lock_digest=None):
    selected = descriptor.get("chat_template")
    if selected is None:
        selected = template
    return _prompt_snapshot((model.to_spec(), config, descriptor,
                             selected, source_lock_digest))


def _owned_prompt_source(source):
    """Only already-owned metadata; never source.source/resolve_model or loaders."""
    return _prompt_source(source.model, json.loads(source.config_source_bytes),
                          source.tokenizer_descriptor, source_lock_digest=source.source_lock_digest)


@dataclass(frozen=True, slots=True)
class _PreparedPrompt:
    request: tuple
    source: tuple
    messages: tuple
    rendered: str
    ids: tuple[int, ...]
    tokenizer: Any
    pending_source: Any = None


class _PreparedPromptUse:
    """Private one-attempt seat, separate from immutable prepared data."""

    def __init__(self, payload, *, _metadata_owner=None):
        self._payload = payload
        self._used = False
        self._binding_started = False
        self._metadata_owner = _metadata_owner
        self._claimed = False
        self._lock = threading.Lock()

    def claim(self):
        with self._lock:
            if self._claimed or self._payload is None:
                raise ResponsesError("prepared prompt already attached or retired")
            self._claimed = True
        return self

    def close_unclaimed(self):
        if not self._claimed:
            self.close()

    def _bind_owned_source(self, body, source):
        """Bind provisional metadata ONCE, only after caller's real source admission.

        This does not admit a source. _produce's original _ensure_source remains
        the source/resource authority and runs after prompt length validation.
        """
        from dataclasses import replace
        with self._lock:
            if self._payload is None or self._used:
                raise ResponsesError("prepared prompt already consumed or retired")
            pending = self._payload.pending_source
            if pending is None:
                return  # Fully bound retained-owner path, checked by consume.
            if self._binding_started:
                raise ResponsesError("prepared prompt binding already attempted")
            self._binding_started = True
            try:
                if self._payload.request != _prompt_request(body):
                    raise ResponsesError("prepared request association changed")
                pending.check_source(source)
                self._payload = replace(self._payload,
                    source=_owned_prompt_source(source), pending_source=None)
            except BaseException:
                self._used = True  # Failure/unwind cannot retry a different source.
                raise
    def consume(self, body, source):
        with self._lock:
            if self._payload is None or self._used:
                raise ResponsesError("prepared prompt already consumed or retired")
            self._used = True  # Mismatch burns this attempt too.
            if self._payload.pending_source is not None:
                raise ResponsesError("prepared prompt still needs original source admission")
            payload = self._payload
            if payload.request != _prompt_request(body) or payload.source != source:
                raise ResponsesError("prepared prompt request/source association changed")
            return payload.tokenizer, payload.ids

    def close(self):
        with self._lock:
            if self._payload is not None:
                close = getattr(self._payload.tokenizer, "close", None)
                if close is not None:
                    close()  # Failed close retains custody for an explicit retry.
                self._payload = None
            if self._metadata_owner is not None:
                self._metadata_owner.close()
                self._metadata_owner = None


_prepared_prompt_request = ContextVar("_prepared_prompt_request", default=None)


def _prepare_prompt(body, *, descriptor, tokenizer, source, messages=None,
                    template=None, tools=None, _pending_source=None, _metadata_owner=None):
    body = copy.deepcopy(body)
    descriptor = copy.deepcopy(descriptor)
    messages = copy.deepcopy(messages)
    tools = copy.deepcopy(tools)
    request = _prompt_request(body)
    messages = normalize_input(body["input"], instructions=body.get("instructions")) if messages is None else messages
    rows = [item.to_prompt_dict() if hasattr(item, "to_prompt_dict") else dict(item) for item in messages]
    normalized = _prompt_snapshot(rows)
    rendered = _render_prompt(messages, descriptor, template=template, tools=tools)
    ids = tuple(tokenizer.encode(rendered, add_bos=bool(descriptor.get("add_bos_token", True))))
    if not ids or any(type(item) is not int or item < 0 for item in ids):
        raise ResponsesError("tokenized input contains invalid prompt IDs", param="input")
    return _PreparedPromptUse(_PreparedPrompt(
        request, source, normalized, rendered, ids, tokenizer, _pending_source),
        _metadata_owner=_metadata_owner)


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
