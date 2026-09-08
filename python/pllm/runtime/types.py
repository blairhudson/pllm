from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Iterable, Iterator, Literal


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


@dataclass(slots=True)
class ResponseUsage:
    input_tokens: int
    output_tokens: int
    total_tokens: int
    input_tokens_details: dict[str, int] = field(default_factory=lambda: {"cached_tokens": 0})
    output_tokens_details: dict[str, int] = field(default_factory=lambda: {"reasoning_tokens": 0})

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "input_tokens_details": self.input_tokens_details,
            "output_tokens_details": self.output_tokens_details,
        }


@dataclass(slots=True)
class Response:
    id: str
    model: str
    output: list[dict[str, Any]]
    created_at: float = field(default_factory=time.time)
    object: Literal["response"] = "response"
    status: str = "completed"
    error: dict[str, Any] | None = None
    incomplete_details: dict[str, Any] | None = None
    instructions: Any = None
    metadata: dict[str, str] | None = None
    parallel_tool_calls: bool = True
    temperature: float | None = None
    top_p: float | None = None
    tools: list[dict[str, Any]] = field(default_factory=list)
    tool_choice: str | dict[str, Any] = "auto"
    truncation: str = "disabled"
    max_output_tokens: int | None = None
    previous_response_id: str | None = None
    background: bool = False
    completed_at: float | None = None
    service_tier: str = "default"
    store: bool = False
    text: dict[str, Any] = field(default_factory=lambda: {"format": {"type": "text"}})
    usage: ResponseUsage | None = None

    @property
    def output_text(self) -> str:
        chunks: list[str] = []
        for item in self.output:
            if item.get("type") != "message":
                continue
            for content in item.get("content", []):
                if content.get("type") == "output_text":
                    chunks.append(str(content.get("text", "")))
        return "".join(chunks)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "object": self.object,
            "created_at": self.created_at,
            "status": self.status,
            "error": self.error,
            "incomplete_details": self.incomplete_details,
            "instructions": self.instructions,
            "metadata": self.metadata,
            "model": self.model,
            "output": self.output,
            "parallel_tool_calls": self.parallel_tool_calls,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "tools": self.tools,
            "tool_choice": self.tool_choice,
            "truncation": self.truncation,
            "max_output_tokens": self.max_output_tokens,
            "previous_response_id": self.previous_response_id,
            "background": self.background,
            "completed_at": self.completed_at or (self.created_at if self.status == "completed" else None),
            "service_tier": self.service_tier,
            "store": self.store,
            "text": self.text,
            "usage": self.usage.to_dict() if self.usage else None,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Response":
        usage_value = value.get("usage")
        usage = None
        if isinstance(usage_value, dict):
            usage = ResponseUsage(
                input_tokens=int(usage_value.get("input_tokens", 0)),
                output_tokens=int(usage_value.get("output_tokens", 0)),
                total_tokens=int(usage_value.get("total_tokens", 0)),
                input_tokens_details=dict(usage_value.get("input_tokens_details") or {"cached_tokens": 0}),
                output_tokens_details=dict(usage_value.get("output_tokens_details") or {"reasoning_tokens": 0}),
            )
        allowed = {
            "id", "model", "output", "created_at", "object", "status", "error",
            "incomplete_details", "instructions", "metadata", "parallel_tool_calls",
            "temperature", "top_p", "tools", "tool_choice", "truncation",
            "max_output_tokens", "previous_response_id", "background", "completed_at",
            "service_tier", "store", "text",
        }
        kwargs = {key: value[key] for key in allowed if key in value}
        kwargs.setdefault("output", [])
        kwargs.setdefault("model", "")
        kwargs["usage"] = usage
        return cls(**kwargs)

    def model_dump(self, **_: Any) -> dict[str, Any]:
        return self.to_dict()

    def model_dump_json(self, **_: Any) -> str:
        import json
        return json.dumps(self.to_dict(), separators=(",", ":"), ensure_ascii=False)

    @classmethod
    def text_response(
        cls,
        *,
        model: str,
        text: str,
        input_tokens: int,
        output_tokens: int,
        response_id: str | None = None,
        instructions: Any = None,
        metadata: dict[str, str] | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
    ) -> "Response":
        response_id = response_id or new_id("resp")
        message_id = new_id("msg")
        output = [
            {
                "id": message_id,
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": text,
                        "annotations": [],
                        "logprobs": [],
                    }
                ],
            }
        ]
        usage = ResponseUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
        )
        return cls(
            id=response_id,
            model=model,
            output=output,
            instructions=instructions,
            metadata=metadata,
            temperature=temperature,
            top_p=top_p,
            usage=usage,
        )


@dataclass(slots=True)
class ResponseEvent:
    type: str
    sequence_number: int
    data: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.type, "sequence_number": self.sequence_number, **self.data}

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ResponseEvent":
        return cls(
            type=str(value.get("type", "message")),
            sequence_number=int(value.get("sequence_number", 0)),
            data={key: item for key, item in value.items() if key not in {"type", "sequence_number"}},
        )

    def __getattr__(self, name: str) -> Any:
        try:
            return self.data[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def model_dump(self, **_: Any) -> dict[str, Any]:
        return self.to_dict()


def response_events(response: Response, deltas: Iterable[str]) -> Iterator[ResponseEvent]:
    """Emit the standard Responses streaming lifecycle for text output."""
    output_item = response.output[0]
    content_part = output_item["content"][0]
    seq = 0

    created = response.to_dict()
    created["status"] = "in_progress"
    created["output"] = []
    created["usage"] = None
    yield ResponseEvent("response.created", seq, {"response": created})
    seq += 1
    yield ResponseEvent("response.in_progress", seq, {"response": created})
    seq += 1

    added_item = {**output_item, "status": "in_progress", "content": []}
    yield ResponseEvent(
        "response.output_item.added",
        seq,
        {"output_index": 0, "item": added_item},
    )
    seq += 1
    added_part = {**content_part, "text": ""}
    yield ResponseEvent(
        "response.content_part.added",
        seq,
        {"item_id": output_item["id"], "output_index": 0, "content_index": 0, "part": added_part},
    )
    seq += 1

    text = ""
    for delta in deltas:
        text += delta
        yield ResponseEvent(
            "response.output_text.delta",
            seq,
            {
                "item_id": output_item["id"],
                "output_index": 0,
                "content_index": 0,
                "delta": delta,
                "logprobs": [],
            },
        )
        seq += 1

    yield ResponseEvent(
        "response.output_text.done",
        seq,
        {
            "item_id": output_item["id"],
            "output_index": 0,
            "content_index": 0,
            "text": text,
            "logprobs": [],
        },
    )
    seq += 1
    yield ResponseEvent(
        "response.content_part.done",
        seq,
        {
            "item_id": output_item["id"],
            "output_index": 0,
            "content_index": 0,
            "part": content_part,
        },
    )
    seq += 1
    yield ResponseEvent(
        "response.output_item.done",
        seq,
        {"output_index": 0, "item": output_item},
    )
    seq += 1
    yield ResponseEvent("response.completed", seq, {"response": response.to_dict()})
