from __future__ import annotations

import os
import time
import uuid
from collections.abc import Iterator
from typing import Any

import uvicorn

from pllm.runtime.responses import ResponsesError
from pllm.runtime.sidecar import create_sidecar_app
from pllm.runtime.types import Response, ResponseEvent, ResponseUsage


class ConformanceRuntime:
    def __init__(self) -> None:
        self.responses: dict[str, Response] = {}

    def create(self, body: dict[str, Any]) -> Response | Iterator[ResponseEvent]:
        if body.get("stream"):
            return self.events(body)
        final: Response | None = None
        for event in self.events(body):
            if event.type == "response.completed":
                final = event.response
        assert final is not None
        return final

    def events(self, body: dict[str, Any]) -> Iterator[ResponseEvent]:
        previous = body.get("previous_response_id")
        if previous is not None and previous not in self.responses:
            raise ResponsesError(
                f"Previous response {previous!r} was not found.",
                param="previous_response_id",
                code="previous_response_not_found",
                status_code=404,
            )

        response_id = f"resp_{uuid.uuid4().hex}"
        created_at = int(time.time())
        output: list[dict[str, Any]]
        tools = body.get("tools") or []
        weather_tool = next(
            (
                tool
                for tool in tools
                if isinstance(tool, dict) and tool.get("name") == "get_weather"
            ),
            None,
        )
        if weather_tool is not None:
            item = {
                "type": "function_call",
                "id": f"fc_{uuid.uuid4().hex}",
                "call_id": f"call_{uuid.uuid4().hex}",
                "name": "get_weather",
                "arguments": '{"location":"Tokyo"}',
                "status": "completed",
            }
            output = [item]
        else:
            item = {
                "type": "message",
                "id": f"msg_{uuid.uuid4().hex}",
                "role": "assistant",
                "phase": "final_answer",
                "status": "completed",
                "content": [
                    {
                        "type": "output_text",
                        "text": "PLLM_OK",
                        "annotations": [],
                        "logprobs": [],
                    }
                ],
            }
            output = [item]

        common = {
            "id": response_id,
            "created_at": created_at,
            "completed_at": None,
            "model": body.get("model", "conformance-model"),
            "instructions": body.get("instructions"),
            "metadata": body.get("metadata") or {},
            "output": [],
            "tools": tools,
            "tool_choice": body.get("tool_choice", "auto"),
            "parallel_tool_calls": body.get("parallel_tool_calls", True),
            "temperature": body.get("temperature", 1.0),
            "top_p": body.get("top_p", 1.0),
            "max_output_tokens": body.get("max_output_tokens"),
            "previous_response_id": previous,
            "truncation": body.get("truncation", "disabled"),
            "store": body.get("store") is not False,
            "service_tier": "default",
            "text": body.get("text") or {"format": {"type": "text"}},
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
            "top_logprobs": 0,
            "reasoning": body.get("reasoning"),
            "max_tool_calls": body.get("max_tool_calls"),
            "safety_identifier": body.get("safety_identifier"),
            "prompt_cache_key": body.get("prompt_cache_key"),
        }
        pending = Response(status="in_progress", usage=None, **common)
        yield ResponseEvent.from_dict(
            {"type": "response.created", "sequence_number": 0, "response": pending.to_dict()}
        )
        yield ResponseEvent.from_dict(
            {"type": "response.in_progress", "sequence_number": 1, "response": pending.to_dict()}
        )
        completed = Response(
            status="completed",
            completed_at=int(time.time()),
            output=output,
            usage=ResponseUsage(input_tokens=4, output_tokens=3, total_tokens=7),
            **{
                key: value for key, value in common.items() if key not in {"completed_at", "output"}
            },
        )
        self.responses[response_id] = completed
        sequence = 2
        pending_item = {**item, "status": "in_progress"}
        yield ResponseEvent.from_dict(
            {
                "type": "response.output_item.added",
                "sequence_number": sequence,
                "output_index": 0,
                "item": pending_item,
            }
        )
        sequence += 1
        if item["type"] == "message":
            part = item["content"][0]
            yield ResponseEvent.from_dict(
                {
                    "type": "response.content_part.added",
                    "sequence_number": sequence,
                    "item_id": item["id"],
                    "output_index": 0,
                    "content_index": 0,
                    "part": {**part, "text": ""},
                }
            )
            sequence += 1
            yield ResponseEvent.from_dict(
                {
                    "type": "response.output_text.delta",
                    "sequence_number": sequence,
                    "item_id": item["id"],
                    "output_index": 0,
                    "content_index": 0,
                    "delta": part["text"],
                }
            )
            sequence += 1
            yield ResponseEvent.from_dict(
                {
                    "type": "response.output_text.done",
                    "sequence_number": sequence,
                    "item_id": item["id"],
                    "output_index": 0,
                    "content_index": 0,
                    "text": part["text"],
                    "logprobs": [],
                }
            )
            sequence += 1
            yield ResponseEvent.from_dict(
                {
                    "type": "response.content_part.done",
                    "sequence_number": sequence,
                    "item_id": item["id"],
                    "output_index": 0,
                    "content_index": 0,
                    "part": part,
                }
            )
            sequence += 1
        yield ResponseEvent.from_dict(
            {
                "type": "response.output_item.done",
                "sequence_number": sequence,
                "output_index": 0,
                "item": item,
            }
        )
        yield ResponseEvent.from_dict(
            {
                "type": "response.completed",
                "sequence_number": sequence + 1,
                "response": completed.to_dict(),
            }
        )

    def retrieve(self, response_id: str) -> Response:
        try:
            return self.responses[response_id]
        except KeyError as exc:
            raise ResponsesError(
                f"Response {response_id!r} was not found.",
                code="not_found",
                status_code=404,
            ) from exc

    def evict_response(self, response_id: str) -> None:
        self.responses.pop(response_id, None)

    def cancel(self, response_id: str) -> Response:
        response = self.retrieve(response_id)
        cancelled = Response.from_dict(
            {**response.to_dict(), "status": "cancelled", "completed_at": int(time.time())}
        )
        self.responses[response_id] = cancelled
        return cancelled


if __name__ == "__main__":
    port = int(os.environ.get("PLLM_CONFORMANCE_PORT", "8765"))
    app = create_sidecar_app(
        client=ConformanceRuntime(),
        remote_base_url="http://127.0.0.1:1",
        remote_api_key="unused",
        local_api_key="test-key",
    )
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
