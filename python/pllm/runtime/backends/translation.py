from __future__ import annotations

import time
import uuid
from typing import Any


def responses_to_chat(body: dict[str, Any]) -> dict[str, Any]:
    messages: list[dict[str, Any]] = []
    instructions = body.get("instructions")
    if instructions:
        messages.append({"role": "system", "content": instructions})
    raw_input = body.get("input", "")
    if isinstance(raw_input, str):
        messages.append({"role": "user", "content": raw_input})
    elif isinstance(raw_input, list):
        for item in raw_input:
            if not isinstance(item, dict):
                messages.append({"role": "user", "content": str(item)})
                continue
            if item.get("type") == "function_call_output":
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": item.get("call_id", "call_unknown"),
                        "content": item.get("output", ""),
                    }
                )
            else:
                messages.append({"role": item.get("role", "user"), "content": item.get("content", "")})
    result: dict[str, Any] = {
        "model": body.get("model"),
        "messages": messages,
        "stream": body.get("stream", False),
    }
    for source, target in (
        ("temperature", "temperature"),
        ("top_p", "top_p"),
        ("max_output_tokens", "max_tokens"),
        ("tools", "tools"),
        ("tool_choice", "tool_choice"),
        ("parallel_tool_calls", "parallel_tool_calls"),
    ):
        if source in body and body[source] is not None:
            result[target] = body[source]
    text = body.get("text") or {}
    if isinstance(text, dict) and text.get("format"):
        result["response_format"] = text["format"]
    return result


def chat_completion_to_response(chat: dict[str, Any], *, model: str) -> dict[str, Any]:
    choice = (chat.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    output: list[dict[str, Any]] = []
    if message.get("content") is not None:
        output.append(
            {
                "id": f"msg_{uuid.uuid4().hex}",
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": str(message.get("content", "")),
                        "annotations": [],
                        "logprobs": [],
                    }
                ],
            }
        )
    for call in message.get("tool_calls") or []:
        function = call.get("function") or {}
        output.append(
            {
                "id": call.get("id", f"fc_{uuid.uuid4().hex}"),
                "type": "function_call",
                "call_id": call.get("id", f"call_{uuid.uuid4().hex}"),
                "name": function.get("name", ""),
                "arguments": function.get("arguments", "{}"),
                "status": "completed",
            }
        )
    usage = chat.get("usage") or {}
    input_tokens = int(usage.get("prompt_tokens", 0))
    output_tokens = int(usage.get("completion_tokens", 0))
    return {
        "id": chat.get("id", f"resp_{uuid.uuid4().hex}"),
        "object": "response",
        "created_at": float(chat.get("created", time.time())),
        "status": "completed",
        "error": None,
        "incomplete_details": None,
        "instructions": None,
        "metadata": None,
        "model": model,
        "output": output,
        "parallel_tool_calls": True,
        "temperature": None,
        "top_p": None,
        "tools": [],
        "tool_choice": "auto",
        "truncation": "disabled",
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens_details": {"reasoning_tokens": 0},
        },
    }
