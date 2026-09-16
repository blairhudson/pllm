from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from typing import Any


_RESPONSE_FIELDS = {
    "input",
    "instructions",
    "background",
    "client_metadata",
    "frequency_penalty",
    "include",
    "max_tool_calls",
    "max_output_tokens",
    "metadata",
    "model",
    "parallel_tool_calls",
    "presence_penalty",
    "previous_response_id",
    "prompt_cache_key",
    "reasoning",
    "safety_identifier",
    "service_tier",
    "store",
    "stream",
    "temperature",
    "text",
    "tool_choice",
    "tools",
    "top_logprobs",
    "top_p",
    "truncation",
}
_CHAT_FIELDS = {
    "max_completion_tokens",
    "max_tokens",
    "messages",
    "model",
    "parallel_tool_calls",
    "stream",
    "stream_options",
    "temperature",
    "tool_choice",
    "tools",
    "top_p",
}
_COMPACT_FIELDS = {"input", "instructions", "model", "prompt_cache_key"}
_CHAT_ROLES = {"system", "developer", "user", "assistant", "tool"}
_COMPACT_ITEM_LIMIT = 64
_COMPACT_BYTE_LIMIT = 64 * 1024


@dataclass(slots=True)
class GatewayError(ValueError):
    message: str
    status_code: int = 400
    error_type: str = "invalid_request_error"
    code: str = "invalid_request_error"
    param: str | None = None

    def __str__(self) -> str:
        return self.message

    def to_dict(self) -> dict[str, Any]:
        return error_envelope(self.message, self.error_type, self.code, self.param)


def error_envelope(
    message: str,
    error_type: str = "invalid_request_error",
    code: str = "invalid_request_error",
    param: str | None = None,
) -> dict[str, Any]:
    return {
        "error": {
            "message": message,
            "type": error_type,
            "param": param,
            "code": code,
        }
    }


def validate_fields(body: Mapping[str, Any], allowed: set[str]) -> None:
    unsupported = sorted(set(body) - allowed)
    if unsupported:
        name = unsupported[0]
        raise GatewayError(f"Unsupported field: {name}", param=name)


def validate_response_body(value: Any) -> dict[str, Any]:
    body = _object_body(value)
    validate_fields(body, _RESPONSE_FIELDS)
    _model(body.get("model"))
    _input(body.get("input"))
    _optional_bool(body, "stream")
    _optional_bool(body, "parallel_tool_calls")
    _optional_bool(body, "store")
    _optional_bool(body, "background")
    if body.get("background") is True:
        raise GatewayError(
            "background responses are not supported", param="background", code="unsupported_feature"
        )
    if "instructions" in body and body["instructions"] is not None:
        _nonempty_string(body["instructions"], "instructions")
    if "max_output_tokens" in body:
        _positive_int(body["max_output_tokens"], "max_output_tokens")
    if "max_tool_calls" in body and body["max_tool_calls"] is not None:
        _positive_int(body["max_tool_calls"], "max_tool_calls")
    for name in ("temperature", "top_p", "frequency_penalty", "presence_penalty"):
        _optional_number(body, name)
    for name in ("frequency_penalty", "presence_penalty"):
        if body.get(name) not in {None, 0, 0.0}:
            raise GatewayError(f"{name} is not supported", param=name, code="unsupported_feature")
    if body.get("top_logprobs") not in {None, 0}:
        raise GatewayError(
            "top_logprobs are not supported", param="top_logprobs", code="unsupported_feature"
        )
    if "tools" in body:
        _validate_response_tools(body["tools"])
        for tool in body["tools"]:
            tool.setdefault("parameters", {})
            tool.setdefault("strict", False)
    if "tool_choice" in body:
        _validate_response_tool_choice(body["tool_choice"])
    if (
        "metadata" in body
        and body["metadata"] is not None
        and not isinstance(body["metadata"], dict)
    ):
        raise GatewayError("metadata must be an object", param="metadata")
    if body.get("client_metadata") is not None and not isinstance(body["client_metadata"], dict):
        raise GatewayError("client_metadata must be an object", param="client_metadata")
    if "previous_response_id" in body and body["previous_response_id"] is not None:
        _nonempty_string(body["previous_response_id"], "previous_response_id")
    if body.get("truncation", "disabled") not in {"auto", "disabled"}:
        raise GatewayError("truncation must be auto or disabled", param="truncation")
    include = body.get("include")
    if include is not None:
        supported_includes = {
            "reasoning.encrypted_content",
            "message.output_text.logprobs",
        }
        if not isinstance(include, list) or any(item not in supported_includes for item in include):
            raise GatewayError("Unsupported include value", param="include")
    reasoning = body.get("reasoning")
    if reasoning is not None and not isinstance(reasoning, dict):
        raise GatewayError("reasoning must be an object", param="reasoning")
    if isinstance(reasoning, dict) and set(reasoning) - {"effort", "summary"}:
        raise GatewayError("Unsupported reasoning field", param="reasoning")
    text = body.get("text")
    if text is not None:
        if not isinstance(text, dict) or set(text) - {"format", "verbosity"}:
            raise GatewayError("Unsupported text configuration", param="text")
        format_value = text.get("format", {"type": "text"})
        if format_value != {"type": "text"}:
            raise GatewayError("Only plain text output is supported", param="text.format")
        verbosity = text.get("verbosity", "medium")
        if verbosity != "medium":
            raise GatewayError(
                "Only medium text verbosity is supported",
                param="text.verbosity",
                code="unsupported_feature",
            )
        body["text"] = {"format": {"type": "text"}, "verbosity": "medium"}
    if body.get("service_tier", "auto") not in {"auto", "default"}:
        raise GatewayError(
            "service_tier is not supported", param="service_tier", code="unsupported_feature"
        )
    text = body.get("text")
    if text is not None:
        if not isinstance(text, dict):
            raise GatewayError("text must be an object", param="text")
        text_format = text.get("format", {"type": "text"})
        if not isinstance(text_format, dict) or text_format.get("type") != "text":
            raise GatewayError(
                "Only plain text output is supported", param="text", code="unsupported_feature"
            )
    for name in ("safety_identifier", "prompt_cache_key"):
        if body.get(name) is not None:
            _nonempty_string(body[name], name)
    return dict(body)


def chat_to_response_body(value: Any) -> dict[str, Any]:
    body = _object_body(value)
    validate_fields(body, _CHAT_FIELDS)
    model = _model(body.get("model"))
    _optional_bool(body, "stream")
    _optional_bool(body, "parallel_tool_calls")
    _optional_number(body, "temperature")
    _optional_number(body, "top_p")
    stream_options = body.get("stream_options")
    if stream_options is not None:
        if not isinstance(stream_options, dict) or set(stream_options) - {"include_usage"}:
            raise GatewayError("Unsupported stream_options value", param="stream_options")
        _optional_bool(stream_options, "include_usage")
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        raise GatewayError("messages must be a non-empty array", param="messages")

    response_input: list[dict[str, Any]] = []
    for index, raw in enumerate(messages):
        param = f"messages[{index}]"
        if not isinstance(raw, dict):
            raise GatewayError("Each message must be an object", param=param)
        unknown = set(raw) - {"role", "content", "tool_call_id", "tool_calls"}
        if unknown:
            field = sorted(unknown)[0]
            raise GatewayError(f"Unsupported message field: {field}", param=f"{param}.{field}")
        role = raw.get("role")
        if role not in _CHAT_ROLES:
            raise GatewayError(f"Unsupported message role: {role!r}", param=f"{param}.role")
        if role != "tool" and raw.get("tool_call_id") is not None:
            raise GatewayError("tool_call_id is only valid on tool messages", param=param)
        if role != "assistant" and raw.get("tool_calls") is not None:
            raise GatewayError("tool_calls is only valid on assistant messages", param=param)
        if role == "tool":
            call_id = _nonempty_string(raw.get("tool_call_id"), f"{param}.tool_call_id")
            content = _chat_text(raw.get("content"), f"{param}.content")
            response_input.append(
                {"type": "function_call_output", "call_id": call_id, "output": content}
            )
            continue

        content = raw.get("content")
        if content is not None:
            response_input.append(
                {
                    "type": "message",
                    "role": role,
                    "content": _chat_content(content, role, f"{param}.content"),
                }
            )
        elif role != "assistant" or not raw.get("tool_calls"):
            raise GatewayError("message content is required", param=f"{param}.content")

        tool_calls = raw.get("tool_calls")
        if tool_calls is not None:
            if role != "assistant" or not isinstance(tool_calls, list) or not tool_calls:
                raise GatewayError(
                    "tool_calls must be a non-empty array on an assistant message",
                    param=f"{param}.tool_calls",
                )
            for call_index, call in enumerate(tool_calls):
                response_input.append(_chat_tool_call(call, f"{param}.tool_calls[{call_index}]"))

    result: dict[str, Any] = {"model": model, "input": response_input}
    for name in ("stream", "temperature", "top_p", "parallel_tool_calls"):
        if name in body:
            result[name] = body[name]
    max_tokens = body.get("max_completion_tokens")
    legacy_max = body.get("max_tokens")
    if max_tokens is not None and legacy_max is not None:
        raise GatewayError(
            "max_completion_tokens and max_tokens cannot both be set",
            param="max_completion_tokens",
        )
    if max_tokens is not None or legacy_max is not None:
        value = max_tokens if max_tokens is not None else legacy_max
        result["max_output_tokens"] = _positive_int(
            value, "max_completion_tokens" if max_tokens is not None else "max_tokens"
        )
    if "tools" in body:
        result["tools"] = _chat_tools(body["tools"])
    if "tool_choice" in body:
        result["tool_choice"] = _chat_tool_choice(body["tool_choice"])
    return result


def compact_response(value: Any) -> dict[str, Any]:
    body = _object_body(value)
    validate_fields(body, _COMPACT_FIELDS)
    model = _model(body.get("model"))
    normalized = normalize_compact_input(body.get("input"), body.get("instructions"))
    pinned: list[dict[str, Any]] = []
    while normalized and normalized[0].get("role") in {"system", "developer"}:
        pinned.append(normalized.pop(0))
    history_limit = _COMPACT_ITEM_LIMIT - len(pinned) - 2
    if history_limit < 1:
        raise GatewayError(
            "Leading instructions exceed local compaction item limit",
            code="unsupported_feature",
            param="input",
        )
    groups = _compaction_groups(normalized)
    retained_groups: list[list[dict[str, Any]]] = []
    retained_count = 0
    if len(pinned) + len(normalized) <= _COMPACT_ITEM_LIMIT - 1:
        retained_groups = groups
        retained_count = len(normalized)
    else:
        for group in reversed(groups):
            if retained_count + len(group) > history_limit:
                continue
            retained_groups.append(group)
            retained_count += len(group)
        retained_groups.reverse()
    elided = len(normalized) - retained_count
    retained = [item for group in retained_groups for item in group]
    while retained_groups and _compact_size(pinned, retained, elided) > _COMPACT_BYTE_LIMIT:
        elided += len(retained_groups.pop(0))
        retained = [item for group in retained_groups for item in group]
    if _compact_size(pinned, retained, elided) > _COMPACT_BYTE_LIMIT:
        raise GatewayError(
            "Newest input item exceeds local compaction limit",
            code="unsupported_feature",
            param="input",
        )
    normalized = pinned + ([_elision_marker(elided)] if elided else []) + retained
    canonical = json.dumps(
        {"model": model, "output": normalized},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode()
    digest = hashlib.sha256(canonical).hexdigest()
    output = [
        *normalized,
        {
            "type": "compaction",
            "id": f"cmp_{digest[:24]}",
            "encrypted_content": digest,
            "created_by": "pllm.gateway",
        },
    ]
    return {
        "id": f"resp_compact_{digest[:24]}",
        "object": "response.compaction",
        "created_at": int(time.time()),
        "output": output,
        "usage": {
            "input_tokens": 0,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens": 0,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": 0,
        },
    }


def normalize_compact_input(value: Any, instructions: Any = None) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    if instructions is not None:
        text = _nonempty_string(instructions, "instructions")
        output.append(_compact_message("developer", text, 0))
    if isinstance(value, str):
        if not value:
            raise GatewayError("input must not be empty", param="input")
        output.append(_compact_message("user", value, len(output)))
        return output
    if not isinstance(value, list) or not value:
        raise GatewayError("input must be a non-empty string or array", param="input")
    for index, raw in enumerate(value):
        param = f"input[{index}]"
        if not isinstance(raw, dict):
            raise GatewayError("Each input item must be an object", param=param)
        kind = raw.get("type", "message")
        if kind == "message":
            role = raw.get("role")
            if role not in {"system", "developer", "user", "assistant"}:
                raise GatewayError(f"Unsupported message role: {role!r}", param=f"{param}.role")
            item = _compact_message(role, raw.get("content"), len(output))
            if isinstance(raw.get("id"), str):
                item["id"] = raw["id"]
            output.append(item)
        elif kind in {"function_call", "function_call_output", "compaction"}:
            output.append(_compact_special_item(raw, kind, param))
        elif kind in {"input_text", "text"}:
            output.append(_compact_message("user", raw.get("text"), len(output)))
        else:
            raise GatewayError(f"Unsupported compact input item type: {kind!r}", param=param)
    return output


def resource_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        result = dict(value)
    elif hasattr(value, "to_dict"):
        result = value.to_dict()
    elif hasattr(value, "model_dump"):
        result = value.model_dump()
    else:
        raise TypeError("Runtime returned an unsupported resource")
    return integer_timestamps(result)


def integer_timestamps(value: Any) -> Any:
    if isinstance(value, list):
        return [integer_timestamps(item) for item in value]
    if not isinstance(value, dict):
        return value
    result = {key: integer_timestamps(item) for key, item in value.items()}
    for key in ("created", "created_at", "completed_at"):
        if isinstance(result.get(key), (int, float)):
            result[key] = int(result[key])
    return result


def response_sse_event(event: Any) -> bytes:
    payload = resource_dict(event)
    event_type = str(payload.get("type", "message"))
    return _named_sse(event_type, payload)


def response_error_sse(exc: Exception, sequence_number: int = 0) -> bytes:
    envelope = exception_envelope(exc)
    error = envelope["error"]
    payload = {
        "type": "error",
        "sequence_number": sequence_number,
        "error": error,
    }
    return _named_sse("error", payload)


def chat_completion(value: Any) -> dict[str, Any]:
    response = resource_dict(value)
    message, finish_reason = _response_message(response.get("output", []))
    finish_reason = _response_finish_reason(response, finish_reason)
    return {
        "id": response.get("id"),
        "object": "chat.completion",
        "created": int(response.get("created_at") or time.time()),
        "model": response.get("model"),
        "choices": [
            {
                "index": 0,
                "message": message,
                "logprobs": None,
                "finish_reason": finish_reason,
            }
        ],
        "usage": _chat_usage(response.get("usage")),
    }


def iter_chat_sse(
    events: Iterable[Any], request_model: str, *, include_usage: bool = False
) -> Iterator[bytes]:
    response_id: str | None = None
    model = request_model
    created = int(time.time())
    tool_indexes: dict[str, int] = {}
    next_tool_index = 0
    emitted_role = False
    finish_reason = "stop"
    final_usage: dict[str, Any] | None = None
    terminal = False

    for raw in events:
        event = resource_dict(raw)
        kind = event.get("type")
        response = event.get("response")
        if isinstance(response, dict):
            response_id = str(response.get("id") or response_id or "")
            model = str(response.get("model") or model)
            created = int(response.get("created_at") or created)
        if kind == "response.created" and not emitted_role:
            yield _chat_chunk(response_id, model, created, {"role": "assistant", "content": ""})
            emitted_role = True
        elif kind == "response.output_text.delta":
            if not emitted_role:
                yield _chat_chunk(response_id, model, created, {"role": "assistant", "content": ""})
                emitted_role = True
            yield _chat_chunk(response_id, model, created, {"content": str(event.get("delta", ""))})
        elif kind == "response.output_item.added":
            item = event.get("item") or {}
            if isinstance(item, dict) and item.get("type") == "function_call":
                key = str(item.get("id") or item.get("call_id") or len(tool_indexes))
                index = next_tool_index
                next_tool_index += 1
                tool_indexes[key] = index
                if item.get("call_id"):
                    tool_indexes[str(item["call_id"])] = index
                finish_reason = "tool_calls"
                if not emitted_role:
                    yield _chat_chunk(
                        response_id, model, created, {"role": "assistant", "content": None}
                    )
                    emitted_role = True
                yield _chat_chunk(
                    response_id,
                    model,
                    created,
                    {
                        "tool_calls": [
                            {
                                "index": index,
                                "id": item.get("call_id") or item.get("id"),
                                "type": "function",
                                "function": {"name": item.get("name", ""), "arguments": ""},
                            }
                        ]
                    },
                )
        elif kind == "response.function_call_arguments.delta":
            key = str(event.get("item_id") or event.get("call_id") or "")
            index = tool_indexes.get(key, 0)
            finish_reason = "tool_calls"
            yield _chat_chunk(
                response_id,
                model,
                created,
                {
                    "tool_calls": [
                        {
                            "index": index,
                            "function": {"arguments": str(event.get("delta", ""))},
                        }
                    ]
                },
            )
        elif kind in {"response.completed", "response.incomplete"} and isinstance(response, dict):
            terminal = True
            final_usage = _chat_usage(response.get("usage"))
            _, finish_reason = _response_message(response.get("output", []))
            finish_reason = _response_finish_reason(response, finish_reason)
        elif kind in {"response.failed", "error"}:
            raise GatewayError(
                "Response generation failed",
                status_code=502,
                code="runtime_error",
                error_type="server_error",
            )

    if not terminal:
        raise GatewayError(
            "Response stream ended without a terminal event",
            status_code=502,
            code="runtime_error",
            error_type="server_error",
        )

    yield _chat_chunk(
        response_id,
        model,
        created,
        {},
        finish_reason=finish_reason,
    )
    if include_usage:
        payload = {
            "id": response_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [],
            "usage": final_usage,
        }
        yield f"data: {json.dumps(payload, separators=(',', ':'), ensure_ascii=False)}\n\n".encode()


def exception_envelope(exc: Exception) -> dict[str, Any]:
    if isinstance(exc, GatewayError):
        return exc.to_dict()
    status_code = getattr(exc, "status_code", 500)
    error_type = "invalid_request_error" if 400 <= status_code < 500 else "server_error"
    code = "runtime_error" if error_type == "server_error" else "invalid_request_error"
    return error_envelope(str(exc) or "Runtime error", error_type, code)


def _object_body(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise GatewayError("Request body must be a JSON object")
    return value


def _model(value: Any) -> str:
    return _nonempty_string(value, "model")


def _input(value: Any) -> None:
    if isinstance(value, str):
        if value:
            return
    elif isinstance(value, list) and value and all(isinstance(item, dict) for item in value):
        return
    raise GatewayError("input must be a non-empty string or array of objects", param="input")


def _nonempty_string(value: Any, param: str) -> str:
    if not isinstance(value, str) or not value:
        raise GatewayError(f"{param} must be a non-empty string", param=param)
    return value


def _string(value: Any, param: str) -> str:
    if not isinstance(value, str):
        raise GatewayError(f"{param} must be a string", param=param)
    return value


def _positive_int(value: Any, param: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise GatewayError(f"{param} must be a positive integer", param=param)
    return value


def _optional_bool(body: Mapping[str, Any], name: str) -> None:
    if name in body and not isinstance(body[name], bool):
        raise GatewayError(f"{name} must be a boolean", param=name)


def _optional_number(body: Mapping[str, Any], name: str) -> None:
    if name not in body or body[name] is None:
        return
    if isinstance(body[name], bool) or not isinstance(body[name], (int, float)):
        raise GatewayError(f"{name} must be a number", param=name)


def _validate_response_tools(value: Any) -> None:
    if not isinstance(value, list):
        raise GatewayError("tools must be an array", param="tools")
    for index, tool in enumerate(value):
        param = f"tools[{index}]"
        if not isinstance(tool, dict):
            raise GatewayError("Tool definition must be an object", param=param)
        if tool.get("type") == "namespace":
            _validate_namespace_tool(tool, param)
            continue
        if tool.get("type") != "function":
            tool_type = tool.get("type") if isinstance(tool, dict) else None
            raise GatewayError(f"Unsupported tool type: {tool_type!r}", param=param)
        unknown = set(tool) - {"type", "name", "description", "parameters", "strict"}
        if unknown:
            raise GatewayError(f"Unsupported function field: {sorted(unknown)[0]}", param=param)
        _nonempty_string(tool.get("name"), f"{param}.name")
        if "description" in tool and not isinstance(tool["description"], str):
            raise GatewayError("function description must be a string", param=param)
        if "parameters" in tool and not isinstance(tool["parameters"], dict):
            raise GatewayError("function parameters must be an object", param=param)
        if "strict" in tool and not isinstance(tool["strict"], bool):
            raise GatewayError("function strict must be a boolean", param=param)


def _validate_namespace_tool(tool: Mapping[str, Any], param: str) -> None:
    unknown = set(tool) - {"type", "name", "description", "tools"}
    if unknown:
        raise GatewayError(f"Unsupported namespace field: {sorted(unknown)[0]}", param=param)
    _nonempty_string(tool.get("name"), f"{param}.name")
    if not isinstance(tool.get("description"), str):
        raise GatewayError("namespace description must be a string", param=f"{param}.description")
    nested = tool.get("tools")
    if not isinstance(nested, list) or not nested:
        raise GatewayError("namespace tools must be a non-empty array", param=f"{param}.tools")
    for index, child in enumerate(nested):
        child_param = f"{param}.tools[{index}]"
        if not isinstance(child, dict) or child.get("type") not in {"function", "custom"}:
            raise GatewayError("Unsupported namespace tool", param=child_param)
        _nonempty_string(child.get("name"), f"{child_param}.name")
        if not isinstance(child.get("description"), str):
            raise GatewayError("tool description must be a string", param=child_param)
        if child.get("type") == "function":
            if not isinstance(child.get("parameters"), dict):
                raise GatewayError("function parameters must be an object", param=child_param)
            if not isinstance(child.get("strict"), bool):
                raise GatewayError("function strict must be a boolean", param=child_param)
        else:
            if not isinstance(child.get("format"), dict):
                raise GatewayError("custom tool format must be an object", param=child_param)


def _validate_response_tool_choice(value: Any) -> None:
    if isinstance(value, str):
        if value not in {"none", "auto", "required"}:
            raise GatewayError("Unsupported tool_choice", param="tool_choice")
        return
    if isinstance(value, dict) and value.get("type") == "allowed_tools":
        if set(value) != {"type", "mode", "tools"} or value.get("mode") not in {
            "auto",
            "required",
        }:
            raise GatewayError("Invalid allowed_tools choice", param="tool_choice")
        tools = value.get("tools")
        if not isinstance(tools, list) or not tools:
            raise GatewayError("allowed_tools requires tools", param="tool_choice.tools")
        for index, tool in enumerate(tools):
            if not isinstance(tool, dict) or set(tool) != {"type", "name"}:
                raise GatewayError("Invalid allowed tool", param=f"tool_choice.tools.{index}")
            if tool.get("type") != "function":
                raise GatewayError(
                    "Only function tools can be allowed", param=f"tool_choice.tools.{index}"
                )
            _nonempty_string(tool.get("name"), f"tool_choice.tools.{index}.name")
        return
    if (
        not isinstance(value, dict)
        or set(value) != {"type", "name"}
        or value.get("type") != "function"
    ):
        raise GatewayError("Only named function tool_choice is supported", param="tool_choice")
    _nonempty_string(value.get("name"), "tool_choice.name")


def _chat_text(value: Any, param: str) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        chunks: list[str] = []
        for item in value:
            if (
                not isinstance(item, dict)
                or item.get("type") != "text"
                or not isinstance(item.get("text"), str)
            ):
                raise GatewayError("Only text content is supported", param=param)
            chunks.append(item["text"])
        return "".join(chunks)
    raise GatewayError("Only text content is supported", param=param)


def _chat_content(value: Any, role: str, param: str) -> list[dict[str, Any]]:
    kind = "output_text" if role == "assistant" else "input_text"
    if isinstance(value, str):
        return [{"type": kind, "text": value}]
    if not isinstance(value, list) or not value:
        raise GatewayError("message content must be text or a non-empty content array", param=param)
    output: list[dict[str, Any]] = []
    for item in value:
        if (
            not isinstance(item, dict)
            or item.get("type") != "text"
            or not isinstance(item.get("text"), str)
        ):
            raise GatewayError("Only text content is supported", param=param)
        output.append({"type": kind, "text": item["text"]})
    return output


def _chat_tool_call(value: Any, param: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) - {"id", "type", "function"}:
        raise GatewayError("Invalid assistant tool call", param=param)
    if value.get("type", "function") != "function":
        raise GatewayError("Only function tool calls are supported", param=f"{param}.type")
    function = value.get("function")
    if not isinstance(function, dict) or set(function) - {"name", "arguments"}:
        raise GatewayError("Invalid function tool call", param=f"{param}.function")
    return {
        "type": "function_call",
        "call_id": _nonempty_string(value.get("id"), f"{param}.id"),
        "name": _nonempty_string(function.get("name"), f"{param}.function.name"),
        "arguments": _string(function.get("arguments"), f"{param}.function.arguments"),
    }


def _chat_tools(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise GatewayError("tools must be an array", param="tools")
    output: list[dict[str, Any]] = []
    for index, tool in enumerate(value):
        param = f"tools[{index}]"
        if isinstance(tool, dict) and tool.get("type") == "custom":
            unknown = set(tool) - {"type", "name", "description", "format", "providerOptions"}
            if unknown:
                raise GatewayError(
                    f"Unsupported custom tool field: {sorted(unknown)[0]}", param=param
                )
            name = _nonempty_string(tool.get("name"), f"{param}.name")
            description = tool.get("description")
            if description is not None and not isinstance(description, str):
                raise GatewayError(
                    "Tool description must be a string", param=f"{param}.description"
                )
            output.append(
                {
                    "type": "function",
                    "name": name,
                    "description": description,
                    "parameters": {
                        "type": "object",
                        "properties": {"input": {"type": "string"}},
                        "required": ["input"],
                        "additionalProperties": False,
                    },
                    "strict": False,
                }
            )
            continue
        if not isinstance(tool, dict) or set(tool) - {"type", "function"}:
            raise GatewayError("Invalid tool definition", param=param)
        if tool.get("type") != "function" or not isinstance(tool.get("function"), dict):
            raise GatewayError(f"Unsupported tool type: {tool.get('type')!r}", param=param)
        function = tool["function"]
        unknown = set(function) - {"name", "description", "parameters", "strict"}
        if unknown:
            raise GatewayError(f"Unsupported function field: {sorted(unknown)[0]}", param=param)
        normalized = {"type": "function", "name": _nonempty_string(function.get("name"), param)}
        if "description" in function and not isinstance(function["description"], str):
            raise GatewayError("function description must be a string", param=param)
        if "parameters" in function and not isinstance(function["parameters"], dict):
            raise GatewayError("function parameters must be an object", param=param)
        if "strict" in function and not isinstance(function["strict"], bool):
            raise GatewayError("function strict must be a boolean", param=param)
        for name in ("description", "parameters", "strict"):
            if name in function:
                normalized[name] = function[name]
        output.append(normalized)
    return output


def _chat_tool_choice(value: Any) -> str | dict[str, Any]:
    if isinstance(value, str):
        if value not in {"none", "auto", "required"}:
            raise GatewayError("Unsupported tool_choice", param="tool_choice")
        return value
    if not isinstance(value, dict) or set(value) - {"type", "function"}:
        raise GatewayError("Invalid tool_choice", param="tool_choice")
    function = value.get("function")
    if (
        value.get("type") != "function"
        or not isinstance(function, dict)
        or set(function) != {"name"}
    ):
        raise GatewayError("Only named function tool_choice is supported", param="tool_choice")
    return {"type": "function", "name": _nonempty_string(function["name"], "tool_choice")}


def _compact_message(role: str, content: Any, index: int) -> dict[str, Any]:
    kind = "output_text" if role == "assistant" else "input_text"

    def compact_part(text: str) -> dict[str, Any]:
        value: dict[str, Any] = {"type": kind, "text": text}
        if kind == "output_text":
            value.update({"annotations": [], "logprobs": []})
        return value

    if isinstance(content, str):
        parts = [compact_part(content)]
    elif isinstance(content, list) and content:
        parts = []
        for item in content:
            if (
                not isinstance(item, dict)
                or item.get("type") not in {"text", "input_text", "output_text"}
                or not isinstance(item.get("text"), str)
            ):
                raise GatewayError("Only text compact message content is supported", param="input")
            parts.append(compact_part(item["text"]))
    else:
        raise GatewayError("Compact message content must be text", param="input")
    return {
        "id": f"msg_compact_{index}",
        "type": "message",
        "status": "completed",
        "role": role,
        "content": parts,
    }


def _compact_special_item(raw: dict[str, Any], kind: str, param: str) -> dict[str, Any]:
    required = {
        "function_call": ("call_id", "name", "arguments"),
        "function_call_output": ("call_id", "output"),
        "compaction": ("id", "encrypted_content"),
    }[kind]
    item = dict(raw)
    item["type"] = kind
    for field in required:
        _string(item.get(field), f"{param}.{field}")
    return item


def _elision_marker(elided: int) -> dict[str, Any]:
    return {
        "id": "msg_compact_elision",
        "type": "message",
        "role": "developer",
        "status": "completed",
        "content": [
            {
                "type": "input_text",
                "text": (
                    f"[PLLM local compaction: {elided} oldest items elided; "
                    "no semantic summary generated.]"
                ),
            }
        ],
    }


def _compaction_groups(items: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    groups: list[list[dict[str, Any]]] = []
    index = 0
    while index < len(items):
        item = items[index]
        if item.get("type") == "function_call" and item.get("call_id"):
            call_id = item["call_id"]
            end = next(
                (
                    candidate
                    for candidate in range(index + 1, len(items))
                    if items[candidate].get("type") == "function_call_output"
                    and items[candidate].get("call_id") == call_id
                ),
                index,
            )
            groups.append(items[index : end + 1])
            index = end + 1
            continue
        groups.append([item])
        index += 1
    return groups


def _compact_size(pinned: list[dict[str, Any]], items: list[dict[str, Any]], elided: int) -> int:
    output = pinned + ([_elision_marker(elided)] if elided else []) + items
    return len(json.dumps(output, separators=(",", ":"), ensure_ascii=False).encode())


def _response_message(output: Any) -> tuple[dict[str, Any], str]:
    text: list[str] = []
    refusal: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    for item in output if isinstance(output, list) else []:
        if not isinstance(item, dict):
            continue
        if item.get("type") == "message":
            for part in item.get("content", []):
                if isinstance(part, dict) and part.get("type") == "output_text":
                    text.append(str(part.get("text", "")))
                elif isinstance(part, dict) and part.get("type") == "refusal":
                    refusal.append(str(part.get("refusal", "")))
        elif item.get("type") == "function_call":
            tool_calls.append(
                {
                    "id": item.get("call_id") or item.get("id"),
                    "type": "function",
                    "function": {
                        "name": item.get("name", ""),
                        "arguments": item.get("arguments", ""),
                    },
                }
            )
    message: dict[str, Any] = {
        "role": "assistant",
        "content": "".join(text) if text else None,
        "refusal": "".join(refusal) if refusal else None,
    }
    if tool_calls:
        message["tool_calls"] = tool_calls
    return message, "tool_calls" if tool_calls else "stop"


def _response_finish_reason(response: dict[str, Any], default: str) -> str:
    if default == "tool_calls" or response.get("status") != "incomplete":
        return default
    details = response.get("incomplete_details")
    reason = details.get("reason") if isinstance(details, dict) else None
    if reason == "content_filter":
        return "content_filter"
    return "length"


def _chat_usage(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    result: dict[str, Any] = {
        "prompt_tokens": int(value.get("input_tokens", 0)),
        "completion_tokens": int(value.get("output_tokens", 0)),
        "total_tokens": int(value.get("total_tokens", 0)),
    }
    if isinstance(value.get("input_tokens_details"), dict):
        details = value["input_tokens_details"]
        result["prompt_tokens_details"] = {
            "cached_tokens": int(details.get("cached_tokens", 0)),
            "audio_tokens": int(details.get("audio_tokens", 0)),
        }
    if isinstance(value.get("output_tokens_details"), dict):
        details = value["output_tokens_details"]
        result["completion_tokens_details"] = {
            "reasoning_tokens": int(details.get("reasoning_tokens", 0)),
            "audio_tokens": int(details.get("audio_tokens", 0)),
            "accepted_prediction_tokens": int(details.get("accepted_prediction_tokens", 0)),
            "rejected_prediction_tokens": int(details.get("rejected_prediction_tokens", 0)),
        }
    return result


def _chat_chunk(
    response_id: str | None,
    model: str,
    created: int,
    delta: dict[str, Any],
    *,
    finish_reason: str | None = None,
    usage: dict[str, Any] | None = None,
) -> bytes:
    payload: dict[str, Any] = {
        "id": response_id or "",
        "object": "chat.completion.chunk",
        "created": int(created),
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": delta,
                "logprobs": None,
                "finish_reason": finish_reason,
            }
        ],
    }
    if usage is not None:
        payload["usage"] = usage
    return f"data: {json.dumps(payload, separators=(',', ':'), ensure_ascii=False)}\n\n".encode()


def _named_sse(event_type: str, payload: dict[str, Any]) -> bytes:
    data = json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
    return f"event: {event_type}\ndata: {data}\n\n".encode()


SSE_DONE = b"data: [DONE]\n\n"
