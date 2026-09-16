from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from jsonschema import SchemaError, ValidationError, validators

from .responses import ResponsesError


_TOOL_CALL = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


@dataclass(frozen=True, slots=True)
class ToolPolicy:
    tools: tuple[dict[str, Any], ...]
    mode: str
    allowed_names: frozenset[str]
    parallel: bool
    max_calls: int | None

    @property
    def enabled(self) -> bool:
        return bool(self.tools) and self.mode != "none"

    def prompt_tools(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    key: value
                    for key, value in tool.items()
                    if key in {"name", "description", "parameters", "strict"}
                },
            }
            for tool in self.tools
            if tool["name"] in self.allowed_names
        ]

    def prompt_instruction(self) -> str | None:
        if self.mode == "none":
            return "Do not call functions. Answer with text."
        if self.mode == "required":
            return "Call at least one provided function. Do not answer with ordinary text."
        if self.mode == "named":
            name = next(iter(self.allowed_names))
            return f"Call the {name} function. Do not answer with ordinary text."
        if not self.parallel:
            return "Call at most one function."
        if self.max_calls is not None:
            return f"Call no more than {self.max_calls} functions."
        return None


@dataclass(frozen=True, slots=True)
class ParsedToolCall:
    name: str
    arguments: str


@dataclass(frozen=True, slots=True)
class ParsedModelOutput:
    text: str
    calls: tuple[ParsedToolCall, ...]


def tool_policy(body: dict[str, Any]) -> ToolPolicy:
    raw_tools = body.get("tools") or []
    if not isinstance(raw_tools, list):
        raise ResponsesError("tools must be an array", param="tools")
    tools: list[dict[str, Any]] = []
    names: set[str] = set()
    for index, value in enumerate(raw_tools):
        param = f"tools[{index}]"
        if not isinstance(value, dict) or value.get("type") != "function":
            raise ResponsesError("only function tools are supported", param=param)
        name = value.get("name")
        if not isinstance(name, str) or not name:
            raise ResponsesError("function name must be a non-empty string", param=f"{param}.name")
        if name in names:
            raise ResponsesError(f"duplicate function tool {name!r}", param=f"{param}.name")
        parameters = value.get("parameters", {"type": "object"})
        if not isinstance(parameters, dict):
            raise ResponsesError(
                "function parameters must be an object", param=f"{param}.parameters"
            )
        if value.get("strict"):
            try:
                validators.validator_for(parameters).check_schema(parameters)
            except SchemaError as exc:
                raise ResponsesError(
                    f"invalid JSON Schema for function {name!r}: {exc.message}",
                    param=f"{param}.parameters",
                ) from exc
        normalized = dict(value)
        normalized["parameters"] = parameters
        tools.append(normalized)
        names.add(name)

    choice = body.get("tool_choice", "auto")
    mode = "auto"
    allowed = set(names)
    if isinstance(choice, str):
        if choice not in {"auto", "none", "required"}:
            raise ResponsesError("unsupported tool_choice", param="tool_choice")
        mode = choice
    elif isinstance(choice, dict) and choice.get("type") == "function":
        name = choice.get("name")
        if not isinstance(name, str) or name not in names:
            raise ResponsesError("named tool_choice is not present in tools", param="tool_choice")
        mode = "named"
        allowed = {name}
    elif isinstance(choice, dict) and choice.get("type") == "allowed_tools":
        choice_mode = choice.get("mode", "auto")
        if choice_mode not in {"auto", "required"}:
            raise ResponsesError("allowed_tools mode must be auto or required", param="tool_choice")
        selected = choice.get("tools")
        if not isinstance(selected, list) or not selected:
            raise ResponsesError(
                "allowed_tools.tools must be a non-empty array", param="tool_choice"
            )
        allowed = set()
        for value in selected:
            if not isinstance(value, dict) or value.get("type") != "function":
                raise ResponsesError("allowed_tools supports only functions", param="tool_choice")
            name = value.get("name")
            if not isinstance(name, str) or name not in names:
                raise ResponsesError("allowed tool is not present in tools", param="tool_choice")
            allowed.add(name)
        mode = choice_mode
    else:
        raise ResponsesError("unsupported tool_choice", param="tool_choice")

    if mode in {"required", "named"} and not tools:
        raise ResponsesError("tool_choice requires at least one function tool", param="tool_choice")
    max_calls = body.get("max_tool_calls")
    if max_calls is not None and (
        not isinstance(max_calls, int) or isinstance(max_calls, bool) or max_calls < 1
    ):
        raise ResponsesError("max_tool_calls must be a positive integer", param="max_tool_calls")
    return ToolPolicy(
        tuple(tools),
        mode,
        frozenset(allowed),
        bool(body.get("parallel_tool_calls", True)),
        max_calls,
    )


def parse_model_output(text: str, policy: ToolPolicy) -> ParsedModelOutput:
    if not policy.enabled:
        return ParsedModelOutput(text, ())

    matches = list(_TOOL_CALL.finditer(text))
    if ("<tool_call>" in text or "</tool_call>" in text) and not matches:
        raise ResponsesError("model emitted a malformed function call", code="model_error")

    calls: list[ParsedToolCall] = []
    tool_by_name = {str(tool["name"]): tool for tool in policy.tools}
    for match in matches:
        try:
            value = json.loads(match.group(1))
        except json.JSONDecodeError as exc:
            raise ResponsesError(
                "model emitted invalid function-call JSON", code="model_error"
            ) from exc
        if not isinstance(value, dict):
            raise ResponsesError("model function call must be a JSON object", code="model_error")
        name = value.get("name")
        arguments = value.get("arguments")
        if not isinstance(name, str) or name not in policy.allowed_names:
            raise ResponsesError("model called a function that was not allowed", code="model_error")
        if not isinstance(arguments, dict):
            raise ResponsesError(
                "model function arguments must be a JSON object", code="model_error"
            )
        tool = tool_by_name[name]
        if tool.get("strict"):
            schema = tool.get("parameters", {"type": "object"})
            try:
                validators.validator_for(schema)(schema).validate(arguments)
            except ValidationError as exc:
                raise ResponsesError(
                    f"model arguments for {name!r} do not match its schema: {exc.message}",
                    code="model_error",
                ) from exc
        calls.append(
            ParsedToolCall(
                name=name,
                arguments=json.dumps(arguments, separators=(",", ":"), ensure_ascii=False),
            )
        )

    if policy.mode in {"required", "named"} and not calls:
        raise ResponsesError("model did not emit the required function call", code="model_error")
    if not policy.parallel and len(calls) > 1:
        raise ResponsesError(
            "model emitted parallel function calls when disabled", code="model_error"
        )
    if policy.max_calls is not None and len(calls) > policy.max_calls:
        raise ResponsesError("model exceeded max_tool_calls", code="model_error")

    plain_parts: list[str] = []
    offset = 0
    for match in matches:
        plain_parts.append(text[offset : match.start()])
        offset = match.end()
    plain_parts.append(text[offset:])
    plain = "".join(plain_parts).strip()
    if "<tool_call>" in plain or "</tool_call>" in plain:
        raise ResponsesError("model emitted a malformed function call", code="model_error")
    if policy.mode in {"required", "named"} and plain:
        raise ResponsesError(
            "model emitted text when a function call was required", code="model_error"
        )
    return ParsedModelOutput(plain, tuple(calls))
