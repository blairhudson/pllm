from __future__ import annotations

import pytest

from pllm.runtime.client import _sampling_temperature
from pllm.runtime.responses import ResponsesError, normalize_input
from pllm.runtime.tools import parse_model_output, tool_policy


def _body(**values):
    return {
        "tools": [
            {
                "type": "function",
                "name": "weather",
                "description": "Get weather",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                    "additionalProperties": False,
                },
                "strict": True,
            }
        ],
        **values,
    }


def test_tool_policy_maps_openresponses_functions_to_chat_template_shape():
    policy = tool_policy(_body(tool_choice={"type": "function", "name": "weather"}))

    assert policy.mode == "named"
    assert policy.prompt_tools() == [
        {
            "type": "function",
            "function": {
                "name": "weather",
                "description": "Get weather",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                    "additionalProperties": False,
                },
                "strict": True,
            },
        }
    ]


def test_parse_model_output_returns_validated_calls_and_preserves_text():
    policy = tool_policy(_body())
    output = parse_model_output(
        'Checking.\n<tool_call>\n{"name":"weather","arguments":{"city":"Oslo"}}\n</tool_call>',
        policy,
    )

    assert output.text == "Checking."
    assert output.calls[0].name == "weather"
    assert output.calls[0].arguments == '{"city":"Oslo"}'


@pytest.mark.parametrize(
    ("text", "message"),
    [
        (
            '<tool_call>{"name":"unknown","arguments":{}}</tool_call>',
            "was not allowed",
        ),
        (
            '<tool_call>{"name":"weather","arguments":{}}</tool_call>',
            "do not match its schema",
        ),
        ("<tool_call>{bad}</tool_call>", "invalid function-call JSON"),
    ],
)
def test_parse_model_output_rejects_untrusted_model_calls(text: str, message: str):
    with pytest.raises(ResponsesError, match=message):
        parse_model_output(text, tool_policy(_body()))


def test_required_and_serial_tool_choices_are_enforced():
    with pytest.raises(ResponsesError, match="required function call"):
        parse_model_output("ordinary answer", tool_policy(_body(tool_choice="required")))

    body = _body(
        parallel_tool_calls=False,
        tools=[
            _body()["tools"][0],
            {
                "type": "function",
                "name": "clock",
                "parameters": {"type": "object"},
            },
        ],
    )
    with pytest.raises(ResponsesError, match="parallel function calls"):
        parse_model_output(
            '<tool_call>{"name":"weather","arguments":{"city":"Oslo"}}</tool_call>'
            '<tool_call>{"name":"clock","arguments":{}}</tool_call>',
            tool_policy(body),
        )

    body["parallel_tool_calls"] = True
    body["max_tool_calls"] = 1
    with pytest.raises(ResponsesError, match="max_tool_calls"):
        parse_model_output(
            '<tool_call>{"name":"weather","arguments":{"city":"Oslo"}}</tool_call>'
            '<tool_call>{"name":"clock","arguments":{}}</tool_call>',
            tool_policy(body),
        )


def test_function_items_normalize_to_model_tool_history():
    messages = normalize_input(
        [
            {
                "type": "function_call",
                "call_id": "call_1",
                "name": "weather",
                "arguments": '{"city":"Oslo"}',
            },
            {
                "type": "function_call_output",
                "call_id": "call_1",
                "output": {"temperature": 12},
            },
        ]
    )

    assert messages[0].to_prompt_dict() == {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "weather", "arguments": {"city": "Oslo"}},
            }
        ],
    }
    assert messages[1].to_prompt_dict() == {
        "role": "tool",
        "content": '{"temperature":12}',
        "tool_call_id": "call_1",
    }


def test_allowed_tool_choice_limits_model_calls() -> None:
    clock = {
        "type": "function",
        "name": "clock",
        "parameters": {"type": "object"},
    }
    policy = tool_policy(
        _body(
            tools=[_body()["tools"][0], clock],
            tool_choice={
                "type": "allowed_tools",
                "mode": "auto",
                "tools": [{"type": "function", "name": "clock"}],
            },
        )
    )
    with pytest.raises(ResponsesError, match="not allowed"):
        parse_model_output(
            '<tool_call>{"name":"weather","arguments":{"city":"Oslo"}}</tool_call>',
            policy,
        )
    parsed = parse_model_output('<tool_call>{"name":"clock","arguments":{}}</tool_call>', policy)
    assert parsed.calls[0].name == "clock"


def test_sampling_temperature_preserves_explicit_zero() -> None:
    assert _sampling_temperature({}) == 0.8
    assert _sampling_temperature({"temperature": None}) == 0.8
    assert _sampling_temperature({"temperature": 0}) == 0.0
