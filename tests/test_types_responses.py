from __future__ import annotations

import pytest

from pllm.runtime.responses import ResponsesError, normalize_input, prompt_text
from pllm.runtime.types import Response, ResponseEvent, response_events


def test_response_output_text_and_model_dump():
    response = Response.text_response(model="m", text="hello", input_tokens=3, output_tokens=1)
    assert response.output_text == "hello"
    assert response.model_dump()["object"] == "response"
    assert response.usage.total_tokens == 4


def test_response_from_dict_round_trip():
    original = Response.text_response(model="m", text="hello", input_tokens=3, output_tokens=1)
    restored = Response.from_dict(original.to_dict())
    assert restored.output_text == "hello"
    assert restored.id == original.id


def test_response_stream_lifecycle_and_sequence():
    response = Response.text_response(model="m", text="hello", input_tokens=1, output_tokens=2)
    events = list(response_events(response, ["he", "llo"]))
    assert events[0].type == "response.created"
    assert events[-1].type == "response.completed"
    assert [event.sequence_number for event in events] == list(range(len(events)))
    assert [e.delta for e in events if e.type == "response.output_text.delta"] == ["he", "llo"]


def test_event_attribute_access():
    event = ResponseEvent.from_dict(
        {"type": "response.output_text.delta", "sequence_number": 2, "delta": "x"}
    )
    assert event.delta == "x"
    with pytest.raises(AttributeError):
        _ = event.missing


def test_normalize_responses_input():
    rows = normalize_input(
        [{"role": "user", "content": [{"type": "input_text", "text": "secret"}]}],
        instructions="be precise",
    )
    assert prompt_text(rows) == "system: be precise\nuser: secret"


def test_normalize_marks_media_for_text_runtime():
    with pytest.raises(ResponsesError, match="input_image"):
        normalize_input([{"role": "user", "content": [{"type": "input_image", "image_url": "x"}]}])


def test_normalize_input_accepts_compaction_state() -> None:
    rows = normalize_input(
        [
            {"type": "compaction", "id": "cmp_1", "encrypted_content": "opaque"},
            {"role": "user", "content": "continue"},
        ]
    )
    assert [(row.role, row.text) for row in rows] == [("user", "continue")]
