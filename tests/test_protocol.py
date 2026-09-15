from __future__ import annotations

import dataclasses
import time

import pytest

from pllm.runtime.framing import FrameError, decode_frame, encode_frame
from pllm.runtime.protocol import (
    ProtocolEnvelope,
    ProtocolError,
    ReplayWindow,
    encode_length_prefixed,
    envelope_from_json,
    envelope_to_json,
    iter_length_prefixed,
    pack_envelope,
    unpack_envelope,
)
from pllm.runtime.responses import ResponsesError, normalize_input, prompt_text
from pllm.runtime.types import Response, response_events


KEY = b"k" * 32


def envelope(**overrides):
    values = dict(
        request_id="req_1",
        session_id="hes_1",
        model="m",
        kind="masked.linear",
        sequence=1,
        payload=b"opaque",
        metadata={"stage": "x"},
        key=KEY,
    )
    values.update(overrides)
    return ProtocolEnvelope.create(**values)


def test_envelope_binary_and_json_roundtrip():
    original = envelope()
    binary = unpack_envelope(pack_envelope(original))
    binary.verify(KEY)
    assert binary == original
    json_value = envelope_from_json(envelope_to_json(original))
    json_value.verify(KEY)
    assert json_value == original


def test_envelope_tamper_and_wrong_key_fail():
    original = envelope()
    with pytest.raises(ProtocolError, match="MAC"):
        dataclasses.replace(original, payload=b"changed").verify(KEY)
    with pytest.raises(ProtocolError, match="MAC"):
        original.verify(b"z" * 32)


def test_stale_envelope_fails():
    original = dataclasses.replace(envelope(), created_at=time.time() - 999).sign(KEY)
    with pytest.raises(ProtocolError, match="stale"):
        original.verify(KEY, max_age_seconds=1)


def test_replay_window_rejects_nonce_and_sequence_reuse():
    replay = ReplayWindow()
    first = envelope(sequence=1)
    replay.accept(first)
    with pytest.raises(ProtocolError, match="nonce"):
        replay.accept(first)
    with pytest.raises(ProtocolError, match="sequence"):
        replay.accept(envelope(sequence=1))


def test_length_prefixed_frames_roundtrip_and_truncation():
    frames = [b"a", b"bc", b""]
    value = encode_length_prefixed(frames)
    assert list(iter_length_prefixed(value)) == frames
    with pytest.raises(ProtocolError):
        list(iter_length_prefixed(value[:-1]))


def test_binary_frame_roundtrip_and_limits():
    raw = encode_frame({"kind": "stage", "n": 3}, b"ciphertext")
    frame = decode_frame(raw)
    assert frame.header == {"kind": "stage", "n": 3}
    assert frame.payload == b"ciphertext"
    with pytest.raises(FrameError):
        decode_frame(b"bad")
    with pytest.raises(FrameError, match="limit"):
        decode_frame(raw, max_payload=2)


def test_response_shape_and_stream_lifecycle():
    response = Response.text_response(model="m", text="hello", input_tokens=2, output_tokens=1)
    assert response.output_text == "hello"
    rebuilt = Response.from_dict(response.to_dict())
    assert rebuilt.output_text == "hello"
    events = list(response_events(response, ["he", "llo"]))
    assert [event.sequence_number for event in events] == list(range(len(events)))
    assert events[0].type == "response.created"
    assert events[-1].type == "response.completed"
    assert "".join(e.delta for e in events if e.type == "response.output_text.delta") == "hello"


@pytest.mark.parametrize(
    "value,expected",
    [
        ("hello", "user: hello"),
        ([{"role": "user", "content": "hello"}], "user: hello"),
        ([{"type": "input_text", "text": "hello"}], "user: hello"),
        ([{"role": "user", "content": [{"type": "input_text", "text": "a"}, {"type": "text", "text": "b"}]}], "user: ab"),
    ],
)
def test_responses_input_normalization(value, expected):
    assert prompt_text(normalize_input(value)) == expected


def test_responses_rejects_multimodal_in_strict_text_runtime():
    with pytest.raises(ResponsesError, match="input_image"):
        normalize_input([{"role": "user", "content": [{"type": "input_image", "image_url": "x"}]}])
