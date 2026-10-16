"""Exact response-wire accounting for the generated equalization regression test."""
from __future__ import annotations

from dataclasses import dataclass
import math

import msgpack

from pllm.runtime.stage_protocol import (
    MaskedStageResponse,
    PreparedStageBatchResponse,
    PREPARED_STAGE_BATCH_RESPONSE_MAGIC,
    PREPARED_STAGE_SINGLE_RESPONSE_MAGIC,
)


@dataclass(frozen=True)
class ResponseAccounting:
    stage_id: str
    shape: tuple[int, int]
    wire_bits: int
    payload: bytes
    arithmetic_payload: bytes
    server_ns_encoding: bytes
    non_timing_framing: bytes


def account_response(payload, *, stage_id, shape, wire_bits, batch_id=None, attempt_id=None):
    """Keep every framing byte; extract the actual encoded clock, not its estimate.

    Masked values and fresh request identifiers legitimately differ between runs.
    Validate their binding/size, retain their original bytes, then zero only their
    contents for framing comparison. Their type/length headers remain untouched.
    """
    assert type(payload) is bytes and len(payload) <= 1 << 20
    assert len(shape) == 2 and all(type(n) is int and 1 <= n <= 256 for n in shape)
    assert math.prod(shape) <= 65536 and wire_bits in {16, 24, 32}
    if batch_id is not None:
        assert payload.startswith((PREPARED_STAGE_BATCH_RESPONSE_MAGIC, PREPARED_STAGE_SINGLE_RESPONSE_MAGIC))
        prefix, body = payload[:12], payload[12:]
        decoded = PreparedStageBatchResponse.unpack(payload, max_rows=shape[0], max_tensor_elements=math.prod(shape))
        assert decoded.batch_id == batch_id
        identifier = bytes.fromhex(batch_id)
        names = ("request_id", "columns", "wire_bits", "server_ns", "data")
    else:
        prefix, body = b"", payload
        decoded = MaskedStageResponse.unpack(payload)
        assert decoded.correlation_id == attempt_id and decoded.stage_id == stage_id
        assert decoded.ring == f"u{wire_bits}" and decoded.modulus == 1 << wire_bits
        identifier = attempt_id.encode("ascii")
        names = {"v", "stage_id", "correlation_id", "ring", "modulus", "wire_bits", "server_ns", "shape", "data"}
    assert decoded.masked_output.shape == tuple(shape) and decoded.wire_bits == wire_bits
    assert len(identifier) == (16 if batch_id is not None else 32)
    reader = msgpack.Unpacker(raw=False, max_buffer_size=1 << 20, max_array_len=5, max_map_len=9, max_ext_len=0)
    reader.feed(body)
    count = reader.read_array_header() if batch_id is not None else reader.read_map_header()
    assert count == len(names)
    framing = [prefix, body[:reader.tell()]]
    fields, clock, arithmetic = {}, None, None
    for index in range(count):
        if batch_id is not None:
            name = names[index]
        else:
            start = reader.tell()
            name = reader.unpack()
            framing.append(body[start:reader.tell()])
            assert name in names and name not in fields
        start = reader.tell()
        value = reader.unpack()
        encoded = body[start:reader.tell()]
        fields[name] = value
        if name == "server_ns":
            assert type(value) is int and 0 <= value < 1 << 63
            clock = encoded
            framing.append(b"\x00")
        elif name in {"request_id", "correlation_id", "data"}:
            assert type(value) is (str if name == "correlation_id" else bytes)
            content = value.encode("ascii") if type(value) is str else value
            if name == "data":
                assert len(content) == math.prod(shape) * (wire_bits // 8)
                arithmetic = content
            else:
                assert content == identifier
            assert content and encoded.endswith(content)
            # Preserve the original MessagePack type/length prefix byte-for-byte.
            framing.append(encoded[:-len(content)] + bytes(len(content)))
        else:
            framing.append(encoded)
    assert reader.tell() == len(body), "unexplained trailing response bytes"
    assert set(fields) == set(names) and clock is not None and arithmetic is not None
    non_timing = b"".join(framing)
    assert len(payload) == len(non_timing) + len(clock) - 1
    return ResponseAccounting(stage_id, tuple(shape), wire_bits, payload, arithmetic, clock, non_timing)


def assert_matched_accounting(bodies, responses):
    """Reconcile counters to retained frames before comparing two numeric choices."""
    assert len(bodies) == len(responses) == 2
    assert bodies[0][:2] == bodies[1][:2], "stage calls or complete upload bytes differ"
    assert bodies[0][0] > 0
    for body, captured in zip(bodies, responses, strict=True):
        assert len(captured) == body[0], "stage calls do not match captured response count"
        assert sum(len(frame.payload) for frame in captured) == body[2], "unexplained download bytes"
        assert body[2] == sum(len(frame.non_timing_framing) + len(frame.server_ns_encoding) - 1
                              for frame in captured)
    for left, right in zip(*responses, strict=True):
        assert (left.stage_id, left.shape, left.wire_bits, len(left.arithmetic_payload)) == (
            right.stage_id, right.shape, right.wire_bits, len(right.arithmetic_payload)), "arithmetic geometry/payload differs"
        assert left.non_timing_framing == right.non_timing_framing, "non-timing framing differs"
    assert bodies[0][2] - bodies[1][2] == (
        sum(len(frame.server_ns_encoding) for frame in responses[0])
        - sum(len(frame.server_ns_encoding) for frame in responses[1])
    ), "download delta is not exactly the encoded timing-width delta"
