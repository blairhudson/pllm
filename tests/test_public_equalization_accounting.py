"""Wire-only controls: no providers, checkpoints or cryptographic setup."""
from __future__ import annotations

from dataclasses import replace
import struct

import msgpack
import numpy as np
import pytest
from public_equalization_accounting import account_response, assert_matched_accounting

from pllm.runtime.protocol import ProtocolError
from pllm.runtime.stage_protocol import MaskedStageResponse, PreparedStageBatchResponse


def _packet(kind="batch", *, clock=65535, shape=None, bits=16, identifier="0" * 32, stage="stage-a", value=0):
    shape = shape or ((2, 6) if kind == "batch" else (1, 12))
    data = np.full(shape, value, dtype=np.uint32)
    payload = (MaskedStageResponse(identifier, data, 1 << bits, bits, clock, stage).pack()
               if kind == "masked" else PreparedStageBatchResponse(identifier, data, bits, clock).pack())
    kwargs = {"stage_id": stage, "shape": shape, "wire_bits": bits,
              "attempt_id" if kind == "masked" else "batch_id": identifier}
    return payload, kwargs


def _compare(left, right):
    assert_matched_accounting([(1, 512, len(left.payload)), (1, 512, len(right.payload))], [[left], [right]])


@pytest.mark.parametrize("kind", ["batch", "single", "masked"])
def test_response_accounting_accepts_only_timing_width_and_fresh_mask_contents(kind):
    first, kwargs = _packet(kind)
    second, other_kwargs = _packet(kind, clock=65536, identifier="f" * 32, value=73)
    left, right = account_response(first, **kwargs), account_response(second, **other_kwargs)
    _compare(left, right)
    assert left.payload == first and right.payload == second
    assert left.arithmetic_payload != right.arithmetic_payload
    assert len(right.payload) - len(left.payload) == 2
    assert (len(left.server_ns_encoding), len(right.server_ns_encoding)) == (3, 5)


def test_response_accounting_uses_actual_encoded_clock_length():
    first, kwargs = _packet(clock=1)
    fields = msgpack.unpackb(first[12:], raw=False)
    # Legal MessagePack uint64 representation of the SAME value, not minimal width.
    second = first[:12] + b"\x95" + b"".join(msgpack.packb(value, use_bin_type=True) for value in fields[:3])
    second += b"\xcf" + struct.pack(">Q", 1) + msgpack.packb(fields[4], use_bin_type=True)
    left, right = account_response(first, **kwargs), account_response(second, **kwargs)
    _compare(left, right)
    assert (len(left.server_ns_encoding), len(right.server_ns_encoding)) == (1, 9)
    assert len(right.payload) - len(left.payload) == 8


@pytest.mark.parametrize("change", ["geometry", "wire-width", "stage-order", "integer-framing", "data-length-header"])
def test_response_accounting_rejects_non_timing_changes(change):
    first, kwargs = _packet()
    if change == "geometry":
        second, other_kwargs = _packet(shape=(3, 4))  # Same arithmetic byte count.
    elif change == "wire-width":
        second, other_kwargs = _packet(shape=(2, 4), bits=24)  # Same byte count again.
    elif change == "stage-order":
        second, other_kwargs = _packet(stage="stage-b")
    else:
        fields = msgpack.unpackb(first[12:], raw=False)
        encoded = [msgpack.packb(value, use_bin_type=True) for value in fields]
        if change == "integer-framing":
            encoded[1] = b"\xcc" + bytes([fields[1]])  # Same columns; extra non-clock byte.
        else:
            encoded[4] = b"\xc5" + struct.pack(">H", len(fields[4])) + fields[4]
        second, other_kwargs = first[:12] + b"\x95" + b"".join(encoded), kwargs
    left, right = account_response(first, **kwargs), account_response(second, **other_kwargs)
    with pytest.raises(AssertionError, match="geometry/payload|non-timing framing"):
        _compare(left, right)


@pytest.mark.parametrize("change", ["extra-field", "trailing-byte", "truncated-data", "wrong-request", "wrong-clock-type"])
def test_response_accounting_rejects_unbound_or_malformed_frames(change):
    payload, kwargs = _packet()
    fields = msgpack.unpackb(payload[12:], raw=False)
    if change == "extra-field":
        fields.append(0)
    elif change == "truncated-data":
        fields[-1] = fields[-1][:-1]
    elif change == "wrong-request":
        fields[0] = bytes.fromhex("f" * 32)
    elif change == "wrong-clock-type":
        fields[3] = float(fields[3])
    changed = payload[:12] + msgpack.packb(fields, use_bin_type=True) + (b"\x00" if change == "trailing-byte" else b"")
    with pytest.raises((AssertionError, ProtocolError)):
        account_response(changed, **kwargs)


@pytest.mark.parametrize("change", ["calls", "uploads", "downloads", "omitted-frame", "clock-span"])
def test_response_accounting_rejects_unexplained_audit_bytes(change):
    payload, kwargs = _packet()
    frame = account_response(payload, **kwargs)
    bodies, responses = [(1, 512, len(payload)), (1, 512, len(payload))], [[frame], [frame]]
    if change == "omitted-frame":
        responses[1] = []
    elif change == "clock-span":
        responses[1] = [replace(frame, server_ns_encoding=b"\x00")]
    else:
        index = {"calls": 0, "uploads": 1, "downloads": 2}[change]
        changed = list(bodies[1])
        changed[index] += 1
        bodies[1] = tuple(changed)
    with pytest.raises(AssertionError):
        assert_matched_accounting(bodies, responses)
