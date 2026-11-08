"""Exact-byte bounded FP32 bridge; no list/float64 transport path."""
import math
import struct

import pytest

from pllm import _native


def pack(*values):
    return struct.pack("<" + "f" * len(values), *values)


@pytest.mark.parametrize("kind", ["gelu_erf", "tanh"])
def test_source_float32_activation_fp32_wire_and_signed_zero(kind):
    values = (-4.0, -1.0, -0.0, 0.0, 1.0, 4.0)
    raw = pack(*values)
    output = _native.source_float32_activation(kind, raw)
    assert type(output) is bytes and len(output) == len(raw)
    actual = struct.unpack("<6f", output)
    for x, result in zip(values, actual):
        expected = math.tanh(x) if kind == "tanh" else x / 2 * (1 + math.erf(x / math.sqrt(2)))
        assert result == pytest.approx(expected, rel=3e-6, abs=2e-7)
    assert output[8:12] == pack(-0.0)
    assert output[12:16] == pack(0.0)
    assert _native.source_float32_activation(kind, b"") == b""
    assert raw == pack(*values)


@pytest.mark.parametrize("kind", ["gelu_erf", "tanh"])
def test_source_float32_activation_refuses_nonfinite_and_invalid_transport(kind):
    for value in (float("nan"), float("inf"), -float("inf")):
        with pytest.raises(ValueError):
            _native.source_float32_activation(kind, pack(0.0, value))
    for raw in (b"x", b"xx", b"xxx", b"\0" * ((1 << 20) + 4)):
        with pytest.raises(ValueError):
            _native.source_float32_activation(kind, raw)
    for raw in ([0.0], bytearray(4), memoryview(bytes(4))):
        with pytest.raises(TypeError):
            _native.source_float32_activation(kind, raw)
    assert len(_native.source_float32_activation(kind, bytes(1 << 20))) == 1 << 20


def test_source_float32_activation_closed_kind_set():
    for kind in ("gelu", "gelu_tanh", "silu", "Tanh", ""):
        with pytest.raises(ValueError):
            _native.source_float32_activation(kind, bytes(4))
