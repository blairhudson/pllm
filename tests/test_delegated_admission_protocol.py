"""Shallow admission rejects hostile headers before any value decoding."""

import msgpack
import pytest

from pllm.runtime import delegated_protocol as protocol


def metadata():
    return {"schema": "pllm.gkm.admit.v1", "request": "a" * 64, "epoch": "b" * 64,
            "configuration": "c" * 64, "source_lock": "d" * 64, "tokenizer": "e" * 64,
            "input_rows": 1, "output_tokens": 2, "publication": b"opaque\x00publication",
            "start": b"opaque\xffstart", "pin": bytes(range(32))}


def envelope(pairs):
    return b"\x8b" + b"".join(msgpack.packb(k, use_bin_type=True) + v for k, v in pairs)


@pytest.mark.parametrize("capacity", ["capsule", "decoder"])
@pytest.mark.parametrize("order", [False, True])
def test_ordinary_and_reordered_admissions_preserve_opaque_bytes(capacity, order):
    value = metadata()
    if order:
        value = dict(reversed(list(value.items())))
    assert protocol.unpack_admission(protocol.pack(value), capacity) == value


@pytest.mark.parametrize("capacity", ["capsule", "decoder"])
@pytest.mark.parametrize("width", [1, 2, 4])
def test_legal_nonminimal_headers(capacity, width):
    value = metadata()
    raw = b"\xde\x00\x0b" if width == 1 else b"\xdf\x00\x00\x00\x0b"

    def string(data):
        return bytes([{1: 0xD9, 2: 0xDA, 4: 0xDB}[width]]) + len(data).to_bytes(width, "big") + data

    for key, item in reversed(list(value.items())):
        raw += string(key.encode())
        if isinstance(item, str):
            raw += string(item.encode())
        elif isinstance(item, bytes):
            raw += bytes([{1: 0xC4, 2: 0xC5, 4: 0xC6}[width]]) + len(item).to_bytes(width, "big") + item
        else:
            raw += b"\xd3" + item.to_bytes(8, "big", signed=True)
    assert protocol.unpack_admission(raw, capacity) == value


@pytest.mark.parametrize("code,width,signed", [(0xCC, 1, False), (0xCD, 2, False),
    (0xCE, 4, False), (0xCF, 8, False), (0xD0, 1, True), (0xD1, 2, True),
    (0xD2, 4, True), (0xD3, 8, True)])
def test_all_legal_integer_widths(code, width, signed):
    pairs = [(key, bytes([code]) + value.to_bytes(width, "big", signed=signed)
              if type(value) is int else msgpack.packb(value, use_bin_type=True))
             for key, value in metadata().items()]
    assert protocol.unpack_admission(envelope(pairs)) == metadata()


ATTACKS = [
    b"\x8b\xabpublication" + header for header in
    (b"\xdc\x00\x20", b"\xdd", b"\x90", b"\x80", b"\xde", b"\xdf",
     b"\xc7", b"\xd4", b"\xc1", b"\xc6\xff\xff\xff\xff", b"\xc4\x00")
] + [b"", b"\x80", b"\x90", b"\xc1", b"\xde", b"\xdf\xff\xff\xff\xff",
     b"\x8b\xdb\xff\xff\xff\xff", b"\x8b\xc4\x06schema",
     b"\x8b\xa7unknown\xc0", b"\x8b\xa6schema\xdb\xff\xff\xff\xff"]


@pytest.mark.parametrize("capacity", ["capsule", "decoder"])
@pytest.mark.parametrize("raw", ATTACKS)
def test_header_attacks_never_reach_decoder(monkeypatch, capacity, raw):
    def forbidden(*args, **kwargs):
        pytest.fail("value decoder reached before complete preflight")

    monkeypatch.setattr(protocol.msgpack, "unpackb", forbidden)
    with pytest.raises(ValueError):
        protocol.unpack_admission(raw, capacity)


@pytest.mark.parametrize("key,bad", [
    ("schema", "x" * 17), ("request", "a" * 63), ("epoch", b"b" * 64),
    ("input_rows", True), ("input_rows", None), ("output_tokens", 1.0),
    ("publication", b""), ("publication", []), ("publication", {}),
    ("start", b""), ("pin", b"p" * 31), ("pin", b"p" * 33),
])
def test_wrong_types_lengths_and_counters_reject_before_unpack(monkeypatch, key, bad):
    value = metadata()
    value[key] = bad
    raw = protocol.pack(value)
    monkeypatch.setattr(protocol.msgpack, "unpackb", lambda *a, **k: pytest.fail("decoded invalid admission"))
    with pytest.raises(ValueError):
        protocol.unpack_admission(raw)


@pytest.mark.parametrize("mode", ["duplicate", "missing", "unknown", "trailing", "truncated"])
def test_complete_shape_scan_before_unpack(monkeypatch, mode):
    pairs = [(k, msgpack.packb(v, use_bin_type=True)) for k, v in metadata().items()]
    if mode == "duplicate":
        pairs[-1] = pairs[0]
    elif mode == "missing":
        pairs.pop()
    elif mode == "unknown":
        pairs[-1] = ("unknown", pairs[-1][1])
    raw = envelope(pairs)
    if mode == "trailing":
        raw += b"\xc0"
    elif mode == "truncated":
        raw = raw[:-1]
    monkeypatch.setattr(protocol.msgpack, "unpackb", lambda *a, **k: pytest.fail("decoded invalid admission"))
    with pytest.raises(ValueError):
        protocol.unpack_admission(raw)


@pytest.mark.parametrize("capacity,pub,start", [("capsule", 512 << 10, 8192), ("decoder", 32 << 20, 4 << 20)])
@pytest.mark.parametrize("field", ["publication", "start"])
def test_binary_header_bound_before_payload_or_decode(monkeypatch, capacity, pub, start, field):
    raw = b"\x8b" + msgpack.packb(field) + b"\xc6" + ((pub if field == "publication" else start) + 1).to_bytes(4, "big")
    monkeypatch.setattr(protocol.msgpack, "unpackb", lambda *a, **k: pytest.fail("decoded oversized span"))
    with pytest.raises(ValueError):
        protocol.unpack_admission(raw, capacity)
