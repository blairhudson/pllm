"""Shared transport admission only; no importer or source-authority claims."""

import inspect

import msgpack
import pytest

from pllm.runtime import bundle_document


BYTE_BOUND = "^bundle document exceeds its byte bound$"
MAP_KEYS = "^bundle maps require unique string keys$"


class BytesSubclass(bytes):
    pass


class BytearraySubclass(bytearray):
    pass


@pytest.fixture
def document_payload():
    document = {
        "v": 2,
        "metadata": {"name": "small transport fixture", "shape": [2, 2]},
        "data": b"\x00\x01\x80\xff",
    }
    payload = msgpack.packb(document, use_bin_type=True)
    # Genuine default decode is a control outside every rejection capture.
    assert bundle_document.unpack_document(payload) == document
    return document, payload


@pytest.mark.parametrize("container", [bytes, bytearray])
def test_unpack_document_decodes_complete_genuine_document(document_payload, container):
    document, payload = document_payload
    assert bundle_document.unpack_document(container(payload)) == document


@pytest.mark.parametrize("payload", [
    None,
    "not bytes",
    memoryview(b"\x80"),
    BytesSubclass(b"\x80"),
    BytearraySubclass(b"\x80"),
    b"",
    bytearray(),
], ids=["none", "str", "memoryview", "bytes-subclass", "bytearray-subclass", "empty-bytes", "empty-bytearray"])
def test_unpack_document_rejects_invalid_input_before_decoder(document_payload, monkeypatch, payload):
    calls = []

    def forbidden(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("invalid transport input reached MessagePack decoder")

    with monkeypatch.context() as patch:
        patch.setattr(bundle_document.msgpack, "unpackb", forbidden)
        with pytest.raises(ValueError, match=BYTE_BOUND) as caught:
            bundle_document.unpack_document(payload)
    assert type(caught.value) is ValueError
    assert calls == []


def test_unpack_document_explicit_cap_exact_boundary(document_payload, monkeypatch):
    document, payload = document_payload
    assert bundle_document.unpack_document(payload, maximum=len(payload)) == document
    calls = []

    def forbidden(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("over-cap transport input reached MessagePack decoder")

    with monkeypatch.context() as patch:
        patch.setattr(bundle_document.msgpack, "unpackb", forbidden)
        with pytest.raises(ValueError, match=BYTE_BOUND) as caught:
            bundle_document.unpack_document(payload, maximum=len(payload) - 1)
    assert type(caught.value) is ValueError
    assert calls == []


def test_unpack_document_declares_eight_gib_captured_default(document_payload):
    parameter = inspect.signature(bundle_document.unpack_document).parameters["maximum"]
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert parameter.default == 8 << 30
    document, payload = document_payload
    assert bundle_document.unpack_document(payload) == document
    # Declaration plus small default control, not an 8 GiB allocation or proof
    # that changing a module constant changes the captured keyword default.


def test_unpack_document_rejects_real_duplicate_map_keys(document_payload):
    packer = msgpack.Packer(use_bin_type=True)
    payload = b"".join([
        packer.pack_map_header(2),
        packer.pack("tensor"), packer.pack(b"first"),
        packer.pack("tensor"), packer.pack(b"second"),
    ])
    with pytest.raises(ValueError, match=MAP_KEYS) as caught:
        bundle_document.unpack_document(payload)
    assert type(caught.value) is ValueError


def test_unpack_document_pair_hook_rejects_binary_map_key(document_payload):
    packer = msgpack.Packer(use_bin_type=True)
    payload = b"".join([
        packer.pack_map_header(1), packer.pack(b"tensor"), packer.pack(b"value"),
    ])
    # Binary keys are accepted by MessagePack's strict-map policy, then rejected
    # by our unique-string-key hook. Integer keys would test a different error.
    with pytest.raises(ValueError, match=MAP_KEYS) as caught:
        bundle_document.unpack_document(payload)
    assert type(caught.value) is ValueError


@pytest.mark.parametrize("artifact_limits", [False, True], ids=["direct", "artifact"])
def test_unpack_document_forwards_real_decoder_limits(document_payload, monkeypatch, artifact_limits):
    document, payload = document_payload
    original = bundle_document.msgpack.unpackb
    calls = []
    maximum = len(payload) + 17

    def record_and_forward(*args, **kwargs):
        calls.append((args, dict(kwargs)))
        return original(*args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(bundle_document.msgpack, "unpackb", record_and_forward)
        decoded = bundle_document.unpack_document(
            payload, maximum=maximum, artifact_limits=artifact_limits,
        )
    assert decoded == document
    assert len(calls) == 1
    args, kwargs = calls[0]
    assert args == (payload,)
    assert callable(kwargs.pop("object_pairs_hook"))
    n = len(payload)
    assert kwargs == {
        "raw": False,
        "max_bin_len": maximum if artifact_limits else n,
        "max_str_len": bundle_document.MAX_MANIFEST_BYTES if artifact_limits else n,
        "max_array_len": 65536 if artifact_limits else n,
        "max_map_len": 65536 if artifact_limits else n // 2,
        "max_ext_len": 32 if artifact_limits else n,
    }
