"""Lightweight bridge routing checks; no rebuilt extension required."""

import hashlib
import json

import pytest

from pllm import _native
from pllm.runtime import delegated_decoder as decoder


@pytest.mark.parametrize("capacity", ["capsule", "decoder"])
@pytest.mark.parametrize("streamed", [False, True])
def test_decoder_contract_routes_optional_streamed_keyword(monkeypatch, capacity, streamed):
    calls = []
    expected = b'{"native":"untouched"}'

    def native(config, rows, steps, **kwargs):
        calls.append((config, rows, steps, kwargs))
        return expected

    monkeypatch.setattr(_native, "gkm_decoder_contract", native)
    assert decoder.decoder_contract({}, input_rows=3, decode_steps=2,
                                    capacity=capacity, streamed=streamed) is expected
    kwargs = {"capacity": capacity, "streamed": True} if streamed else (
        {"capacity": capacity} if capacity != "capsule" else {})
    assert calls == [(b"{}", 3, 2, kwargs)]


def test_decoder_contract_defaults_to_generic(monkeypatch):
    def native(config, rows, steps):
        return b"generic"

    monkeypatch.setattr(_native, "gkm_decoder_contract", native)
    assert decoder.decoder_contract({}, input_rows=1, decode_steps=0) == b"generic"


@pytest.mark.parametrize("streamed", [None, 0, 1, "true", [], {}])
def test_decoder_contract_rejects_non_boolean_streamed_before_native(monkeypatch, streamed):
    def forbidden(*args, **kwargs):
        raise AssertionError("invalid selector reached native")

    monkeypatch.setattr(_native, "gkm_decoder_contract", forbidden)
    with pytest.raises(ValueError, match="streamed must be a boolean"):
        decoder.decoder_contract({}, input_rows=1, decode_steps=0, streamed=streamed)


def test_decoder_contract_streamed_is_keyword_only():
    with pytest.raises(TypeError):
        decoder.decoder_contract({}, 1, 0, "capsule", True)


@pytest.mark.skipif("streamed" not in (_native.gkm_decoder_contract.__text_signature__ or ""),
                    reason="requires rebuilt native streamed-contract bridge")
@pytest.mark.parametrize("capacity", ["capsule", "decoder"])
def test_native_streamed_contract_preserves_general_source_identity(capacity):
    from test_delegated_decoder import config

    encoded = decoder._encode_config(config())
    generic = _native.gkm_decoder_contract(encoded, 3, 2, capacity=capacity)
    assert _native.gkm_decoder_contract(encoded, 3, 2, capacity=capacity, streamed=False) == generic
    streamed = _native.gkm_decoder_contract(encoded, 3, 2, capacity=capacity, streamed=True)
    assert decoder.decoder_contract(config(), input_rows=3, decode_steps=2,
                                    capacity=capacity, streamed=True) == streamed
    general_record, streamed_record = json.loads(generic), json.loads(streamed)
    assert general_record != streamed_record
    assert streamed_record["source_binding_digest"] == hashlib.sha256(
        b"pllm.gkm.decoder-reference.v1\0" + generic
    ).hexdigest()
    assert streamed_record["transport_resources"]["profile"] == "pllm.gkm.streamed-single-window.v1"
    for key in ("plan", "source_shapes", "source_float32_bytes", "numeric_profile", "local_numeric_profile"):
        assert streamed_record[key] == general_record[key]
    with pytest.raises(TypeError):
        _native.gkm_decoder_contract(encoded, 3, 2, capacity, True)
