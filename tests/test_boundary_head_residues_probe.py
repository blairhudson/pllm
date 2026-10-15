"""Small native-codec checks; run against installed wheel, not source shadow."""
import copy
import importlib.util
import json
from pathlib import Path
import secrets
import struct
import sys

import numpy as np
import pytest
from pllm import _native

PATH = Path(__file__).resolve().parents[1] / "scripts/probe_boundary_head_residues.py"
SPEC = importlib.util.spec_from_file_location("boundary_head_residues_probe", PATH)
probe = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = probe
SPEC.loader.exec_module(probe)


@pytest.fixture
def fixture():
    weights = bytes([0, 0, 1, 255, 128, 128])
    scales = np.ones(3, dtype="<f4").tobytes()
    pin = {key: probe.sha(key.encode()) for key in (
        "source_lock_sha256", "checkpoint_sha256", "model_config_sha256", "descriptor_sha256")}
    pin.update(weight_sha256=probe.sha(weights), scales_sha256=probe.sha(scales),
               vocabulary=3, hidden=2, qmax=127, weight_dtype="i8",
               activation_quantizer=probe.QUANTIZER)
    cert = probe.certificate(weights, scales, pin)
    return weights, scales, pin, cert


def test_widths_are_public_signed_bounds(fixture):
    weights, _, _, cert = fixture
    assert cert.widths == bytes([1, 9, 16])
    assert cert.widths == _native.offset_row_bits(weights, 2, 127)
    # Includes -128, not incorrectly clamped to -127.
    widths, _ = probe.oracle_layout(bytes([128]) * 896, 896)
    assert widths == bytes([25])


def test_two_fresh_shares_negative_wrap_and_oracle(fixture):
    _, _, _, cert = fixture
    for dots in ([0, -254, -32512], [0, 254, 32512], [0, -1, -1]):
        right = list(struct.unpack("<3I", secrets.token_bytes(12)))
        left = [(d - m) & 0xffffffff for d, m in zip(dots, right)]
        a, b = [_native.offset_pack_rows(struct.pack("<3I", *v), cert.widths, 1)
                for v in (left, right)]
        assert a == probe.pack_oracle(left, cert.widths)
        assert b == probe.pack_oracle(right, cert.widths)
        assert list(struct.unpack("<3q", _native.offset_reconstruct_rows(a, b, cert.widths, 1))) == dots
        assert probe.reconstruct_oracle(a, b, cert.widths) == dots


@pytest.mark.parametrize("profile", ["l1", "uniform"])
def test_actual_input_share_matmul_then_low_residue_reconstruction(fixture, profile):
    weights, _, pin, cert = fixture
    if profile == "uniform":
        cert = probe.geometry_certificate(pin)
    matrix = [[x if x < 128 else x - 256 for x in weights[i:i + 2]]
              for i in range(0, len(weights), 2)]
    for q in ([127, 127], [-127, -127], [127, -127]):
        mask = struct.unpack("<2I", secrets.token_bytes(8))
        complement = [(x - m) & 0xffffffff for x, m in zip(q, mask)]
        outputs = [[sum(w * x for w, x in zip(row, share)) & 0xffffffff
                    for row in matrix] for share in (complement, mask)]
        packed = [_native.offset_pack_rows(struct.pack("<3I", *values), cert.widths, 1)
                  for values in outputs]
        actual = struct.unpack("<3q", _native.offset_reconstruct_rows(*packed, cert.widths, 1))
        expected = tuple(sum(w * x for w, x in zip(row, q)) for row in matrix)
        assert actual == expected


def test_width32_extremes_and_cross_byte_boundaries():
    widths = bytes(range(1, 33))
    dots = [-(1 << (w - 1)) for w in widths]
    masks = struct.unpack("<32I", secrets.token_bytes(128))
    left = [(d - m) & 0xffffffff for d, m in zip(dots, masks)]
    packed = [_native.offset_pack_rows(struct.pack("<32I", *values), widths, 1)
              for values in (left, masks)]
    assert packed[0] == probe.pack_oracle(left, widths)
    assert packed[1] == probe.pack_oracle(list(masks), widths)
    assert list(struct.unpack("<32q", _native.offset_reconstruct_rows(*packed, widths, 1))) == dots
    assert probe.reconstruct_oracle(*packed, widths) == dots


@pytest.mark.parametrize("field", ["bound_sha256", "width_sha256", "source_lock_sha256",
                                  "checkpoint_sha256", "descriptor_sha256", "encoding",
                                  "qmax", "rows_per_call", "vocabulary", "hidden"])
def test_reject_self_declared_certificate_fields(fixture, field):
    weights, scales, pin, cert = fixture
    metadata = json.loads(cert.metadata)
    metadata[field] = metadata[field] + 1 if type(metadata[field]) is int else "0" * 64
    forged = probe.Certificate(probe.canonical(metadata), cert.widths)
    with pytest.raises(ValueError, match="recomputed pinned weights"):
        probe.verify_certificate(forged, weights, scales, pin)


def test_reject_under_and_over_widths(fixture):
    weights, scales, pin, cert = fixture
    for widths in (bytes([1, 8, 16]), bytes([1, 10, 16]), cert.widths[:-1]):
        with pytest.raises(ValueError):
            probe.verify_certificate(probe.Certificate(cert.metadata, widths), weights, scales, pin)


def test_reject_weight_source_substitution_even_with_forged_pin(fixture):
    weights, scales, pin, cert = fixture
    substituted = b"\x01" + weights[1:]
    with pytest.raises(ValueError, match="actual weight/scales commitment"):
        probe.verify_certificate(cert, substituted, scales, pin)
    forged_pin = copy.deepcopy(pin)
    forged_pin["weight_sha256"] = probe.sha(substituted)
    forged_pin["source_lock_sha256"] = "0" * 64
    forged_cert = probe.certificate(substituted, scales, forged_pin)
    with pytest.raises(ValueError):
        probe.verify_certificate(forged_cert, weights, scales, pin)


def test_frames_bind_all_context_and_encoding(fixture):
    cert = fixture[-1]
    payload = _native.offset_pack_rows(bytes(12), cert.widths, 1)
    context = probe.sha(b"admission/session/epoch/head/sequence").encode()[:32]
    wire = probe.frame(payload, cert, context, 0, 7)
    assert probe.unframe(wire, cert, context, 0, 7) == payload
    for offset in (0, 8, 9, 10, 12, 44, 76, 80, 84, 92):
        damaged = bytearray(wire)
        damaged[offset] ^= 1
        with pytest.raises(ValueError):
            probe.unframe(bytes(damaged), cert, context, 0, 7)
    for damaged in (wire[:-1], wire + b"\0"):
        with pytest.raises(ValueError):
            probe.unframe(damaged, cert, context, 0, 7)
    with pytest.raises(ValueError):
        probe.unframe(wire, cert, context, 1, 7)
    with pytest.raises(ValueError):
        probe.unframe(wire, cert, b"x" * 32, 0, 7)


def test_reject_padding_at_frame_and_native_boundaries(fixture):
    cert = fixture[-1]
    payload = _native.offset_pack_rows(bytes(12), cert.widths, 1)
    assert sum(cert.widths) == 26
    bad = payload[:-1] + bytes([payload[-1] | 128])
    context = b"x" * 32
    with pytest.raises(ValueError, match="padding"):
        probe.unframe(probe.frame(bad, cert, context, 0), cert, context, 0)
    with pytest.raises(ValueError):
        _native.offset_reconstruct_rows(bad, payload, cert.widths, 1)
    with pytest.raises(ValueError):
        _native.offset_reconstruct_rows(payload, bad, cert.widths, 1)
    with pytest.raises(ValueError):
        _native.offset_reconstruct_rows(payload, payload, cert.widths, 2)


def test_client_postcheck_rejects_out_of_bound_sum_and_forged_bound(fixture):
    weights, _, _, cert = fixture
    _, bounds = probe.oracle_layout(weights, 2)
    # Row 1 has width 9 but a certified absolute bound of 254, not 255.
    a = _native.offset_pack_rows(struct.pack("<3I", 0, 255, 0), cert.widths, 1)
    b = _native.offset_pack_rows(bytes(12), cert.widths, 1)
    with pytest.raises(ValueError, match="exceeds certified"):
        probe.reconstruct_checked(a, b, cert, bounds)
    forged = bytearray(bounds)
    forged[8] ^= 1
    with pytest.raises(ValueError, match="local public bounds"):
        probe.reconstruct_checked(a, b, cert, bytes(forged))


def test_geometry_guard_and_exact_qwen_payload_projection():
    for v, h in ((0, 1), (1, 0), (262145, 896), (151936, 65532), (True, 1)):
        with pytest.raises(ValueError):
            probe.geometry(v, h)
    with pytest.raises(ValueError, match="MAC guard"):
        probe.run(100, 100, oracle_mac_limit=1)
    costs = probe.costs(bytes([25]) * 151936, 1200, 896, 8)
    assert costs["packed_bytes_per_party"] == 474800
    assert costs["raw_u32_two_party_payload_bytes_per_call"] == 1215488
    assert costs["terminal_padding_bits_per_party"] == 0
    assert costs["raw_u32_two_party_payload_bytes_per_call"] - 2 * 474800 == 265888
    assert probe.HEADER.size == 100


def test_small_report_has_byte_exact_float32_and_explicit_scope():
    report = probe.run(vocabulary=7, hidden=3, repetitions=1)
    assert report["evidence"] == "synthetic-only"
    assert all(report["correctness"].values())
    assert report["costs"]["delivery_recipients"] == ["boundary_a", "boundary_b", "client"]
    assert "not actual Qwen" in report["qwen_certificate_size_note"]
    uniform = report["uniform_geometry_profile"]
    assert uniform["full_float32_logits_byte_equal"]
    assert uniform["client_verification_reads_weight_bytes"] is False
    assert uniform["costs"]["public_layout_bytes_per_recipient"] == 1
    assert uniform["costs"]["client_public_verification_weight_and_scales_bytes"] == 0


def test_uniform_profile_verifies_without_client_weights(fixture, monkeypatch):
    _, _, pin, l1 = fixture
    def forbidden(*args, **kwargs):
        raise AssertionError("source-free verification must not read weights or derive L1")
    monkeypatch.setattr(probe, "oracle_layout", forbidden)
    monkeypatch.setattr(probe._native, "offset_row_bits", forbidden)
    cert = probe.geometry_certificate(pin)
    assert probe.verify_geometry_certificate(cert, pin) == cert
    assert cert.transmitted_layout == bytes([16])
    assert cert.widths == bytes([16]) * 3
    assert cert.widths != l1.widths  # Wider uniform widths are valid in this profile only.
    metadata = json.loads(cert.metadata)
    assert metadata["uniform_bound"] == 2 * 128 * 127
    assert metadata["profile"] == probe.UNIFORM_PROFILE


def test_profiles_cannot_substitute_for_each_other(fixture):
    weights, scales, pin, l1 = fixture
    uniform = probe.geometry_certificate(pin)
    with pytest.raises(ValueError):
        probe.verify_certificate(uniform, weights, scales, pin)
    with pytest.raises(ValueError):
        probe.verify_geometry_certificate(l1, pin)
    forged = json.loads(uniform.metadata)
    forged["profile"] = probe.L1_PROFILE
    with pytest.raises(ValueError):
        probe.verify_geometry_certificate(probe.Certificate(
            probe.canonical(forged), uniform.widths, uniform.transmitted_layout), pin)


@pytest.mark.parametrize("field", ["profile", "uniform_bound", "uniform_width", "weight_dtype",
                                  "activation_quantizer", "weight_sha256", "scales_sha256",
                                  "descriptor_sha256", "source_lock_sha256", "hidden", "vocabulary"])
def test_uniform_rejects_forged_profile_bound_and_source(fixture, field):
    pin = fixture[2]
    cert = probe.geometry_certificate(pin)
    metadata = json.loads(cert.metadata)
    metadata[field] = metadata[field] + 1 if type(metadata[field]) is int else "forged"
    with pytest.raises(ValueError):
        probe.verify_geometry_certificate(probe.Certificate(
            probe.canonical(metadata), cert.widths, cert.transmitted_layout), pin)
    with pytest.raises(ValueError):
        probe.verify_geometry_certificate(probe.Certificate(cert.metadata, cert.widths,
                                                            bytes([15])), pin)


def test_uniform_exact_bound_both_signs_and_one_past_rejected(fixture):
    pin = fixture[2]
    cert = probe.geometry_certificate(pin)
    bound = pin["hidden"] * 128 * 127
    for value in (-bound, bound, -bound - 1, bound + 1):
        mask = struct.unpack("<3I", secrets.token_bytes(12))
        complement = [(value - x) & 0xffffffff for x in mask]
        packed = [_native.offset_pack_rows(struct.pack("<3I", *share), cert.widths, 1)
                  for share in (complement, mask)]
        if abs(value) <= bound:
            assert struct.unpack("<3q", probe.reconstruct_checked(*packed, cert)) == (value,) * 3
        else:
            with pytest.raises(ValueError, match="exceeds certified"):
                probe.reconstruct_checked(*packed, cert)


def test_uniform_source_pin_is_external_assumption_not_self_authentication(fixture):
    # A self-consistent opaque pin can be constructed: provenance must be external.
    pin = {**fixture[2], "weight_sha256": "0" * 64}
    forged = probe.geometry_certificate(pin)
    assert probe.verify_geometry_certificate(forged, pin) == forged
    with pytest.raises(ValueError):
        probe.verify_geometry_certificate(forged, fixture[2])


def test_uniform_cold_delivery_is_scalar_width_and_memory_preflight():
    pin = {key: "0" * 64 for key in ("source_lock_sha256", "checkpoint_sha256",
           "model_config_sha256", "descriptor_sha256", "weight_sha256", "scales_sha256")}
    pin.update(vocabulary=151936, hidden=896, qmax=127, weight_dtype="i8",
               activation_quantizer=probe.QUANTIZER)
    cert = probe.geometry_certificate(pin)
    result = probe.costs(cert.widths, len(cert.metadata), 896, 8, probe.UNIFORM_PROFILE)
    assert result["cold_certificate_layout_delivery_bytes"] == 3 * (8 + len(cert.metadata) + 1)
    assert result["packed_bytes_per_party"] == 474800
    assert result["client_expanded_native_width_array_bytes"] == 151936
    with pytest.raises(ValueError, match="modeled temporary envelope"):
        probe.allocation_preflight(151936, 896, 512)
    info = probe.allocation_preflight(151936, 896, 1024)
    assert info["available_host_headroom_queried"] is False
    assert info["generation_chunk_int16_plus_i8_max_bytes"] == 64 * 896 * 3
