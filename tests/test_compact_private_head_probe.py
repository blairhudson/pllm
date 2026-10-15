"""Tiny fixtures only: no model values, network, or sustained timing."""
import json
from pathlib import Path
import struct
from types import SimpleNamespace
import zlib

import numpy as np
import pytest

from scripts import probe_compact_private_head as probe


def fixture(weights=None):
    w = np.array([[0, 0, 0], [-128, 127, 1], [1, -1, 1], [127, 127, 127]],
                 np.int8) if weights is None else weights
    s = np.linspace(0.001, 0.02, w.shape[0], dtype=np.float32)
    pin = {key: probe.digest(key.encode()) for key in (
        "source_lock_sha256", "checkpoint_sha256", "model_config_sha256", "descriptor_sha256")}
    pin.update(weight_sha256=probe.digest(w.tobytes()), scales_sha256=probe.digest(s.tobytes()),
               vocabulary=w.shape[0], hidden=w.shape[1], qmax=127,
               weight_dtype="i8", activation_quantizer=probe.QUANTIZER)
    return w, s, pin


def certificate(w, s, pin, **kwargs):
    return probe.scan(lambda a, b: w[a:b], lambda a, b: s[a:b], pin, **kwargs)


def test_full_two_pass_certificate_and_attainable_floor():
    w, s, pin = fixture()
    calls = []

    def read(a, b):
        calls.append((a, b))
        return w[a:b]

    cert = probe.scan(read, lambda a, b: s[a:b], pin, chunk_rows=2)
    assert len(calls) == 4
    bounds = struct.unpack("<4Q", cert["bounds"])
    assert bounds == (0, 127 * 256, 127 * 3, 127 * 381)
    for row, bound, width in zip(w, bounds, cert["widths"], strict=True):
        x = np.sign(row.astype(np.int16)) * 127
        assert int(row.astype(np.int64) @ x.astype(np.int64)) == bound
        assert bound < 1 << (width - 1)
        if bound:
            assert bound >= 1 << (width - 2)  # one less bit fails +B signed-center
    assert json.loads(cert["metadata"])["weight_sha256"] == pin["weight_sha256"]


@pytest.mark.parametrize("which", ["weights", "scales", "quantizer", "dtype", "hash", "zero_scale", "nan_scale"])
def test_rejects_source_contract_mismatch(which):
    w, s, pin = fixture()
    if which == "weights":
        w[0, 0] = 1
    elif which == "scales":
        s[0] *= 2
    elif which == "quantizer":
        pin["qmax"] = 128
    elif which == "dtype":
        w = w.astype(np.uint8)
    elif which == "hash":
        pin["weight_sha256"] = "x" * 64
    elif which == "zero_scale":
        s[0] = 0
    else:
        s[0] = np.nan
    with pytest.raises(ValueError):
        certificate(w, s, pin)


def test_mutation_between_passes_rejected():
    w, s, pin = fixture()
    count = 0

    def read(a, b):
        nonlocal count
        count += 1
        return w[a:b] if count == 1 else np.zeros_like(w[a:b])

    with pytest.raises(ValueError, match="independent"):
        probe.scan(read, lambda a, b: s[a:b], pin, chunk_rows=4)


def test_native_certificate_mismatch_rejected():
    w, s, pin = fixture()
    with pytest.raises(ValueError, match="native certificate"):
        certificate(w, s, pin, native_widths=lambda *_: b"\x01" * len(w))


@pytest.mark.parametrize("widths", [bytes([25]) * 37, bytes(range(1, 33)), b"\x01\x20\x01"])
def test_all_public_layout_codecs_roundtrip(widths):
    for encoded in probe.layout_encodings(widths).values():
        assert probe.decode_layout(encoded, len(widths)) == widths


@pytest.mark.parametrize("data,rows", [(b"Z" + zlib.compress(b"\x19" * 1000), 3),
    (b"R" + struct.pack("<BI", 25, 1000), 3), (b"B\x00", 1),
    (b"5\xff", 1), (b"Z" + zlib.compress(b"\x19") + b"trailing", 1)])
def test_layout_expansion_and_canonical_guards(data, rows):
    with pytest.raises(ValueError):
        probe.decode_layout(data, rows)


def test_both_roles_cold_delivery_and_existing_framing():
    w, s, pin = fixture(np.full((151, 3), 1, np.int8))
    cert = certificate(w, s, pin)
    c = probe.costs(cert, 3, calls=8)
    assert c["reply_payload_bytes_both"] == 2 * ((sum(cert["widths"]) + 7) // 8)
    assert c["reply_framed_bytes_both_estimate"] == 2 * (c["reply_payload_bytes_per_role"] + 74 + 32 + 142 + 137)
    metadata_bytes = min(c["metadata_encoding_bytes"].values())
    setup = 3 * (probe.SETUP_HEADER.size + metadata_bytes + 2 + 142 + 137)
    assert c["cold_layout_metadata_delivery_bytes_three_recipients_estimate"] == setup
    assert c["net_saved_bytes_at_head_calls_estimate"] == 8 * c["saved_reply_bytes_both_per_head_estimate"] - setup
    assert c["client_public_weight_verification_delivery_bytes_if_not_preinstalled"] == 151 * 7


def test_local_full_logits_and_sampler_state_match_original():
    from pllm.runtime.quantization import dequantize_matmul
    from pllm.runtime.transformer_client import MaskedTransformerClientRuntime

    rng = np.random.default_rng(6161)
    w, s, _ = fixture(rng.integers(-128, 128, (97, 17), dtype=np.int16).astype(np.int8))
    x = rng.integers(-127, 128, 17, dtype=np.int16).astype(np.int8)
    a = np.float32(0.03125)
    actual = probe.local_head(lambda a, b: w[a:b], lambda a, b: s[a:b], 97, 17, x, a, chunk_rows=7)
    dots = (w.astype(np.int64) @ x.astype(np.int64)).reshape(1, -1)
    expected = dequantize_matmul(dots, np.array([a]), s)[0]
    assert actual.tobytes() == expected.tobytes()
    for temperature, top_p in ((0, None), (0.8, None), (0.8, 1.0), (0.8, 0.9), (0.8, 0.0)):
        left = SimpleNamespace(rng=np.random.default_rng(17))
        right = SimpleNamespace(rng=np.random.default_rng(17))
        for _ in range(4):
            assert MaskedTransformerClientRuntime.sample(left, actual, temperature=temperature, top_p=top_p) == MaskedTransformerClientRuntime.sample(right, expected, temperature=temperature, top_p=top_p)
        assert left.rng.bit_generator.state == right.rng.bit_generator.state


def test_unsafe_shortcuts_counterexamples():
    # +B is attainable but narrower signed-center decoder produces -B.
    bound = 128
    residue = bound & 255
    assert residue - 256 == -128
    # Dividing shares before reconstruction drops carry: 1+1=2, yet floors sum=0.
    assert (1 // 2 + 1 // 2) != ((1 + 1) // 2)
    # Compressing mask bytes as small signed outputs is invalid even for zero dot.
    share_a, share_b = 127, 129
    assert (share_a + share_b) % 256 == 0 and share_b > 127
    # Cancelling positive common activation scale changes float32 ties on underflow.
    scores = (np.array([1, 2], np.float32) * np.float32(1e-20)) * np.float32(1e-30)
    assert np.argmax(scores) == 0 and np.argmax(np.array([1, 2])) == 1
    # Full softmax has positive mass outside any fixed proper top-k for equal logits.
    assert np.exp(np.zeros(5)).sum() == 5 and 2 / 5 < 1


def test_top_p_tie_contract_not_greedy_or_conventional_crossing():
    from pllm.runtime.transformer_client import MaskedTransformerClientRuntime

    runtime = SimpleNamespace(rng=np.random.default_rng(1))
    logits = np.zeros(4, np.float32)
    assert MaskedTransformerClientRuntime.sample(runtime, logits) == 0
    # Reverse default argsort tie order; p=0 retains only order[0], not greedy[0].
    expected = int(np.argsort(np.full(4, 0.25))[::-1][0])
    assert MaskedTransformerClientRuntime.sample(runtime, logits, temperature=1, top_p=0) == expected


def test_private_circuit_gate_counts_and_no_false_promotion():
    result = probe.private_selection_gate(bytes([20]) * 151936)
    assert result["modular_reconstruction_adder_AND_gates"] == 151936 * 37
    assert result["adder_only_garbled_table_bytes"] == 179892224
    assert result["evaluator_input_OT_count"] == 3038720
    assert result["full_vocabulary_sampler_gate"].startswith("FAIL")


def test_static_admission_before_reader():
    w, s, pin = fixture()
    pin["vocabulary"] = probe.MAX_ROWS + 1
    with pytest.raises(ValueError, match="geometry"):
        probe.scan(lambda *_: pytest.fail("read before admission"), lambda *_: None, pin)


def test_guarded_raw_worker_tiny(tmp_path):
    # Worker callable only: no fresh_workers lock needed for four public fixture rows.
    w, s, pin = fixture()
    wp, sp = tmp_path / "w.i8", tmp_path / "s.f32"
    wp.write_bytes(w.tobytes())
    sp.write_bytes(s.tobytes())
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"weights": str(wp), "scales": str(sp), "pin": pin}))
    result = probe.worker({"raw_manifest": str(manifest), "chunk_rows": 2, "admission_mib": 256})
    assert result["verification"]["installed_native_every_row_verified"]
    assert result["pin"] == pin
    assert bytes.fromhex(result["widths_hex"]) == certificate(w, s, pin)["widths"]
