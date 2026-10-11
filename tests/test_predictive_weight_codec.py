"""Tiny functional tests: no checkpoint loads, network, or performance cohort."""
import struct
import zlib
import hashlib
import importlib.metadata
import json

import numpy as np
import pytest

from scripts.probe_predictive_weight_codec import (
    MODES, LOCAL_MODES, DICT_LIMIT, BoundedBF16Rows, decode_dictionary,
    dictionary_selection, dictionary_parents, decode_frame, encode_dictionary,
    encode_frame, projection, reference_quantize, runtime_weight_slice, sha,
    token_parents, validated_control, cohort_schedule,
)


@pytest.mark.parametrize("mode", LOCAL_MODES)
@pytest.mark.parametrize("shape", [(1, 1), (7, 11), (1024, 17)])
def test_roundtrip(mode, shape):
    rows, columns = shape
    raw = np.random.default_rng(42).integers(0, 256, shape, dtype=np.uint8).tobytes()
    parents = np.arange(rows, dtype=np.int32) - 1
    frame = encode_frame(raw, rows=rows, columns=columns, mode=mode, parents=parents)
    assert decode_frame(frame, rows=rows, columns=columns, mode=mode) == raw
    if mode == "zlib1":
        body = zlib.compress(raw, level=1)
        assert frame == struct.pack(">II", len(raw), len(body)) + body


def test_structure_predictor_and_modular_wrap():
    tokens = ["a", "ab", "b", "cab", "abcd", "x" * 257, None]
    parents = token_parents(tokens)
    assert parents.tolist() == [-1, 0, -1, 1, 1, -1, -1]
    raw = bytes([255, 0, 128, 127, 1, 254, 16, 240, 240, 16, 5, 6, 7, 8])
    frame = encode_frame(raw, rows=7, columns=2, mode="token-parent", parents=parents)
    assert decode_frame(frame, rows=7, columns=2, mode="token-parent") == raw


def test_plain_frame_matches_installed_runtime_control():
    from pllm.runtime.bundle_compression import encode_bundle_frames
    raw = bytes(range(256)) * 3
    assert encode_frame(raw, rows=3, columns=256, mode="zlib1") == b"".join(encode_bundle_frames(raw))


@pytest.mark.parametrize("mode", LOCAL_MODES)
def test_truncation_corruption_trailing_and_shape(mode):
    frame = encode_frame(bytes(range(32)), rows=4, columns=8, mode=mode,
                         parents=np.array([-1, 0, 1, 2]))
    damaged = bytearray(frame)
    damaged[-1] ^= 0x80
    for bad in (frame[:-1], frame + b"x", frame[:7], bytes(damaged)):
        with pytest.raises(ValueError):
            decode_frame(bad, rows=4, columns=8, mode=mode)
    with pytest.raises(ValueError):
        decode_frame(frame, rows=4, columns=7, mode=mode)


@pytest.mark.parametrize("parent", [-2, 0, 1, 2**31 - 1])
def test_reject_invalid_parent_on_both_sides(parent):
    with pytest.raises(ValueError):
        encode_frame(b"x", rows=1, columns=1, mode="token-parent", parents=np.array([parent]))
    body = zlib.compress(struct.pack("<i", parent) + b"x")
    frame = struct.pack(">II", 5, len(body)) + body
    with pytest.raises(ValueError):
        decode_frame(frame, rows=1, columns=1, mode="token-parent")


def test_inflate_bomb_and_duplicate_zlib_stream():
    for body in (zlib.compress(b"x" * 100000), zlib.compress(b"x") + zlib.compress(b"y")):
        frame = struct.pack(">II", 1, len(body)) + body
        with pytest.raises(ValueError):
            decode_frame(frame, rows=1, columns=1, mode="zlib1")


@pytest.mark.parametrize("rows,columns", [(0, 1), (1025, 1), (1, 4097), (1024, 4096), (True, 1)])
def test_policy_dimensions(rows, columns):
    with pytest.raises(ValueError):
        decode_frame(b"", rows=rows, columns=columns, mode="zlib1")


def test_reference_integer_and_scale_bytes():
    raw, scales = reference_quantize(np.array([[0, 0], [-1, 1]], np.float32))
    assert raw == np.array([[0, 0], [-127, 127]], np.int8).tobytes()
    assert scales == np.array([1, np.float32(1 / 127)], "<f4").tobytes()


def test_token_search_bounded_and_acyclic():
    tokens = ["a" * (i % 256 + 1) for i in range(1024)]
    parents = token_parents(tokens)
    assert np.all(parents < np.arange(1024))
    with pytest.raises(ValueError):
        token_parents(tokens + ["overflow"])


@pytest.mark.parametrize("rows,columns", [(1, 1), (7, 11), (1024, 17)])
def test_global_dictionary_complete_layout(rows, columns):
    bank_rows = min(9, rows)
    dictionary = np.random.default_rng(17).integers(0, 256, (bank_rows, columns), dtype=np.uint8)
    blob = encode_dictionary(dictionary.tobytes(), rows=bank_rows, columns=columns)
    bank = decode_dictionary(blob, columns=columns, expected_sha256=hashlib.sha256(dictionary).hexdigest())
    assert np.array_equal(bank, dictionary)
    raw = np.random.default_rng(41).integers(0, 256, (rows, columns), dtype=np.uint8).tobytes()
    parents = np.arange(rows, dtype=np.int32) % bank_rows
    parents[0] = -1
    frame = encode_frame(raw, rows=rows, columns=columns, mode=MODES[3], parents=parents, dictionary=bank)
    assert decode_frame(frame, rows=rows, columns=columns, mode=MODES[3], dictionary=bank) == raw
    for bad in (frame[:-1], frame + b"x"):
        with pytest.raises(ValueError):
            decode_frame(bad, rows=rows, columns=columns, mode=MODES[3], dictionary=bank)


def test_global_coverage_selection_and_distant_parent():
    tokens = ["a", "b", "ab", "ac", "xa", "abz", "xbc", None]
    ids, coverage = dictionary_selection(tokens, budget=2)
    assert ids == [0, 1]  # a:4 descendants; b:1; ab:1; token-id breaks tie.
    assert coverage == [4, 1]
    parents = dictionary_parents(["abz", "xbc", "unknown", "a"], [tokens[row] for row in ids])
    assert parents.tolist() == [0, -1, -1, 0]
    distant = ["a"] + [None] * 1024 + ["abcd"]
    selected, _ = dictionary_selection(distant, budget=1)
    assert selected == [0]
    assert token_parents(distant[-1:]).tolist() == [-1]
    assert dictionary_parents(distant[-1:], ["a"]).tolist() == [0]


@pytest.mark.parametrize("budget", [0, 513, True, 1.0])
def test_global_selection_budget(budget):
    with pytest.raises(ValueError):
        dictionary_selection(["a", "ab"], budget=budget)


def test_dictionary_budget_and_integrity():
    raw = b"\xff\x00" * DICT_LIMIT
    blob = encode_dictionary(raw, rows=DICT_LIMIT, columns=2)
    digest = hashlib.sha256(raw).hexdigest()
    assert decode_dictionary(blob, columns=2, expected_sha256=digest).nbytes == 1024
    damaged = bytearray(blob)
    damaged[-1] ^= 0x80
    overbudget = bytearray(blob)
    struct.pack_into(">I", overbudget, 8, DICT_LIMIT + 1)
    for bad in (blob[:-1], blob + b"x", bytes(damaged), bytes(overbudget)):
        with pytest.raises(ValueError):
            decode_dictionary(bad, columns=2, expected_sha256=digest)
    with pytest.raises(ValueError):
        decode_dictionary(blob, columns=2, expected_sha256="0" * 64)
    with pytest.raises(ValueError):
        encode_dictionary(b"x" * 513, rows=513, columns=1)
    with pytest.raises(ValueError):
        decode_frame(b"", rows=1, columns=1, mode=MODES[3], dictionary=np.zeros((513, 1), np.uint8))


def test_max_dictionary_uses_bounded_independent_frames():
    raw = bytes(512 * 4096)
    blob = encode_dictionary(raw, rows=512, columns=4096)
    assert struct.unpack_from(">I", blob, 16)[0] == 2
    assert decode_dictionary(blob, columns=4096, expected_sha256=hashlib.sha256(raw).hexdigest()).nbytes == len(raw)


@pytest.mark.parametrize("parent", [-2, 2, 2**31 - 1])
def test_global_parent_corruption(parent):
    bank = np.array([[0], [255]], np.uint8)
    body = zlib.compress(struct.pack("<i", parent) + b"x")
    frame = struct.pack(">II", 5, len(body)) + body
    with pytest.raises(ValueError):
        decode_frame(frame, rows=1, columns=1, mode=MODES[3], dictionary=bank)
    with pytest.raises(ValueError):
        encode_frame(b"x", rows=1, columns=1, mode=MODES[3], parents=np.array([parent]), dictionary=bank)


def test_projection_uses_validated_fields_and_preserves_losses():
    control = {"pair_bytes": 300, "classes": {"public_bundle": 100}}
    gain = projection(control, baseline=70, candidate=60)
    assert gain["complete_bundle_counterfactual_bytes"] == 90
    assert gain["cold_repeat_pair_counterfactual_bytes"] == 290
    loss = projection(control, baseline=70, candidate=80)
    assert loss["matrix_bytes_saved"] == -10
    assert loss["pair_fraction_saved"] == -10 / 300
    control["pair_bytes"] = 500
    assert projection(control, baseline=70, candidate=60)["cold_repeat_pair_counterfactual_bytes"] == 490


def test_minimal_frozen_runner_schedule():
    schedule = cohort_schedule()
    assert len(schedule) == 12
    assert schedule[:4] == [(0, mode) for mode in MODES]
    assert all(schedule.count((repeat, mode)) == 1 for repeat in range(3) for mode in MODES)


def test_control_identity_binding(tmp_path, monkeypatch):
    from scripts import probe_all_link_frontier as helper
    raw = tmp_path / "raw.json"
    raw.write_text("{}")
    expected_hash = sha(raw)
    control = {"pair_bytes": 300, "classes": {"public_bundle": 100}, "evidence_sha256": expected_hash}
    monkeypatch.setattr(helper, "conserved_control", lambda document: control)
    frontier = tmp_path / "frontier.json"
    frontier.write_text(json.dumps({"complete": True, "control": control}))
    checked, identity = validated_control(frontier, raw, expected_hash)
    assert checked == control
    assert identity["frontier_sha256"] == sha(frontier)
    with pytest.raises(ValueError, match="hash mismatch"):
        validated_control(frontier, raw, "0" * 64)
    frontier.write_text(json.dumps({"complete": True, "control": {"pair_bytes": 301}}))
    with pytest.raises(ValueError, match="reconciliation"):
        validated_control(frontier, raw, expected_hash)


def write_tiny_bf16(path):
    value = np.array([[0, 1], [-1, .5], [128, -128]], "<f4")
    raw = (value.view("<u4") >> 16).astype("<u2").tobytes()
    header = json.dumps({"model.embed_tokens.weight": {"dtype": "BF16", "shape": [3, 2],
                        "data_offsets": [0, len(raw)]}}, separators=(",", ":")).encode()
    header += b" " * (-len(header) % 8)
    path.write_bytes(struct.pack("<Q", len(header)) + header + raw)
    return raw, value


def test_generated_bf16_slice_and_independent_fallback(tmp_path, record_property, monkeypatch):
    from pllm.runtime.safetensors_store import SafeTensorStore
    from pllm.runtime.quantization import quantize_weight_per_row
    import pllm
    from safetensors import safe_open
    assert importlib.metadata.version("pllm.run")
    record_property("installed_pllm_path", pllm.__file__)
    path = tmp_path / "model.safetensors"
    raw, expected = write_tiny_bf16(path)
    key = "model.embed_tokens.weight"
    adapter_rejected = False
    with safe_open(path, framework="np") as handle:
        assert tuple(handle.get_slice(key).get_shape()) == (3, 2)
        try:
            handle.get_slice(key)[1:3, :]
        except (TypeError, ValueError) as error:
            assert any(word in str(error).lower() for word in ("bf16", "bfloat"))
            adapter_rejected = True
    record_property("framework_np_rejected_bf16", adapter_rejected)
    store = SafeTensorStore(tmp_path)
    with BoundedBF16Rows(path, key) as independent:
        actual_raw, value = independent.read(1, 2)
        assert actual_raw == raw[4:]
        assert np.array_equal(value, expected[1:])
        weight, loader = runtime_weight_slice(store, independent, key, 1, 2)
        assert np.array_equal(weight, expected[1:])
        assert reference_quantize(weight) == reference_quantize(expected[1:])
        quantized = quantize_weight_per_row(weight, bits=8)
        reference_raw, reference_scales = reference_quantize(expected[1:])
        assert quantized.values.tobytes() == reference_raw
        assert quantized.scales.astype("<f4", copy=False).tobytes() == reference_scales
        if adapter_rejected:
            assert loader == "bounded-raw-bf16-fallback"
        def reject_bf16(*_args, **_kwargs):
            raise TypeError("unsupported BFloat16")
        monkeypatch.setattr(store, "get_slice", reject_bf16)
        fallback, loader = runtime_weight_slice(store, independent, key, 1, 2)
        assert loader == "bounded-raw-bf16-fallback"
        assert np.array_equal(fallback, expected[1:])
        with pytest.raises(ValueError):
            independent.read(2, 2)


def test_bf16_header_bounds(tmp_path):
    path = tmp_path / "bad.safetensors"
    path.write_bytes(struct.pack("<Q", (1 << 20) + 1))
    with pytest.raises(ValueError, match="header outside"):
        BoundedBF16Rows(path, "key")
    path.write_bytes(b"short")
    with pytest.raises(ValueError, match="truncated"):
        BoundedBF16Rows(path, "key")
