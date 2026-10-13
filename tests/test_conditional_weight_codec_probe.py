"""Independent bytes, corruption and bounded-input checks for research rANS."""
import hashlib
import json
import random
import struct

import pytest

from scripts import probe_conditional_weight_codec as codec


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


@pytest.fixture
def tables():
    return codec.train([(bytes(range(256)) * 4, 32, cls) for cls in range(3)])


@pytest.mark.parametrize("raw,columns", [
    (bytes(range(256)) * 3, 256), (b"\0" * 1024, 32),
    (b"\xff\x80\x7f\x00" * 512, 64),
    (random.Random(91).randbytes(65536), 1024), (b"\x80", 1),
])
def test_exact_independent_bytes(raw, columns, tables):
    for cls in range(3):
        frame = codec.encode(raw, columns, cls, tables)
        assert codec.decode(frame, columns, cls, tables, sha(raw)) == raw
        assert codec.encode(raw, columns, cls, tables) == frame


def test_frozen_model_roundtrip_and_determinism(tables):
    assert codec.parse_model(codec.model_blob(tables)) == tables
    assert all(sum(row) == 4096 and min(row) >= 1 for row in tables)
    assert codec.context(255) == codec.context(1) == 0
    assert codec.context(128) == 7


def test_corruption_truncation_overlong(tables):
    raw = random.Random(22).randbytes(2048)
    frame = codec.encode(raw, 32, 0, tables)
    for cut in (0, 1, codec.HEADER.size, codec.HEADER.size + 4, len(frame) - 1):
        with pytest.raises(ValueError):
            codec.decode(frame[:cut], 32, 0, tables, sha(raw))
    for offset in (0, 8, 12, 15, 19, 51, codec.HEADER.size, len(frame) - 1):
        changed = bytearray(frame)
        changed[offset] ^= 0x80
        with pytest.raises(ValueError):
            codec.decode(changed, 32, 0, tables, sha(raw))
    # Match declared body size: cannot rely only on outer frame length rejection.
    fields = list(codec.HEADER.unpack_from(frame))
    fields[4] += 1
    overlong = codec.HEADER.pack(*fields) + frame[codec.HEADER.size:] + b"\0"
    with pytest.raises(ValueError, match="noncanonical"):
        codec.decode(overlong, 32, 0, tables, sha(raw))
    with pytest.raises(ValueError):
        codec.decode(frame, 32, 0, tables, "00" * 32)
    with pytest.raises(ValueError):
        codec.decode(frame, 64, 0, tables, sha(raw))
    with pytest.raises(ValueError):
        codec.decode(frame, 32, 1, tables, sha(raw))
    other = codec.train([(b"\0" * 1024, 32, 0)])
    with pytest.raises(ValueError):
        codec.decode(frame, 32, 0, other, sha(raw))


def test_budgets_models_and_initial_state(tables):
    for size, columns, cls in ((0, 1, 0), (65537, 1, 0), (3, 2, 0), (2, 0, 0), (2, 1, 3)):
        with pytest.raises(ValueError):
            codec.encode(b"\0" * size, columns, cls, tables)
    for blob in (b"", codec.model_blob(tables)[:-1], b"\0" * 12288):
        with pytest.raises(ValueError):
            codec.parse_model(blob)
    frame = bytearray(codec.encode(b"\0" * 32, 32, 0, tables))
    struct.pack_into("<I", frame, codec.HEADER.size, 0)
    with pytest.raises(ValueError, match="initial"):
        codec.decode(frame, 32, 0, tables, sha(b"\0" * 32))


def test_row_reset_causal_context_and_distinct_classes():
    tables = codec.train([(b"\x70\x01" * 64, 2, 0), (b"\0\xff" * 64, 2, 1)])
    assert tables[0] != tables[8]
    # Context7 gets second symbols, while every new row starts at context0.
    assert tables[7][1] > tables[7][112]
    raw = b"\x70\x01" * 64
    assert codec.decode(codec.encode(raw, 2, 0, tables), 2, 0, tables, sha(raw)) == raw


def test_declared_size_rejected_before_output_allocation(tables):
    raw = b"\0" * 32
    frame = codec.encode(raw, 32, 0, tables)
    fields = list(codec.HEADER.unpack_from(frame))
    fields[1] = 1 << 31
    with pytest.raises(ValueError, match="outside policy"):
        codec.decode(codec.HEADER.pack(*fields) + frame[codec.HEADER.size:], 32, 0, tables, sha(raw))


def test_production_frame_accounting_independent_compressor():
    import zlib
    from pllm.runtime.bundle_compression import encode_bundle_frames

    raw = random.Random(833).randbytes((1 << 20) + 129)
    ledger = codec.ProductionZlibLedger()
    for first in range(0, len(raw), 28672):
        ledger.feed(raw[first:first + 28672])
    frames = ledger.finish()
    expected = [raw[:1 << 20], raw[1 << 20:]]
    assert len(frames) == 2
    for frame, block in zip(frames, expected, strict=True):
        compressed = zlib.compress(block, 1)
        assert frame == {"raw_bytes": len(block), "body_bytes": len(compressed),
                         "frame_bytes": len(compressed) + 8, "raw_sha256": sha(block),
                         "body_sha256": sha(compressed)}
    assert ledger.finish() == frames  # No extra empty or duplicate frame.
    actual_frames = list(encode_bundle_frames(raw))
    for frame, actual in zip(frames, actual_frames, strict=True):
        assert struct.unpack_from(">II", actual) == (frame["raw_bytes"], frame["body_bytes"])
        assert len(actual) == frame["frame_bytes"]
        assert sha(actual[8:]) == frame["body_sha256"]


def test_shared_table_amortization_sign_and_scales_cancel():
    small = {"zlib1_frame_bytes": 1000, "conditional_frame_bytes": 990, "scale_bytes": 128}
    once = codec.charged_aggregate([small], 100, 12)
    repeated = codec.charged_aggregate([small] * 12, 100, 12)
    assert once["saved_bytes"] == -102
    assert repeated["saved_bytes"] == 8
    assert repeated["candidate_bytes"] == 12 * (990 + 128) + 112
    assert repeated["baseline_bytes"] == 12 * (1000 + 128)
    # Standalone selected object still pays all frozen tables, not one class.
    assert once["model_bytes_charged"] == 100
    assert codec.charged_aggregate([small | {"scale_bytes": 0}], 100, 12)["saved_bytes"] == -102


def test_predeclared_ranges_budget_and_shape_before_payload(tmp_path):
    path = tmp_path / "source.safetensors"
    metadata = {key: {"dtype": "BF16", "shape": [4096, 896]} for key in codec.KEYS}

    def write():
        blob = json.dumps(metadata).encode()
        path.write_bytes(struct.pack("<Q", len(blob)) + blob)

    write()
    assert codec.preflight_larger_ranges(path) == 5332992
    metadata[codec.KEYS[1]]["shape"][0] = 512
    write()
    with pytest.raises(ValueError, match="shape"):
        codec.preflight_larger_ranges(path)


def test_tensor_only_frozen_tables_and_exact_decode():
    samples = [(bytes(range(256)) * 4, 32, cls) for cls in range(3)]
    tables = codec.train(samples, conditioned=False)
    for cls in range(3):
        assert len(set(tables[cls * 8:(cls + 1) * 8])) == 1
    assert len(codec.model_blob(tables)) == 12288  # Duplication not free wire pruning.
    raw = b"\xff\x70\x00\x80" * 64
    assert codec.decode(codec.encode(raw, 32, 2, tables), 32, 2, tables, sha(raw)) == raw


def test_published_original_and_larger_aggregate_independent_accounting():
    evidence = json.loads((codec.ROOT / "docs/evidence/research-conditional-weight-codec.json").read_text())
    assert evidence["baseline_bytes_including_scales"] == 235886
    assert evidence["candidate_bytes_including_scales_and_model"] == 246907
    large = evidence["larger_frame_follow_up"]
    assert large["original_model_sha256"] == codec.FROZEN_MODEL_SHA256
    objects = large["objects"]
    baseline = conditional = tensor_only = 0
    for obj in objects:
        assert obj["i8_sha256"] == obj["decoded_sha256"] == obj["tensor_only_decoded_sha256"]
        assert len(obj["zlib1_frames"]) == 1
        assert obj["zlib1_frames"][0]["raw_bytes"] == obj["rows"] * obj["columns"]
        assert obj["zlib1_frames"][0]["frame_bytes"] == obj["zlib1_frames"][0]["body_bytes"] + 8
        assert obj["conditional_frame_bytes"] == sum(b["conditional_frame_bytes"] for b in obj["source_slices"])
        baseline += obj["zlib1_frame_bytes"] + obj["scale_bytes"]
        conditional += obj["conditional_frame_bytes"] + obj["scale_bytes"]
        tensor_only += obj["tensor_only_frame_bytes"] + obj["scale_bytes"]
    assert baseline == large["all_three_objects"]["baseline_bytes"] == 1970731
    assert conditional + 12288 + 212 == large["all_three_objects"]["candidate_bytes"] == 1970796
    assert tensor_only + 12288 + 198 == large["tensor_only_all_three_objects"]["candidate_bytes"] == 1967341
    assert large["all_three_objects"]["saved_bytes"] == -65
    assert large["tensor_only_all_three_objects"]["saved_bytes"] == 3390
    for selected in large["independent_single_object_deliveries"]:
        assert selected["conditional"]["model_bytes_charged"] == 12288
        assert selected["unused_class_table_bytes_charged"] == 8192
