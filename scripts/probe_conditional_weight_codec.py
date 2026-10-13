"""Small public, source-locked conditional rANS screen; never load a model."""
from __future__ import annotations

import argparse
from bisect import bisect_right
import hashlib
import json
from pathlib import Path
import struct
import sys
import zlib

ROOT = Path(__file__).resolve().parents[1]
REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
MAX_BLOCK = 65536
TOTAL = 4096
LOW = 1 << 23
CONTEXTS = 8
HEADER = struct.Struct("<8sIHB I32s32s")
MAGIC = b"PCWRANS1"
KEYS = ("model.embed_tokens.weight", "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.mlp.gate_proj.weight")
# Fixed disjoint source ranges, not selected using coded results.
CALIBRATION = (0, 64, 128)
HELD_OUT = (256, 384, 512)
ROWS = 32
SOURCE_SHA256 = "fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe"
SOURCE_HEADER_SHA256 = "bd808652be103f0e4e29786b9aa8829e0f38607880486926c7246ee2de577a76"
FROZEN_MODEL_SHA256 = "52d375e5900191d22892d2d773a0acde17e7eba7175630a14066682e2f6b65a0"
# Parent-predeclared ranges, disjoint from original cohorts; 4,300,800 BF16 bytes.
LARGE_RANGES = ((1024, 1024), (544, 352), (1024, 1024))
PRODUCTION_CHUNK = 1 << 20


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def file_hash(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def context(previous):
    """Unsigned source bytes stay unchanged; signed magnitude only indexes tables."""
    magnitude = abs(previous if previous < 128 else previous - 256)
    return min(7, magnitude // 16)


def bounds(size, columns, tensor_class):
    if (type(size) is not int or not 1 <= size <= MAX_BLOCK or
            type(columns) is not int or not 1 <= columns <= 4096 or size % columns or
            type(tensor_class) is not int or not 0 <= tensor_class < len(KEYS)):
        raise ValueError("block dimensions/class outside policy")


def train(samples, conditioned=True):
    counts = [[1] * 256 for _ in range(len(KEYS) * CONTEXTS)]
    for raw, columns, cls in samples:
        bounds(len(raw), columns, cls)
        for i, symbol in enumerate(raw):
            ctx = context(raw[i - 1]) if conditioned and i % columns else 0
            counts[cls * CONTEXTS + ctx][symbol] += 1
    tables = []
    for row in counts:
        total = sum(row)
        # Reserve one count per symbol; largest remainder, byte-id tie break.
        scaled = [x * (TOTAL - 256) for x in row]
        freq = [1 + x // total for x in scaled]
        order = sorted(range(256), key=lambda i: (-(scaled[i] % total), i))
        for i in order[:TOTAL - sum(freq)]:
            freq[i] += 1
        tables.append(tuple(freq))
    if not conditioned:
        # Same wire format/decoder: repeated context rows, honestly charged in full.
        tables = [tables[cls * CONTEXTS] for cls in range(len(KEYS)) for _ in range(CONTEXTS)]
    return tuple(tables)


def model_blob(tables):
    if len(tables) != len(KEYS) * CONTEXTS:
        raise ValueError("model table count")
    for row in tables:
        if (len(row) != 256 or any(type(x) is not int or not 1 <= x <= TOTAL for x in row)
                or sum(row) != TOTAL):
            raise ValueError("model frequency bounds")
    return b"".join(struct.pack("<256H", *row) for row in tables)


def parse_model(blob):
    if len(blob) != len(KEYS) * CONTEXTS * 512:
        raise ValueError("model length")
    tables = tuple(struct.unpack_from("<256H", blob, offset)
                   for offset in range(0, len(blob), 512))
    model_blob(tables)
    return tables


def cumulative(tables):
    result = []
    for row in tables:
        sums = [0]
        for freq in row:
            sums.append(sums[-1] + freq)
        result.append(sums)
    return result


def encode(raw, columns, cls, tables):
    bounds(len(raw), columns, cls)
    model = model_blob(tables)
    starts = cumulative(tables)
    state, emitted = LOW, bytearray()
    for i in range(len(raw) - 1, -1, -1):
        symbol = raw[i]
        ctx = cls * CONTEXTS + (context(raw[i - 1]) if i % columns else 0)
        freq = tables[ctx][symbol]
        while state >= ((LOW >> 12) << 8) * freq:
            emitted.append(state & 255)
            state >>= 8
        state = (state // freq) * TOTAL + state % freq + starts[ctx][symbol]
    body = struct.pack("<I", state) + bytes(reversed(emitted))
    return HEADER.pack(MAGIC, len(raw), columns, cls, len(body),
                       bytes.fromhex(digest(model)), bytes.fromhex(digest(raw))) + body


def decode(frame, columns, cls, tables, expected_sha256):
    if not HEADER.size + 4 <= len(frame) <= HEADER.size + 2 * MAX_BLOCK + 4:
        raise ValueError("frame length outside policy")
    magic, size, width, kind, length, model_hash, raw_hash = HEADER.unpack_from(frame)
    bounds(size, width, kind)
    if (magic != MAGIC or width != columns or kind != cls or
            not 4 <= length <= 2 * size + 4 or len(frame) != HEADER.size + length or
            raw_hash.hex() != expected_sha256 or model_hash.hex() != digest(model_blob(tables))):
        raise ValueError("frame metadata/digest/length mismatch")
    starts = cumulative(tables)
    state = struct.unpack_from("<I", frame, HEADER.size)[0]
    if not LOW <= state < LOW * 256:
        raise ValueError("invalid initial rANS state")
    offset, output = HEADER.size + 4, bytearray()
    for i in range(size):
        ctx = cls * CONTEXTS + (context(output[-1]) if i % width else 0)
        slot = state & (TOTAL - 1)
        symbol = bisect_right(starts[ctx], slot) - 1
        output.append(symbol)
        state = tables[ctx][symbol] * (state >> 12) + slot - starts[ctx][symbol]
        while state < LOW:
            if offset == len(frame):
                raise ValueError("truncated rANS renormalization")
            state = state * 256 + frame[offset]
            offset += 1
    if state != LOW or offset != len(frame) or digest(output) != expected_sha256:
        raise ValueError("noncanonical state/trailing bytes/decoded digest mismatch")
    return bytes(output)


def read_sample(path, key, first, count=ROWS):
    """Read <=128 KiB source payload per block; independent header/quantizer oracle."""
    import numpy as np
    from pllm.runtime.quantization import quantize_weight_per_row

    with Path(path).open("rb") as handle:
        prefix = handle.read(8)
        if len(prefix) != 8:
            raise ValueError("truncated source header")
        size = struct.unpack("<Q", prefix)[0]
        if not 1 <= size <= 1 << 20:
            raise ValueError("source header bounds")
        encoded = handle.read(size)
        if len(encoded) != size:
            raise ValueError("truncated source metadata")
        entry = json.loads(encoded)[key]
        rows, columns = entry["shape"]
        start, end = entry["data_offsets"]
        bounds(count * columns, columns, KEYS.index(key))
        if (entry["dtype"] != "BF16" or first < 0 or first + count > rows or
                start < 0 or end - start != rows * columns * 2 or
                8 + size + end > Path(path).stat().st_size):
            raise ValueError("source tensor bounds")
        handle.seek(8 + size + start + first * columns * 2)
        source = handle.read(count * columns * 2)
        if len(source) != count * columns * 2:
            raise ValueError("truncated source slice")
    values = (np.frombuffer(source, "<u2").astype("<u4") << 16).view("<f4").reshape(count, columns)
    scales = np.where(np.max(np.abs(values), axis=1) > 0,
                      np.max(np.abs(values), axis=1) / 127, 1).astype("<f4")
    raw = np.clip(np.rint(values / scales[:, None]), -127, 127).astype(np.int8).tobytes()
    oracle = quantize_weight_per_row(values, bits=8)
    if raw != oracle.values.tobytes() or scales.tobytes() != oracle.scales.astype("<f4").tobytes():
        raise AssertionError("independent i8/scale oracle mismatch")
    identity = {"tensor": key, "source_shape": [rows, columns], "first_row": first,
                "rows": count, "source_bf16_sha256": digest(source),
                "i8_sha256": digest(raw), "scale_sha256": digest(scales.tobytes()),
                "source_payload_bytes": len(source), "scale_bytes": scales.nbytes}
    return raw, columns, scales.tobytes(), identity


class ProductionZlibLedger:
    """Exactly zlib.compress(raw, 1), framed in 1 MiB byte chunks; small feed buffers."""
    def __init__(self):
        self.records = []
        self._reset()

    def _reset(self):
        self.encoder = zlib.compressobj(1)
        self.raw_hash, self.body_hash = hashlib.sha256(), hashlib.sha256()
        self.raw_size = self.body_size = 0

    def _emit(self, blob):
        self.body_hash.update(blob)
        self.body_size += len(blob)

    def _finish(self):
        if self.raw_size:
            self._emit(self.encoder.flush())
            self.records.append({"raw_bytes": self.raw_size, "body_bytes": self.body_size,
                                 "frame_bytes": self.body_size + 8,
                                 "raw_sha256": self.raw_hash.hexdigest(),
                                 "body_sha256": self.body_hash.hexdigest()})
            self._reset()

    def feed(self, raw):
        offset = 0
        while offset < len(raw):
            count = min(len(raw) - offset, PRODUCTION_CHUNK - self.raw_size)
            part = memoryview(raw)[offset:offset + count]
            self.raw_hash.update(part)
            self.raw_size += count
            self._emit(self.encoder.compress(part))
            offset += count
            if self.raw_size == PRODUCTION_CHUNK:
                self._finish()

    def finish(self):
        self._finish()
        return self.records


def charged_aggregate(objects, model_bytes, descriptor_bytes, candidate_key="conditional_frame_bytes"):
    """Charge immutable full model once per selected delivery, including unused classes."""
    baseline = sum(r["zlib1_frame_bytes"] + r["scale_bytes"] for r in objects)
    candidate = sum(r[candidate_key] + r["scale_bytes"] for r in objects) + model_bytes + descriptor_bytes
    return {"baseline_bytes": baseline, "candidate_bytes": candidate,
            "model_bytes_charged": model_bytes, "descriptor_bytes_charged": descriptor_bytes,
            "saved_bytes": baseline - candidate,
            "fraction_saved": (baseline - candidate) / baseline}


def preflight_larger_ranges(checkpoint):
    """Validate every new range and complete lightweight payload budget before reads."""
    with Path(checkpoint).open("rb") as handle:
        size = struct.unpack("<Q", handle.read(8))[0]
        if not 1 <= size <= 1 << 20:
            raise ValueError("source header bounds")
        metadata = json.loads(handle.read(size))
    total = 0
    for cls, (key, (first, count)) in enumerate(zip(KEYS, LARGE_RANGES, strict=True)):
        entry = metadata[key]
        rows, width = entry["shape"]
        bounds(ROWS * width, width, cls)
        if entry["dtype"] != "BF16" or first < 0 or first + count > rows:
            raise ValueError("predeclared follow-up slice exceeds source shape")
        for old_first in CALIBRATION + HELD_OUT:
            if max(first, old_first) < min(first + count, old_first + ROWS):
                raise ValueError("follow-up overlaps original cohort")
        total += 2 * width * (count + ROWS * (len(CALIBRATION) + len(HELD_OUT)))
    if total > 8 << 20:
        raise ValueError("complete lightweight source payload budget exceeded")
    return total


def larger_screen(checkpoint, tables, tensor_only, descriptor):
    """Both models already frozen before any new range is read; no model selection."""
    total_budget = preflight_larger_ranges(checkpoint)
    objects = []
    source_bytes = 0
    for cls, (key, (first, count)) in enumerate(zip(KEYS, LARGE_RANGES, strict=True)):
        control = ProductionZlibLedger()
        hashes = [hashlib.sha256() for _ in range(4)]
        blocks = []
        candidate = tensor_candidate = scale_bytes = 0
        for row in range(first, first + count, ROWS):
            raw, width, scales, identity = read_sample(checkpoint, key, row, min(ROWS, first + count - row))
            source_bytes += identity["source_payload_bytes"]
            if source_bytes > 8 << 20:
                raise ValueError("new held-out source payload budget exceeded")
            frame = encode(raw, width, cls, tables)
            decoded = decode(frame, width, cls, tables, digest(raw))
            tensor_frame = encode(raw, width, cls, tensor_only)
            tensor_decoded = decode(tensor_frame, width, cls, tensor_only, digest(raw))
            if raw != decoded or raw != tensor_decoded:
                raise AssertionError("larger held-out exact bytes mismatch")
            for hasher, value in zip(hashes, (raw, decoded, tensor_decoded, scales), strict=True):
                hasher.update(value)
            control.feed(raw)
            candidate += len(frame)
            tensor_candidate += len(tensor_frame)
            scale_bytes += len(scales)
            blocks.append(identity | {"conditional_frame_bytes": len(frame),
                                       "tensor_only_frame_bytes": len(tensor_frame)})
        control_frames = control.finish()
        objects.append({"tensor": key, "first_row": first, "rows": count, "columns": width,
                        "raw_bytes": count * width, "scale_bytes": scale_bytes,
                        "i8_sha256": hashes[0].hexdigest(), "decoded_sha256": hashes[1].hexdigest(),
                        "tensor_only_decoded_sha256": hashes[2].hexdigest(), "scale_sha256": hashes[3].hexdigest(),
                        "source_slices": blocks, "zlib1_frames": control_frames,
                        "zlib1_frame_bytes": sum(r["frame_bytes"] for r in control_frames),
                        "conditional_frame_bytes": candidate, "tensor_only_frame_bytes": tensor_candidate,
                        "candidate_blocks": len(blocks), "exact": True})
    tensor_descriptor = json.dumps({"keys": KEYS, "context": "tensor-only; repeated context rows",
                                    "total": TOTAL, "version": 1}, sort_keys=True).encode()
    model_bytes = len(model_blob(tables))
    aggregate = charged_aggregate(objects, model_bytes, len(descriptor))
    selected = []
    for obj in objects:
        selected.append({"tensor": obj["tensor"],
                         "conditional": charged_aggregate([obj], model_bytes, len(descriptor)),
                         "tensor_only": charged_aggregate([obj], model_bytes, len(tensor_descriptor),
                                                          "tensor_only_frame_bytes"),
                         "unused_class_table_bytes_charged": model_bytes - CONTEXTS * 512})
    return {"scope": "disjoint contiguous held-out ranges; exact production 1 MiB byte framing",
            "original_model_sha256": digest(model_blob(tables)),
            "tensor_only_model_sha256": digest(model_blob(tensor_only)),
            "tensor_only_descriptor_bytes": len(tensor_descriptor),
            "tensor_only_wire_model_bytes": model_bytes,
            "tensor_only_deduplication": "not implemented; all repeated and unused tables charged",
            "new_source_bf16_bytes_read": source_bytes, "source_budget_bytes": 8 << 20,
            "complete_lightweight_source_payload_bytes": total_budget,
            "objects": objects, "all_three_objects": aggregate,
            "tensor_only_all_three_objects": charged_aggregate(objects, model_bytes, len(tensor_descriptor),
                                                               "tensor_only_frame_bytes"),
            "independent_single_object_deliveries": selected,
            "decision": "open-complete-object-gate" if aggregate["saved_bytes"] > 0 else "reject-larger-held-out-gate",
            "qualification": "size/function only; source slices not whole objects; no serving/RSS/CPU claim"}


def screen(checkpoint):
    if source_header_hash(checkpoint) != SOURCE_HEADER_SHA256:
        raise ValueError("cached public source header differs from frozen lock")
    # All calibration precedes any held-out read. No refit or candidate selection.
    calibration, identities = [], []
    for cls, key in enumerate(KEYS):
        for first in CALIBRATION:
            raw, columns, _, identity = read_sample(checkpoint, key, first)
            calibration.append((raw, columns, cls))
            identities.append(identity)
    tables = parse_model(model_blob(train(calibration)))
    if digest(model_blob(tables)) != FROZEN_MODEL_SHA256:
        raise ValueError("original contextual model changed")
    tensor_only = train(calibration, conditioned=False)
    del calibration
    records = []
    for cls, key in enumerate(KEYS):
        for first in HELD_OUT:
            raw, columns, scales, identity = read_sample(checkpoint, key, first)
            frame = encode(raw, columns, cls, tables)
            decoded = decode(frame, columns, cls, tables, digest(raw))
            if decoded != raw:
                raise AssertionError("decoded bytes differ")
            records.append(identity | {"raw_bytes": len(raw), "conditional_frame_bytes": len(frame),
                           "zlib1_frame_bytes": len(zlib.compress(raw, 1)) + 8,
                           "decoded_sha256": digest(decoded), "exact": decoded == raw,
                           "scale_delivery_bytes": len(scales)})
    model = model_blob(tables)
    # Explicit descriptor charge, in addition to dimensions/class/model hash in frames.
    descriptor = json.dumps({"keys": KEYS, "context": "previous signed magnitude //16, cap7; row reset0",
                             "total": TOTAL, "version": 1}, sort_keys=True).encode()
    larger = larger_screen(checkpoint, tables, tensor_only, descriptor)
    baseline = sum(r["zlib1_frame_bytes"] + r["scale_bytes"] for r in records)
    candidate = sum(r["conditional_frame_bytes"] + r["scale_bytes"] for r in records) + len(model) + len(descriptor)
    quantization_path = ROOT / "python/pllm/runtime/quantization.py"
    from pllm.runtime import quantization
    from pllm.runtime import bundle_compression
    from pllm.runtime._native_support import extension
    native = extension()
    files = [Path(__file__), ROOT / "tests/test_conditional_weight_codec_probe.py",
             quantization_path, ROOT / "python/pllm/runtime/bundle_compression.py",
             Path(quantization.__file__), Path(bundle_compression.__file__), Path(sys.executable).resolve()]
    if native is not None:
        files.append(Path(native.__file__))
    return {"schema": "pllm.conditional_weight_codec.v1", "scope": "bounded functional/size screen",
            "source": {"repo": "Qwen/Qwen2.5-0.5B-Instruct", "revision": REVISION,
                       "resolved_checkpoint": str(Path(checkpoint).resolve()),
                       "published_expected_checkpoint_sha256": SOURCE_SHA256,
                       "whole_checkpoint_sha256": None,
                       "whole_checkpoint_hash_not_rechecked": "no full scan during overlapping workers",
                       "header_sha256": source_header_hash(checkpoint)},
            "implementation_sha256": {str(p.resolve()): file_hash(p) for p in files},
            "calibration": identities, "held_out": records,
            "larger_frame_follow_up": larger,
            "model_bytes": len(model), "model_sha256": digest(model), "descriptor_bytes": len(descriptor),
            "frame_header_bytes": HEADER.size, "baseline_bytes_including_scales": baseline,
            "candidate_bytes_including_scales_and_model": candidate, "saved_bytes": baseline - candidate,
            "fraction_saved": (baseline - candidate) / baseline,
            "payload_only_saved_bytes": sum(r["zlib1_frame_bytes"] - r["conditional_frame_bytes"] for r in records),
            "control_scope": "same 32-row samples, zlib1 plus 8-byte header; not production 1 MiB framing",
            "source_payload_bytes_read": sum(r["source_payload_bytes"] for r in identities + records),
            "decision": "open-byte-gate" if candidate < baseline else "reject-frozen-model",
            "remote_body_work": "unchanged: coding only public delivery, no execution placement change",
            "uniform_masked_tensor_saving": 0, "all_link_measured_saving": None,
            "resources": {"whole_client_rss": None, "decode_cpu": None, "low_qualified": False,
                          "client_floor": "must add codec owners to matching full SDK fresh-client peak",
                          "packed_model_bytes": len(model), "cumulative_u16_projection_bytes": 24 * 257 * 2,
                          "max_block_bytes": MAX_BLOCK, "max_frame_bytes": HEADER.size + 2 * MAX_BLOCK + 4,
                          "python_table_and_integer_overhead": "not measured; packed projection is not Python RSS"},
            "next_gate": "Larger-frame follow-up rejects contextual model and favors tensor-only frames. No integration promotion; separately authorized parent-only complete-object gate remains optional, before any SDK RSS/CPU cohort. No sample extrapolation."}


def source_header_hash(path):
    with Path(path).open("rb") as handle:
        size = struct.unpack("<Q", handle.read(8))[0]
        if not 1 <= size <= 1 << 20:
            raise ValueError("source header bounds")
        return digest(handle.read(size))


def full_worker(payload):
    """Parent-only complete-object byte gate; fresh_workers owns serial admission.

    No performance claim: includes source hashing and independent quantization.
    Compare production 1 MiB-byte zlib control, not tiny sample control.
    """
    checkpoint = Path(payload["checkpoint"])
    expected = SOURCE_SHA256
    if file_hash(checkpoint) != expected:
        raise ValueError("full checkpoint hash differs from published source lock")
    calibration = []
    for cls, key in enumerate(KEYS):
        for first in CALIBRATION:
            raw, width, _, _ = read_sample(checkpoint, key, first)
            calibration.append((raw, width, cls))
    tables = train(calibration)
    with checkpoint.open("rb") as handle:
        size = struct.unpack("<Q", handle.read(8))[0]
        metadata = json.loads(handle.read(size))
    result = []
    for cls, key in enumerate(KEYS):
        rows, width = metadata[key]["shape"]
        raw_hash, decoded_hash, scale_hash = (hashlib.sha256() for _ in range(3))
        candidate = baseline = scale_bytes = blocks = 0
        control = ProductionZlibLedger()
        for first in range(0, rows, ROWS):
            raw, _, scales, _ = read_sample(checkpoint, key, first, min(ROWS, rows - first))
            frame = encode(raw, width, cls, tables)
            decoded = decode(frame, width, cls, tables, digest(raw))
            if decoded != raw:
                raise AssertionError("complete-object byte mismatch")
            raw_hash.update(raw)
            decoded_hash.update(decoded)
            scale_hash.update(scales)
            control.feed(raw)
            candidate += len(frame)
            scale_bytes += len(scales)
            blocks += 1
        baseline = sum(r["frame_bytes"] for r in control.finish())
        result.append({"tensor": key, "rows": rows, "columns": width,
                       "i8_sha256": raw_hash.hexdigest(), "decoded_sha256": decoded_hash.hexdigest(),
                       "scales_sha256": scale_hash.hexdigest(), "scale_delivery_bytes": scale_bytes,
                       "conditional_frame_bytes": candidate, "zlib1_1mib_frame_bytes": baseline,
                       "blocks": blocks, "model_shared_charge_bytes": len(model_blob(tables)),
                       "full_serving_rss_cpu_and_wire": None})
    if file_hash(checkpoint) != expected:
        raise ValueError("checkpoint changed during serial complete-object gate")
    return {"scope": "serial complete-object byte gate only", "checkpoint_sha256": expected,
            "script_sha256": file_hash(__file__), "model_sha256": digest(model_blob(tables)),
            "objects": result, "low_qualified": False,
            "caveat": "charge shared model once plus descriptors/scales; only embedding is always boundary delivery; body objects require matching selective placement"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path.home() / ".cache/huggingface/hub" /
                        "models--Qwen--Qwen2.5-0.5B-Instruct/snapshots" / REVISION / "model.safetensors")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--serial-full", action="store_true",
                        help="parent only: admitted fresh worker; scans cached whole checkpoint")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        print(json.dumps(full_worker(json.load(sys.stdin)), sort_keys=True))
        return
    if args.output is None:
        parser.error("--output required")
    if args.serial_full:
        from frontier_probe_support import fresh_workers
        with fresh_workers(__file__, memory_bytes=512 << 20, timeout=14400) as (run, _, _):
            result = run({"checkpoint": str(args.checkpoint)})
    else:
        result = screen(args.checkpoint)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
