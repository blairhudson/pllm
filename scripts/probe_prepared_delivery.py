"""Bounded correction, WebSocket and ready-to-serve public-artifact screens.

Synthetic masked words never represent user data. The artifact screen uses a
source-locked public embedding slice and the original public tokenizer, charging
both publication and digest-checked replay. It is a local file prototype, not an
SDK cache, HTTP measurement or complete bundle/request speed claim.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import time
import tracemalloc


def timed(call, count=7):
    samples = []
    for _ in range(count):
        wall, cpu = time.perf_counter(), time.process_time()
        result = call()
        samples.append({"wall_seconds": time.perf_counter() - wall,
                        "cpu_seconds": time.process_time() - cpu})
        del result
    tracemalloc.start()
    result = call()
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    del result
    return {"samples": samples, "median_cpu_seconds": statistics.median(
        s["cpu_seconds"] for s in samples), "python_traced_peak_bytes": peak}


def corrections():
    import numpy as np

    rng = np.random.default_rng(971)
    records = []
    for bits in (16, 24, 32):
        shape = (150, 9728)
        transformed = rng.integers(0, 1 << bits, shape, dtype=np.uint32)
        mask = rng.integers(0, 1 << bits, shape, dtype=np.uint32)

        def separate():
            return ((transformed.astype(np.int64) - mask.astype(np.int64)) % (1 << bits)).astype(np.uint32)

        def narrowed():
            value = np.subtract(transformed, mask, dtype=np.uint32)
            if bits < 32:
                np.bitwise_and(value, np.uint32((1 << bits) - 1), out=value)
            return value

        expected = separate()
        assert np.array_equal(expected, narrowed())
        # Independent arbitrary-precision edge oracle, including wrap32.
        for a, b in ((0, 1), ((1 << bits) - 1, 0), (0, (1 << bits) - 1)):
            actual = int(np.subtract(np.uint32(a), np.uint32(b), dtype=np.uint32)) & ((1 << bits) - 1)
            assert actual == (a - b) % (1 << bits)
        records.append({"bits": bits, "shape": list(shape), "checked_outputs": expected.size,
            "output_sha256": hashlib.sha256(expected.tobytes()).hexdigest(),
            "separate": timed(separate), "narrowed": timed(narrowed)})
    return {"records": records}


def websocket():
    import numpy as np
    from websockets.extensions.permessage_deflate import PerMessageDeflate
    from websockets.frames import Frame, OP_BINARY

    rng = np.random.default_rng(514)
    records = []
    # Typical one-row stage frames with fresh masks, public metadata and tickets.
    for size in (3072, 8192, 28672):
        frames = [Frame(OP_BINARY, b"PLLM-public-stage-v1" + bytes(32)
                        + rng.bytes(16) + rng.bytes(size)) for _ in range(144)]

        def exchange(compressed):
            sender = PerMessageDeflate(False, False, 12, 12) if compressed else None
            receiver = PerMessageDeflate(False, False, 12, 12) if compressed else None
            total = 0
            for original in frames:
                encoded = sender.encode(original) if sender else original
                length = len(encoded.data)
                total += length + (2 if length < 126 else 4 if length < 65536 else 10)
                decoded = receiver.decode(encoded, max_size=size + 128) if receiver else encoded
                if decoded.data != original.data:
                    raise ValueError("WebSocket extension changed application bytes")
            return total

        records.append({"body_bytes": size, "frames": len(frames),
            "deflate_frame_bytes": exchange(True), "plain_frame_bytes": exchange(False),
            "deflate": timed(lambda: exchange(True)), "plain": timed(lambda: exchange(False))})
    return {"records": records, "scope": "Local extension encode/decode; no HTTP, TCP, masking header, RTT or SDK claim."}


def artifacts():
    from pllm import load_model
    from pllm.runtime.benchmark_memory import host_memory
    from pllm.runtime.bundle_compression import encode_bundle_frames, iter_bundle_frames
    from pllm.runtime.quantization import quantize_weight_per_row
    from safetensors import safe_open
    from benchmarks.research.qwen3_artifacts import SOURCE

    host = host_memory()
    if host.available - host.reserve < 1 << 30:
        return {"status": "memory-rejected", "required_bytes": 1 << 30}
    manifest = load_model(SOURCE)
    root = Path(manifest.source)
    # Public shape-only selection, with no private inputs or model execution.
    choices = []
    for row in manifest.metadata["source_lock"]["files"]:
        if row["path"].endswith(".safetensors"):
            with safe_open(root / row["path"], framework="np") as store:
                for key in store.keys():
                    shape = store.get_slice(key).get_shape()
                    if len(shape) == 2:
                        choices.append((shape[0] * shape[1], row["path"], key, shape))
    _, filename, key, shape = max(choices)
    rows = min(shape[0], (8 << 20) // shape[1])
    with safe_open(root / filename, framework="pt") as store:
        weights = store.get_slice(key)[:rows].float().numpy()
        payload = quantize_weight_per_row(weights, bits=8).values.tobytes()
    del weights
    records = []
    for label, raw in (("public_w8_matrix_slice", payload), ("public_tokenizer", (root / "tokenizer.json").read_bytes())):
        digest = hashlib.sha256(raw).hexdigest()
        with tempfile.TemporaryFile() as stream:
            def publish():
                stream.seek(0)
                count, framed = 0, hashlib.sha256()
                for block in encode_bundle_frames(raw):
                    stream.write(block)
                    framed.update(block)
                    count += len(block)
                stream.truncate()
                stream.flush()
                return count, framed.hexdigest()

            encoded_size, framed_digest = publish()

            def replay():
                stream.seek(0)
                actual, size = hashlib.sha256(), 0
                while block := stream.read(1 << 20):
                    actual.update(block)
                    size += len(block)
                if actual.hexdigest() != framed_digest or size != encoded_size:
                    raise ValueError("prebuilt artifact changed")
                return size

            stream.seek(0)
            decoded = hashlib.sha256()
            for block in iter_bundle_frames(iter(lambda: stream.read(1 << 20), b""),
                                            expected_size=len(raw), on_wire_bytes=lambda _: None):
                decoded.update(block)
            assert decoded.hexdigest() == digest
            records.append({"kind": label, "raw_bytes": len(raw), "raw_sha256": digest,
                "additional_prebuilt_disk_bytes": encoded_size, "framed_sha256": framed_digest,
                "on_request_encode": timed(lambda: sum(map(len, encode_bundle_frames(raw)))),
                "publish": timed(publish), "verified_file_replay": timed(replay)})
    return {"status": "completed", "source_lock_digest": manifest.source_lock_digest,
            "source_tensor": key, "source_shape": shape, "slice_shape": [rows, shape[1]],
            "records": records}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--child", choices=("corrections", "websocket", "artifacts"))
    args = parser.parse_args()
    if args.child:
        print(json.dumps(globals()[args.child]()))
        return
    if args.output is None or args.output.exists():
        parser.error("a new output path is required")
    records = []
    for case in ("corrections", "websocket", "artifacts"):
        for repetition in range(3):
            raw = subprocess.check_output([sys.executable, "-m", "scripts.probe_prepared_delivery",
                                           "--child", case], text=True, timeout=300)
            records.append({"case": case, "repetition": repetition, **json.loads(raw)})
    args.output.write_text(json.dumps({"schema": "pllm.prepared_delivery_screen.v1", "scope": __doc__,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "records": records},
        indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"output": str(args.output), "processes": len(records)}))


if __name__ == "__main__":
    main()
