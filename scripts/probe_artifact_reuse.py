"""Charge public pre-encoding once, then serve independent fresh HTTP clients.

Bounded file/HTTP prototype using the ordinary frame codec and authenticated
private FilePayload import. Every client downloads all public objects. Measures
publication, provider serving, client decode/hash/storage and separate disk
owners. No SDK provider cache, complete bundle, decoder, TLS or WAN-speed claim.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time


def digest(value):
    return hashlib.sha256(value).hexdigest()


def corpus(root, synthetic):
    """Publish only public source objects; never accept a private input fixture."""
    if synthetic:
        import numpy as np
        raw = np.random.default_rng(716).integers(-128, 128, 2 << 20, dtype=np.int8).tobytes()
        objects = [("synthetic_i8", raw), ("synthetic_text", b'public token dictionary\n' * 50000)]
        source = {"synthetic": True}
    else:
        from pllm import load_model
        from pllm.runtime.benchmark_memory import host_memory
        from pllm.runtime.quantization import quantize_weight_per_row
        from safetensors import safe_open
        from benchmarks.research.qwen3_artifacts import SOURCE

        host = host_memory()
        if host.available - host.reserve < 1 << 30:
            raise ValueError("public slice preparation requires 1 GiB post-reserve headroom")
        manifest = load_model(SOURCE)
        directory, choices = Path(manifest.source), []
        for row in manifest.metadata["source_lock"]["files"]:
            if row["path"].endswith(".safetensors"):
                with safe_open(directory / row["path"], framework="np") as store:
                    for key in store.keys():
                        shape = store.get_slice(key).get_shape()
                        if len(shape) == 2:
                            choices.append((shape[0] * shape[1], row["path"], key, shape))
        _, filename, key, shape = max(choices)
        rows = min(shape[0], (8 << 20) // shape[1])
        with safe_open(directory / filename, framework="pt") as store:
            weights = store.get_slice(key)[:rows].float().numpy()
            raw = quantize_weight_per_row(weights, bits=8).values.tobytes()
        del weights
        tokenizer = directory / "tokenizer.json"
        if not 0 < tokenizer.stat().st_size <= 64 << 20:
            raise ValueError("public tokenizer exceeds source bound")
        objects = [("public_w8_matrix_slice", raw), ("public_tokenizer", tokenizer.read_bytes())]
        source = {"source_lock_digest": manifest.source_lock_digest,
                  "source_tensor": key, "source_shape": shape, "slice_shape": [rows, shape[1]]}
    records = []
    for index, (kind, raw) in enumerate(objects):
        (root / f"{index}.raw").write_bytes(raw)
        records.append({"index": index, "kind": kind, "raw_bytes": len(raw), "raw_sha256": digest(raw)})
    return {"source": source, "objects": records}


def publish(root):
    from pllm.runtime.bundle_compression import encode_bundle_frames
    records = json.loads((root / "source.json").read_bytes())
    wall, cpu = time.perf_counter(), time.process_time()
    for row in records["objects"]:
        raw = (root / f'{row["index"]}.raw').read_bytes()
        if len(raw) != row["raw_bytes"] or digest(raw) != row["raw_sha256"]:
            raise ValueError("public source object changed")
        count, hashed = 0, hashlib.sha256()
        with (root / f'{row["index"]}.encoded').open("xb") as stream:
            for block in encode_bundle_frames(raw):
                stream.write(block)
                count += len(block)
                hashed.update(block)
            stream.flush()
            os.fsync(stream.fileno())
        row.update(encoded_bytes=count, encoded_sha256=hashed.hexdigest())
    records["publication"] = {"wall_seconds": time.perf_counter() - wall,
                              "cpu_seconds": time.process_time() - cpu}
    (root / "published.json").write_text(json.dumps(records))
    return records


def provider(root, variant):
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    import threading
    from pllm.runtime.bundle_compression import encode_bundle_frames

    config = json.loads((root / "published.json").read_bytes())
    # Both variants retain the source, as a serving model still owns its weights.
    raw = [(root / f'{row["index"]}.raw').read_bytes() for row in config["objects"]]
    if any(len(value) != row["raw_bytes"] or digest(value) != row["raw_sha256"]
           for row, value in zip(config["objects"], raw, strict=True)):
        raise ValueError("provider source identity changed")
    samples, failures = [], []
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_GET(self):
            table = {f'/objects/{row["index"]}': row for row in config["objects"]}
            if self.path not in table:
                self.send_error(404)
                return
            row = table[self.path]
            wall, cpu = time.perf_counter(), time.process_time()
            count, hashed = 0, hashlib.sha256()
            try:
                self.send_response(200)
                self.send_header("Content-Type", "application/octet-stream")
                self.send_header("X-Raw-SHA256", row["raw_sha256"])
                self.end_headers()
                def blocks():
                    if variant == "on_request":
                        yield from encode_bundle_frames(raw[row["index"]])
                    else:
                        with (root / f'{row["index"]}.encoded').open("rb") as stream:
                            while chunk := stream.read(1 << 20):
                                yield chunk
                for block in blocks():
                    hashed.update(block)
                    count += len(block)
                    self.wfile.write(block)
                self.wfile.flush()
                if count != row["encoded_bytes"] or hashed.hexdigest() != row["encoded_sha256"]:
                    raise ValueError("provider encoding identity changed")
                samples.append({"object": row["index"], "body_bytes": count,
                    "wall_seconds": time.perf_counter() - wall, "cpu_seconds": time.process_time() - cpu})
            except BaseException as error:
                failures.append(type(error).__name__)
                raise

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    print(json.dumps({"port": server.server_port}), flush=True)
    try:
        if sys.stdin.readline().strip() != "stop":
            raise ValueError("missing provider shutdown")
    finally:
        server.shutdown()
        thread.join()
        server.server_close()
    if failures:
        raise ValueError("artifact serving failed")
    print(json.dumps({"samples": samples, "source_payload_bytes": sum(map(len, raw)),
                      "resources_retired": True}), flush=True)


def client(root, url):
    import httpx
    from pllm.runtime.benchmark_memory import process_memory
    from pllm.runtime.bundle_compression import iter_bundle_frames
    from pllm.runtime.bundle_storage import FilePayload
    from pllm.runtime.http_response import owned_response

    config = json.loads((root / "published.json").read_bytes())
    before = process_memory()
    wall, cpu = time.perf_counter(), time.process_time()
    records = []
    with httpx.Client(base_url=url, timeout=60, trust_env=False) as transport:
        for row in config["objects"]:
            received, hashed = 0, hashlib.sha256()
            with transport.stream("GET", f'/objects/{row["index"]}') as response, owned_response(response):
                response.raise_for_status()
                if response.headers.get("X-Raw-SHA256") != row["raw_sha256"]:
                    raise ValueError("client source binding mismatch")
                def chunks():
                    nonlocal received
                    for block in response.iter_raw(chunk_size=65536):
                        received += len(block)
                        hashed.update(block)
                        yield block
                blocks = iter_bundle_frames(chunks(), expected_size=row["raw_bytes"], on_wire_bytes=lambda _: None)
                private = FilePayload(blocks, size=row["raw_bytes"], digest=row["raw_sha256"])
                try:
                    if received != row["encoded_bytes"] or hashed.hexdigest() != row["encoded_sha256"]:
                        raise ValueError("client framed identity mismatch")
                    records.append({"object": row["index"], "body_bytes": received,
                                    "decoded_bytes": len(private), "raw_sha256": private.digest})
                finally:
                    private.close()
    return {"wall_seconds": time.perf_counter() - wall, "cpu_seconds": time.process_time() - cpu,
        "cold_process_cpu_seconds": time.process_time(), "memory_before": before,
        "memory_after": process_memory(), "objects": records, "resources_retired": True,
        "client_snapshot_peak_disk_bytes": max(row["raw_bytes"] for row in config["objects"])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--clients", type=int, default=4)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--child", choices=("corpus", "publish", "provider", "client"))
    parser.add_argument("--root", type=Path)
    parser.add_argument("--variant", choices=("on_request", "preencoded"))
    parser.add_argument("--url")
    args = parser.parse_args()
    if not 1 <= args.clients <= 8 or not 1 <= args.repetitions <= 5:
        parser.error("at most eight fresh clients and five repetitions")
    if args.child:
        if args.child == "provider":
            provider(args.root, args.variant)
            return
        result = (corpus(args.root, args.synthetic) if args.child == "corpus" else
                  publish(args.root) if args.child == "publish" else client(args.root, args.url))
        print(json.dumps(result))
        return
    if args.output is None or args.output.exists():
        parser.error("a new output path is required")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, "-m", "scripts.probe_artifact_reuse"]
    records = []
    with tempfile.TemporaryDirectory(prefix="artifact-reuse-", dir=Path("target")) as directory:
        root = Path(directory).resolve()
        source = json.loads(subprocess.check_output(command + ["--child", "corpus", "--root", str(root)]
            + (["--synthetic"] if args.synthetic else []), text=True, timeout=180))
        (root / "source.json").write_text(json.dumps(source))
        for repetition in range(args.repetitions):
            for path in root.glob("*.encoded"):
                path.unlink()
            published = json.loads(subprocess.check_output(command + ["--child", "publish", "--root", str(root)],
                                                          text=True, timeout=120))
            for variant in (("on_request", "preencoded") if repetition % 2 == 0 else ("preencoded", "on_request")):
                process = subprocess.Popen(command + ["--child", "provider", "--variant", variant, "--root", str(root)],
                    stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
                try:
                    ready = json.loads(process.stdout.readline())
                    clients = [json.loads(subprocess.check_output(command + ["--child", "client", "--root", str(root),
                        "--url", f'http://127.0.0.1:{ready["port"]}'], text=True, timeout=120)) for _ in range(args.clients)]
                    stdout, _ = process.communicate("stop\n", timeout=30)
                    if process.returncode:
                        raise ValueError("provider failed")
                    served = json.loads(stdout)
                    if len(served["samples"]) != args.clients * len(source["objects"]):
                        raise ValueError("provider/client request conservation failed")
                    records.append({"repetition": repetition, "variant": variant, "publication": published["publication"],
                        "provider": served, "clients": clients})
                finally:
                    if process.poll() is None:
                        process.terminate()
                        try:
                            process.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait()
    from pllm.runtime import bundle_compression, bundle_storage
    paths = [Path(__file__), Path(bundle_compression.__file__), Path(bundle_storage.__file__)]
    result = {"schema": "pllm.artifact_reuse_probe.v1", "scope": __doc__, "source": source["source"],
        "objects": published["objects"], "source_disk_bytes": sum(row["raw_bytes"] for row in source["objects"]),
        "additional_encoded_disk_bytes": sum(row["encoded_bytes"] for row in published["objects"]),
        "source_sha256": {str(path): digest(path.read_bytes()) for path in paths}, "records": records}
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"output": str(args.output), "fresh_clients": len(records) * args.clients}))


if __name__ == "__main__":
    main()
