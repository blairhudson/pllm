"""Fresh-process exact private-lookup screen with the full tied output head retained.

Real public Qwen boundary weights and public query fixtures. Native framed bodies
are not HTTP or physical wire. Separate sequential workers are not independent
operators. No restricted vocabulary, sampling change, or serving promotion.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import struct
import sys
import tempfile
import time

from probe_public_grammar_boundary import (
    MODEL, REVISION, ROOT, artifact_lengths, digest, footprint, lock_files, verify_files,
)


VARIANTS = ("original_scalar", "accelerated_scalar", "accelerated_packed128")
MAX_TABLE = 256 << 20


def publish(payload):
    import numpy as np
    from tokenizers import Tokenizer
    from pllm.runtime.quantization import quantize_weight_per_row
    from pllm.runtime.safetensors_store import SafeTensorStore

    root, snapshot = Path(payload["root"]), Path(payload["snapshot"])
    started = time.process_time()
    source = lock_files(snapshot)
    config = json.loads((snapshot / "config.json").read_bytes())
    n, d = config["vocab_size"], config["hidden_size"]
    if (n, d, config["tie_word_embeddings"]) != (151936, 896, True):
        raise ValueError("pinned source geometry differs")
    if n * (d + 4) > MAX_TABLE:
        raise ValueError("public table exceeds allocation admission")
    tokenizer = Tokenizer.from_file(str(snapshot / "tokenizer.json"))
    public_text = ("Explain how a library organizes its books and helps readers find reliable "
                   "information. Include examples of fiction, history, science and reference works. ") * 24
    ids = tokenizer.encode(public_text, add_special_tokens=False).ids[:150]
    if len(ids) != 150:
        raise AssertionError("public query fixture too short")
    (root / "prompt.json").write_text(json.dumps(ids))
    del tokenizer
    store = SafeTensorStore(snapshot)
    key = "model.embed_tokens.weight"
    if store.tensor_shape(key) != (n, d) or store.tensor_dtype(key) != "BF16":
        raise ValueError("pinned tied source shape/dtype differs")
    scales = np.empty(n, dtype="<f4")
    with (root / "head.i8").open("xb") as head, (root / "records.bin").open("xb") as records:
        for start, chunk in zip(range(0, n, 512), store.iter_slices(
                key, (slice(i, min(i + 512, n)) for i in range(0, n, 512))), strict=True):
            quantized = quantize_weight_per_row(chunk, bits=8)
            head.write(quantized.values.tobytes())
            scales[start:start + len(chunk)] = quantized.scales
            packed = np.empty((len(chunk), d + 4), np.uint8)
            packed[:, :d] = quantized.values.view(np.uint8)
            packed[:, d:] = quantized.scales.astype("<f4", copy=False).view(np.uint8).reshape(-1, 4)
            records.write(packed.tobytes())
    scales.tofile(root / "scales.f32")
    files = {name: {"bytes": (root / name).stat().st_size, "sha256": digest(root / name)}
             for name in ("head.i8", "scales.f32", "records.bin", "prompt.json")}
    binding = {"source": source, "model": MODEL, "revision": REVISION,
               "weight_bits": 8, "activation_bits": 8, "head_sha256": files["head.i8"]["sha256"],
               "scales_sha256": files["scales.f32"]["sha256"], "shape": [n, d], "rows": "all"}
    artifact = artifact_lengths(binding, [root / "head.i8", root / "scales.f32"])
    descriptor = hashlib.sha256(b"pllm/public-page-table/v1\0" + struct.pack("<QQ", n, d + 4))
    with (root / "records.bin").open("rb") as stream:
        while block := stream.read(1 << 20):
            descriptor.update(block)
    verify_files(snapshot, source)
    return {"source": source, "files": files, "shape": [n, d], "descriptor": descriptor.hexdigest(),
            "full_boundary_artifact": artifact, "publisher_cpu_seconds": time.process_time() - started,
            "memory": footprint(), "vocabulary_restricted": False}


def native_module(path, expected):
    path = Path(path).resolve()
    if not path.is_file() or digest(path) != expected:
        raise ValueError("native implementation differs from frozen identity")
    # Each worker loads exactly one extension. The old implementation is never
    # imported into the current installed package or the parent process.
    spec = importlib.util.spec_from_file_location("_native", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def party(payload):
    import numpy as np  # Equal import footprint across all implementation arms.

    root, public = Path(payload["root"]), payload["public"]
    n, d = public["shape"]
    keys = payload["keys"]
    if (payload["party"] not in (0, 1) or not 1 <= len(keys) <= 150 or n * (d + 4) > MAX_TABLE
            or any(type(key) is not str or len(key) > 4096 for key in keys)):
        raise ValueError("party workload exceeds preflight")
    native = native_module(payload["native_path"], payload["native_sha256"])
    started = time.process_time()
    verify_files(root, {"records.bin": public["files"]["records.bin"]})
    data = (root / "records.bin").read_bytes()
    server = native.PrivatePageServer(data, d + 4, payload["party"])
    del data
    if server.descriptor() != (bytes.fromhex(public["descriptor"]), n, d + 4):
        raise ValueError("private table descriptor differs")
    setup_cpu = time.process_time() - started
    queries = [bytes.fromhex(key) for key in keys]
    replies = []
    started, wall = time.process_time(), time.perf_counter()
    for offset in range(0, len(queries), 64):
        replies.extend(server.evaluate_batch(queries[offset:offset + 64]))
    cpu, elapsed = time.process_time() - started, time.perf_counter() - wall
    try:
        server.evaluate(queries[0])
    except ValueError:
        pass
    else:
        raise AssertionError("used query replay accepted")
    memory = footprint()
    del server
    if digest(payload["native_path"]) != payload["native_sha256"]:
        raise ValueError("worker native implementation changed")
    return {"party": payload["party"], "native_sha256": payload["native_sha256"],
            "setup_cpu_seconds": setup_cpu, "evaluation_cpu_seconds": cpu,
            "evaluation_elapsed_seconds": elapsed, "memory": memory, "replay_rejected": True,
            "query_bytes": sum(map(len, queries)), "reply_bytes": sum(map(len, replies)),
            "replies": [reply.hex() for reply in replies], "numpy_version": np.__version__}


def summarize(samples, repeats):
    if len(samples) != repeats * len(VARIANTS) * 2:
        raise ValueError("incomplete lookup cohort")
    rows = []
    for count in (16, 150):
        for variant in VARIANTS:
            group = [s for s in samples if s["prompt_lookup_rows"] == count and s["variant"] == variant]
            if (sorted(s["repeat"] for s in group) != list(range(repeats))
                    or len({s["native_body_bytes"] for s in group}) != 1
                    or any(not s["exact_embedding_bytes_and_float32_bits"] for s in group)):
                raise ValueError("lookup evidence does not reconcile")
            rows.append({"variant": variant, "prompt_lookup_rows": count, "replicates": repeats,
                         "native_body_bytes": group[0]["native_body_bytes"],
                         "both_party_evaluation_cpu_seconds": statistics.median(
                             sum(p["evaluation_cpu_seconds"] for p in s["parties"]) for s in group),
                         "both_party_setup_cpu_seconds": statistics.median(
                             sum(p["setup_cpu_seconds"] for p in s["parties"]) for s in group),
                         "client_issue_cpu_seconds": statistics.median(s["client_issue_cpu_seconds"] for s in group),
                         "client_decode_cpu_seconds": statistics.median(s["client_decode_cpu_seconds"] for s in group),
                         "largest_party_process_peak_rss_bytes": max(
                             p["memory"]["process_lifetime_peak_rss_bytes"] for s in group for p in s["parties"])})
    return rows


def run(output, repeats, control_dir):
    import numpy as np
    from pllm import _native
    from frontier_probe_support import fresh_workers

    control = json.loads((control_dir / "identity.json").read_text())
    control_path = (control_dir / Path(_native.__file__).name).resolve()
    if digest(control_path) != control["native_sha256"]:
        raise ValueError("control native does not match archived identity")
    snapshot = Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots" / REVISION
    if not snapshot.is_dir() or type(repeats) is not int or not 1 <= repeats <= 3:
        raise ValueError("cached pinned checkpoint and 1..3 repeats required")
    source_paths = [Path(__file__), ROOT / "scripts/frontier_probe_support.py",
                    ROOT / "scripts/probe_public_grammar_boundary.py", ROOT / ".cargo/config.toml",
                    ROOT / "crates/pllm-garble/src/private_pages.rs",
                    ROOT / "crates/pllm-python/src/private_pages.rs", Path(_native.__file__),
                    control_path, control_dir / "identity.json"]
    def identities():
        return {str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p): digest(p) for p in source_paths}
    source_hashes = identities()
    native_sha = digest(_native.__file__)
    with tempfile.TemporaryDirectory(prefix="pllm-private-lookup-", dir=ROOT / "target") as temp:
        with fresh_workers(Path(__file__), memory_bytes=2 << 30, timeout=240) as (worker, guard, host):
            public = worker({"mode": "publish", "root": temp, "snapshot": str(snapshot)})
            ids = json.loads((Path(temp) / "prompt.json").read_text())
            n, d = public["shape"]
            samples = []
            for repeat in range(repeats):
                for count in ((16, 150) if repeat % 2 == 0 else (150, 16)):
                    order = VARIANTS[repeat:] + VARIANTS[:repeat]
                    for variant in order:
                        leaf_bits = 128 if variant == "accelerated_packed128" else 1
                        started = time.process_time()
                        queries = [_native.private_page_issue(bytes.fromhex(public["descriptor"]), n, d + 4,
                                                             index, leaf_bits=leaf_bits) for index in ids[:count]]
                        issue_cpu = time.process_time() - started
                        parties = []
                        old = variant == "original_scalar"
                        try:
                            for role in (0, 1):
                                parties.append(worker({"mode": "party", "root": temp, "public": public,
                                    "party": role, "keys": [q[role].hex() for q in queries],
                                    "native_path": str(control_path) if old else _native.__file__,
                                    "native_sha256": control["native_sha256"] if old else native_sha}))
                            started = time.process_time()
                            values = [q[2].decode(bytes.fromhex(parties[0]["replies"][i]),
                                                  bytes.fromhex(parties[1]["replies"][i])) for i, q in enumerate(queries)]
                            decode_cpu = time.process_time() - started
                            with (Path(temp) / "records.bin").open("rb") as stream:
                                for index, value in zip(ids[:count], values, strict=True):
                                    stream.seek(index * (d + 4))
                                    expected = stream.read(d + 4)
                                    if value != expected:
                                        raise AssertionError("private lookup differs from the full source table")
                                    a = np.frombuffer(value[:d], np.int8).astype(np.float32) * np.frombuffer(value[d:], "<f4")[0]
                                    b = np.frombuffer(expected[:d], np.int8).astype(np.float32) * np.frombuffer(expected[d:], "<f4")[0]
                                    np.testing.assert_array_equal(a.view("u4"), b.view("u4"))
                            for party_result in parties:
                                del party_result["replies"]
                            samples.append({"repeat": repeat, "variant": variant, "prompt_lookup_rows": count,
                                "leaf_bits": leaf_bits, "checked_embedding_values": count * d,
                                "exact_embedding_bytes_and_float32_bits": True,
                                "client_issue_cpu_seconds": issue_cpu, "client_decode_cpu_seconds": decode_cpu,
                                "parties": parties, "native_body_bytes": sum(
                                    p["query_bytes"] + p["reply_bytes"] for p in parties)})
                        finally:
                            for query in queries:
                                query[2].cancel()
            medians = summarize(samples, repeats)
            if source_hashes != identities():
                raise RuntimeError("probe or installed implementation changed during cohort")
            guard.check()
            if guard.error:
                raise RuntimeError(guard.error)
            report = {"schema": "pllm.private_lookup_acceleration.v1", "scope": __doc__,
                "model": MODEL, "revision": REVISION, "public": public, "samples": samples, "medians": medians,
                "build": worker.build, "source_sha256": source_hashes, "control": control,
                "host": asdict(host), "maximum_swap_growth_bytes": guard.maximum_swap_growth_bytes,
                "client_issuance_scope": "current installed implementation in every arm; original arm isolates original server evaluation",
                "full_tied_client_head_retained": True, "client_weight_bytes_removed": 0,
                "full_request_traffic_savings": None, "full_decoder_checkpoint_quality": None,
                "whole_client_peak_rss_bytes": None, "authenticated_provider_transport": False,
                "additional_worker_table_payload_if_missing_source_bytes": 2 * public["files"]["records.bin"]["bytes"],
                "source_delivery_scope": "derived tables add no network only when both workers already own the source; full client head remains in every arm"}
    with output.open("x") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps({"output": str(output), "medians": medians,
                      "maximum_swap_growth_bytes": report["maximum_swap_growth_bytes"]}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--control-dir", type=Path, default=ROOT / "target/private-lookup-control")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        payload = json.loads(sys.stdin.read(4 << 20))
        print(json.dumps({"publish": publish, "party": party}[payload["mode"]](payload)))
    else:
        if args.output is None:
            parser.error("--output required")
        run(args.output, args.repeats, args.control_dir)


if __name__ == "__main__":
    main()
