"""Pinned finite-output head/PIR gate; separate serial roles, no serving promotion.

Real public Qwen boundary weights, synthetic hidden rows and public prompt tokens.
Pipe execution measures native bodies/CPU, not authenticated provider transport.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import resource
import statistics
import struct
import sys
import tempfile
import time
from types import SimpleNamespace
import zlib

from finite_language_boundary import canonical, compile_language, dense_restricted, tokenizer_pieces


ROOT = Path(__file__).resolve().parents[1]
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
WORDS = ('{"label":"negative"}', '{"label":"neutral"}', '{"label":"positive"}')
MAX_TABLE = 256 << 20
MAX_HEAD_ROWS = 512


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def footprint():
    import psutil
    return {"rss_bytes": psutil.Process().memory_info().rss,
            "process_lifetime_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            * (1 if sys.platform == "darwin" else 1024)}


def lock_files(root):
    names = ("config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "model.safetensors")
    result = {}
    for name in names:
        path = root / name
        size = path.stat().st_size
        if not 0 < size <= (2 << 30 if name.endswith("safetensors") else 64 << 20):
            raise ValueError("source file exceeds bounded screen")
        result[name] = {"bytes": size, "sha256": digest(path)}
    return result


def verify_files(root, expected):
    for name, identity in expected.items():
        path = root / name
        if path.stat().st_size != identity["bytes"] or digest(path) != identity["sha256"]:
            raise ValueError("source/artifact changed")


def artifact_lengths(header, paths):
    """Exact isolated research artifact framing, not the SDK object serializer."""
    metadata = canonical(header)
    raw, compressed, frames = 12 + len(metadata), 12 + len(metadata), 0
    for path in paths:
        with Path(path).open("rb") as stream:
            while block := stream.read(1 << 20):
                raw += 8 + len(block)
                compressed += 8 + len(zlib.compress(block, level=6))
                frames += 1
    return {"raw_framed_bytes": raw, "zlib_framed_bytes": compressed,
            "header_bytes": 12 + len(metadata), "frames": frames,
            "format": "research magic8+header_length_u32+canonical_header; u32 raw/encoded sizes per <=1MiB independent block"}


def publish(payload):
    import numpy as np
    from tokenizers import Tokenizer
    from pllm.runtime.quantization import quantize_weight_per_row
    from pllm.runtime.safetensors_store import SafeTensorStore

    root, snapshot = Path(payload["root"]), Path(payload["snapshot"])
    started = time.process_time()
    source = lock_files(snapshot)
    config = json.loads((snapshot / "config.json").read_bytes())
    if (config["vocab_size"], config["hidden_size"], config["tie_word_embeddings"]) != (151936, 896, True):
        raise ValueError("pinned source geometry differs")
    n, d, eos = config["vocab_size"], config["hidden_size"], config["eos_token_id"]
    if n * (d + 4) > MAX_TABLE:
        raise ValueError("public table exceeds allocation admission")
    raw = (snapshot / "tokenizer.json").read_bytes()
    language = compile_language(raw, WORDS, vocabulary=n, eos=eos)
    if len(language.union) > MAX_HEAD_ROWS:
        raise ValueError("finite head union exceeds screen admission")
    tok = Tokenizer.from_str(raw.decode())
    paths = []
    for word in WORDS:
        ids = tok.encode(word, add_special_tokens=False).ids
        state, remaining = 0, 64
        for token in ids + [eos]:
            state = language.advance(state, token, remaining)
            remaining -= 1
        if state != -1 or tok.decode(ids) != word:
            raise AssertionError("original tokenizer language parity")
        paths.append(ids)
    pieces = tokenizer_pieces(raw, n, eos)
    # A lower bound only: these tokens can occur inside arbitrary JSON strings.
    interior = []
    for token, piece in pieces.items():
        try:
            text = piece.decode("utf-8", errors="strict")
        except UnicodeError:
            continue
        if text and all(ord(c) >= 32 and c not in '\\"' for c in text):
            interior.append(token)
    # Independent exhaustive byte-prefix oracle over the complete source vocabulary.
    for state, prefix in enumerate(language.prefixes):
        expected = {token for token, piece in pieces.items()
                    if any(word.startswith(prefix + piece) for word in language.words)}
        if prefix in language.words:
            expected.add(eos)
        if expected != {token for token, _ in language.transitions[state]}:
            raise AssertionError("complete vocabulary reachability oracle differs")
    # Verify alternate, noncanonical generated token paths with the original decoder.
    alternate = []
    for word in language.words:
        state, ids = 0, []
        while language.prefixes[state] != word:
            choices = [(len(language.prefixes[dest]), token, dest)
                       for token, dest in language.transitions[state] if dest != -1
                       and word.startswith(language.prefixes[dest])]
            _, token, state = min(choices)
            ids.append(token)
        if tok.decode(ids) != word.decode():
            raise AssertionError("alternate tokenizer segmentation differs")
        alternate.append(ids)
    public_text = ("Classify the sentiment of this public test sentence using the fixed label schema. "
                   "The weather is pleasant and the service is helpful. ") * 24
    prompt_ids = tok.encode(public_text, add_special_tokens=False).ids[:150]
    if len(prompt_ids) != 150:
        raise AssertionError("public query fixture too short")
    for name, value in (("language.json", language.to_spec()), ("prompt.json", prompt_ids)):
        (root / name).write_bytes(canonical(value))
    del pieces, tok, raw

    store = SafeTensorStore(snapshot)
    key = "model.embed_tokens.weight"
    if store.tensor_shape(key) != (n, d) or store.tensor_dtype(key) != "BF16":
        raise ValueError("pinned tied source shape/dtype differs")
    scales = np.empty(n, dtype="<f4")
    ids = np.array(language.union, dtype=np.int64)
    shard = np.empty((len(ids), d), dtype=np.int8)
    source_values = hashlib.sha256()
    with (root / "head.i8").open("xb") as head, (root / "records.bin").open("xb") as records:
        for start, chunk in zip(range(0, n, 512), store.iter_slices(
                key, (slice(i, min(i + 512, n)) for i in range(0, n, 512))), strict=True):
            source_values.update(chunk.astype("<f4", copy=False).tobytes())
            q = quantize_weight_per_row(chunk, bits=8)
            head.write(q.values.tobytes())
            scales[start:start + len(chunk)] = q.scales
            packed = np.empty((len(chunk), d + 4), dtype=np.uint8)
            packed[:, :d] = q.values.view(np.uint8)
            packed[:, d:] = q.scales.astype("<f4", copy=False).view(np.uint8).reshape(-1, 4)
            records.write(packed.tobytes())
            selected = np.flatnonzero((ids >= start) & (ids < start + len(chunk)))
            shard[selected] = q.values[ids[selected] - start]
    scales.tofile(root / "scales.f32")
    shard.tofile(root / "shard.i8")
    scales[ids].tofile(root / "shard.scales.f32")
    artifact_names = ("head.i8", "records.bin", "scales.f32", "shard.i8", "shard.scales.f32", "language.json", "prompt.json")
    files = {name: {"bytes": (root / name).stat().st_size, "sha256": digest(root / name)} for name in artifact_names}
    binding = {"source": source, "model": MODEL, "revision": REVISION,
               "weight_bits": 8, "activation_bits": 8, "head_sha256": files["head.i8"]["sha256"],
               "scales_sha256": files["scales.f32"]["sha256"], "shape": [n, d]}
    full_bill = artifact_lengths(dict(binding, rows="all"), [root / "head.i8", root / "scales.f32"])
    shard_bill = artifact_lengths(dict(binding, rows=list(language.union), language=language.to_spec()),
                                 [root / "shard.i8", root / "shard.scales.f32"])
    descriptor = hashlib.sha256(b"pllm/public-page-table/v1\0" + struct.pack("<QQ", n, d + 4))
    with (root / "records.bin").open("rb") as stream:
        while block := stream.read(1 << 20):
            descriptor.update(block)
    verify_files(snapshot, source)
    return {"source": source, "files": files, "shape": [n, d], "eos": eos,
            "descriptor": descriptor.hexdigest(), "language": language.to_spec(),
            "language_digest": language.digest, "canonical_paths": paths,
            "alternate_paths": alternate, "arbitrary_json_interior_token_lower_bound": len(interior),
            "public_source_float32_values_sha256": source_values.hexdigest(),
            "full_boundary_artifact": full_bill, "finite_boundary_artifact": shard_bill,
            "publisher_complete_cpu_seconds": time.process_time() - started, "memory": footprint()}


def make_bundle(weights, scales, hidden):
    from pllm.runtime.native import MaskedGEMM
    from pllm.runtime.transformer_client import ClientBundle, StageMetadata
    rows = len(scales)
    stage = StageMetadata("lm_head", "linear", hidden, rows, 8, 8, "pow2", 1 << 32, 32,
                          scales, client_weight=weights, client_weight_scales=scales)
    bundle = ClientBundle("research-finite-head", {}, {}, {}, {"lm_head": stage}, {}, {}, schema_version=2)
    bundle._local_kernel = MaskedGEMM(threads=1, build=False)
    if not bundle._local_kernel.native:
        raise RuntimeError("installed native kernel required")
    return bundle


def head_gate(payload):
    import numpy as np
    from pllm.native import PagedGEMM
    from pllm.runtime.quantization import quantize_activation_per_row
    from pllm.runtime.transformer_client import MaskedTransformerClientRuntime

    root, public = Path(payload["root"]), payload["public"]
    verify_files(root, public["files"])
    raw = (Path(payload["snapshot"]) / "tokenizer.json").read_bytes()
    n, d = public["shape"]
    language = compile_language(raw, WORDS, vocabulary=n, eos=public["eos"])
    if language.to_spec() != public["language"]:
        raise ValueError("public language binding changed")
    del raw
    full_scales = np.fromfile(root / "scales.f32", "<f4")
    small = np.fromfile(root / "shard.i8", np.int8).reshape(-1, d)
    scales = np.fromfile(root / "shard.scales.f32", "<f4")
    x = np.random.default_rng(61061).standard_normal((8, d), dtype=np.float32)
    x[0] = 0  # Exact tied logits exercise the original vocabulary-wide sampler order.
    x[1] = 0
    x[1, 0] = 127
    x[2] *= np.float32(0.001)
    x[3] *= np.float32(100)
    matrix = PagedGEMM.from_raw(root / "head.i8", public["files"]["head.i8"]["sha256"], (n, d), threads=1)
    try:
        full, sliced = make_bundle(matrix, full_scales, d), make_bundle(small, scales, d)
        reference = full.local_linear("lm_head", x)
        candidate = sliced.local_linear("lm_head", x)
        np.testing.assert_array_equal(candidate.view("u4"), reference[:, language.union].view("u4"))
        q = quantize_activation_per_row(x, bits=8)
        # Independent bounded signed-integer oracle, including the original float32 order.
        with (root / "head.i8").open("rb") as stream:
            for start in range(0, n, 512):
                rows = min(512, n - start)
                w = np.frombuffer(stream.read(rows * d), np.int8).reshape(rows, d)
                integers = q.values.astype(np.int64) @ w.astype(np.int64).T
                expected = integers.astype(np.float32) * q.scales[:, None] * full_scales[None, start:start + rows]
                np.testing.assert_array_equal(reference[:, start:start + rows].view("u4"), expected.view("u4"))
        checked, selections = 0, 0
        for row in range(len(x)):
            for state in range(len(language.prefixes)):
                for remaining in sorted({64, language.minimum_selections[state]}):
                    allowed = language.allowed(state, remaining)
                    oracle = np.full(n, -np.inf, np.float32)
                    oracle[list(allowed)] = reference[row, list(allowed)]
                    actual = dense_restricted(candidate[row], language.union, language, state, remaining)
                    np.testing.assert_array_equal(actual.view("u4"), oracle.view("u4"))
                    checked += n
                    for temperature, top_p in ((0, 1), (0.7, 0.9), (1, 1)):
                        a = SimpleNamespace(rng=np.random.default_rng(row * 101 + state))
                        b = SimpleNamespace(rng=np.random.default_rng(row * 101 + state))
                        left = MaskedTransformerClientRuntime.sample(a, oracle, temperature=temperature, top_p=top_p)
                        right = MaskedTransformerClientRuntime.sample(b, actual, temperature=temperature, top_p=top_p)
                        if left != right or a.rng.bit_generator.state != b.rng.bit_generator.state:
                            raise AssertionError("SDK sampler or RNG continuation differs")
                        selections += 1
        return {"full_head_values_checked": reference.size,
                "masked_vocabulary_values_checked": checked, "sampler_and_rng_comparisons": selections,
                "complete_head_sha256": hashlib.sha256(reference.tobytes()).hexdigest(),
                "sliced_head_sha256": hashlib.sha256(candidate.tobytes()).hexdigest(),
                "synthetic_hidden_rows": len(x), "exact_float32_and_integer_oracle": True,
                "memory": footprint(), "scope": "boundary-only validation; both arms and oracle charged; no body/KV or task quality"}
    finally:
        matrix.close()


def party(payload):
    from pllm import _native

    root, public = Path(payload["root"]), payload["public"]
    n, d = public["shape"]
    keys = payload["keys"]
    if (payload["party"] not in (0, 1) or not 1 <= len(keys) <= 150
            or n * (d + 4) > MAX_TABLE or any(type(k) is not str or len(k) > 4096 for k in keys)):
        raise ValueError("party workload exceeds preflight")
    started = time.process_time()
    verify_files(root, {"records.bin": public["files"]["records.bin"]})
    data = (root / "records.bin").read_bytes()
    server = _native.PrivatePageServer(data, d + 4, payload["party"])
    del data
    if server.descriptor() != (bytes.fromhex(public["descriptor"]), n, d + 4):
        raise ValueError("private table descriptor differs")
    setup = time.process_time() - started
    queries = [bytes.fromhex(key) for key in keys]
    started, wall = time.process_time(), time.perf_counter()
    replies = []
    for offset in range(0, len(queries), 64):
        replies.extend(server.evaluate_batch(queries[offset:offset + 64]))
    cpu, elapsed = time.process_time() - started, time.perf_counter() - wall
    try:
        server.evaluate(queries[0])
    except ValueError:
        pass
    else:
        raise AssertionError("used query replay accepted")
    measured = footprint()
    del server
    return {"replies": [reply.hex() for reply in replies], "setup_cpu_seconds": setup,
            "evaluation_cpu_seconds": cpu, "evaluation_elapsed_seconds": elapsed,
            "native_query_bytes": sum(map(len, queries)), "native_reply_bytes": sum(map(len, replies)),
            "replay_rejected": True, "memory": measured, "party": payload["party"]}


def run(output, repeats):
    import numpy as np
    from pllm import _native
    from frontier_probe_support import fresh_workers

    snapshot = Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots" / REVISION
    if not snapshot.is_dir() or type(repeats) is not int or not 1 <= repeats <= 3:
        raise ValueError("cached pinned checkpoint and 1..3 repeats required")
    source_paths = [Path(__file__), ROOT / "scripts/finite_language_boundary.py",
                    ROOT / "scripts/frontier_probe_support.py", ROOT / "tests/test_public_grammar_boundary.py"]
    import pllm.runtime.transformer_client as tc
    import pllm.runtime.quantization as qt
    source_paths.extend((Path(tc.__file__), Path(qt.__file__), Path(_native.__file__)))
    source_hashes = {str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p): digest(p) for p in source_paths}
    with tempfile.TemporaryDirectory(prefix="pllm-public-grammar-", dir=ROOT / "target") as temp:
        with fresh_workers(Path(__file__), memory_bytes=2 << 30, timeout=240) as (worker, guard, host):
            shared = {"root": temp, "snapshot": str(snapshot)}
            public = worker(dict(shared, mode="publish"))
            shared["public"] = public
            head = worker(dict(shared, mode="head"))
            ids = json.loads((Path(temp) / "prompt.json").read_bytes())
            samples = []
            for repeat in range(repeats):
                for count in ((16, 150) if repeat % 2 == 0 else (150, 16)):
                    started = time.process_time()
                    queries = [_native.private_page_issue(bytes.fromhex(public["descriptor"]), *
                               (public["shape"][0], public["shape"][1] + 4), index) for index in ids[:count]]
                    client_issue = time.process_time() - started
                    parties = []
                    try:
                        for role in (0, 1):
                            parties.append(worker(dict(shared, mode="party", party=role,
                                                       keys=[q[role].hex() for q in queries])))
                        started = time.process_time()
                        values = [q[2].decode(bytes.fromhex(parties[0]["replies"][i]),
                                              bytes.fromhex(parties[1]["replies"][i])) for i, q in enumerate(queries)]
                        client_decode = time.process_time() - started
                        d = public["shape"][1]
                        with (Path(temp) / "records.bin").open("rb") as source:
                            for index, value in zip(ids[:count], values, strict=True):
                                source.seek(index * (d + 4))
                                expected = source.read(d + 4)
                                if value != expected:
                                    raise AssertionError("private embedding bytes differ")
                                a = np.frombuffer(value[:d], np.int8).astype(np.float32) * np.frombuffer(value[d:], "<f4")[0]
                                b = np.frombuffer(expected[:d], np.int8).astype(np.float32) * np.frombuffer(expected[d:], "<f4")[0]
                                np.testing.assert_array_equal(a.view("u4"), b.view("u4"))
                        for item in parties:
                            del item["replies"]
                        samples.append({"repeat": repeat, "prompt_lookup_rows": count,
                                        "checked_embedding_values": count * d, "client_issue_cpu_seconds": client_issue,
                                        "client_decode_cpu_seconds": client_decode, "parties": parties,
                                        "native_body_bytes": sum(p["native_query_bytes"] + p["native_reply_bytes"] for p in parties)})
                    finally:
                        for q in queries:
                            q[2].cancel()
            medians = []
            for count in (16, 150):
                selected = [r for r in samples if r["prompt_lookup_rows"] == count]
                medians.append({"prompt_lookup_rows": count, "replicates": len(selected),
                                "native_body_bytes": selected[0]["native_body_bytes"],
                                "both_party_evaluation_cpu_seconds": statistics.median(
                                    sum(p["evaluation_cpu_seconds"] for p in s["parties"]) for s in selected),
                                "both_party_setup_cpu_seconds": statistics.median(
                                    sum(p["setup_cpu_seconds"] for p in s["parties"]) for s in selected),
                                "largest_party_process_peak_rss_bytes": max(
                                    p["memory"]["process_lifetime_peak_rss_bytes"] for s in selected for p in s["parties"])})
            if source_hashes != {str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p): digest(p) for p in source_paths}:
                raise RuntimeError("probe or installed implementation changed during cohort")
            guard.check()
            if guard.error:
                raise RuntimeError(guard.error)
            report = {"schema": "pllm.public_grammar_boundary_gate.v1", "model": MODEL, "revision": REVISION,
                      "scope": __doc__, "public": public, "head": head, "samples": samples, "medians": medians,
                      "build": worker.build, "host": asdict(host), "maximum_swap_growth_bytes": guard.maximum_swap_growth_bytes,
                      "source_sha256": source_hashes,
                      "complete_request_bytes": None, "client_only_peak_rss_bytes": None,
                      "full_decoder_or_task_quality": None, "authenticated_provider_transport": False,
                      "role_scope": "one native party per fresh sequential interpreter; co-located, not independent operators",
                      "cold_delivery_cases": {
                          "existing_full_checkpoint_at_each_lookup_role": "derive records locally; additional table network bytes zero, CPU/RSS/storage charged",
                          "two_additional_workers_missing_source": 2 * public["files"]["records.bin"]["bytes"],
                          "all_parties_cold": "charge common source distribution in both arms; do not add derived-table transfer when locally derived"}}
    # Publish only after the shared runner's final build check and owner retirement.
    with Path(output).open("x") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps({"output": str(output), "union": len(public["language"]["union"]), "medians": medians,
                      "head": head, "maximum_swap_growth_bytes": report["maximum_swap_growth_bytes"]}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        payload = json.loads(sys.stdin.read(4 << 20))
        modes = {"publish": publish, "head": head_gate, "party": party}
        print(json.dumps(modes[payload["mode"]](payload)))
    else:
        if args.output is None:
            parser.error("--output required")
        run(args.output, args.repeats)


if __name__ == "__main__":
    main()
