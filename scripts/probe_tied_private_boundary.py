"""Bounded exact tied embedding/full-head gate, not a decoder or wire benchmark.

Pinned public Qwen weights, public prompt fixture and synthetic hidden vectors.
Separate fresh sequential party interpreters own only one share. Generated tokens
come from the original full-vocabulary sampler, but no transformer body executes.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import importlib
import json
import os
from pathlib import Path
import statistics
import sys
import tempfile
import time

from probe_private_lookup_acceleration import publish
from probe_public_grammar_boundary import (
    MODEL, REVISION, ROOT, artifact_lengths, digest, footprint, make_bundle, verify_files,
)

MAX_TABLE = 256 << 20
OUTPUTS = 8


def workload(prompt_rows):
    if type(prompt_rows) is not int or prompt_rows not in (16, 150):
        raise ValueError("only pinned 16+8 and 150+8 schedules admitted")
    return {"prompt_rows": prompt_rows, "output_rows": OUTPUTS,
            "generated_lookup_rows": OUTPUTS - 1,
            "executed_lookup_rows": prompt_rows + OUTPUTS - 1,
            "issued_head_rows": OUTPUTS, "executed_head_rows": OUTPUTS}


def hidden_rows(d):
    import numpy as np
    x = np.random.default_rng(61101).standard_normal((OUTPUTS, d), dtype=np.float32)
    x[0] = 0  # Full-vocabulary ties, no vocabulary restriction.
    x[1] *= np.float32(0.001)
    x[2] *= np.float32(100)
    return x


def split_head(values):
    """Fresh uniform additive shares in Z/(2**32), including negative i8 codes."""
    import numpy as np
    q = np.asarray(values)
    if q.ndim != 2 or q.dtype != np.int8 or q.shape[0] > OUTPUTS:
        raise ValueError("head codes must be bounded rank-two i8")
    first = np.frombuffer(os.urandom(q.size * 4), "<u4").reshape(q.shape)
    second = np.subtract(q.astype(np.uint32), first, dtype=np.uint32)
    return first, second


def restore_head(first, second, activation_scales, weight_scales):
    import numpy as np
    from pllm.runtime.quantization import dequantize_matmul
    if first.dtype != np.uint32 or second.dtype != np.uint32 or first.shape != second.shape:
        raise ValueError("head replies must have matching u32 shapes")
    integer = np.add(first, second, dtype=np.uint32).view(np.int32)
    return integer, dequantize_matmul(integer, activation_scales, weight_scales)


def embedding(record, d):
    import numpy as np
    if len(record) != d + 4:
        raise ValueError("scaled embedding record length differs")
    return np.frombuffer(record[:d], np.int8).astype(np.float32) * np.frombuffer(record[d:], "<f4")[0]


def select(logits):
    from types import SimpleNamespace
    import numpy as np
    from pllm.runtime.transformer_client import MaskedTransformerClientRuntime
    owner = SimpleNamespace(rng=np.random.default_rng(61102))
    tokens = [MaskedTransformerClientRuntime.sample(owner, row, temperature=0.7, top_p=0.9)
              for row in logits]
    return tokens, owner.rng.bit_generator.state


def ledger(public, schedule, parties):
    """Keep raw storage, research artifacts, native frames and absent wire distinct."""
    n, d = public["shape"]
    files = public["files"]
    matrix, scales, records = (files[k]["bytes"] for k in ("head.i8", "scales.f32", "records.bin"))
    if (matrix, scales, records) != (n * d, n * 4, n * (d + 4)):
        raise ValueError("tied boundary storage does not reconcile")
    query = sum(p["lookup_query_bytes"] for p in parties)
    reply = sum(p["lookup_reply_bytes"] for p in parties)
    head_in = sum(p["head_input_bytes"] for p in parties)
    head_out = sum(p["head_output_bytes"] for p in parties)
    if (sum(p["lookup_rows"] for p in parties) != 2 * schedule["executed_lookup_rows"]
            or sum(p["head_rows"] for p in parties) != 2 * OUTPUTS
            or head_in != 2 * OUTPUTS * d * 4 or head_out != 2 * OUTPUTS * n * 4):
        raise ValueError("issued/executed rows or head bodies do not reconcile")
    online = query + reply + head_in + head_out
    scale_artifact = public["scale_artifact"]["zlib_framed_bytes"]
    worker_artifact = public["worker_artifact"]["zlib_framed_bytes"]
    source_bytes = sum(f["bytes"] for f in public.get("source", {}).values())
    return {
        "local_client_raw_matrix_and_scales_bytes": matrix + scales,
        "local_client_research_artifact_bytes": public["full_boundary_artifact"]["zlib_framed_bytes"],
        "offload_client_matrix_bytes": 0, "offload_client_public_scales_bytes": scales,
        "offload_client_scale_research_artifact_bytes": scale_artifact,
        "client_raw_matrix_bytes_removed": matrix,
        "each_logical_worker": {"native_lookup_table_bytes": records,
                                "private_head_snapshot_disk_bytes": matrix,
                                "head_snapshot_resident_bytes": None,
                                "lookup_python_import_overlap_bytes": records,
                                "head_snapshot_creation_overlap_bytes": None,
                                "scales_needed_for_worker_head": 0},
        "both_workers_retained_table_plus_snapshot_bytes": 2 * (records + matrix),
        "storage_total_is_not_measured_concurrent_rss": True,
        "shared_harness_staging_head_file_bytes": matrix,
        "shared_harness_staging_records_file_bytes": records,
        "source_missing_each_worker_retained_raw_delivery_files_bytes": matrix + records,
        "fresh_party_processes": len(parties),
        "cumulative_native_table_import_bytes": len(parties) * records,
        "cumulative_private_snapshot_disk_creation_bytes": sum(p["head_rows"] > 0 for p in parties) * matrix,
        "lookup_native_query_frame_bytes": query, "lookup_native_reply_frame_bytes": reply,
        "head_raw_u32_input_bytes": head_in, "head_raw_u32_output_bytes": head_out,
        "head_native_protocol_frame_bytes": None,
        "online_native_lookup_frames_plus_raw_head_bytes": online,
        "source_present": {"worker_derived_payload_delivery_bytes": 0,
                           "client_scale_artifact_plus_online_bytes": scale_artifact + online,
                           "both_workers_source_bundle_storage_bytes": 2 * source_bytes,
                           "worker_public_derivation_cpu_measured": False},
        "source_missing": {"both_workers_raw_table_and_head_delivery_bytes": 2 * (records + matrix),
                           "both_workers_research_artifact_delivery_bytes": 2 * worker_artifact,
                           "client_and_worker_artifacts_plus_online_bytes": scale_artifact + online + 2 * worker_artifact},
        "physical_wire_bytes": None, "http_tls_authentication_bytes": None,
        "json_hex_pipe_harness_counted_as_wire": False,
        "tokenizer_body_kv_sampler_transport_bytes": None,
        "publisher_disk_bytes": sum(f["bytes"] for f in files.values()),
        "source_bundle_bytes_per_source_owner": source_bytes,
    }


def publish_boundary(payload):
    started = time.process_time()
    result = publish(payload)
    root = Path(payload["root"])
    binding = {"model": MODEL, "revision": REVISION, "source": result["source"],
               "files": result["files"], "shape": result["shape"], "weight_bits": 8, "activation_bits": 8}
    result["scale_artifact"] = artifact_lengths(dict(binding, role="client-public-scales"), [root / "scales.f32"])
    result["worker_artifact"] = artifact_lengths(dict(binding, role="one-worker-table-and-head"),
                                                [root / "records.bin", root / "head.i8"])
    result["publisher_complete_cpu_seconds"] = time.process_time() - started
    result["memory"] = footprint()
    return result


def admit(public):
    n, d = public["shape"]
    if (type(n) is not int or type(d) is not int or n < 1 or d < 1
            or n * (d + 4) > MAX_TABLE or d * 127 * 127 >= 1 << 31):
        raise ValueError("geometry exceeds table or exact signed-ring bound")
    return n, d


def baseline(payload):
    import numpy as np
    from pllm.native import PagedGEMM
    root, public = Path(payload["root"]), payload["public"]
    n, d = admit(public)
    schedule = workload(payload["prompt_rows"])
    started = time.process_time()
    verify_files(root, public["files"])
    scales = np.fromfile(root / "scales.f32", "<f4")
    matrix = PagedGEMM.from_raw(root / "head.i8", public["files"]["head.i8"]["sha256"], (n, d), threads=1)
    bundle = make_bundle(matrix, scales, d)
    bundle.stages["token_lookup"] = replace(bundle.stages["lm_head"], id="token_lookup", op="embedding",
                                           in_features=n, out_features=d, client_weight_layout="embedding")
    x = hidden_rows(d)
    setup = time.process_time() - started
    try:
        started = time.process_time()
        logits = np.concatenate([bundle.local_linear("lm_head", row[None, :]) for row in x])
        head_cpu = time.process_time() - started
        started = time.process_time()
        tokens, state = select(logits)
        sampler_cpu = time.process_time() - started
        ids = json.loads((root / "prompt.json").read_text())[:schedule["prompt_rows"]] + tokens[:-1]
        started = time.process_time()
        # Original local lookup gathers the same tied matrix; no duplicate table.
        for indices in (ids[:schedule["prompt_rows"]], *([i] for i in tokens[:-1])):
            bundle.local_token_lookup(np.asarray(indices, np.int64))
        lookup_cpu = time.process_time() - started
        return {"setup_cpu_seconds": setup, "original_local_full_head_cpu_seconds": head_cpu,
                "local_embedding_cpu_seconds": lookup_cpu, "sampler_cpu_seconds": sampler_cpu,
                "schedule": schedule, "memory": footprint(), "logits": logits.astype("<f4").tobytes().hex(),
                "tokens": tokens, "rng_state": state,
                "baseline": "original ClientBundle.local_linear, full tied PagedGEMM, W8A8; not a two-worker kernel"}
    finally:
        matrix.close()


def party(payload):
    import numpy as np
    from pllm import _native
    from pllm.native import PagedGEMM
    root, public = Path(payload["root"]), payload["public"]
    n, d = admit(public)
    keys, shares, role = payload["keys"], payload["shares"], payload["party"]
    batch_size = payload.get("lookup_batch_size", 64)
    if (type(role) is not int or role not in (0, 1) or not 1 <= len(keys) <= 150
            or type(batch_size) is not int or batch_size not in (1, 64)
            or len(shares) not in (0, OUTPUTS)
            or any(type(k) is not str or len(k) > 4096 for k in keys)
            or any(type(s) is not str or len(s) != d * 8 for s in shares)):
        raise ValueError("party workload exceeds preflight")
    if digest(_native.__file__) != payload["native_sha256"]:
        raise ValueError("installed native identity changed")
    started = time.process_time()
    verify_files(root, {k: public["files"][k] for k in ("head.i8", "records.bin")})
    data = (root / "records.bin").read_bytes()
    server = _native.PrivatePageServer(data, d + 4, role)
    del data
    if server.descriptor() != (bytes.fromhex(public["descriptor"]), n, d + 4):
        raise ValueError("table descriptor differs")
    kernel = (PagedGEMM.from_raw(root / "head.i8", public["files"]["head.i8"]["sha256"],
                               (n, d), threads=1) if shares else None)
    setup = time.process_time() - started
    queries = [bytes.fromhex(key) for key in keys]
    inputs = [np.frombuffer(bytes.fromhex(s), "<u4").reshape(1, d) for s in shares]
    try:
        replies = []
        started, wall = time.process_time(), time.perf_counter()
        for offset in range(0, len(queries), batch_size):
            replies.extend(server.evaluate_batch(queries[offset:offset + batch_size]))
        lookup_cpu, lookup_wall = time.process_time() - started, time.perf_counter() - wall
        try:
            server.evaluate(queries[0])
        except ValueError:
            pass
        else:
            raise AssertionError("one-use lookup replay accepted")
        started, wall = time.process_time(), time.perf_counter()
        outputs = [kernel.wrap32(share) for share in inputs]
        head_cpu, head_wall = time.process_time() - started, time.perf_counter() - wall
        if digest(_native.__file__) != payload["native_sha256"]:
            raise ValueError("native identity changed during party")
        return {"party": role, "setup_cpu_seconds": setup, "lookup_cpu_seconds": lookup_cpu,
                "head_cpu_seconds": head_cpu, "lookup_elapsed_seconds": lookup_wall,
                "head_elapsed_seconds": head_wall, "memory": footprint(),
                "lookup_rows": len(keys), "head_rows": len(shares), "replay_rejected": True,
                "lookup_batch_size": batch_size,
                "head_native_metadata_bytes": kernel.metadata_bytes if kernel else 0,
                "head_transient_weight_bytes": kernel.transient_weight_bytes if kernel else 0,
                "lookup_query_bytes": sum(map(len, queries)), "lookup_reply_bytes": sum(map(len, replies)),
                "head_input_bytes": sum(s.nbytes for s in inputs), "head_output_bytes": sum(s.nbytes for s in outputs),
                "replies": [r.hex() for r in replies],
                "head_outputs": [s.astype("<u4", copy=False).tobytes().hex() for s in outputs]}
    finally:
        if kernel is not None:
            kernel.close()


def lookup_round(worker, root, public, ids, shares, native_sha, *, batch_size=64):
    import numpy as np
    from pllm import _native
    n, d = public["shape"]
    started = time.process_time()
    queries = []
    try:
        for index in ids:
            queries.append(_native.private_page_issue(bytes.fromhex(public["descriptor"]), n, d + 4,
                                                     int(index), leaf_bits=128))
        issue_cpu = time.process_time() - started
        parties = [worker({"mode": "party", "root": str(root), "public": public, "party": role,
                           "keys": [q[role].hex() for q in queries],
                           "shares": [s[None, :].astype("<u4", copy=False).tobytes().hex() for s in shares[role]],
                           "lookup_batch_size": batch_size,
                           "native_sha256": native_sha}) for role in (0, 1)]
        started = time.process_time()
        recovered = [q[2].decode(bytes.fromhex(parties[0]["replies"][i]), bytes.fromhex(parties[1]["replies"][i]))
                     for i, q in enumerate(queries)]
        embeddings = [embedding(record, d) for record in recovered]
        decode_cpu = time.process_time() - started
        # Validation excluded from measured client decode and worker evaluation.
        with (root / "records.bin").open("rb") as stream:
            for index, record, actual in zip(ids, recovered, embeddings, strict=True):
                stream.seek(int(index) * (d + 4))
                expected = stream.read(d + 4)
                if record != expected:
                    raise AssertionError("private record differs from original quantized lookup")
                np.testing.assert_array_equal(actual.view("u4"), embedding(expected, d).view("u4"))
        for result in parties:
            del result["replies"]
        return parties, issue_cpu, decode_cpu
    finally:
        for query in queries:
            query[2].cancel()


def run(output, repeats):
    import numpy as np
    from pllm import _native
    from pllm.runtime.quantization import quantize_activation_per_row
    from frontier_probe_support import fresh_workers
    if output.exists() or type(repeats) is not int or not 1 <= repeats <= 3:
        raise ValueError("new output path and 1..3 repeats required")
    snapshot = Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots" / REVISION
    if not snapshot.is_dir():
        raise ValueError("pinned cached checkpoint required; no download/build/install")
    paths = [Path(__file__), Path(_native.__file__)] + [ROOT / p for p in (
        "scripts/frontier_probe_support.py", "scripts/probe_private_lookup_acceleration.py",
        "scripts/probe_public_grammar_boundary.py", "scripts/finite_language_boundary.py", ".cargo/config.toml",
        "python/pllm/runtime/quantization.py", "python/pllm/runtime/transformer_client.py",
        "python/pllm/native.py", "python/pllm/runtime/paged.py", "python/pllm/runtime/native.py",
        "python/pllm/runtime/_native_support.py", "python/pllm/runtime/safetensors_store.py",
        "crates/pllm-garble/src/private_pages.rs", "crates/pllm-python/src/private_pages.rs",
        "crates/pllm-python/src/paged.rs", "crates/pllm-python/src/lib.rs",
        "crates/pllm-core/src/paged.rs", "crates/pllm-core/src/kernels.rs", "crates/pllm-core/src/public_file.rs",
        "Cargo.lock", "Cargo.toml", "crates/pllm-core/Cargo.toml", "crates/pllm-python/Cargo.toml",
        "crates/pllm-garble/Cargo.toml")]
    installed = {name: Path(importlib.import_module(name).__file__) for name in (
        "pllm.runtime.quantization", "pllm.runtime.transformer_client", "pllm.runtime.paged",
        "pllm.runtime.native", "pllm.runtime._native_support", "pllm.runtime.safetensors_store",
        "pllm.native", "pllm.metrics")}
    paths.extend(installed.values())
    def identities():
        return {str(p): digest(p) for p in paths}
    locked = identities()
    native_sha = digest(_native.__file__)
    with tempfile.TemporaryDirectory(prefix="pllm-tied-boundary-", dir=ROOT / "target") as temp:
        root = Path(temp)
        with fresh_workers(Path(__file__), memory_bytes=2 << 30, timeout=240) as (worker, guard, host):
            if host.disk_free < 2 << 30:
                raise RuntimeError("tied boundary requires 2 GiB free disk")
            public = worker({"mode": "publish", "root": temp, "snapshot": str(snapshot)})
            n, d = public["shape"]
            prompt = json.loads((root / "prompt.json").read_text())
            scales = np.fromfile(root / "scales.f32", "<f4")
            samples = []
            for repeat in range(repeats):
                for count in ((16, 150) if repeat % 2 == 0 else (150, 16)):
                    local = worker({"mode": "baseline", "root": temp, "public": public, "prompt_rows": count})
                    reference = np.frombuffer(bytes.fromhex(local.pop("logits")), "<f4").reshape(OUTPUTS, n)
                    schedule = workload(count)
                    started = time.process_time()
                    quantized = [quantize_activation_per_row(row[None, :], bits=8) for row in hidden_rows(d)]
                    codes = np.concatenate([q.values for q in quantized])
                    activation_scales = np.concatenate([q.scales for q in quantized])
                    split = [split_head(q.values) for q in quantized]
                    shares = tuple(np.concatenate([s[role] for s in split]) for role in (0, 1))
                    client_quantize_share = time.process_time() - started
                    parties, issue, decode = lookup_round(worker, root, public, prompt[:count], shares, native_sha)
                    outputs = [np.concatenate([np.frombuffer(bytes.fromhex(s), "<u4").reshape(1, n)
                                              for s in p.pop("head_outputs")]) for p in parties]
                    started = time.process_time()
                    integers, logits = restore_head(*outputs, activation_scales, scales)
                    restore_cpu = time.process_time() - started
                    np.testing.assert_array_equal(logits.view("u4"), reference.view("u4"))
                    # Independent signed int64 oracle over every source head row.
                    started = time.process_time()
                    with (root / "head.i8").open("rb") as stream:
                        for start in range(0, n, 512):
                            width = min(512, n - start)
                            w = np.frombuffer(stream.read(width * d), np.int8).reshape(width, d)
                            expected = codes.astype(np.int64) @ w.astype(np.int64).T
                            np.testing.assert_array_equal(integers[:, start:start + width], expected)
                    validation_cpu = time.process_time() - started
                    started = time.process_time()
                    tokens, state = select(logits)
                    sampler_cpu = time.process_time() - started
                    if tokens != local.pop("tokens") or state != local.pop("rng_state"):
                        raise AssertionError("full-vocabulary sampler/RNG differs")
                    more, issue2, decode2 = lookup_round(worker, root, public, tokens[:-1],
                                                       (np.empty((0, d), np.uint32),) * 2, native_sha, batch_size=1)
                    for p in more:
                        del p["head_outputs"]
                    parties.extend(more)
                    samples.append({"repeat": repeat, "schedule": schedule, "baseline": local,
                        "parties": parties, "leaf_bits": 128,
                        "issued_lookup_rows": count + OUTPUTS - 1,
                        "exact_record_embedding_integer_logit_sampler_rng": True,
                        "checked_embedding_values": schedule["executed_lookup_rows"] * d,
                        "checked_head_integer_and_logit_values": OUTPUTS * n,
                        "client_quantize_and_share_cpu_seconds": client_quantize_share,
                        "client_lookup_issue_cpu_seconds": issue + issue2,
                        "client_lookup_decode_dequantize_cpu_seconds": decode + decode2,
                        "client_head_restore_dequantize_cpu_seconds": restore_cpu,
                        "client_sampler_cpu_seconds": sampler_cpu,
                        "separate_int64_oracle_cpu_seconds": validation_cpu,
                        "full_logits_sha256": hashlib.sha256(logits.tobytes()).hexdigest(),
                        "byte_ledger": ledger(public, schedule, parties)})
            verify_files(snapshot, public["source"])
            verify_files(root, public["files"])
            if locked != identities():
                raise RuntimeError("source or native implementation changed during cohort")
            guard.check()
            if guard.error:
                raise RuntimeError(guard.error)
            medians = []
            for count in (16, 150):
                group = [s for s in samples if s["schedule"]["prompt_rows"] == count]
                if len(group) != repeats:
                    raise ValueError("incomplete cohort")
                medians.append({"prompt_rows": count, "replicates": repeats,
                    "baseline_original_full_head_cpu_seconds": statistics.median(
                        s["baseline"]["original_local_full_head_cpu_seconds"] for s in group),
                    "all_party_setup_cpu_seconds": statistics.median(
                        sum(p["setup_cpu_seconds"] for p in s["parties"]) for s in group),
                    "all_party_lookup_and_head_cpu_seconds": statistics.median(
                        sum(p["lookup_cpu_seconds"] + p["head_cpu_seconds"] for p in s["parties"]) for s in group),
                    "client_boundary_cpu_seconds": statistics.median(sum(s[k] for k in (
                        "client_quantize_and_share_cpu_seconds", "client_lookup_issue_cpu_seconds",
                        "client_lookup_decode_dequantize_cpu_seconds", "client_head_restore_dequantize_cpu_seconds",
                        "client_sampler_cpu_seconds")) for s in group)})
            report = {"schema": "pllm.tied_private_boundary.v1", "status": "complete", "scope": __doc__,
                "model": MODEL, "revision": REVISION, "public": public, "samples": samples, "medians": medians,
                "build": worker.build, "source_sha256": locked, "host": asdict(host),
                "installed_python_sha256": {name: digest(path) for name, path in installed.items()},
                "numpy_version": np.__version__, "python": sys.version,
                "maximum_swap_growth_bytes": guard.maximum_swap_growth_bytes,
                "source_present_worker_derivation": "not timed; publisher setup is separate, do not count as free",
                "worker_schedule": "two head+prompt parties, then two generated-lookup-only parties; all sequential and fresh",
                "worker_peak_rss_is_client_peak": False, "authenticated_provider_transport": False,
                "full_response_or_sdk_improvement": None, "whole_client_peak_rss_bytes": None,
                "vocabulary_restricted": False, "tee": False, "prompt_disclosed_to_workers": False}
    with output.open("x") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps({"output": str(output), "medians": medians}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        raw = sys.stdin.read(4 << 20)
        if len(raw) >= 4 << 20:
            raise ValueError("worker payload exceeds bound")
        payload = json.loads(raw)
        print(json.dumps({"publish": publish_boundary, "baseline": baseline, "party": party}[payload["mode"]](payload)))
    else:
        if args.output is None:
            parser.error("new --output path required")
        run(args.output, args.repeats)


if __name__ == "__main__":
    main()
