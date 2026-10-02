"""Public prepared artifact locality through ordinary SDK and canonical benchmark.

No model downloads. Horizons 10/100 are projections from measured cold/warm
requests, not repeated execution. Body accounting is not HTTP wire or peak RAM.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

import msgpack
import numpy as np
import psutil

from pllm import Deployment, ExecutionBudget, Experiment, Model
from pllm.preparation import PreparedInventory
from pllm.profiles import MaskedLinearCpu
from pllm.protocols import ClientBundleTransport
from pllm.quantization import SymmetricPerRow
from pllm.roles import ClientLinearRoles, OutputHeadAtInference
from pllm.runtime.bundle_artifacts import configure_artifact_cache
from pllm.runtime.bundle_compression import encode_bundle_frames
from pllm.runtime.client import OpenAI, ProtocolError
from pllm.runtime.dashboard import client_body_placement_snapshot
from pllm.runtime.semantic_executor import SemanticDecoderRuntime
from pllm.runtime.servers import build_roles
from pllm.runtime.tiny_llama import create_tiny_llama_checkpoint

QWEN = "Qwen/Qwen2.5-0.5B-Instruct"
REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
PROMPT = "Explain fresh masks briefly."


def selected(source, root, *, attention=False, encoding="artifacts", remote_head=False):
    return Experiment(
        "attention-" + encoding if attention else "baseline-" + encoding,
        MaskedLinearCpu(
            source, quantization=SymmetricPerRow(weight_bits=8, activation_bits=8),
            placement=ClientLinearRoles(["qkv_projection", "attention_output"]) if attention else None,
            boundary=OutputHeadAtInference() if remote_head else None,
            inventory=PreparedInventory("request-sized", rows=1),
            delivery=ClientBundleTransport(encoding),
        ),
        Deployment.local(root=str(root)),
        ExecutionBudget(requests=16, max_input_tokens=128, max_new_tokens=2),
    )


def sdk_client(topology, experiment, cache):
    return OpenAI(
        base_url=topology._inference_url, api_key=topology._inference_key,
        preparation_base_url=topology._preparation_url,
        preparation_api_key=topology._preparation_key,
        default_model=experiment.resolve().model, experiment=experiment,
        background_inventory_refill=False, bundle_cache_dir=cache,
    )


def measured_request(client, model_id):
    vectors = []
    forward = SemanticDecoderRuntime._forward

    def capture_forward(runtime, ids, **kwargs):
        value = forward(runtime, ids, **kwargs)
        vectors.append(np.asarray(value).copy())
        return value

    before = client.privacy_audit.to_dict()
    cpu, started = time.process_time(), time.perf_counter()
    with patch.object(SemanticDecoderRuntime, "_forward", capture_forward):
        response = client.responses.create(model=model_id, input=PROMPT, max_output_tokens=2,
                                           temperature=0, store=False)
    seconds, cpu_seconds = time.perf_counter() - started, time.process_time() - cpu
    after = client.privacy_audit.to_dict()
    audit = {key: after[key] - before[key] for key in before}
    state = client._core._transformer_states[model_id]
    inventory = state.prepared_inventory
    bundle = state.bundle
    # Vocabulary projection is a dense linear MAC even though its semantic op
    # is named lm_head. Token embedding is lookup, not a dense matrix dot.
    linear = {stage.id: stage for stage in bundle.stages.values() if stage.op in {"linear", "lm_head"}}
    all_macs = sum(s.in_features * s.out_features for s in linear.values())
    remote_macs = sum(s.in_features * s.out_features for s in linear.values() if s.client_weight is None)
    arrays = {id(array): array for array in bundle.arrays.values()}
    for stage in bundle.stages.values():
        for array in (stage.client_weight, stage.client_weight_scales, stage.weight_scales,
                      stage.bias, stage.input_equalization, stage.client_aux_weight, stage.client_aux_scales):
            if array is not None:
                arrays[id(array)] = array
    native_snapshot_bytes = sum(bundle.stages[sid].client_weight.nbytes
                                for sid in bundle._local_matrices)
    return {
        "seconds": seconds, "client_cpu_seconds": cpu_seconds,
        "audit": audit, "usage": {"input_tokens": response.usage.input_tokens,
                                    "output_tokens": response.usage.output_tokens},
        "output_sha256": hashlib.sha256(response.output_text.encode()).hexdigest(),
        "captured_logits": [{"shape": list(vector.shape), "sha256": hashlib.sha256(vector.tobytes()).hexdigest()}
                            for vector in vectors],
        "raw_bundle_sha256": state.bundle_fingerprint,
        "source_lock_digest": state.bundle.manifest["metadata"]["source_lock_digest"],
        "body_fingerprint": state.bundle.privacy["body_fingerprint"],
        "stage_commitment": state.bundle.privacy["stage_commitment"],
        "material_inventory_id": inventory.id,
        "material_status": inventory.status(),
        "placement": client_body_placement_snapshot(client, model_id),
        "all_linear_macs_per_row": all_macs, "remote_linear_macs_per_row": remote_macs,
        "remote_all_linear_mac_fraction": remote_macs / all_macs,
        "client_bundle_numpy_payload_bytes": sum(a.nbytes for a in arrays.values()),
        "native_cpu_i8_snapshot_bytes": native_snapshot_bytes,
        "inference_native_i8_stage_snapshot_payload_floor_bytes": sum(
            s.in_features * s.out_features for s in {s.id: s for s in bundle.stages.values()}.values()),
        "native_snapshot_floor_scope": "shape-derived stage payload floor per role; excludes allocator overhead, source and NumPy copies; not peak RAM",
        "private_prefill_cache_payload_bytes": 0 if state.prefill_cache is None else state.prefill_cache.size_bytes,
        "peak_client_memory_bytes": None,
    }, vectors


def body_cost(row):
    audit = row["audit"]
    fields = ("preparation_upload_bytes", "preparation_download_bytes", "correction_push_bytes",
              "session_authorization_upload_bytes", "session_authorization_download_bytes",
              "masked_online_upload_bytes", "masked_online_download_bytes", "bundle_network_bytes")
    return sum(audit[k] for k in fields)


def cohort(source, work, *, cap):
    from pllm.model_loader import resolve_model

    started, cpu = time.perf_counter(), time.process_time()
    resolved = resolve_model(source)
    source_resolution = {"seconds": time.perf_counter() - started,
                         "cpu_seconds": time.process_time() - cpu,
                         "source_lock_digest": resolved.source_lock_digest,
                         "checkpoint_digest": resolved.checkpoint_digest,
                         "checkpoint_storage_bytes": sum(path.stat().st_size for path in
                                                         resolved.path.glob("*.safetensors")),
                         "network_download_bytes": 0}
    model_id = source.model_id or source.source
    cache = work / "sdk-cache"
    samples, children, controls, material_ids = [], [], {}, set()
    failure = None
    for placement in ("baseline", "attention", "return_baseline"):
        attention = placement == "attention"
        experiment = selected(source, work / placement, attention=attention)
        started = time.perf_counter()
        with build_roles(experiment, engine_threads=1) as topology:
            startup_seconds = time.perf_counter() - started
            processes = [{"role": item.role, "pid": item.pid, "running": item.running}
                         for item in topology.statuses]
            assert len(processes) == 2 and all(p["running"] and p["pid"] != os.getpid() for p in processes)
            children.append({"placement": placement, "processes": processes,
                             "source_and_role_startup_seconds": startup_seconds,
                             "child_cpu_seconds_at_ready": {
                                 p["role"]: sum(psutil.Process(p["pid"]).cpu_times()[:2]) for p in processes}})
            encodings = ("none", "zlib", "artifacts") if placement != "return_baseline" else ("artifacts",)
            for encoding in encodings:
                candidate = selected(source, work / placement, attention=attention, encoding=encoding)
                selected_cache = cache if encoding == "artifacts" else work / "control-cache" / placement / encoding
                with sdk_client(topology, candidate, selected_cache) as client:
                    core = client._core
                    if encoding == "artifacts":
                        object_cache = configure_artifact_cache(client, max_bytes=cap)
                    child_cpu = {p["role"]: sum(psutil.Process(p["pid"]).cpu_times()[:2]) for p in processes}
                    row, vectors = measured_request(client, model_id)
                    row["child_response_cpu_seconds"] = {
                        p["role"]: sum(psutil.Process(p["pid"]).cpu_times()[:2]) - child_cpu[p["role"]]
                        for p in processes}
                    row.update(placement_name=placement, encoding=encoding,
                               experiment_digest=candidate.configuration_digest())
                    if encoding == "artifacts":
                        row["artifact_cache"] = object_cache.stats.to_dict()
                    if placement != "return_baseline" and encoding == "none":
                        controls[placement] = (row["output_sha256"], vectors)
                        # Compare actual HTTP raw/framed sizes with local encoders;
                        # record hash/compression CPU explicitly, outside inference.
                        started_hash, cpu_hash = time.perf_counter(), time.process_time()
                        response = core.http.get(f"/v1/runtime/models/{model_id}/client-bundle", headers=core.headers)
                        response.raise_for_status()
                        payload = response.content
                        digest = hashlib.sha256(payload).hexdigest()
                        framed = sum(len(frame) for frame in encode_bundle_frames(payload))
                        row["raw_zlib_analysis"] = {
                            "raw_bytes": len(payload), "zlib_framed_bytes": framed, "raw_sha256": digest,
                            "hash_and_zlib_seconds": time.perf_counter() - started_hash,
                            "hash_and_zlib_cpu_seconds": time.process_time() - cpu_hash,
                            "extra_probe_control_download_bytes": len(payload),
                        }
                    golden_text, golden_vectors = controls["attention" if attention else "baseline"]
                    row["exact_raw_w8_logits"] = (len(vectors) == len(golden_vectors) and bool(vectors)
                                                   and all(np.array_equal(a, b) for a, b in zip(vectors, golden_vectors)))
                    row["exact_raw_output"] = row["output_sha256"] == golden_text
                    assert row["exact_raw_w8_logits"]
                    assert row["exact_raw_output"]
                    assert row["material_inventory_id"] not in material_ids
                    material_ids.add(row["material_inventory_id"])
                    assert row["audit"]["preparation_requests_during_online"] == 0
                    if encoding == "artifacts" and placement == "return_baseline":
                        assert row["artifact_cache"]["object_requests"] == 0
                        assert row["artifact_cache"]["manifest_download_bytes"] > 0
                    samples.append(row)
                    if encoding == "artifacts":
                        # Fresh SDK process-equivalent object-cache horizon. No
                        # private state/material is reused between these clients.
                        with sdk_client(topology, candidate, cache) as warm:
                            warm_cache = configure_artifact_cache(warm, max_bytes=cap)
                            warm_row, warm_vectors = measured_request(warm, model_id)
                            warm_row.update(placement_name=placement + "_warm", encoding=encoding,
                                            artifact_cache=warm_cache.stats.to_dict())
                            assert warm_cache.stats.object_requests == 0
                            assert warm_row["material_inventory_id"] not in material_ids
                            material_ids.add(warm_row["material_inventory_id"])
                            warm_row["exact_raw_w8_logits"] = (len(warm_vectors) == len(golden_vectors) and bool(warm_vectors)
                                                               and all(np.array_equal(a, b) for a, b in zip(warm_vectors, golden_vectors)))
                            warm_row["exact_raw_output"] = warm_row["output_sha256"] == golden_text
                            assert warm_row["exact_raw_w8_logits"] and warm_row["exact_raw_output"]
                            samples.append(warm_row)
            if placement == "return_baseline":
                with sdk_client(topology, experiment, cache) as damaged:
                    object_cache = configure_artifact_cache(damaged, max_bytes=cap)
                    # Corrupt a shared scale/vector, guaranteeing baseline use.
                    from pllm.runtime.bundle_artifacts import parse_manifest
                    response = damaged._core.http.get(
                        f"/v1/runtime/models/{model_id}/client-bundle-artifacts", headers=damaged._core.headers)
                    descriptor = damaged._core._client_bundle_descriptor(model_id)
                    manifest = parse_manifest(response.content, fingerprint=descriptor["sha256"], size=descriptor["size"])
                    object_row = min(manifest["objects"], key=lambda r: r["size"])
                    (object_cache.root / (object_row["sha256"] + ".blob")).write_bytes(b"corrupt")
                    repaired, repaired_vectors = measured_request(damaged, model_id)
                    repaired.update(placement_name="corrupt_object_repair", encoding="artifacts",
                                    artifact_cache=object_cache.stats.to_dict())
                    repaired["exact_raw_w8_logits"] = (len(repaired_vectors) == len(golden_vectors)
                                                       and all(np.array_equal(a, b) for a, b in zip(repaired_vectors, golden_vectors)))
                    repaired["exact_raw_output"] = repaired["output_sha256"] == golden_text
                    assert repaired["exact_raw_w8_logits"] and repaired["exact_raw_output"]
                    assert object_cache.stats.corruptions == object_cache.stats.object_requests == 1
                    assert repaired["material_inventory_id"] not in material_ids
                    material_ids.add(repaired["material_inventory_id"])
                    samples.append(repaired)
                with sdk_client(topology, experiment, cache) as invalid:
                    configure_artifact_cache(invalid, max_bytes=cap)

                    def alter_manifest(response):
                        if response.request.url.path.endswith("/client-bundle"):
                            response.read()
                            value = msgpack.unpackb(response.content, raw=False)
                            graph = msgpack.unpackb(value["skeleton"], raw=False)
                            graph["tokenizer"]["chat_template"] = "altered"
                            value["skeleton"] = msgpack.packb(graph, use_bin_type=True)
                            response._content = msgpack.packb(value, use_bin_type=True)

                    invalid._core.http.event_hooks["response"].append(alter_manifest)
                    try:
                        invalid.responses.create(model=model_id, input=PROMPT, max_output_tokens=2, temperature=0)
                    except ProtocolError:
                        assert invalid.privacy_audit.preparation_rows == 0
                        assert invalid.privacy_audit.preparation_upload_bytes == 0
                        assert invalid.privacy_audit.session_authorization_upload_bytes == 0
                        failure = {"bad_raw_digest_rejected": True, "reserved_rows": 0,
                                   "audit": invalid.privacy_audit.to_dict(),
                                   "artifact_cache": invalid._core.artifact_cache_stats.to_dict()}
                    else:
                        raise AssertionError("altered artifact graph was admitted")
    horizons = []
    for placement in ("baseline", "attention"):
        cold = next(r for r in samples if r["placement_name"] == placement and r["encoding"] == "artifacts")
        warm = next(r for r in samples if r["placement_name"] == placement + "_warm")
        for requests in (1, 10, 100):
            horizons.append({"placement": placement, "requests": requests,
                             "kind": "measured_one_request" if requests == 1 else "projection_from_one_cold_one_warm_request",
                             "covered_source_preparation_session_decode_bundle_body_bytes": body_cost(cold) + (requests - 1) * body_cost(warm),
                             "artifact_object_download_bytes": cold["artifact_cache"]["object_download_bytes"],
                             "manifest_download_bytes": cold["artifact_cache"]["manifest_download_bytes"] + (requests - 1) * warm["artifact_cache"]["manifest_download_bytes"],
                             "source_checkpoint_network_download_bytes": 0})
    # Explicit actual >74% all-linear remote-MAC profile, including the head.
    # Qwen's ordinary local head is large; body-only percentages do not qualify.
    if resolved.manifest.tied_embeddings:
        remote_row = max((row for row in samples if row["encoding"] == "artifacts"),
                         key=lambda row: row["remote_all_linear_mac_fraction"])
        remote_row = {**remote_row, "remote_head_scope_blocker": "installed remote-head runtime rejects tied token/head checkpoints"}
    else:
        most_remote = selected(source, work / "most-remote", remote_head=True)
        with build_roles(most_remote, engine_threads=1) as topology:
            with sdk_client(topology, most_remote, cache) as client:
                object_cache = configure_artifact_cache(client, max_bytes=cap)
                remote_row, remote_vectors = measured_request(client, model_id)
                remote_row["artifact_cache"] = object_cache.stats.to_dict()
                assert remote_row["remote_all_linear_mac_fraction"] > .74
                assert remote_row["material_inventory_id"] not in material_ids
                golden_text, golden_vectors = controls["baseline"]
                remote_row["exact_raw_w8_logits"] = (len(remote_vectors) == len(golden_vectors)
                                                     and all(np.array_equal(a, b) for a, b in zip(remote_vectors, golden_vectors)))
                remote_row["exact_raw_output"] = remote_row["output_sha256"] == golden_text
                assert remote_row["exact_raw_w8_logits"] and remote_row["exact_raw_output"]
    remote_row["greater_than_74_percent_all_linear_remote_macs"] = remote_row["remote_all_linear_mac_fraction"] > .74
    return {"samples": samples, "child_processes": children, "horizons": horizons,
            "invalid_manifest": failure, "all_private_inventory_ids_distinct": True,
            "actual_most_remote_mac_profile": remote_row,
            "source_resolution": source_resolution,
            "cache_cap_bytes": cap, "source_checkpoint_network_download_bytes": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cached-qwen", action="store_true")
    parser.add_argument("--cached-model-source", type=Path,
                        help="Optional local real checkpoint, including 9 GB sources; never downloaded")
    parser.add_argument("--cache-mib", type=int, default=2048)
    parser.add_argument("--canonical-benchmark", action="store_true")
    parser.add_argument("--archive", type=Path)
    args = parser.parse_args()
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    with tempfile.TemporaryDirectory(prefix="pllm-artifacts-") as directory:
        work = Path(directory).resolve()
        root = create_tiny_llama_checkpoint(work / "tiny", hidden_size=128,
                                           intermediate_size=1024, head_dim=32,
                                           num_hidden_layers=2, model_type="qwen2", with_qkv_bias=True,
                                           tie_word_embeddings=False)
        tiny_source = Model.path(str(root), model_id="artifact-locality-tiny")
        result = {"schema": "pllm.artifact_locality_evidence.v1", "date": "2026-10-01",
                  "scope": "public compiled schema-2 prepared bundles; ordinary SDK; two child roles; covered bodies, not wire/RSS",
                  "tiny": cohort(tiny_source, work / "tiny-run", cap=args.cache_mib << 20),
                  "pinned_qwen": None, "cached_model_source": None,
                  "canonical_benchmark": None}
        result["admission"] = {"temporary_whitelist_probe": False}
        if args.cached_qwen:
            from huggingface_hub import hf_hub_download
            root = Path(hf_hub_download(QWEN, "config.json", revision=REVISION, local_files_only=True)).parent
            source = Model.hf(QWEN, revision=REVISION, local_files_only=True, model_id=QWEN)
            result["pinned_qwen"] = cohort(source, work / "qwen-run", cap=args.cache_mib << 20)
            result["pinned_qwen"]["source"] = {"model_id": QWEN, "revision": REVISION, "local_files_only": True}
        if args.cached_model_source:
            source = Model.path(str(args.cached_model_source.resolve()), model_id="artifact-locality-real")
            result["cached_model_source"] = cohort(source, work / "real-run", cap=args.cache_mib << 20)
        # Preserve measured SDK evidence if the optional canonical benchmark
        # fails afterward; final successful archive replaces it.
        if args.archive:
            args.archive.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
        if args.canonical_benchmark:
            from pllm.runtime.benchmark_cli import run_loopback_benchmark
            experiment = selected(tiny_source, work / "canonical")
            result["canonical_benchmark"] = run_loopback_benchmark(
                model=tiny_source.source, model_id=tiny_source.model_id, tiny=False,
                prompt=PROMPT, max_output_tokens=2, warmups=0, repetitions=1,
                timeout_seconds=120, experiment=experiment, temperature=0, capture_output_digest=True)
        result["code_sha256"] = {
            name: hashlib.sha256((Path(__file__).resolve().parents[1] / name).read_bytes()).hexdigest()
            for name in ("python/pllm/runtime/bundle_artifacts.py", "python/pllm/runtime/client.py",
                         "python/pllm/runtime/server.py", "python/pllm/runtime/transformer_engine.py",
                         "python/pllm/protocols/bundle_transport.py", "python/pllm/profiles/__init__.py",
                         "crates/pllm-compiler/src/lib.rs", "scripts/probe_artifact_locality.py")}
        document = json.dumps(result, sort_keys=True, indent=2) + "\n"
        if args.archive:
            args.archive.write_text(document)
        print(document)


if __name__ == "__main__":
    main()
