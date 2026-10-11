"""Five all-link screens: conserved serving bytes, geometry, and a public codec.

Only the public W8 matrix codec is newly executed here. Placement, history and
publisher sizes are explicitly cost projections, not executable serving choices.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics
import struct
import sys
import time
import zlib

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CONTROL = ROOT / "docs/evidence/research-prepared-study-qwen25-confirm.json"
GEOMETRY = ROOT / "docs/evidence/network-algebra-qwen25-2026-10-06.json"
REPO = "Qwen/Qwen2.5-0.5B-Instruct"
REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
MODES = ("zlib1", "bitplanes-zlib1", "zigzag-bitplanes-zlib1")
FRAME_LIMIT = 1 << 20


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def frozen_sources():
    from pllm.runtime import bundle_compression, quantization, safetensors_store
    paths = [Path(__file__), ROOT / "scripts/frontier_probe_support.py", CONTROL,
             GEOMETRY, Path(quantization.__file__), Path(safetensors_store.__file__),
             Path(bundle_compression.__file__)]
    return {str(p): digest(p) for p in paths}


def conserved_control(document):
    """Use one complete measured cell; never add upload and receipt counters."""
    if not document["complete"]:
        raise ValueError("incomplete control")
    selected = [r for r in document["results"]
                if r["execution"]["case"]["name"] == "prepared/digit8"]
    if len(selected) != 3:
        raise ValueError("expected the three-client confirmation")
    reports = [r["execution"]["details"]["report"] for r in selected]
    fingerprints = {run["model_fingerprint"] for report in reports for run in report["runs"]}
    if len(fingerprints) != 1 or len({r["configuration_digest"] for r in selected}) != 1:
        raise ValueError("control identity mismatch")
    bodies = []
    for report in reports:
        account = report["topology_accounting"]
        if len(report["runs"]) != 2 or account["warmups"]:
            raise ValueError("expected two measured requests without warmups")
        for ledger in account["runs"]:
            if sum(x["serialized_body_bytes"] for x in ledger["body_bytes_by_edge"]) != ledger["all_link_serialized_body_bytes"]:
                raise ValueError("directed ledger does not conserve")
        startup = account["startup"]
        if any(x["serialized_body_bytes"] for x in startup["body_bytes_by_edge"]) or startup["all_link_serialized_body_bytes"]:
            raise ValueError("this screen expects the demand control's empty startup")
        bodies.append(sum(ledger["all_link_serialized_body_bytes"] for ledger in account["runs"]))
    report = reports[0]
    links = defaultdict(int)
    stages = defaultdict(lambda: defaultdict(int))
    account = report["topology_accounting"]
    for run, ledger, stage_ledger in zip(report["runs"], account["runs"], account["stages"]["runs"], strict=True):
        if run["status"] != "completed" or (run["tokens"]["input_tokens"], run["tokens"]["output_tokens"]) != (150, 8):
            raise ValueError("control workload changed")
        for link in ledger["body_bytes_by_edge"]:
            name = f"{link['source']}->{link['destination']}/{link['phase']}"
            links[name] += link["serialized_body_bytes"]
        if not stage_ledger["reconciled_with_protocol_bodies"]:
            raise ValueError("stage ledger is not reconciled")
        for stage, stage_links in stage_ledger["body_bytes_by_stage_and_edge"].items():
            for edge, count in stage_links.items():
                stages[stage][edge] += count
    if sum(links.values()) != bodies[0]:
        raise ValueError("pair ledger does not conserve")
    classified = {
        "corrections": links["preparation->inference/offline"],
        "outputs": links["inference->client/online"],
        "inputs": links["client->inference/online"],
        "public_bundle": links["inference->client/cold"],
    }
    classified["control_and_seeds"] = bodies[0] - sum(classified.values())
    if classified["control_and_seeds"] < 0:
        raise ValueError("overlapping classifications")
    return {
        "scope": "first complete digit8 cold/exact-repeat pair; application bodies, not wire",
        "evidence_sha256": digest(CONTROL),
        "configuration_digest": selected[0]["configuration_digest"],
        "body_fingerprint": next(iter(fingerprints)),
        "pair_bytes": bodies[0], "replicate_pair_bytes": bodies,
        "classes": classified, "directed_counters": dict(links),
        "stages": {k: dict(v) for k, v in stages.items()},
    }


def choose_placement(items, *, budget, horizon):
    """Exact 0/1 optimization; raw public weights/scales delivered once.

    The budget is dense per-row body MACs, not a measured runtime CPU budget.
    Items are disjoint compiled matrix groups, and may be selected only once.
    """
    if type(horizon) is not int or horizon < 1 or budget < 0:
        raise ValueError("invalid placement horizon/budget")
    if len({x["id"] for x in items}) != len(items):
        raise ValueError("duplicate matrix group")
    unit = math.gcd(*(x["macs"] for x in items)) if items else 1
    capacity = budget // unit
    if capacity > 100_000:
        raise ValueError("placement state exceeds bounded screen")
    states = {0: (0, ())}
    for index, item in enumerate(items):
        cost = item["macs"] // unit
        value = horizon * item["pair_stage_bytes"] - item["weight_and_scale_bytes"]
        for used, (saving, indices) in list(states.items()):
            next_used = used + cost
            if next_used <= capacity and saving + value > states.get(next_used, (-1, ()))[0]:
                states[next_used] = (saving + value, (*indices, index))
    _, (saving, indices) = max(states.items(), key=lambda pair: (pair[1][0], -pair[0]))
    chosen = [items[i] for i in indices]
    return {"horizon_pairs": horizon, "saved_body_bytes_estimate": saving,
            "selected": [x["id"] for x in chosen],
            "client_dense_macs_per_row": sum(x["macs"] for x in chosen),
            "extra_public_payload_bytes": sum(x["weight_and_scale_bytes"] for x in chosen),
            "removed_stage_bytes_per_pair": sum(x["pair_stage_bytes"] for x in chosen)}


def placement_screen(control, geometry):
    if not geometry["complete"] or geometry["source"]["body_fingerprint"] != control["body_fingerprint"]:
        raise ValueError("geometry is not the measured quantized body")
    items = []
    for stage in geometry["stages"]:
        rows, columns = stage["shape"]
        traffic = control["stages"][stage["stage"]]
        # Seeds, admission and common bundle/control bodies stay charged.
        removable = sum(traffic.get(k, 0) for k in (
            "preparation->inference", "client->inference", "inference->client"))
        items.append({"id": stage["stage"], "macs": rows * columns,
                      "weight_and_scale_bytes": rows * columns + 4 * rows,
                      "pair_stage_bytes": removable})
    if len(items) != len(control["stages"]):
        raise ValueError("geometry and measured stage sets differ")
    total = sum(x["macs"] for x in items)
    frontier = []
    for horizon in (1, 4, 16):
        row = choose_placement(items, budget=total // 10, horizon=horizon)
        baseline = (control["classes"]["public_bundle"]
                    + horizon * (control["pair_bytes"] - control["classes"]["public_bundle"]))
        row.update(baseline_body_bytes_estimate=baseline,
                   fractional_saving=row["saved_body_bytes_estimate"] / baseline,
                   client_dense_macs_fraction=row["client_dense_macs_per_row"] / total)
        frontier.append(row)
    return {"scope": "optimistic placement projection from observed stage bodies; no execution",
            "body_dense_macs_per_row": total,
            "budget": "at most 10% of dense per-row body MACs; new client ownership requires opt-in",
            "omitted": ["new bundle framing/manifests", "CPU and RSS", "actual pruned-schedule MAC pricing"],
            "frontier": frontier}


def history_rows(input_lengths, output_tokens):
    if output_tokens < 1 or not input_lengths:
        raise ValueError("invalid history")
    completed = replayed = resumed = 0
    for width in input_lengths:
        if type(width) is not int or width <= completed:
            raise ValueError("history must extend all previously executed tokens")
        replayed += width + output_tokens - 1
        resumed += width - completed + output_tokens - 1
        completed = width + output_tokens - 1
    return {"replayed_full_rows": replayed, "continued_full_rows": resumed,
            "saved_full_rows": replayed - resumed, "last_executed_position": completed,
            "fractional_row_saving": 1 - resumed / replayed}


def history_screen(config):
    row = history_rows([150, 288, 512], 8)
    kv_per_row = (2 * config["num_hidden_layers"] * config["num_key_value_heads"]
                  * (config["hidden_size"] // config["num_attention_heads"]) * 4)
    return row | {
        "scope": "three-turn exact-extension geometry; no prepared-response measurement",
        "input_lengths": [150, 288, 512], "output_tokens_per_turn": 8,
        "local_final_kv_payload_bytes": kv_per_row * row["last_executed_position"],
        "local_final_logits_payload_bytes": config["vocab_size"] * 4,
        "additional_network_bytes_for_same_client_local_storage": 0,
        "new_saving_when_existing_qualified_RAM_cache_hits": 0,
        "gates": ["exact token/numeric/state provenance and public workload bound",
                  "fresh provider admission and fresh one-use material after restart",
                  "pending sampled token is not executed KV",
                  "charge storage, failed attempts and any remote backup separately",
                  "terminal pruning and complete framed serving bytes need live measurement"],
    }


def publisher_screen(store, config, source_bytes):
    matrix = norm = boundary = scales = boundary_scales = 0
    for key in store.keys:
        shape = store.tensor_shape(key)
        count = math.prod(shape)
        if len(shape) == 2:
            matrix += count
            scales += shape[0] * 4
            if shape == (config["vocab_size"], config["hidden_size"]):
                boundary += count
                boundary_scales += shape[0] * 4
        else:
            norm += count * 4
    if not config["tie_word_embeddings"] or boundary != config["vocab_size"] * config["hidden_size"]:
        raise ValueError("publisher fixture expects one tied public boundary matrix")
    inference = matrix + scales + norm
    preparation = matrix - boundary + scales - boundary_scales
    return {"scope": "raw role-specific W8 weight/scale and float32 local-tensor payload estimate",
            "two_provider_source_safetensors_bytes": 2 * source_bytes,
            "inference_artifact_payload_bytes": inference,
            "preparation_artifact_payload_bytes": preparation,
            "combined_artifact_payload_bytes": inference + preparation,
            "saved_source_payload_bytes": 2 * source_bytes - inference - preparation,
            "fractional_source_payload_saving": 1 - (inference + preparation) / (2 * source_bytes),
            "saving_in_existing_response_counters": 0,
            "gates": ["trusted publisher and source/quantizer/body commitments before import",
                      "tokenizer, manifests, signatures, publication CPU and artifact storage remain extra",
                      "provider import parity and actual cold all-link distribution must be measured"],
            "source_checkpoint_still_retained_by_publisher": True}


def correlation_screen(control):
    removed = control["classes"]["corrections"]
    # Counterexample to treating finite-field PCGs as exact wrap16 correlations.
    left, right, prime, ring = 65535, 1, 65537, 65536
    return {"scope": "optimistic byte ceiling and arithmetic counterexample; no cryptographic implementation",
            "free_correction_maximum_saved_bytes": removed,
            "free_correction_remaining_pair_bytes": control["pair_bytes"] - removed,
            "free_correction_maximum_fractional_saving": removed / control["pair_bytes"],
            "field_ring_counterexample": {"left": left, "right": right, "field": prime,
                                          "ring": ring, "field_sum": (left + right) % prime,
                                          "ring_sum": (left + right) % ring},
            "sources": ["https://eprint.iacr.org/2025/169", "https://eprint.iacr.org/2025/892"],
            "gates": ["arbitrary public matrix correlations, not merely scalar OLE",
                      "exact wrap16/24/32 conversion and masks with reviewed privacy assumptions",
                      "charge both parties' seeds/tables, expansion and setup",
                      "one-use lifetime, independent roles and parameter-specific security review"],
            "paper_scope": "finite-field/circuit-dependent PCGs; published QA-SD parameter caveat",
            "decoder_admitted": False}


def transform(raw, mode, *, inverse=False):
    if mode not in MODES or not 0 < len(raw) <= FRAME_LIMIT:
        raise ValueError("unsupported or overlarge public codec frame")
    if mode == "zlib1":
        return raw
    value = np.frombuffer(raw, dtype=np.uint8)
    if inverse:
        value = np.packbits(np.unpackbits(value).reshape(8, -1).T, axis=1).ravel()
        if mode.startswith("zigzag"):
            integer = value.astype(np.int16)
            value = ((integer >> 1) ^ -(integer & 1)).astype(np.int8).view(np.uint8)
    else:
        if mode.startswith("zigzag"):
            integer = value.view(np.int8).astype(np.int16)
            value = ((integer << 1) ^ (integer >> 7)).astype(np.uint8)
        value = np.packbits(np.unpackbits(value[:, None], axis=1).T.reshape(-1))
    return value.tobytes()


def codec_worker(payload):
    from pllm.runtime.safetensors_store import SafeTensorStore
    from pllm.runtime.quantization import quantize_weight_per_row
    if payload["sources"] != frozen_sources():
        raise ValueError("codec worker source mismatch")
    directory = Path(payload["checkpoint"])
    source = directory / "model.safetensors"
    if digest(source) != payload["checkpoint_sha256"]:
        raise ValueError("checkpoint bytes changed")
    store = SafeTensorStore(directory)
    key = "model.embed_tokens.weight"  # Pinned fixture identity, not runtime dispatch.
    shape = store.tensor_shape(key)
    if shape != (151936, 896) or store.tensor_dtype(key) != "BF16":
        raise ValueError("changed public codec fixture")
    mode = payload["mode"]
    raw_hash, scale_hash = hashlib.sha256(), hashlib.sha256()
    raw_bytes = encoded_bytes = encode_cpu = decode_cpu = 0
    maximum_frame = 0
    for first in range(0, shape[0], 1024):
        weight = store.get_slice(key, (slice(first, min(shape[0], first + 1024)), slice(None)))
        quantized = quantize_weight_per_row(weight, bits=8)
        raw = quantized.values.tobytes()
        raw_hash.update(raw)
        scale_hash.update(quantized.scales.tobytes())
        start = time.process_time()
        body = zlib.compress(transform(raw, mode), level=1)
        frame = struct.pack(">II", len(raw), len(body)) + body
        encode_cpu += time.process_time() - start
        start = time.process_time()
        recovered = transform(zlib.decompress(frame[8:]), mode, inverse=True)
        decode_cpu += time.process_time() - start
        if recovered != raw:
            raise AssertionError("public matrix codec changed weight bits")
        raw_bytes += len(raw)
        encoded_bytes += len(frame)
        maximum_frame = max(maximum_frame, len(frame))
    if payload["sources"] != frozen_sources() or digest(source) != payload["checkpoint_sha256"]:
        raise ValueError("source identity changed during codec work")
    return {"mode": mode, "checked_values": raw_bytes, "framed_bytes": encoded_bytes,
            "weight_sha256": raw_hash.hexdigest(), "scales_sha256": scale_hash.hexdigest(),
            "encode_cpu_seconds": encode_cpu, "decode_cpu_seconds": decode_cpu,
            "maximum_frame_bytes": maximum_frame}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--processes", type=int, default=3)
    parser.add_argument("--output", type=Path, default=ROOT / "target/all-link-frontier.json")
    args = parser.parse_args()
    if args.worker:
        print(json.dumps(codec_worker(json.load(sys.stdin))))
        return
    if not 1 <= args.processes <= 5:
        parser.error("--processes must be 1..5")
    from huggingface_hub import snapshot_download
    from frontier_probe_support import fresh_workers
    from pllm.runtime.safetensors_store import SafeTensorStore
    source_identity = frozen_sources()
    directory = Path(snapshot_download(REPO, revision=REVISION, local_files_only=True,
                                       allow_patterns=["config.json", "model.safetensors"]))
    checkpoint_hash = digest(directory / "model.safetensors")
    config = json.loads((directory / "config.json").read_text())
    control = conserved_control(json.loads(CONTROL.read_text()))
    geometry = json.loads(GEOMETRY.read_text())
    result = {"schema": "pllm.all_link_frontier_screen.v1", "complete": False,
              "model": {"repo": REPO, "revision": REVISION,
                        "checkpoint_sha256": checkpoint_hash,
                        "config_sha256": digest(directory / "config.json")},
              "source_identity": source_identity,
              "control": control,
              "placement": placement_screen(control, geometry),
              "history": history_screen(config),
              "publisher": publisher_screen(SafeTensorStore(directory), config,
                                               (directory / "model.safetensors").stat().st_size),
              "correlations": correlation_screen(control),
              "public_codec": {"scope": "complete public W8 tied matrix; row-aligned <=1 MiB frames",
                               "excluded": ["source loading/quantization from codec CPU",
                                            "hash/parity checks from codec CPU",
                                            "other bundle objects and HTTP framing"],
                               "rows": []}}
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        temporary = args.output.with_suffix(".pending")
        temporary.write_text(json.dumps(result, indent=2) + "\n")
        temporary.replace(args.output)

    save()
    with fresh_workers(__file__, memory_bytes=768 << 20, timeout=300) as (run, guard, _host):
        result["build"] = run.build
        for repeat in range(args.processes):
            order = MODES[repeat % len(MODES):] + MODES[:repeat % len(MODES)]
            for mode in order:
                row = run({"sources": source_identity, "checkpoint": str(directory),
                           "checkpoint_sha256": checkpoint_hash, "mode": mode})
                result["public_codec"]["rows"].append(row | {"repeat": repeat})
                save()
        result["swap_growth_bytes"] = guard.maximum_swap_growth_bytes
    rows = result["public_codec"]["rows"]
    if len({(r["weight_sha256"], r["scales_sha256"], r["checked_values"]) for r in rows}) != 1:
        raise ValueError("codec workers disagree on actual public weights/scales")
    result["public_codec"]["medians"] = {
        mode: {field: statistics.median(r[field] for r in rows if r["mode"] == mode)
               for field in ("framed_bytes", "encode_cpu_seconds", "decode_cpu_seconds")}
        for mode in MODES}
    if source_identity != frozen_sources() or checkpoint_hash != digest(directory / "model.safetensors"):
        raise ValueError("source identity changed across cohort")
    result["complete"] = True
    save()


if __name__ == "__main__":
    main()
