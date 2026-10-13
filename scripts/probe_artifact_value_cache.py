"""Deterministic public metadata eviction screen; no runtime or timing imports.

Fixture manifests are source-locked *summaries*, not authenticated SDK manifests.
Only recorded raw/encoded lengths are reused; all access traces are synthetic.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from functools import lru_cache
import hashlib
import itertools
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SLOTS = 128
SLOT_BYTES = 64
INDEX_BYTES = (SLOTS + 1) * SLOT_BYTES  # One header slot for floor/schema/owner.
MAX_EVENTS = 4096
POLICY_PUBLICATION_BYTES = 512
SCORE_SCALE = 1 << 32


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def digest(value):
    return hashlib.sha256(value).hexdigest()


@dataclass(frozen=True)
class Object:
    key: str
    raw: int
    encoded: int
    kind: str
    source_lock: str


def validate(objects, manifests, trace):
    """Reject private/unbound metadata before any policy sees an access."""
    if not 0 < len(objects) <= SLOTS or len(trace) > MAX_EVENTS:
        raise ValueError("bounded public object/event table required")
    for key, row in objects.items():
        if key != row.key or row.kind not in {"token", "head", "placement"}:
            raise ValueError("public token/head/placement objects only")
        for identity in (key, row.source_lock):
            if len(identity) != 64 or any(c not in "0123456789abcdef" for c in identity):
                raise ValueError("source/object digest required")
        if any(type(n) is not int or not 0 < n <= 4 << 30 for n in (row.raw, row.encoded)):
            raise ValueError("bounded positive raw and encoded byte lengths required")
    for manifest in manifests.values():
        if set(manifest) != {"model", "placement", "source_lock", "objects", "manifest_bytes", "control_bytes"}:
            raise ValueError("public manifest summary fields differ")
        if not manifest["objects"] or len(set(manifest["objects"])) != len(manifest["objects"]):
            raise ValueError("unique ordered first-use objects required")
        if any(k not in objects or objects[k].source_lock != manifest["source_lock"]
               for k in manifest["objects"]):
            raise ValueError("manifest source lock/object mismatch")
        if any(type(manifest[k]) is not int or manifest[k] < 0
               for k in ("manifest_bytes", "control_bytes")):
            raise ValueError("nonnegative per-import traffic required")
    for event in trace:
        if set(event) - {"request", "worker", "manifest", "faults"} or not {"request", "worker", "manifest"} <= set(event):
            raise ValueError("explicit public request/worker/manifest required")
        if event["manifest"] not in manifests:
            raise ValueError("unknown public manifest")
        for key, fault in event.get("faults", {}).items():
            if key not in objects or fault not in {"corrupt", "missing"}:
                raise ValueError("invalid public read fault")


def replay(objects, manifests, trace, capacity, policy="lru", frequency_cap=4):
    """One shared owner, serialized fetches (existing single-flight linearization).

    LRU mimics mandatory admission and oversize bypass. LFU and value may bypass
    incoming objects. Value = aging floor + capped past frequency * encoded/raw.
    Q32 integer comparisons avoid platform-dependent ties. No future accesses used.
    """
    validate(objects, manifests, trace)
    if policy not in {"lru", "lfu", "value"} or type(capacity) is not int or capacity < 0:
        raise ValueError("invalid policy/capacity")
    if frequency_cap not in {1, 4, 16}:
        raise ValueError("frequency cap not frozen candidate")
    reserve = INDEX_BYTES if policy != "lru" else 0
    if capacity < reserve:
        raise ValueError("capacity below fixed policy index reservation")
    budget = max(0, capacity - reserve)
    resident, history = {}, {}  # All identities public; at most SLOTS each.
    object_misses = {key: 0 for key in objects}
    object_hits = {key: 0 for key in objects}
    floor = 0
    result = dict(object_bytes=0, manifest_bytes=0, control_bytes=0, hits=0, misses=0,
                  corruptions=0, missing_reads=0, evictions=0, bypasses=0,
                  local_read_bytes=0, cache_write_bytes=0, index_write_bytes=0,
                  peak_payload_bytes=0, peak_index_entries=0)
    for event in trace:
        manifest = manifests[event["manifest"]]
        result["manifest_bytes"] += manifest["manifest_bytes"]
        result["control_bytes"] += manifest["control_bytes"]
        for key, fault in event.get("faults", {}).items():
            if key in resident:
                if fault == "corrupt":
                    result["corruptions"] += 1
                    result["local_read_bytes"] += objects[key].raw
                else:
                    result["missing_reads"] += 1
                del resident[key]
        for key in manifest["objects"]:
            row = objects[key]
            tick = result["hits"] + result["misses"] + 1
            count = min(frequency_cap, history.get(key, 0) + 1)
            if policy != "lru":
                history[key] = count
                result["index_write_bytes"] += SLOT_BYTES
                result["peak_index_entries"] = max(result["peak_index_entries"], len(history))
            score = floor + count * row.encoded * SCORE_SCALE // row.raw
            if key in resident:
                result["hits"] += 1
                object_hits[key] += 1
                result["local_read_bytes"] += row.raw  # Authenticated full read, not free.
                resident[key] = (tick, count, score)
                continue
            result["misses"] += 1
            object_misses[key] += 1
            result["object_bytes"] += row.encoded
            if row.raw > budget:
                result["bypasses"] += 1
                continue
            resident[key] = (tick, count, score)
            while sum(objects[k].raw for k in resident) > budget:
                if policy == "lru":
                    victim = min((k for k in resident if k != key), key=lambda k: (resident[k][0], k))
                elif policy == "lfu":
                    victim = min(resident, key=lambda k: (resident[k][1], resident[k][0], k))
                else:
                    victim = min(resident, key=lambda k: (resident[k][2], resident[k][0], k))
                    floor = resident[victim][2]
                del resident[victim]
                result["bypasses" if victim == key else "evictions"] += 1
            if key in resident:
                result["cache_write_bytes"] += row.raw
            result["peak_payload_bytes"] = max(result["peak_payload_bytes"], sum(objects[k].raw for k in resident))
    publication = POLICY_PUBLICATION_BYTES if policy != "lru" else 0
    result.update(policy=policy, capacity_bytes=capacity, payload_budget_bytes=budget,
                  index_reserved_bytes=min(capacity, reserve), publication_bytes=publication,
                  final_payload_bytes=sum(objects[k].raw for k in resident),
                  retained_objects=sorted(resident), imports=len(trace),
                  misses_by_object=object_misses, hits_by_object=object_hits)
    result["covered_network_bytes"] = (result["object_bytes"] + result["manifest_bytes"]
                                       + result["control_bytes"] + publication)
    return result


def tiny_oracle(objects, accesses, capacity):
    """Exact encoded-transfer lower bound, optional admission, <=8 objects/24 accesses.

    Future knowledge here ONLY, never calibration or online score. On miss, retain
    any subset of old cache plus downloaded object; hits may retain current cache.
    """
    if len(objects) > 8 or len(accesses) > 24:
        raise ValueError("tiny oracle bound exceeded")
    if capacity < 0 or any(k not in objects for k in accesses):
        raise ValueError("invalid oracle input")

    @lru_cache(None)
    def solve(i, cache):
        if i == len(accesses):
            return 0
        key = accesses[i]
        if key in cache:
            return solve(i + 1, cache)
        available = sorted(set(cache) | {key})
        choices = []
        for n in range(len(available) + 1):
            for subset in itertools.combinations(available, n):
                if sum(objects[k].raw for k in subset) <= capacity:
                    choices.append(solve(i + 1, subset))
        return objects[key].encoded + min(choices)
    return solve(0, ())


def make_manifest(model, placement, lock, keys, manifest_bytes=2048, control_bytes=256):
    return dict(model=model, placement=placement, source_lock=lock, objects=keys,
                manifest_bytes=manifest_bytes, control_bytes=control_bytes)


def trace(names, workers=1):
    return [dict(request=f"r{i:04d}", worker=f"w{i % workers}", manifest=name)
            for i, name in enumerate(names)]


def fixtures():
    lock = digest(b"synthetic-public-source-lock-v1")
    objects = {}
    for name, raw, encoded, kind in [("token", 2 << 20, 1800000, "token"),
                                     ("head", 4 << 20, 3900000, "head")]:
        key = digest(name.encode())
        objects[key] = Object(key, raw, encoded, kind, lock)
    keys = list(objects)
    manifests = {}
    for i in range(8):
        key = digest(f"placement-{i}".encode())
        objects[key] = Object(key, 1 << 20, 80000, "placement", lock)
        manifests[f"p{i}"] = make_manifest("synthetic-model-A", f"body-{i}", lock, keys + [key])
        manifests[f"scan{i}"] = make_manifest("synthetic-model-A", f"body-{i}", lock, [key])
    manifests["shared"] = make_manifest("synthetic-model-A", "remote-body", lock, keys)
    # Training and held-out fixed independently; no cache state crosses split.
    training = trace(["p0", "p1", "p2", "p3"] * 4)
    heldout = {
        "switching": trace(["p4", "p6", "p5", "p7", "p5", "p4"] * 5),
        "scan_churn": trace(["shared"] + [f"scan{i}" for i in range(8)] + ["shared"] * 3),
        "first_cold": trace(["p7"]),
        "exact_repeat": trace(["p7"] * 12),
        "phase_change": trace(["shared"] * 16 + ["scan0", "scan1"] * 16),
        "shared_workers": trace(["p4"] * 8, workers=4),
    }
    lock_b = digest(b"synthetic-public-source-lock-model-B-v1")
    model_b = []
    for name, raw, encoded, kind in [("B-token", 2 << 20, 1500000, "token"),
                                     ("B-head", 4 << 20, 3600000, "head"),
                                     ("B-placement", 1 << 20, 60000, "placement")]:
        key = digest(name.encode())
        objects[key] = Object(key, raw, encoded, kind, lock_b)
        model_b.append(key)
    manifests["model_b"] = make_manifest("synthetic-model-B", "body-0", lock_b, model_b)
    heldout["model_switching"] = trace(["p4", "model_b", "p5", "p4", "model_b"] * 4)
    corrupt = trace(["p4"] * 4)
    corrupt[1]["faults"] = {keys[0]: "corrupt"}
    corrupt[2]["faults"] = {keys[1]: "missing"}
    heldout["repair"] = corrupt
    return objects, manifests, training, heldout


def recorded_fixture():
    evidence = json.loads((ROOT / "docs/evidence/research-artifact-reuse-qwen3.json").read_text())
    lock = evidence["source"]["source_lock_digest"]
    objects = {}
    for row in evidence["objects"]:
        key = row["raw_sha256"]  # Content identity here; not runtime domain identity.
        objects[key] = Object(key, row["raw_bytes"], row["encoded_bytes"],
                              "token" if row["index"] == 1 else "head", lock)
    keys = list(objects)
    manifests = {
        "both": make_manifest("Qwen3-4B-recorded-slice", "synthetic-both", lock, keys),
        "head": make_manifest("Qwen3-4B-recorded-slice", "synthetic-head-slice", lock, keys[:1]),
        "token": make_manifest("Qwen3-4B-recorded-slice", "synthetic-token-only", lock, keys[1:]),
    }
    return objects, manifests, trace(["both", "head", "head", "token"] * 6)


def snapshot(objects, manifests, traces):
    return dict(objects={k: asdict(v) for k, v in objects.items()}, manifests=manifests, traces=traces)


def calibrate(objects, manifests, training, capacities):
    scores = {cap: sum(replay(objects, manifests, training, size, "value", cap)["covered_network_bytes"]
                       for size in capacities) for cap in (1, 4, 16)}
    return min(scores, key=lambda cap: (scores[cap], cap)), scores


def report():
    objects, manifests, training, heldout = fixtures()
    capacities = [6 << 20, 7 << 20, 8 << 20, 10 << 20, 16 << 20]
    frozen, calibration = calibrate(objects, manifests, training, capacities)
    rows = []
    for name, events in heldout.items():
        for size in capacities:
            results = {p: replay(objects, manifests, events, size, p, frozen) for p in ("lru", "lfu", "value")}
            delta = results["lru"]["covered_network_bytes"] - results["value"]["covered_network_bytes"]
            accesses = results["lru"]["hits"] + results["lru"]["misses"]
            rows.append(dict(trace=name, capacity_bytes=size, results=results, net_saved_bytes=delta,
                             weighted_miss_bytes_avoided=results["lru"]["object_bytes"] - results["value"]["object_bytes"],
                             accesses=accesses, break_even_publication_bytes=512))
    real_objects, real_manifests, real_trace = recorded_fixture()
    real_rows = [{"capacity_bytes": size, "results": {
        p: replay(real_objects, real_manifests, real_trace, size, p, frozen)
        for p in ("lru", "lfu", "value")}}
        for size in [12 << 20, 19 << 20, 20 << 20, 24 << 20]]
    # Exhaustive held-out adversary, not another tuning input.
    tiny = {digest(name.encode()): Object(digest(name.encode()), 10000, wire, "placement", digest(b"tiny"))
            for name, wire in [("a", 9000), ("b", 1000), ("c", 1000)]}
    tiny_manifests = {k: make_manifest("tiny", k, row.source_lock, [k], 0, 0) for k, row in tiny.items()}
    worst = None
    for names in itertools.product(tiny, repeat=6):
        events = trace(names)
        lru = replay(tiny, tiny_manifests, events, 20000, "lru", frozen)
        value = replay(tiny, tiny_manifests, events, 20000, "value", frozen)
        regression = value["covered_network_bytes"] - lru["covered_network_bytes"]
        if worst is None or regression > worst["regression_bytes"]:
            worst = dict(trace=list(names), regression_bytes=regression, lru=lru, value=value,
                         oracle_object_bytes=tiny_oracle(tiny, names, 20000))
    sources = ["python/pllm/runtime/bundle_artifacts.py", "python/pllm/runtime/bundle_compression.py",
               "python/pllm/runtime/bundle_storage.py", "docs/research/native-snapshot-reference.md",
               "docs/plans/low-tier-all-link-hypotheses.md", "docs/evidence/research-artifact-reuse-qwen3.json",
               "scripts/probe_artifact_value_cache.py", "tests/test_artifact_value_cache_probe.py",
               "docs/research/artifact-value-cache-reference.md"]
    return dict(schema="pllm.artifact_value_cache_probe.v1", scope="pure metadata synthetic replay",
                qualification=dict(whole_client_low="unknown", numeric_execution="not run", sdk="not run"),
                frozen_frequency_cap=frozen, calibration_scores=calibration,
                calibration_fixture=snapshot(objects, manifests, {"training": training}),
                heldout_fixture=snapshot(objects, manifests, heldout), synthetic_rows=rows,
                recorded_size_fixture=snapshot(real_objects, real_manifests, {"synthetic": real_trace}),
                recorded_size_rows=real_rows, exhaustive_tiny_fixture=snapshot(tiny, tiny_manifests, {}),
                exhaustive_tiny_worst=worst,
                accounting=dict(index_bytes=INDEX_BYTES, index_slots=SLOTS, slot_bytes=SLOT_BYTES,
                                policy_publication_bytes=POLICY_PUBLICATION_BYTES,
                                publication_scope="once per replay on Preparation-to-Client; modeled allowance",
                                per_import_manifest_control="explicit synthetic lengths; charged every import",
                                physical_wire="unknown", other_links="unchanged assumed; not measured"),
                source_sha256={p: digest((ROOT / p).read_bytes()) for p in sources})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify", type=Path, help="exact deterministic evidence replay")
    args = parser.parse_args()
    evidence = report()
    if args.verify:
        if json.loads(args.verify.read_text()) != json.loads(json.dumps(evidence)):
            raise SystemExit("evidence differs from deterministic replay/source hashes")
        print("exact evidence replay passed")
    elif args.output:
        args.output.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
    else:
        print(json.dumps(evidence, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
