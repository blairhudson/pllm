"""Independent byte accounting and causal replay regressions; no SDK imports."""
from collections import OrderedDict
from dataclasses import replace
import itertools

import pytest

from scripts import probe_artifact_value_cache as probe


def independent_lru(objects, manifests, events, capacity):
    cache = OrderedDict()
    wire = reads = writes = 0
    for event in events:
        for key, fault in event.get("faults", {}).items():
            if key in cache:
                if fault == "corrupt":
                    reads += objects[key].raw
                del cache[key]
        for key in manifests[event["manifest"]]["objects"]:
            row = objects[key]
            if key in cache:
                reads += row.raw
                cache.move_to_end(key)
            else:
                wire += row.encoded
                if row.raw <= capacity:
                    while sum(cache.values()) + row.raw > capacity:
                        cache.popitem(last=False)
                    cache[key] = row.raw
                    writes += row.raw
    return wire, reads, writes, sorted(cache)


@pytest.mark.parametrize("capacity", [0, 6 << 20, 7 << 20, 8 << 20, 16 << 20])
def test_lru_independent_all_traces(capacity):
    objects, manifests, _, traces = probe.fixtures()
    for events in traces.values():
        expected = independent_lru(objects, manifests, events, capacity)
        actual = probe.replay(objects, manifests, events, capacity)
        assert expected == (actual["object_bytes"], actual["local_read_bytes"],
                            actual["cache_write_bytes"], actual["retained_objects"])


@pytest.mark.parametrize("policy", ["lru", "lfu", "value"])
def test_capacity_network_control_and_faults(policy):
    objects, manifests, _, traces = probe.fixtures()
    for events in traces.values():
        result = probe.replay(objects, manifests, events, 8 << 20, policy)
        assert result["peak_payload_bytes"] + result["index_reserved_bytes"] <= 8 << 20
        assert result["peak_index_entries"] <= probe.SLOTS
        assert result["manifest_bytes"] == 2048 * len(events)
        assert result["control_bytes"] == 256 * len(events)
        assert result["covered_network_bytes"] == sum(result[k] for k in (
            "object_bytes", "manifest_bytes", "control_bytes", "publication_bytes"))
    result = probe.replay(objects, manifests, traces["repair"], 16 << 20, policy)
    assert result["corruptions"] == result["missing_reads"] == 1
    assert result["misses"] == 5


def test_source_identity_and_private_metadata_rejected():
    objects, manifests, training, _ = probe.fixtures()
    key = next(iter(objects))
    altered = dict(objects)
    altered[key] = replace(objects[key], source_lock="0" * 64)
    with pytest.raises(ValueError, match="source lock"):
        probe.replay(altered, manifests, training, 8 << 20)
    altered[key] = replace(objects[key], kind="private-kv")
    with pytest.raises(ValueError, match="public token"):
        probe.replay(altered, manifests, training, 8 << 20)
    with pytest.raises(ValueError, match="explicit public"):
        probe.replay(objects, manifests, [{**training[0], "prompt": "secret"}], 8 << 20)


def test_cold_and_ample_cache_cannot_save_bytes():
    objects, manifests, _, traces = probe.fixtures()
    for name in ("first_cold", "exact_repeat"):
        lru = probe.replay(objects, manifests, traces[name], 16 << 20)
        value = probe.replay(objects, manifests, traces[name], 16 << 20, "value")
        assert value["object_bytes"] == lru["object_bytes"]
        assert value["covered_network_bytes"] == lru["covered_network_bytes"] + 512


def test_encoded_not_raw_and_index_capacity_boundary():
    key, lock = probe.digest(b"one"), probe.digest(b"lock")
    objects = {key: probe.Object(key, 10000, 100, "head", lock)}
    manifests = {"m": probe.make_manifest("model", "body", lock, [key], 7, 9)}
    events = probe.trace(["m", "m"])
    lru = probe.replay(objects, manifests, events, 10000)
    value = probe.replay(objects, manifests, events, 10000, "value")
    assert (lru["object_bytes"], value["object_bytes"]) == (100, 200)
    assert value["cache_write_bytes"] == 0
    with pytest.raises(ValueError, match="index reservation"):
        probe.replay(objects, manifests, events, 0, "value")


def test_shared_workers_charge_all_manifests_single_object_fetch():
    objects, manifests, _, traces = probe.fixtures()
    result = probe.replay(objects, manifests, traces["shared_workers"], 16 << 20, "value")
    assert result["misses"] == 3
    assert result["hits"] == 21
    assert result["manifest_bytes"] == 8 * 2048


def test_prefix_causality_and_frozen_training():
    objects, manifests, training, traces = probe.fixtures()
    frozen, scores = probe.calibrate(objects, manifests, training, [7 << 20, 8 << 20])
    assert frozen == min(scores, key=lambda cap: (scores[cap], cap))
    events = traces["switching"]
    before = probe.replay(objects, manifests, events[:5], 8 << 20, "value", frozen)
    # Appending any future never mutates prefix state or training recipe.
    probe.replay(objects, manifests, events + traces["model_switching"], 8 << 20, "value", frozen)
    assert before == probe.replay(objects, manifests, events[:5], 8 << 20, "value", frozen)
    assert (frozen, scores) == probe.calibrate(objects, manifests, training, [7 << 20, 8 << 20])


def test_tiny_oracle_independent_two_object_closed_form():
    lock = probe.digest(b"tiny")
    a, b = probe.digest(b"a"), probe.digest(b"b")
    objects = {k: probe.Object(k, 1, wire, "placement", lock) for k, wire in [(a, 7), (b, 3)]}
    # One-slot cache, A B A: retaining A and bypassing B costs 10, not LRU's 17.
    assert probe.tiny_oracle(objects, [a, b, a], 1) == 10
    for names in itertools.product([a, b], repeat=5):
        assert probe.tiny_oracle(objects, names, 0) == sum(objects[k].encoded for k in names)
        assert probe.tiny_oracle(objects, names, 2) == sum(objects[k].encoded for k in set(names))
    with pytest.raises(ValueError, match="bound"):
        probe.tiny_oracle(objects, [a] * 25, 1)


def test_recorded_sizes_remain_recorded_not_synthetic_weights():
    objects, manifests, events = probe.recorded_fixture()
    assert sorted((row.raw, row.encoded) for row in objects.values()) == [
        (8386560, 7619553), (11422654, 2534486)]
    result = probe.replay(objects, manifests, events, 24 << 20)
    assert result["object_bytes"] == 10154039
    assert all(m["model"] == "Qwen3-4B-recorded-slice" for m in manifests.values())


def test_independent_value_density_eviction_arithmetic():
    lock = probe.digest(b"manual")
    keys = [probe.digest(name.encode()) for name in "abc"]
    objects = {key: probe.Object(key, 10000, wire, "placement", lock)
               for key, wire in zip(keys, [9000, 1000, 1000], strict=True)}
    manifests = {key: probe.make_manifest("manual", key, lock, [key], 0, 0) for key in keys}
    a, b, c = keys
    result = probe.replay(objects, manifests, probe.trace([a, b, c, b, a]),
                          20000 + probe.INDEX_BYTES, "value", 1)
    # A score .9 protected; C evicts B at .1, returning B at .2 evicts C.
    assert result["object_bytes"] == 12000
    assert result["evictions"] == 2
    assert result["hits"] == 1
    assert result["misses_by_object"] == {a: 1, b: 2, c: 1}


def test_oracle_independent_forward_state_enumeration():
    lock = probe.digest(b"forward")
    keys = [probe.digest(name.encode()) for name in "abc"]
    objects = {key: probe.Object(key, raw, wire, "placement", lock)
               for key, raw, wire in zip(keys, [1, 2, 3], [9, 2, 4], strict=True)}
    for names in itertools.product(keys, repeat=4):
        costs = {frozenset(): 0}
        for key in names:
            following = {}
            for cache, old_cost in costs.items():
                cost = old_cost + (0 if key in cache else objects[key].encoded)
                available = cache | {key}
                for flags in itertools.product([False, True], repeat=len(available)):
                    retained = frozenset(k for k, keep in zip(sorted(available), flags, strict=True) if keep)
                    if sum(objects[k].raw for k in retained) <= 3:
                        following[retained] = min(following.get(retained, float("inf")), cost)
            costs = following
        assert probe.tiny_oracle(objects, names, 3) == min(costs.values())


def test_exact_evidence_replay():
    import json

    expected = json.loads((probe.ROOT / "docs/evidence/research-artifact-value-cache.json").read_text())
    assert json.loads(json.dumps(probe.report())) == expected
