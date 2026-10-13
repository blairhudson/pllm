import copy
from dataclasses import replace
import hashlib

import pytest

from scripts.probe_prefill_cancellation_cuts import (
    LOW, Ledger, Node, Prices, admission, execute, export_state, graph,
    oracle, report, restore_state, simulate, validate,
)


def test_all_frontiers_exact_bits_and_bounded_liveness():
    checks = oracle()
    assert len(checks) == 9
    assert all(c["exact_toy_final_bits"] for c in checks)
    assert max(c["payload_bytes"] for c in checks) == 64


@pytest.mark.parametrize("cuts", [(), (0, 10), (5,), (5, 5, 10), (10, 5), (True, 10)])
def test_bad_cuts(cuts):
    with pytest.raises(ValueError):
        validate(graph(), cuts)


def test_dependency_order_is_not_rearranged():
    with pytest.raises(ValueError):
        validate((Node("a", ("future",), "remote"),), (1,))


def state():
    nodes = graph()
    values = {"input": (0.125, 0.25, 0.375, 0.5)}
    execute(nodes, values, 0, 5)
    return nodes, export_state(nodes, values, 5, generation=2)


@pytest.mark.parametrize("field,value", [
    ("pc", 4), ("generation", 1), ("graph_digest", "bad"),
    ("numeric", "changed"), ("schema", "bad"),
])
def test_continuation_binding_rejects_replay_and_mismatch(field, value):
    nodes, saved = state()
    saved[field] = value
    with pytest.raises(ValueError):
        restore_state(nodes, saved, 5, 4, 2)


@pytest.mark.parametrize("mutation", ["missing", "extra", "dtype", "shape", "payload", "corrupt"])
def test_complete_typed_live_state_required(mutation):
    nodes, saved = state()
    tensors = saved["tensors"]
    key = next(iter(tensors))
    if mutation == "missing":
        del tensors[key]
    elif mutation == "extra":
        tensors["junk"] = copy.deepcopy(tensors[key])
    elif mutation == "payload":
        tensors[key]["hex"] = "00"
    elif mutation == "corrupt":
        tensors[key]["hex"] = "00000000" * 4
    else:
        tensors[key][mutation] = "bad"
    with pytest.raises(ValueError):
        restore_state(nodes, saved, 5, 4, 2)


def test_full_kv_alone_is_insufficient_counterexample():
    nodes, saved = state()
    alternate = copy.deepcopy(saved)
    alternate["tensors"]["0.out"]["hex"] = "00000000" * 4
    alternate["tensors"]["0.out"]["sha256"] = hashlib.sha256(bytes(16)).hexdigest()
    assert alternate["tensors"]["0.kv"] == saved["tensors"]["0.kv"]
    a = restore_state(nodes, saved, 5, 4, 2)
    b = restore_state(nodes, alternate, 5, 4, 2)
    execute(nodes, a, 5, 10)
    execute(nodes, b, 5, 10)
    assert a["1.out"] != b["1.out"]


@pytest.mark.parametrize("rows", [0, 17, True])
def test_import_bounded_before_payload_decode(rows):
    nodes, saved = state()
    with pytest.raises(ValueError):
        restore_state(nodes, saved, 5, rows, 2)


@pytest.mark.parametrize("point,stop", [
    ("cut", 5), ("cut", 10), ("preparing", 5), ("preparing", 10),
    ("sealed", 5), ("sealed", 10), ("online_inflight", 1),
    ("online_inflight", 4), ("online_inflight", 6), ("acknowledged", 5),
])
def test_cancellation_conservation_and_body_sums(point, stop):
    result = simulate((5, 10), point=point, stop=stop)
    ledger = result["rows"]
    assert ledger["issued"] == ledger["reserved"] + ledger["discarded"]
    assert ledger["reserved"] == ledger["claimed"] + ledger["burned_unused"]
    assert ledger["burn_total_reserved"] == ledger["reserved"]
    assert not ledger["active"]
    assert result["total_body_upper_bound"] == sum(result["directed_body_upper_bounds"].values())


@pytest.mark.parametrize("reason", ["cancel", "malformed", "replay", "expiry"])
def test_reserved_failure_burns_even_inflight_claim(reason):
    ledger = Ledger()
    ledger.prepare(12)
    ledger.seal()
    ledger.claim(4, ack=False)
    assert ledger.online_inflight == 4
    ledger.retire(authenticated=reason != "malformed")
    assert ledger.audit()["burn_total_reserved"] == 12
    with pytest.raises(ValueError):
        ledger.claim(4)


def test_no_preparation_online_or_without_confirmed_retirement():
    ledger = Ledger()
    ledger.prepare(12)
    ledger.seal()
    with pytest.raises(ValueError):
        ledger.prepare(1)
    ledger.retire(authenticated=False)
    with pytest.raises(ValueError):
        ledger.prepare(1)


def test_failed_unsealed_issuance_is_discarded_not_saved():
    ledger = Ledger()
    ledger.prepare(12)
    ledger.retire()
    assert ledger.audit()["discarded"] == 12
    result = simulate((5, 10), stop=5, point="preparing")
    assert result["directed_body_upper_bounds"]["preparation_to_inference"] == 12 * 96
    observed = result["inflight_observations"][0]
    assert observed["rows"]["correction_inflight"] == 12
    assert observed["bytes"]["correction_inflight_upper"] == 12 * 96


def test_success_and_cancellation_exact_break_even():
    r = report()
    assert r["successful_whole_prefill_fixed_epoch"]["rows"]["issued"] == 36
    assert r["successful_staged_fixed_epoch"]["rows"]["issued"] == 36
    assert r["break_even"]["successful_body_penalty"] == 1536
    assert r["break_even"]["canceled_body_saving"] == 1792
    assert r["break_even"]["cancellation_fraction_strictly_greater_than"] == "1536/3328"
    assert r["successful_staged_fixed_epoch"]["pauses"] == 1
    assert r["successful_staged_fixed_epoch"]["pause_seconds"] is None


def test_zero_correction_prices_cannot_create_savings():
    prices = replace(Prices(), correction=0)
    baseline = simulate((10,), stop=5, point="acknowledged", prices=prices)
    candidate = simulate((5, 10), stop=5, prices=prices)
    assert candidate["total_body_upper_bound"] > baseline["total_body_upper_bound"]


@pytest.mark.parametrize("base,expected", [(LOW - 32, True), (LOW - 31, False)])
def test_source_bound_memory_copy_admission(base, expected):
    r = admission(source_digest="a" * 64, state_manifest={"residual": 16},
                  base_bytes=base, scratch_bytes=0, correction_bytes=0)
    assert r["owned_bound_fits"] is expected
    assert not r["whole_client_low_qualified"]


def test_bad_source_and_negative_owner():
    with pytest.raises(ValueError):
        admission(source_digest="unknown", state_manifest={}, base_bytes=0,
                  scratch_bytes=0, correction_bytes=0)
    with pytest.raises(ValueError):
        admission(source_digest="a" * 64, state_manifest={"bad": -1}, base_bytes=0,
                  scratch_bytes=0, correction_bytes=0)
