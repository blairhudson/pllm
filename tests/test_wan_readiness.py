import copy
import dataclasses

import pytest

from pllm.deployment import PartyAccess, WanConditions
from pllm.metrics import wan_readiness


def report(edges, *, tokens=10):
    ledger = {"tracked_body_counter_set_present": True, "body_bytes_by_edge": edges,
              "all_link_serialized_body_bytes": sum(e["serialized_body_bytes"] for e in edges),
              "online_all_link_serialized_body_bytes": sum(e["serialized_body_bytes"] for e in edges if e["phase"] == "online")}
    empty = {**ledger, "body_bytes_by_edge": [], "all_link_serialized_body_bytes": 0,
             "online_all_link_serialized_body_bytes": 0}
    return {"schema_version": "pllm.loopback_benchmark.v1", "warmup_runs": [],
            "runs": [{"status": "completed", "tokens": {"authoritative": True, "output_tokens": tokens}}],
            "topology_accounting": {"startup": empty, "warmups": [], "runs": [ledger]}}


def edge(a, b, size, phase="online"):
    return dict(source=a, destination=b, serialized_body_bytes=size, phase=phase)


def test_units_and_concurrent_access_bottleneck():
    result = wan_readiness(report([edge("client", "a", 5_000_000), edge("client", "b", 5_000_000)]))
    window = result["summary"]["online"]
    assert window["minimum_transfer_seconds"] == 2  # Shared 40 Mbps, not two independent 40 Mbps ports.
    assert window["parties"]["a"]["download_seconds"] == 0.4
    assert window["bottleneck"] == {"party_id": "client", "direction": "upload"}
    assert result["summary"]["online_generated_tokens_per_second_ceiling"] == 5
    assert result["measured_wan_latency_seconds"] is None
    assert result["internet_ready"] is None


def test_peer_upload_cold_floor_and_heterogeneous_rates():
    data = report([edge("client", "inference", 1_000_000), edge("preparation", "inference", 20_000_000, "offline")])
    data["topology_accounting"]["startup"] = report([edge("inference", "client", 100_000_000, "cold")])["topology_accounting"]["runs"][0]
    result = wan_readiness(data, WanConditions(parties=(PartyAccess("preparation", upload_mbps=8),)))
    assert result["summary"]["online"]["minimum_transfer_seconds"] == 0.2
    assert result["summary"]["setup_inclusive"]["minimum_transfer_seconds"] == 20
    assert result["summary"]["setup_inclusive"]["wan_body_bytes"] == 121_000_000


def test_grouped_roles_share_capacity_and_local_edges_are_not_wan():
    data = report([edge("a", "b", 10_000_000), edge("a", "client", 5_000_000), edge("b", "client", 5_000_000)])
    conditions = WanConditions(role_parties=(("a", "host"), ("b", "host")))
    result = wan_readiness(data, conditions)["summary"]["online"]
    assert result["wan_body_bytes"] == result["intra_party_body_bytes"] == 10_000_000
    assert result["minimum_transfer_seconds"] == 2
    assert result["parties"]["host"]["upload_bytes"] == 10_000_000
    assert WanConditions.from_spec(conditions.to_spec()) == conditions


def test_unknown_counters_never_imply_zero_capacity_cost():
    data = report([edge("client", "a", 5)])
    data["topology_accounting"]["startup"] = None
    assert wan_readiness(data)["summary"]["setup_inclusive"] is None
    data["topology_accounting"]["runs"][0]["all_link_serialized_body_bytes"] += 1
    assert wan_readiness(data)["runs"][0]["online"] is None
    data = report([])
    assert wan_readiness(data)["summary"]["online"]["minimum_transfer_seconds"] == 0
    assert wan_readiness(data)["summary"]["online_generated_tokens_per_second_ceiling"] is None


@pytest.mark.parametrize("bad", [0, -1, True, float("nan"), float("inf"), 1_000_001])
def test_invalid_rates_fail_closed(bad):
    with pytest.raises(ValueError):
        WanConditions(upload_mbps=bad)


def test_strict_immutable_scenario_and_single_flow_shaper_conversion():
    value = WanConditions(parties=(PartyAccess("client", 20, 8),))
    assert value.link_conditions("client", "inference").bytes_per_second == 1_000_000
    assert WanConditions.from_spec(value.to_spec()).digest == value.digest
    with pytest.raises(dataclasses.FrozenInstanceError):
        value.upload_mbps = 50
    spec = copy.deepcopy(value.to_spec())
    spec["parties"].append(spec["parties"][0])
    with pytest.raises(ValueError):
        WanConditions.from_spec(spec)
    assert WanConditions(100, 40).digest == WanConditions(100.0, 40.0).digest


def test_network_execution_maps_roles_to_parties_and_refuses_conflicting_scenarios():
    data = report([edge("worker_a", "client", 5_000_000)])
    data["runs"][0]["network_audit"] = {"placement": [{"role_id": "worker_a", "party_id": "home"}]}
    conditions = WanConditions(parties=(PartyAccess("home", upload_mbps=8),))
    window = wan_readiness(data, conditions)["summary"]["online"]
    assert window["minimum_transfer_seconds"] == 5
    assert window["bottleneck"] == {"party_id": "home", "direction": "upload"}
    with pytest.raises(ValueError, match="placement"):
        wan_readiness(data, WanConditions(role_parties=(("worker_a", "different"),)))


def test_warmups_are_charged_once_and_do_not_inflate_output_denominator():
    data = report([edge("client", "a", 5_000_000)], tokens=10)
    data["warmup_runs"] = copy.deepcopy(data["runs"])
    data["topology_accounting"]["warmups"] = copy.deepcopy(data["topology_accounting"]["runs"])
    result = wan_readiness(data)["summary"]
    assert result["online"]["minimum_transfer_seconds"] == 1
    assert result["setup_inclusive"]["minimum_transfer_seconds"] == 2
    assert result["setup_inclusive_generated_tokens_per_second_ceiling"] == 5


def test_offset_setup_bundle_and_teardown_are_charged_once():
    from pllm.runtime.topology_accounting import two_worker_body_accounting
    privacy = {f"role_link.{role}.{phase}_{direction}_bytes": 10
               for role in ("worker_a", "worker_b") for phase in ("setup", "online", "teardown")
               for direction in ("upload", "download")}
    privacy["bundle_network_bytes"] = 100
    ledger = two_worker_body_accounting({"privacy": privacy})
    data = report([])
    data["topology_accounting"]["runs"] = [ledger]
    summary = wan_readiness(data)["summary"]
    assert summary["online"]["wan_body_bytes"] == 40
    assert summary["setup_inclusive"]["wan_body_bytes"] == 220
