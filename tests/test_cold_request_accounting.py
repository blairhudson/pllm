"""Regression gates for extracting a single cold request from paired evidence."""
import copy
import importlib.util
from fractions import Fraction
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "cold_request_accounting", ROOT / "scripts/probe_cold_request_accounting.py"
)
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


@pytest.fixture(scope="module")
def source():
    return probe.load_source()


def test_actual_cold_window_excludes_repeat_and_counts_startup_once(source):
    row = probe.first_request(source["results"][0])
    assert row["covered_body_bytes"] == 477_104_957
    assert row["native_tcp_stream_bytes"] == 478_654_310
    assert row["public_delivery_body_bytes"] == 124_905_091
    assert row["known_prefill_body_bytes"] == 335_685_613
    assert row["known_decode_body_bytes"] == 16_372_533
    assert row["unattributed_body_bytes"] == 138_710
    assert row["other_known_setup_body_bytes"] == 3_010
    assert row["complete_cold_wire_bytes"] is None
    assert row["first_request_only_client_peak_rss_bytes"] is None


def test_every_historical_first_request_is_accounted_separately(source):
    report = probe.audit(source)
    assert len(report["cases"]) == 30
    assert len(report["case_medians"]) == 10
    for row in report["cases"]:
        assert sum(row["directed_body_bytes"].values()) == row["covered_body_bytes"]
        assert (sum(row["components_body_bytes"].values()) + row["unattributed_body_bytes"]
                == row["covered_body_bytes"])


def test_fractional_replacement_budget_never_rounds_up():
    value = probe.free_removal_budget(103, 31, Fraction(1, 5))
    assert value["maximum_additional_body_bytes"] == 10
    assert 103 - 31 + 10 <= Fraction(4, 5) * 103
    assert 103 - 31 + 11 > Fraction(4, 5) * 103
    assert probe.free_removal_budget(100, 10, Fraction(1, 5))["maximum_additional_body_bytes"] == -10


@pytest.mark.parametrize("total,removed,target", [
    (0, 0, Fraction(1, 5)), (100, 101, Fraction(1, 5)),
    (True, 0, Fraction(1, 5)), (100, -1, Fraction(1, 5)),
    (100, 1, 0.2), (100, 1, Fraction(0)), (100, 1, Fraction(1)),
])
def test_invalid_budget_domains_reject(total, removed, target):
    with pytest.raises(ValueError):
        probe.free_removal_budget(total, removed, target)


@pytest.mark.parametrize("corruption", [
    "repeat_as_cold", "extra_warmup", "allowance_as_outputs", "duplicate_edge",
    "component_undercount", "partition_mismatch", "stream_undercount", "startup_gap",
])
def test_corrupt_or_misaligned_evidence_rejects(source, corruption):
    result = copy.deepcopy(source["results"][0])
    report = result["execution"]["details"]["report"]
    ledger = report["topology_accounting"]["runs"][0]
    if corruption == "repeat_as_cold":
        report["runs"][0]["cold"] = False
    elif corruption == "extra_warmup":
        report["warmup_runs"] = 1
    elif corruption == "allowance_as_outputs":
        report["runs"][0]["tokens"]["output_tokens"] = 7
    elif corruption == "duplicate_edge":
        ledger["body_bytes_by_edge"].append(copy.deepcopy(ledger["body_bytes_by_edge"][0]))
    elif corruption == "component_undercount":
        ledger["traffic_attribution"]["components_body_bytes"]["prefill_material"] -= 1
    elif corruption == "partition_mismatch":
        report["traffic_decomposition"]["cold_first_request"]["known_base_all_link_body_bytes"] += 1
    else:
        samples = report["native_network_accounting"]["samples"]
        if corruption == "stream_undercount":
            samples["runs"][0]["after"]["wan_emulation"]["total_stream_bytes"] -= 1
        else:
            first = samples["runs"][0]["before"]["wan_emulation"]
            first["total_stream_bytes"] += 1
            first["directed_stream_bytes"]["client->inference"] += 1
    with pytest.raises(ValueError):
        probe.first_request(result)


def test_missing_or_duplicate_replicates_reject(source):
    changed = dict(source, results=source["results"][:-1])
    with pytest.raises(ValueError):
        probe.audit(changed)
    changed = dict(source, results=source["results"][:-1] + source["results"][:1])
    with pytest.raises(ValueError):
        probe.audit(changed)


def test_source_changes_cannot_reuse_frozen_provenance(tmp_path):
    path = tmp_path / "wrong.json"
    path.write_text('{"complete": true}')
    with pytest.raises(ValueError, match="digest differs"):
        probe.load_source(path)
