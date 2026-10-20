"""Ordinary FSS reports consume serialized worker deltas, not inferred intervals."""
from contextlib import closing
from copy import deepcopy
import json

import pytest

from pllm.runtime.benchmark_cli import build_loopback_report
from pllm.runtime.benchmark_delegated import FSS_PHASE_COUNTERS
from pllm.runtime.benchmark_history import BenchmarkHistory, BenchmarkRun
from pllm.runtime.fss_protocol import HOLDERS, ROLES
from test_fss_benchmark import counters


NS = 10**9


def phase(**values):
    return dict.fromkeys(FSS_PHASE_COUNTERS, 0) | values


def saved_run(name="request", *, preparation=17, holders=(11, 13), measured=True):
    privacy = counters() | {"plaintext_prompt_bytes_sent": 0, "plaintext_token_ids_sent": 0,
                            "preparation_requests_during_online": 0}
    processes = {role: {"cpu_seconds": 1, "rss_peak_bytes": 128} for role in ("client", *ROLES)}
    if measured:
        processes["dealer"]["phase_timing"] = phase(preparation_calls=1, preparation_ns=preparation * NS,
            delivery_callback_ns=4 * NS, seal_callback_ns=6 * NS)
        for role, seconds in zip(HOLDERS, holders, strict=True):
            processes[role]["phase_timing"] = phase(online_calls=1, online_ns=seconds * NS)
    run = BenchmarkRun(run_id=name, status="completed", model_id="fss-unit", model_fingerprint="a" * 64,
        max_output_tokens=3, cold=True, warm=False,
        started_at_ns=NS, finished_at_ns=46 * NS, online_started_at_ns=5 * NS,
        first_token_at_ns=45 * NS, last_token_at_ns=45 * NS,
        input_tokens=2, output_tokens=3, token_usage_authoritative=True,
        privacy_delta=privacy, process_metrics=processes, output_delivery="buffered_final_ids")
    with closing(BenchmarkHistory(":memory:")) as history:
        history.add(run)
        return history.get(name)


def report(runs, warmups=()):
    return build_loopback_report(model_id="fss-unit", tiny=False, max_output_tokens=3,
        runs=runs, warmup_runs=list(warmups), roles=("client", *ROLES),
        runtime_contract={"runtime_kind": "fss", "output_delivery": "buffered_final_ids"},
        initial_preparation_audit=dict.fromkeys(counters(), 0), inventory_policy="none", temperature=0)


def test_serialized_phase_counters_reach_ordinary_report_without_relabeling_online():
    saved = saved_run()
    original = deepcopy(saved)
    result = report([saved])
    row, summary = result["runs"][0], result["summary"]
    timing = row["phase_timing"]
    assert result["checks"]["passed"]
    assert timing == result["topology_accounting"]["runs"][0]["phase_timing"]
    assert timing["preparation_seconds"] == 17
    assert timing["delivery_callback_seconds"] == 4 and timing["seal_callback_seconds"] == 6
    assert timing["preparation_outside_callbacks_seconds"] == 7
    assert timing["online_seconds_by_holder"] == {"worker_a": 11, "worker_b": 13}
    assert timing["online_seconds"] == summary["median_fss_max_holder_native_run_seconds"] == 13
    assert timing["online_scope"] == "maximum_holder_native_run_wall_not_client_online_interval"
    assert "not_pure_compute" in timing["outside_callbacks_scope"]
    assert summary["median_fss_dealer_preparation_seconds"] == 17
    assert summary["median_fss_delivery_callback_seconds"] == 4
    assert summary["median_fss_seal_callback_seconds"] == 6
    assert summary["median_fss_preparation_outside_callbacks_seconds"] == 7
    assert row["durations"]["online_seconds"] is summary["median_online_seconds"] is None
    assert row["durations"]["ttft_seconds"] is summary["median_ttft_seconds"] is None
    assert row["durations"]["final_disclosure_seconds"] == summary["median_final_disclosure_seconds"] == 40
    assert row["durations"]["full_seconds"] == summary["median_full_seconds"] == 45
    assert row["durations"]["preparation_seconds"] == saved["durations"]["preparation_seconds"]
    assert saved == original  # conversion never mutates history or its counters
    json.dumps(result, allow_nan=False)


def test_summary_uses_measured_requests_only_and_preserves_missing_cohort_members():
    first = saved_run("first")
    second = saved_run("second", preparation=21, holders=(17, 15))
    warmup = saved_run("warmup", preparation=30, holders=(2, 3))
    result = report([first, second], [warmup])
    assert result["summary"]["median_fss_dealer_preparation_seconds"] == 19
    assert result["summary"]["median_fss_max_holder_native_run_seconds"] == 15
    assert result["warmup_runs"][0]["phase_timing"]["preparation_seconds"] == 30
    assert result["warmup_runs"][0]["phase_timing"] == result["topology_accounting"]["warmups"][0]["phase_timing"]
    missing = report([first, saved_run("legacy", measured=False)])
    assert all(value is None for key, value in missing["summary"].items() if key.startswith("median_fss_"))
    assert missing["summary"]["median_final_disclosure_seconds"] == 40


@pytest.mark.parametrize("damage", ["absent", "partial", "multiple_calls", "zero_calls", "malformed"])
def test_legacy_or_unattributable_timing_stays_unknown_without_total_time_inference(damage):
    saved = saved_run(measured=damage != "absent")
    if damage == "partial": saved["processes"]["worker_b"].pop("phase_timing")
    elif damage == "multiple_calls": saved["processes"]["worker_b"]["phase_timing"]["online_calls"] = 2
    elif damage == "zero_calls": saved["processes"]["worker_b"]["phase_timing"] = phase()
    elif damage == "malformed": saved["processes"]["worker_b"]["phase_timing"]["online_ns"] = True
    result = report([saved])
    assert result["checks"]["passed"]  # missing timing is not missing privacy/body coverage
    assert result["runs"][0]["phase_timing"]["online_seconds"] is None
    assert result["summary"]["median_fss_max_holder_native_run_seconds"] is None
    assert result["summary"]["median_online_seconds"] is None
    assert result["summary"]["median_fss_dealer_preparation_seconds"] == (None if damage == "absent" else 17)
    assert result["summary"]["median_final_disclosure_seconds"] == 40
    assert result["summary"]["total_accounted_benchmark_body_bytes"] == report([saved_run()])["summary"]["total_accounted_benchmark_body_bytes"]


def test_measured_zero_is_not_confused_with_missing_native_run():
    saved = saved_run(holders=(0, 0))
    saved["processes"]["dealer"]["phase_timing"] = phase(preparation_calls=1)
    result = report([saved])
    assert all(value == 0 for key, value in result["summary"].items() if key.startswith("median_fss_"))
    assert result["summary"]["median_online_seconds"] is None
