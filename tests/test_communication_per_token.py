from __future__ import annotations

from copy import deepcopy

import pytest

from pllm.metrics import communication_per_token
from pllm.runtime.benchmark_cli import run_loopback_benchmark
from pllm.runtime.topology_accounting import online_body_counter_total


def _ledger(covered, online=None):
    return {
        "tracked_body_counter_set_present": True,
        "all_link_serialized_body_bytes": covered,
        "online_all_link_serialized_body_bytes": online,
    }


def _run(tokens, decode_bytes=None):
    return {
        "status": "completed",
        "tokens": {"authoritative": True, "input_tokens": 100, "output_tokens": tokens},
        "network": {
            "decode_scope": "after-first-output-through-completion",
            "decode_online_body_bytes": decode_bytes,
            "decode_output_tokens": tokens - 1,
        },
    }


def _report():
    return {
        "schema_version": "pllm.loopback_benchmark.v1",
        "runs": [_run(3, 2_000_000), _run(1, 0)],
        "warmup_runs": [_run(10)],
        "topology_accounting": {
            "startup": _ledger(1_000_000),
            "warmups": [_ledger(2_000_000)],
            "runs": [_ledger(4_000_000, 3_000_000), _ledger(3_000_000, 2_000_000)],
        },
    }


def test_weighted_denominators_charge_warmups_once_and_separate_decode():
    result = communication_per_token(_report())
    assert result["summary"] == {
        "generated_output_tokens": 4,
        "online_mb_per_output_token": 1.25,
        "covered_mb_per_output_token": 1.75,
        "setup_inclusive_mb_per_output_token": 2.5,
        "cold_first_response_mb_per_output_token": 0.3,
        "decode_online_mb_per_output_token": 1.0,
    }
    assert result["runs"][1]["decode_online_mb_per_output_token"] is None


@pytest.mark.parametrize("tokens", [0, None, True, 3.0])
def test_missing_or_zero_authoritative_denominator_is_not_an_output_cap(tokens):
    report = _report()
    report["runs"][0]["tokens"]["output_tokens"] = tokens
    result = communication_per_token(report)
    assert result["runs"][0]["online_mb_per_output_token"] is None
    if tokens != 0:
        assert result["summary"]["online_mb_per_output_token"] is None


def test_missing_bytes_failed_runs_and_unobserved_decode_remain_unknown():
    for mutation in ("missing", "failed", "unauthoritative", "decode"):
        report = deepcopy(_report())
        if mutation == "missing":
            report["topology_accounting"]["startup"] = None
            report["topology_accounting"]["runs"][0]["tracked_body_counter_set_present"] = False
        elif mutation == "failed":
            report["runs"][0]["status"] = "failed"
        elif mutation == "unauthoritative":
            report["runs"][0]["tokens"]["authoritative"] = False
        else:
            report["runs"][0].pop("network")
        result = communication_per_token(report)
        assert result["summary"]["decode_online_mb_per_output_token"] is None
        if mutation != "decode":
            assert result["summary"]["online_mb_per_output_token"] is None


def test_online_counter_snapshot_does_not_double_count_audit_aliases():
    counters = {
        "inference_upload_bytes": 10,
        "inference_download_bytes": 20,
        "masked_online_upload_bytes": 1000,
        "correction_push_bytes": 4000,
    }
    assert online_body_counter_total(counters, ("client", "preparation", "inference")) == 30
    assert online_body_counter_total(counters, ("client",)) is None
    assert online_body_counter_total(counters, ("client", "worker_a", "worker_b")) is None
    for role in ("worker_a", "worker_b"):
        for direction in ("upload", "download"):
            counters[f"role_link.{role}.online_{direction}_bytes"] = 5
    assert online_body_counter_total(counters, ("client", "worker_a", "worker_b")) == 20


def test_live_tiny_measures_decode_window_and_uses_actual_generation():
    report = run_loopback_benchmark(
        model="unused",
        model_id=None,
        timeout_seconds=60,
        tiny=True,
        warmups=0,
        repetitions=1,
        max_output_tokens=2,
        prompt="Say hello.",
        temperature=0.0,
    )
    assert report["checks"]["passed"]
    row = report["communication_per_token"]["runs"][0]
    assert row["generated_output_tokens"] == 2
    assert row["decode_output_tokens"] == 1
    assert 0 < row["decode_online_body_bytes"] < row["online_body_bytes"]
    assert report["summary"]["setup_inclusive_mb_per_output_token"] > 0
