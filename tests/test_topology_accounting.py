"""Lifecycle body coverage, without checkpoint loading or native execution."""
from __future__ import annotations

import copy

import httpx
import pytest

from pllm.metrics.communication import communication_per_token
from pllm.metrics.wan import _window
from pllm.deployment import WanConditions
from pllm.runtime.benchmark_cli import _comparison_key, accounted_benchmark_body_totals
from pllm.runtime.topology_accounting import (
    ACCOUNTING_SCHEMA, CONTROL_ACCOUNTING_SCHEMA, CONTROL_COVERAGE,
    accounting_coverage_identity, measured_decode_window, prepared_body_accounting,
)


def counters():
    return dict(session_authorization_upload_bytes=1, session_authorization_download_bytes=2,
                preparation_upload_bytes=3, preparation_download_bytes=4, correction_push_bytes=5,
                bundle_network_bytes=6, inference_upload_bytes=7, inference_download_bytes=8)


def controls():
    return dict(inference_control_upload_bytes=11, inference_control_download_bytes=13,
                preparation_control_upload_bytes=17, preparation_control_download_bytes=19)


def ledger(values):
    return prepared_body_accounting({"privacy": values})


def report(rows):
    run = {"status": "completed", "model_fingerprint": "same-body", "warm": False,
           "tokens": {"authoritative": True, "input_tokens": 4, "output_tokens": 2},
           "durations": {"ttft_seconds": 1, "tokens_per_second": 1},
           "max_output_tokens": 8}
    return {"schema_version": "pllm.loopback_benchmark.v1",
            "configuration": {"sampling": {"effective_temperature": 0, "mode": "greedy", "top_p": None}},
            "runs": [copy.deepcopy(run) for _ in rows[1:]], "warmup_runs": [],
            "topology_accounting": {"startup": rows[0], "warmups": [], "runs": rows[1:]}}


def test_historical_totals_scope_and_unknown_controls_preserved():
    row = ledger(counters())
    assert row["schema"] == ACCOUNTING_SCHEMA
    assert row["all_link_serialized_body_bytes"] == 36
    assert row["control_body_counter_set_present"] is None
    assert row["lifecycle_control_serialized_body_bytes"] is None
    assert row["scope"] == "recorded prepared-protocol serialized bodies and run-window CPU samples only"
    assert len(row["body_bytes_by_edge"]) == 6


def test_directed_lifecycle_controls_counted_once_arithmetic_unchanged():
    row = ledger({**counters(), **controls(), "masked_online_upload_bytes": 999999})
    assert row["schema"] == CONTROL_ACCOUNTING_SCHEMA
    assert row["body_counter_coverage"] == CONTROL_COVERAGE
    assert row["lifecycle_control_serialized_body_bytes"] == 60
    assert row["all_link_serialized_body_bytes"] == 96
    assert row["client_serialized_body_bytes"] == 91  # excludes preparation->inference correction
    assert row["online_all_link_serialized_body_bytes"] == 15
    directed = {(edge["source"], edge["destination"]): edge["serialized_body_bytes"]
                for edge in row["body_bytes_by_edge"] if edge["phase"] == "control"}
    assert directed == {("client", "inference"): 11, ("inference", "client"): 13,
                        ("client", "preparation"): 17, ("preparation", "client"): 19}
    assert sum(edge["serialized_body_bytes"] for edge in row["body_bytes_by_edge"]) == 96
    audit = {**counters(), **controls(), "generation_completed_responses": 1,
             "generation_output_tokens": 2, "generation_decode_output_tokens": 1,
             "generation_prefill_online_body_bytes": 9, "generation_decode_online_body_bytes": 6}
    assert measured_decode_window(audit, output_tokens=2, roles=("client", "preparation", "inference")) == (6, 1)


@pytest.mark.parametrize("missing", list(controls()))
def test_partial_control_keys_never_create_known_total(missing):
    values = {**counters(), **controls()}
    del values[missing]
    row = ledger(values)
    assert row["control_body_counter_set_present"] is False
    assert row["tracked_body_counter_set_present"] is False
    assert row["all_link_serialized_body_bytes"] is None
    assert row["lifecycle_control_serialized_body_bytes"] is None
    assert accounted_benchmark_body_totals(report([row, row])["topology_accounting"])[
        "total_accounted_benchmark_body_bytes"] is None


@pytest.mark.parametrize("invalid", [None, True, -1, 1.5])
def test_invalid_control_value_rejects_full_coverage(invalid):
    row = ledger({**counters(), **controls(), "inference_control_upload_bytes": invalid})
    assert row["all_link_serialized_body_bytes"] is None
    assert row["control_body_counter_set_present"] is False


def test_old_new_and_mixed_coverage_cannot_rank_as_identical():
    old, new = ledger(counters()), ledger({**counters(), **controls()})
    historical, current, mixed = report([old, old]), report([new, new]), report([old, new])
    assert _comparison_key(historical) != _comparison_key(current)
    assert _comparison_key(historical) is not None and _comparison_key(current) is not None
    assert _comparison_key(mixed) is None
    assert accounting_coverage_identity(mixed["topology_accounting"]) is None
    assert communication_per_token(mixed)["summary"]["setup_inclusive_mb_per_output_token"] is None
    assert accounted_benchmark_body_totals(mixed["topology_accounting"])[
        "total_accounted_benchmark_body_bytes"] is None
    assert communication_per_token(current)["body_counter_coverage"] == CONTROL_COVERAGE
    assert accounted_benchmark_body_totals(current["topology_accounting"])[
        "total_accounted_benchmark_body_bytes"] == 192


def test_wan_accepts_separate_control_phase_but_not_mixed_coverage():
    old, new = ledger(counters()), ledger({**counters(), **controls()})
    conditions = WanConditions()
    full = _window([new], conditions)
    assert full is not None and full["body_counter_coverage"] == CONTROL_COVERAGE
    assert _window([old, new], conditions) is None
    # Arithmetic-only online floor must not absorb lifecycle controls.
    assert _window([new], conditions, online=True)["wan_body_bytes"] == 15


def test_sdk_metrics_coverage_scope_separates_old_new_and_rejects_stale_mixed_total(tmp_path):
    from pllm._benchmark_worker import Measurement
    from pllm.metrics import Sampling
    from pllm._benchmark_evidence import metric_contract
    from pllm.runtime.benchmark_response_target import response_metrics
    old, new = ledger(counters()), ledger({**counters(), **controls()})
    captured = []
    for rows in ([old, old], [new, new], [old, new]):
        data = report(rows)
        data["summary"] = {"total_accounted_benchmark_body_bytes": 999}  # never trust stale mixed aggregate
        m = Measurement({"sampling": Sampling(processes=1).to_spec(), "salt": "ab" * 32,
                         "inputs": {}, "directory": str(tmp_path)})
        response_metrics(data, m)
        captured.append(next(metric for metric in m.metrics if metric["id"] == "network.covered_body_bytes"))
    assert metric_contract(captured[0]) != metric_contract(captured[1])
    assert captured[2]["value"] is None


def test_real_control_wrapper_setup_repeated_epochs_rejection_and_failed_upload():
    from pllm.runtime.client import PrivacyAudit, RuntimeClient
    core = object.__new__(RuntimeClient)
    core.audit = PrivacyAudit()
    seen = []
    def receive(request):
        seen.append((len(request.content), 5))
        return httpx.Response(409 if request.url.path.endswith("cancel") else 200,
                              content=b"reply", request=request)
    with httpx.Client(transport=httpx.MockTransport(receive), base_url="http://inference") as inference, \
            httpx.Client(transport=httpx.MockTransport(receive), base_url="http://preparation") as preparation:
        core.http, core.preparation_http = inference, preparation
        for _ in range(3):
            for path in ("/v1/runtime/inventories", "/v1/runtime/inventories/public-id/ready",
                         "/v1/runtime/sessions", "/v1/runtime/sessions/public-id/complete",
                         "/v1/runtime/sessions/public-id/cancel"):
                core._control_post(inference, path, content=b"setup")
        core._control_post(preparation, "/v1/preparation/inventories/public-id/cancel", content=b"cancel")
        before = core.audit.to_dict()
        assert before["inference_control_upload_bytes"] == 75
        assert before["inference_control_download_bytes"] == 75
        assert before["preparation_control_upload_bytes"] == 6
        assert before["preparation_control_download_bytes"] == 5
    def fail(request):
        raise httpx.ConnectError("synthetic failure", request=request)
    with httpx.Client(transport=httpx.MockTransport(fail), base_url="http://inference") as inference:
        core.http = inference
        with pytest.raises(httpx.HTTPError):
            core._control_post(inference, "/v1/runtime/sessions", content=b"attempt")
    after = core.audit.to_dict()
    assert after["inference_control_upload_bytes"] == 82
    assert after["inference_control_download_bytes"] == 75  # unknown reply not invented
    row = ledger({**counters(), **{key: after[key] for key in controls()}})
    assert row["lifecycle_control_serialized_body_bytes"] == sum(after[key] for key in controls())
    assert row["all_link_serialized_body_bytes"] == 36 + 168
    assert len(seen) == 16
