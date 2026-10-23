"""Source/transport doubles only: qualify issued bounded acceptance, never infer it.

All counts, native barriers, owners and memory in comparison fixtures are mock
data. Real SDK remains queued. Production gate must NOT accept reasonless legacy
'incomplete' by counting deltas, treating API status as protocol failure/success,
or adding an unissued accepted=True flag.
"""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import time
from types import SimpleNamespace as NS

import pytest


@pytest.fixture(autouse=True)
def mock_only(monkeypatch):
    import subprocess
    from pllm.runtime import client_process_usage as usage

    def forbidden(*args, **kwargs):
        raise AssertionError("bounded endpoint controls cannot launch children or observe OS birth")

    monkeypatch.setattr(subprocess, "Popen", forbidden)
    monkeypatch.setattr(usage, "_birth_identity", forbidden)
    monkeypatch.setattr(usage, "supported", lambda: True)


def _dashboard_terminal(monkeypatch, *, terminal="incomplete", reason="max_output_tokens",
                        object_response=False, capture_output_digest=True, after_terminal=None,
                        event_terminal=None):
    from pllm.runtime.dashboard import DashboardConfig, DashboardRuntime, OTelStore, _RunCapture

    runtime = DashboardRuntime(DashboardConfig("127.0.0.1", 8791, None, "mock", 8,
        temperature=0, capture_output_digest=capture_output_digest), OTelStore())
    capture = _RunCapture("mock", 8, True, time.time_ns(), {}, time.monotonic_ns(), 0)
    runtime._active_run = capture
    events = [{"type": "response.output_text.delta", "delta": "x"} for _ in range(8)]
    if terminal is not None:
        response = {"status": terminal,
            "incomplete_details": {"reason": reason} if reason is not None else None,
            "usage": {"input_tokens": 64, "output_tokens": 8},
            "output": [{"type": "message", "content": [{"type": "output_text", "text": "x" * 8}]}]}
        if object_response:
            response = NS(**{**response, "usage": NS(**response["usage"]), "output_text": "x" * 8})
        events.append({"type": f"response.{event_terminal or terminal}", "response": response})

    def stream():
        yield from events
        if after_terminal is not None:
            after_terminal(response)

    runtime._client = NS(responses=NS(create=lambda **kw: stream()))
    finished = []
    monkeypatch.setattr(runtime, "_finish_run", lambda c, status, error: finished.append((status, error)))
    # Failure observer is not under test; no telemetry/network/OS failure probes.
    monkeypatch.setattr("pllm.runtime.benchmark_failure.capture_failure", lambda *a, **kw: None)
    runtime._run_chat("mock private context", 8)
    return runtime, capture, finished


@pytest.mark.parametrize("reason", ["max_output_tokens", "content_filter", "unknown", None])
@pytest.mark.parametrize("object_response", [False, True])
@pytest.mark.parametrize("capture_output_digest", [False, True])
def test_actual_dashboard_copies_only_terminal_public_reason(monkeypatch, reason, object_response, capture_output_digest):
    runtime, capture, finished = _dashboard_terminal(monkeypatch, reason=reason,
        object_response=object_response, capture_output_digest=capture_output_digest)
    assert finished == [("completed", None)]
    assert capture.token_usage_authoritative is True
    assert (capture.input_tokens, capture.output_tokens) == (64, 8)
    expected = {"response_status": "incomplete"}
    if reason in {"max_output_tokens", "content_filter"}:
        expected["incomplete_details"] = {"reason": reason}
    if capture_output_digest:
        expected["output_text_digest"] = hashlib.sha256(json.dumps("x" * 8).encode()).hexdigest()
    assert runtime._state["generation"] == expected


def test_reason_is_snapshot_not_alias_or_late_callback(monkeypatch):
    runtime, _, _ = _dashboard_terminal(monkeypatch,
        after_terminal=lambda response: response["incomplete_details"].update(reason="content_filter"))
    assert runtime._state["generation"]["incomplete_details"] == {"reason": "max_output_tokens"}


def test_mismatched_terminal_event_cannot_issue_token_limit_reason(monkeypatch):
    runtime, _, _ = _dashboard_terminal(monkeypatch, event_terminal="completed")
    assert runtime._state["generation"]["response_status"] == "incomplete"  # Never rewrite status.
    assert "incomplete_details" not in runtime._state["generation"]


@pytest.mark.parametrize("terminal", [None, "cancelled", "failed"])
def test_eight_deltas_without_successful_terminal_event_do_not_authorize_counts(monkeypatch, terminal):
    _, capture, finished = _dashboard_terminal(monkeypatch, terminal=terminal)
    assert finished[0][0] == "failed" and finished[0][1] is not None
    assert capture.output_tokens == 8 and capture.token_usage_authoritative is False


@pytest.mark.parametrize("capture", [False, True])
def test_report_filter_preserves_reason_without_payload_alias_or_default_changes(capture):
    from pllm.runtime.benchmark_cli import _generation_metadata

    limited = {"response_status": "incomplete", "output_text_digest": "a" * 64,
        "incomplete_details": {"reason": "max_output_tokens"}}
    limited["incomplete_details"]["private_payload"] = "never published"
    observed = _generation_metadata(limited, capture_output_digest=capture)
    assert observed["incomplete_details"] == {"reason": "max_output_tokens"}
    limited["incomplete_details"]["reason"] = "content_filter"
    assert observed["incomplete_details"] == {"reason": "max_output_tokens"}
    assert _generation_metadata(limited, capture_output_digest=capture)["incomplete_details"] == {"reason": "content_filter"}
    assert _generation_metadata({"response_status": "completed"}, capture_output_digest=capture) == {"response_status": "completed"}
    assert _generation_metadata({"response_status": "incomplete"}, capture_output_digest=capture) == {"response_status": "incomplete"}


@pytest.mark.parametrize("details", [None, [], "max_output_tokens", {"reason": None},
    {"reason": True}, {"reason": ["max_output_tokens"]}, {"reason": "private-unknown"}])
def test_malformed_or_unknown_reason_is_not_published(details):
    from pllm.runtime.benchmark_cli import _generation_metadata

    assert _generation_metadata({"response_status": "incomplete", "incomplete_details": details},
        capture_output_digest=False) == {"response_status": "incomplete"}


def test_64_8_fixture_has_no_unrelated_owner_count_or_audit_blocker():
    from test_client_endpoint_memory_comparison import SELECT, run

    # Only a fixture sanity check, not proof native bounded output emits completed.
    assert run(input_tokens=64, output_tokens=8).compare("control", "candidate", SELECT)["status"] == "comparable_in_scope"


def _bounded_run(change=None, *, generation=None):
    from pllm.metrics import BenchmarkRun
    from pllm.runtime.benchmark_cli import _generation_metadata
    from test_client_endpoint_memory_comparison import run

    observed = run(input_tokens=64, output_tokens=8).to_dict()
    for index, row in enumerate(observed["results"]):
        measured, = row["execution"]["details"]["report"]["runs"]
        issued = generation or {**measured["generation"], "response_status": "incomplete",
            "incomplete_details": {"reason": "max_output_tokens"}}
        measured["generation"] = _generation_metadata(issued, capture_output_digest=True)
        measured["privacy"].update(generation_completed_responses=1,
            generation_output_tokens=8, generation_requested_output_tokens=8)
        if change is not None and index:
            change(row)
    return BenchmarkRun(observed)


def test_terminal_observation_through_normal_report_and_ordinary_comparison(monkeypatch):
    from pllm.evidence import BenchmarkResult
    from pllm.metrics import BenchmarkRun
    from test_client_endpoint_memory_comparison import SELECT

    runtime, _, _ = _dashboard_terminal(monkeypatch)
    observed = _bounded_run(generation=runtime._state["generation"])
    before = observed.to_dict()
    checked = observed.compare("control", "candidate", SELECT)
    assert checked["status"] == "comparable_in_scope"
    assert checked["window"] == "client_process_birth_through_report_endpoint"
    assert checked["whole_lifetime"] is False and checked["physical_peak"] is False
    assert checked["reduction_status"] == "not_established"
    assert observed.to_dict() == before  # Comparison never rewrites status/counts.
    # Existing published schema deliberately permits ordinary observations in
    # execution.details; reuse SDK incomplete_details, no new ABI or schema field.
    import jsonschema
    schema = json.loads((Path(__file__).resolve().parents[1] / "schemas/benchmark-result-v2.schema.json").read_text())
    for row in before["results"]:
        jsonschema.Draft202012Validator(schema).validate(row)
        assert BenchmarkResult.from_dict(row).to_dict() == row
    # Old stored records still validate; missing optional reason is unknown.
    legacy = deepcopy(before)
    for row in legacy["results"]:
        row["execution"]["details"]["report"]["runs"][0]["generation"].pop("incomplete_details")
        jsonschema.Draft202012Validator(schema).validate(row)
        assert BenchmarkResult.from_dict(row).to_dict() == row
    assert BenchmarkRun(legacy).compare("control", "candidate", SELECT)["status"] == "unranked"


@pytest.mark.parametrize("status,reason,barrier", [
    ("incomplete", "max_output_tokens", False),
    ("incomplete", "content_filter", True),
    ("incomplete", None, True),
    ("cancelled", "max_output_tokens", True),
    ("failed", "max_output_tokens", True),
])
def test_ordinary_64_8_api_rejects_nonaccepted_terminal_evidence(status, reason, barrier):
    from pllm.runtime.benchmark_cli import _generation_metadata
    from test_client_endpoint_memory_comparison import SELECT

    def change(row):
        report = row["execution"]["details"]["report"]
        measured, = report["runs"]
        measured["generation"] = _generation_metadata({**measured["generation"],
            "response_status": status, "incomplete_details": {"reason": reason}}, capture_output_digest=True)
        measured["privacy"].update(generation_completed_responses=int(barrier),
            generation_output_tokens=8 if barrier else 0, generation_requested_output_tokens=8 if barrier else 0)

    observed = _bounded_run(change=change)
    assert all(row["status"] == "completed" for row in observed.to_dict()["results"])
    assert observed.compare("control", "candidate", SELECT)["status"] == "unranked"
    # Missing optional qualification never changes default/string selection.
    assert observed.compare("control", "candidate", "client.peak_rss")["status"] == "comparable"


def _report(row):
    return row["execution"]["details"]["report"]


@pytest.mark.parametrize("change", [
    lambda row: _report(row)["runs"][0]["privacy"].pop("generation_completed_responses"),
    lambda row: _report(row)["runs"][0]["privacy"].update(generation_completed_responses=True),
    lambda row: _report(row)["runs"][0]["privacy"].update(generation_completed_responses=2),
    lambda row: _report(row)["runs"][0]["privacy"].update(generation_output_tokens=7),
    lambda row: _report(row)["runs"][0]["privacy"].update(generation_requested_output_tokens=9),
    lambda row: _report(row)["runs"][0]["privacy"].pop("generation_requested_output_tokens"),
    lambda row: _report(row)["runs"][0]["tokens"].update(output_tokens=7),
    lambda row: _report(row)["runs"][0]["tokens"].update(input_tokens=63),
    lambda row: _report(row)["runs"][0]["tokens"].update(authoritative=False),
    lambda row: _report(row)["runs"][0].update(max_output_tokens=9),
    lambda row: _report(row)["configuration"].update(max_output_tokens=7),
    lambda row: _report(row)["configuration"].update(max_output_tokens=8.),
    lambda row: _report(row)["checks"].update(expected_output_tokens=False),
    lambda row: _report(row)["checks"].pop("expected_input_tokens"),
    lambda row: _report(row)["checks"].update(all_runs_completed=False),
    lambda row: _report(row)["checks"].update(no_online_preparation_requests=False),
    lambda row: _report(row)["checks"].pop("no_plaintext_prompt_bytes_sent"),
    lambda row: _report(row)["checks"].update(passed=False),
    lambda row: _report(row)["runs"][0]["generation"].pop("incomplete_details"),
    lambda row: _report(row)["client_owned_memory"]["lifetime_owner"].update(root_birth_bound=False),
    lambda row: _report(row)["client_owned_memory"]["lifetime_owner"].update(membership_complete=False),
    lambda row: _report(row)["client_owned_memory"]["lifetime_owner"].update(reason="unretired_topology"),
    lambda row: _report(row)["client_owned_memory"]["lifetime_owner"]["members"].update(preparation={}),
    lambda row: _report(row)["client_owned_memory"]["individual_client_hwm"].update(source="wait4_family_maximum"),
    lambda row: row["execution"]["isolation"].update(fresh_process=False),
    lambda row: row["execution"]["details"].pop("response_state_audit"),
])
def test_bounded_path_keeps_barrier_counts_checks_and_owner_guards(change):
    from test_client_endpoint_memory_comparison import SELECT

    observed = _bounded_run(change=change)
    assert observed.compare("control", "candidate", SELECT)["status"] == "unranked"
    assert all(row["status"] == "completed" for row in observed.to_dict()["results"])
