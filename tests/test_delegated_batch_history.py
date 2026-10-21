"""Persist fake transport measurements; no SDK workers or benchmark execution."""

import threading
from dataclasses import replace
from types import SimpleNamespace

import pytest

from pllm.runtime import dashboard
from pllm.runtime.benchmark_history import BenchmarkHistory, BenchmarkRun, SCHEMA_VERSION
from pllm.runtime.delegated_client import _BATCH_TRANSPORT_FIELDS, _COUNTER_FIELDS, _DelegatedAudit
from pllm.runtime.delegated_transport import _Peers
from pllm.runtime.topology_accounting import measured_decode_window, two_worker_body_accounting


def snapshot(batch=True):
    peers = object.__new__(_Peers)
    peers.roles = ("worker_a", "worker_b")
    values = {"setup": (3, 4), "online": (10, 15), "teardown": (1, 2)}
    peers.counts = {f"role_link.{role}.{phase}_{direction}_bytes": sizes[index]
                   for role in peers.roles for phase, sizes in values.items()
                   for index, direction in enumerate(("upload", "download"))}
    peers.phase_bytes = {"prefill": 30, "decode": 20}
    peers.unknown = set()
    peers.calls, peers.server_ns = 4, 100
    peers.calls_known = peers.server_known = True
    peers.batch = {"descriptors": [{"rows": 2}, {"rows": 1}]} if batch else None
    peers.batch_index = peers.batch_exchanges = 2
    peers.scalar_claims = 3
    return peers.audit()


def run(privacy, run_id="batch_history"):
    return BenchmarkRun(run_id=run_id, status="completed", model_id="fake-gkm",
        max_output_tokens=2, cold=False, warm=True, started_at_ns=1_000_000_000,
        finished_at_ns=4_000_000_000, input_tokens=2, output_tokens=2,
        token_usage_authoritative=True, privacy_delta=privacy,
        output_delivery="buffered_until_verified")


@pytest.mark.parametrize("missing", [None, "batch_exchanges", "native_scalar_claims", "both"])
def test_batch_history_roundtrip_preserves_exact_and_absent_counters(tmp_path, missing):
    privacy = snapshot()
    if missing is not None:
        for name in (_BATCH_TRANSPORT_FIELDS if missing == "both" else (missing,)):
            privacy.pop(name)
    database = tmp_path / "history.sqlite3"
    history = BenchmarkHistory(database)
    try:
        saved = history.add(run(privacy))
        assert saved["privacy"] == privacy
        assert saved["schema_version"] == SCHEMA_VERSION == 3
        assert saved["durations"]["tokens_per_second"] is None
        assert saved["measurement"]["decode_tps_supported"] is False
        assert history.get("batch_history") == saved
        assert history.list() == [saved]
    finally:
        history.close()
    reopened = BenchmarkHistory(database)
    try:
        assert reopened.get("batch_history") == saved
    finally:
        reopened.close()


def test_existing_scalar_history_record_is_unchanged_when_batch_record_added(tmp_path):
    history = BenchmarkHistory(tmp_path / "history.sqlite3")
    try:
        scalar = history.add(run(snapshot(False), "scalar_history"))
        history.add(run(snapshot(), "batch_history"))
        assert history.get("scalar_history") == scalar
        assert scalar["privacy"]["online_steps"] == 2
        assert not _BATCH_TRANSPORT_FIELDS & scalar["privacy"].keys()
    finally:
        history.close()


@pytest.mark.parametrize("name", sorted(_BATCH_TRANSPORT_FIELDS))
@pytest.mark.parametrize("value", [True, -1, 1.0, None, 1 << 63])
def test_batch_counter_history_validation_keeps_transport_integer_bounds(name, value):
    with pytest.raises(ValueError, match="batch privacy deltas must be bounded"):
        run({name: value})


@pytest.mark.parametrize("name", sorted(_BATCH_TRANSPORT_FIELDS))
def test_largest_admitted_batch_counter_roundtrips_without_float_conversion(tmp_path, name):
    history = BenchmarkHistory(tmp_path / "history.sqlite3")
    try:
        saved = history.add(run({name: (1 << 63) - 1}))
        assert history.get("batch_history")["privacy"] == saved["privacy"] == {name: (1 << 63) - 1}
    finally:
        history.close()


@pytest.mark.parametrize("name", ["batch_tokens", "batch_exchanges.total", "native_scalar_claims_secret"])
def test_allowlist_remains_closed_for_unrecognized_counter_names(name):
    with pytest.raises(ValueError, match="unsupported privacy metrics"):
        run({name: 1})


def test_explanatory_batch_counts_do_not_add_bytes_or_invent_missing_links():
    privacy = {**snapshot(), "bundle_network_bytes": 0,
        "generation_completed_responses": 1, "generation_output_tokens": 2,
        "generation_decode_output_tokens": 1}
    record = run(privacy).to_dict()
    baseline = replace(run(privacy), privacy_delta={name: value for name, value in privacy.items()
                       if name not in _BATCH_TRANSPORT_FIELDS}).to_dict()
    ledger = two_worker_body_accounting(record)
    assert ledger == two_worker_body_accounting(baseline)
    assert ledger["online_all_link_serialized_body_bytes"] == 50
    assert ledger["all_link_serialized_body_bytes"] == sum(snapshot()[key]
        for key in snapshot() if key.startswith("role_link."))
    roles = ("client", "worker_a", "worker_b")
    assert measured_decode_window(record["privacy"], output_tokens=2, roles=roles) == (20, 1)
    record["privacy"].pop("role_link.worker_a.online_upload_bytes")
    assert two_worker_body_accounting(record)["all_link_serialized_body_bytes"] is None
    assert measured_decode_window(record["privacy"], output_tokens=2, roles=roles) is None


@pytest.mark.parametrize("batch", [False, True], ids=["scalar", "batch"])
def test_dashboard_finish_persists_window_deltas_and_buffered_unknowns(tmp_path, monkeypatch, batch):
    """Exercise the shared CLI/ordinary-target persistence path, not a job."""
    database = tmp_path / "dashboard.sqlite3"
    history = BenchmarkHistory(database)
    runtime = dashboard.DashboardRuntime(
        dashboard.DashboardConfig("127.0.0.1", 8791, None, "fake-gkm", 2, database),
        dashboard.OTelStore(), history)
    fields = _COUNTER_FIELDS | (_BATCH_TRANSPORT_FIELDS if batch else frozenset())
    audit = _DelegatedAudit(threading.RLock(), fields)
    runtime._client = SimpleNamespace(privacy_audit=audit)
    runtime._topology = SimpleNamespace(requires_preparation=False, started=True, closed=False,
        statuses=tuple(SimpleNamespace(role=role, running=True) for role in ("worker_a", "worker_b")),
        worker_process_metrics=lambda: {})
    runtime._state["output_delivery"] = "buffered_until_verified"
    monkeypatch.setattr(dashboard.time, "sleep", lambda _: None)
    monkeypatch.setattr(runtime, "_capture_online_telemetry", lambda: None)
    try:
        for index in range(2):
            capture = dashboard._RunCapture(run_id=f"window_{index}", max_output_tokens=2,
                cold=False, started_at_ns=1_000_000_000, audit_before=audit.to_dict(),
                started_monotonic_ns=1_000_000_000, protocol_start_cursor=0,
                online_started_at_ns=2_000_000_000, online_started_monotonic_ns=2_000_000_000,
                first_token_at_ns=3_000_000_000, first_token_monotonic_ns=3_000_000_000,
                last_token_at_ns=3_000_000_000, last_token_monotonic_ns=3_000_000_000,
                finished_at_ns=4_000_000_000, finished_monotonic_ns=4_000_000_000,
                input_tokens=2, output_tokens=2, token_usage_authoritative=True)
            audit.begin_transport()
            audit.finish_transport(snapshot(batch))
            audit.generation_completed_responses += 1
            audit.generation_output_tokens += 2
            audit.generation_decode_output_tokens += 1
            runtime._finish_run(capture, "completed", None)
            assert runtime._state["error"] is None
            saved = history.get(capture.run_id)
            assert saved is not None and saved["status"] == "completed"
            assert saved["privacy"]["inference_stage_calls"] == 4
            assert saved["privacy"]["online_steps"] == (3 if batch else 2)
            if batch:
                assert saved["privacy"]["batch_exchanges"] == 2
                assert saved["privacy"]["native_scalar_claims"] == 3
            else:
                assert not _BATCH_TRANSPORT_FIELDS & saved["privacy"].keys()
            assert saved["network"]["decode_online_body_bytes"] == 20
            assert saved["durations"]["tokens_per_second"] is None
            assert saved["measurement"]["decode_tps_supported"] is False
            assert two_worker_body_accounting(saved)["online_all_link_serialized_body_bytes"] == 50
    finally:
        history.close()
